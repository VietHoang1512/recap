#!/usr/bin/env python3
import os
import re
import json
import random
import warnings
import argparse

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

DATA_ROOT = os.environ.get("DATA_ROOT", "./share_data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "./outputs")

def setup_distributed():
    """Initialize torch.distributed from environment (torchrun)."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # Training/inference PG stays on NCCL
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # NEW: Create a CPU (Gloo) group for object collectives to avoid CUDA allocs in NCCL
    try:
        cpu_pg = dist.new_group(backend="gloo")
    except Exception:
        cpu_pg = None  # Fallback: will use default PG if Gloo isn't available
    return local_rank, world_size, rank, cpu_pg

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL eval with bbox extraction")

    # Paths & run info
    parser.add_argument("--model-path", type=str, required=True,
                        help="HF model id or a local checkpoint directory")
    parser.add_argument("--data-root", type=str,
                        default=os.path.join(DATA_ROOT, "rec_jsons_processed"),
                        help="directory holding <dataset>.json REC files")
    parser.add_argument("--image-root", type=str,
                        default=os.path.join(DATA_ROOT, "lisa-test"),
                        help="directory the per-example image paths are relative to")
    parser.add_argument("--test-datasets", type=str, nargs="+", default=["lisa_test"],
                        help="One or more dataset names (expects <name>.json inside --data-root)")
    parser.add_argument("--output-path", type=str,
                        default=None,
                        help=("If ends with .json: save to that file (multi-dataset: suffix dataset name). "
                              "If a directory or None: save <dir>/<dataset>.json. "
                              "Default None → ./outputs/<model-leaf>/<dataset>.json"))

    # Inference / generation
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=64,
                        help="-1 means use all samples in dataset JSON")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--iou-threshold", type=float, default=0.5)

    # Model loading knobs
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--flash-attn", action="store_true", default=True)
    parser.add_argument("--no-flash-attn", action="store_true", default=False)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--main-rank", type=int, default=0)

    # Prompt template
    parser.add_argument("--question-template", type=str, default=(
        "{Question}. Output the final answer in <answer> </answer> tags. "
        "The output answer format should be as follows:\n"
        "<answer>[x1, y1, x2, y2]</answer>\n"
        "Please strictly follow the format."
    ))

    return parser.parse_args()

def map_dtype(s):
    if s == "bf16":
        return torch.bfloat16
    if s == "fp16":
        return torch.float16
    return torch.float32

def extract_bbox_answer(content: str):
    # Find within <answer>...</answer>, then parse [x1,y1,x2,y2] inside dict-ish text
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'

    # bbox_pattern = r'\[\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*' \
    #             r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*' \
    #             r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*' \
    #             r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\]'    
    
    m = re.search(answer_tag_pattern, content, re.DOTALL)
    if m:
        inner = m.group(1).strip()
        b = re.search(bbox_pattern, inner, re.DOTALL)
        if b:
            return [int(b.group(1)), int(b.group(2)), int(b.group(3)), int(b.group(4))]
    else:
        inner = content.strip()
        b = re.search(bbox_pattern, inner, re.DOTALL)
        if b:
            return [int(b.group(1)), int(b.group(2)), int(b.group(3)), int(b.group(4))]        
    return [0, 0, 0, 0]

def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2] - 1, box2[2] - 1)
    inter_y2 = min(box1[3] - 1, box2[3] - 1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = ((box1[2] - box1[0]) * (box1[3] - box1[1])
             + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter)
    return float(inter) / union if union > 0 else 0.0

def resize_bbox(bbox, input_height, input_width, image_height, image_width):
    # In-place normalization to original image size
    bbox[0] = bbox[0] / input_width * image_width
    bbox[1] = bbox[1] / input_height * image_height
    bbox[2] = bbox[2] / input_width * image_width
    bbox[3] = bbox[3] / input_height * image_height
    return bbox

def derive_output_path(args, model_leaf, ds):
    """Decide final JSON path per dataset."""
    if args.output_path is None:
        base = os.path.join(OUTPUT_ROOT, model_leaf)
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, f"{ds}.json")

    # If user passed a directory
    if not args.output_path.endswith(".json"):
        os.makedirs(args.output_path, exist_ok=True)
        return os.path.join(args.output_path, f"{ds}.json")

    # If user passed a single .json file
    if len(args.test_datasets) == 1:
        return args.output_path
    root, ext = os.path.splitext(args.output_path)
    return f"{root}_{ds}{ext}"

def main():
    args = parse_args()
    print(args)
    # Distributed setup
    local_rank, world_size, rank, cpu_pg = setup_distributed()
    device = f"cuda:{local_rank}"
    if rank == args.main_rank:
        print(f"Process {rank}/{world_size} using {device}")

    # Model + processor
    use_flash = args.flash_attn and not args.no_flash_attn
    dtype = map_dtype(args.dtype)
    attn_impl = "flash_attention_2" if use_flash else "eager"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map={"": local_rank},
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Repro
    random.seed(args.seed)

    model_leaf = args.model_path.strip("/").split("/")[-1]

    for ds in args.test_datasets:
        if rank == args.main_rank:
            print(f"\nProcessing dataset: {ds} ...")

        ds_path = os.path.join(args.data_root, f"{ds}.json")
        with open(ds_path, "r") as f:
            data = json.load(f)

        # Shuffle and sample
        random.shuffle(data)
        if args.num_samples is not None and args.num_samples > 0:
            data = data[:args.num_samples]

        # Split per rank
        per_rank = len(data) // world_size
        start_idx = rank * per_rank
        end_idx = start_idx + per_rank if rank < world_size - 1 else len(data)
        rank_data = data[start_idx:end_idx]

        # Build chat messages (load images fully and close files immediately)
        messages = []
        for x in rank_data:
            image_path = os.path.join(args.image_root, x["image"])
            # Load into memory and close file handle to be safe

            message = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.open(image_path)},
                    {"type": "text", "text": args.question_template.format(Question=x["problem"])},
                ],
            }]
            messages.append(message)

        # Inference
        rank_outputs = []
        rng = range(0, len(messages), args.bsz)
        for i in tqdm(rng, disable=(rank != args.main_rank)):
            batch_messages = messages[i:i + args.bsz]
            text = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                    for m in batch_messages]

            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            ).to(device)

            generated_ids = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
            )
            # Trim prompt
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            batch_output = []
            for j, output_text in enumerate(batch_output_text):
                # Qwen-VL packs an image grid; 14 is patch size → reconstruct input resolution
                input_h = int(inputs["image_grid_thw"][j][1] * 14)
                input_w = int(inputs["image_grid_thw"][j][2] * 14)
                im = batch_messages[j][0]["content"][0]["image"]
                img_w, img_h = im.size
                batch_output.append((output_text, input_h, input_w, img_h, img_w))
                print("output_text", output_text)
            rank_outputs.extend(batch_output)

            # NEW: aggressively free GPU tensors between batches to keep NCCL happy
            del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs
            torch.cuda.synchronize(local_rank)
            torch.cuda.empty_cache()

        if rank == args.main_rank:
            print(f"Rank {rank} finished {len(rank_outputs)} examples for {ds}")

        # Gather results from all ranks (on CPU/Gloo to avoid tiny CUDA allocs in NCCL)
        local_pairs = [(start_idx + i, out) for i, out in enumerate(rank_outputs)]

        # Free VRAM before collective just in case
        torch.cuda.synchronize(local_rank)
        torch.cuda.empty_cache()

        gathered = [None] * world_size
        if cpu_pg is not None:
            dist.all_gather_object(gathered, local_pairs, group=cpu_pg)
        else:
            # Fallback if Gloo isn't available
            dist.all_gather_object(gathered, local_pairs)

        # Main rank aggregates and evaluates
        if rank == args.main_rank:
            all_outputs = [None] * len(data)
            for per_rank_list in gathered:
                for idx, output in per_rank_list:
                    all_outputs[idx] = output
            assert all(o is not None for o in all_outputs), "Some outputs are missing after gather."

            final_output = []
            correct = 0
            total_iou = 0
            for sample, model_output in zip(data, all_outputs):
                original_output, in_h, in_w, img_h, img_w = model_output
                gt = sample["solution"]
                pred = extract_bbox_answer(original_output)
                resized = resize_bbox(pred, in_h, in_w, img_h, img_w)

                iou_ = iou(resized, gt)
                ok = 1 if iou_ > args.iou_threshold else 0
                correct += ok
                total_iou += iou_

                final_output.append({
                    "image": sample["image"],
                    "question": sample["problem"],
                    "ground_truth": gt,
                    "model_output": original_output,
                    "input_size": (in_h, in_w),
                    "image_size": (img_h, img_w),
                    "extracted_answer": resized,
                    "iou": iou_,
                    "correct": ok
                })

            acc = correct / len(data) * 100 if len(data) else 0.0
            print(f"\nAccuracy of {ds}: {acc:.2f}%")
            mean_iou = total_iou / len(data) * 100 if len(data) else 0.0
            print(f"\nIOU of {ds}: {mean_iou:.2f}%")
            out_path = derive_output_path(args, model_leaf, ds)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({"accuracy": acc, "results": final_output, "iou": mean_iou}, f, indent=2)
            print(f"Results saved to {out_path}")
            print("-" * 100)

        # Sync between datasets
        dist.barrier()

if __name__ == "__main__":
    main()