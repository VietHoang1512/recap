#!/usr/bin/env python3
import os
import json
import random
import argparse
import warnings
from dataclasses import dataclass
from typing import List, Tuple
from tqdm.auto import tqdm
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "./outputs")


# -----------------------------
# Distributed utils
# -----------------------------
def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # try:
    #     cpu_pg = dist.new_group(backend="gloo")
    # except Exception:
    #     cpu_pg = None
    cpu_pg = None
        
    return local_rank, world_size, rank, cpu_pg


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Measure KL(q||p) on lmms-lab/COCO-Caption2017")

    # Models
    p.add_argument("--model-path", type=str, required=True,
                   help="Path or HF id for your FINETUNED Qwen2.5-VL model (policy q).")
    p.add_argument("--base-model-path", type=str,
                   default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="Path or HF id for BASE/REFERENCE Qwen2.5-VL model (policy p).")
    p.add_argument("--base-on-cpu", action="store_true", default=False,
                   help="Load the base/reference model on CPU to save VRAM (slower).")

    # Dataset
    p.add_argument("--dataset-name", type=str, default="lmms-lab/COCO-Caption2017")
    p.add_argument("--dataset-split", type=str, default="val",
                   help="Split name in the dataset (e.g., train/validation/test).")
    p.add_argument("--question-fallback", type=str, default="Describe this image.",
                   help="If a row has empty question, use this fallback.")
    p.add_argument("--num-samples", type=int, default=2048,
                   help="-1 for all available samples in split.")

    # Decoding
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--do-sample", action="store_true", default=False)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)

    # Inference / dtype / attention
    p.add_argument("--bsz", type=int, default=2)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--flash-attn", action="store_true", default=True)
    p.add_argument("--no-flash-attn", action="store_true", default=False)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--main-rank", type=int, default=0)
    p.add_argument("--output-path", type=str, default=None,
                   help="Where to write results JSON. Default: ./outputs/<model-leaf>/kl_coco.json")

    return p.parse_args()


def map_dtype(s):
    if s == "bf16": return torch.bfloat16
    if s == "fp16": return torch.float16
    return torch.float32


# -----------------------------
# KL helpers
# -----------------------------
@dataclass
class BatchKL:
    kl_sums: torch.Tensor      # (B,) sum_t [log q - log p]
    token_counts: torch.Tensor # (B,) number of generated tokens considered


def compute_path_kl_forced(
    model_q: Qwen2_5_VLForConditionalGeneration,
    model_p: Qwen2_5_VLForConditionalGeneration,
    processor,
    prompt_inputs: dict,
    generated_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    device_q: torch.device,
    device_p: torch.device,
) -> BatchKL:
    """
    Teacher-force both models on prompt+generated, then collect per-token logprobs
    for the generated region to compute path-KL: sum_t [log q - log p].
    """
    # Build TF batch inputs: copy vision inputs from prompt_inputs, but swap in full sequences
    tf_inputs = {
        k: v for k, v in prompt_inputs.items()
        if k not in ["input_ids", "attention_mask"]
    }
    tf_inputs["input_ids"] = generated_ids
    tf_inputs["attention_mask"] = torch.ones_like(generated_ids, dtype=torch.long, device=generated_ids.device)

    with torch.no_grad():
        # --- q logits
        out_q = model_q(**tf_inputs, use_cache=False)
        logits_q = out_q.logits.to(torch.float32)  # (B, L, V)
        logprobs_q = torch.log_softmax(logits_q, dim=-1)

        # --- p logits (may be on CPU)
        # Move TF inputs to p's device
        tf_inputs_p = {k: (v.to(device_p) if torch.is_tensor(v) else v) for k, v in tf_inputs.items()}
        out_p = model_p(**tf_inputs_p, use_cache=False)
        logits_p = out_p.logits.to(torch.float32)  # (B, L, V)
        logprobs_p = torch.log_softmax(logits_p, dim=-1)

    # Gather next-token logprobs for the realized tokens
    # Shift to align logits with next-token targets
    targets = generated_ids[:, 1:]  # (B, L-1)
    lp_q = logprobs_q[:, :-1, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
    lp_p = logprobs_p[:, :-1, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

    # Mask to only include generated region (exclude prompt)
    B, Lm1 = lp_q.shape
    pos = torch.arange(Lm1, device=lp_q.device).unsqueeze(0).expand(B, -1)  # 0..L-2
    # For each sample i: include j where j >= prompt_len_i-1
    mask = pos >= (prompt_lens - 1).unsqueeze(1)
    # Count only positions up to the actual sequence length minus 1 (everything is valid in generate())
    # If you'd like to stop at first EOS, add an EOS-based cut here.

    diff = (lp_p - lp_q) * mask  # (B, L-1)
    kl_sums = diff.sum(dim=1)              # (B,)
    token_counts = mask.sum(dim=1)         # (B,)
    return BatchKL(kl_sums=kl_sums, token_counts=token_counts)


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    print("args", args)
    # Dist
    local_rank, world_size, rank, cpu_pg = setup_distributed()
    device_q = torch.device(f"cuda:{local_rank}")
    device_p = torch.device("cpu") if args.base_on_cpu else device_q

    if rank == args.main_rank:
        print(f"Process {rank}/{world_size} using device_q={device_q}, device_p={device_p}")

    # Dtype/attn
    use_flash = args.flash_attn and not args.no_flash_attn
    dtype = map_dtype(args.dtype)
    attn_impl = "flash_attention_2" if use_flash else "eager"



    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset
    if rank == args.main_rank:
        print(f"Loading dataset: {args.dataset-name if hasattr(args, 'dataset-name') else args.dataset_name} [{args.dataset_split}]")

    ds = load_dataset(args.dataset_name, args.dataset_split, split="train", trust_remote_code=True)

    # Subsample / shuffle deterministically
    all_idx = list(range(len(ds)))
    random.shuffle(all_idx)
    if args.num_samples is not None and args.num_samples > 0:
        if args.num_samples > 0:
            all_idx = all_idx[:args.num_samples]
    # Shard by rank
    per_rank = len(all_idx) // world_size
    start = rank * per_rank
    end = start + per_rank if rank < world_size - 1 else len(all_idx)
    idx_this_rank = all_idx[start:end]

    if rank == args.main_rank:
        print(f"Total {len(all_idx)} samples → per-rank ~{per_rank}, this rank has {len(idx_this_rank)}.")



    # Models + processor
    if rank == args.main_rank:
        print("Loading models...")

    model_q = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map={"": local_rank},
    )
    # Use the *fine-tuned* processor to ensure exact same tokenizer/vocab
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Base/reference p
    if args.base_on_cpu:
        model_p = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.float32 if device_p.type == "cpu" else dtype,
            attn_implementation="eager" if device_p.type == "cpu" else attn_impl,
            device_map=None if device_p.type == "cpu" else {"": local_rank},
        ).to(device_p)
    else:
        model_p = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model_path,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            device_map={"": local_rank},
        )

    # Batched loop
    pad_id = processor.tokenizer.pad_token_id
    results_local = []
    total_kl_sum = 0.0
    total_tokens = 0


    # Decide output path
    model_leaf = args.model_path.strip("/").split("/")[-1]
    if args.output_path is None:
        base = os.path.join(OUTPUT_ROOT, model_leaf)
        os.makedirs(base, exist_ok=True)
        out_path = os.path.join(base, "kl_coco.json")
    else:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        out_path = args.output_path
    print("out_path", out_path)

    # Build messages batch-by-batch
    for i in tqdm(range(0, len(idx_this_rank), args.bsz)):
        batch_idx = idx_this_rank[i:i + args.bsz]
        rows = [ds[j] for j in batch_idx]

        # Prepare chat messages with image + question (fallback if question empty)
        messages = []
        for r in rows:
            img = r["image"]  # PIL.Image.Image (feature 'Image')
            # q = r["question"] 
            q = r["conversations"][0]["value"].replace("<image>", "")
            
            w, h = img.size
            longer = max(w, h)
            scale = 1.0
            new_w=w
            new_h=h
            MAX_SIDE=1024
            if longer > MAX_SIDE:
                scale = MAX_SIDE / float(longer)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                # LANCZOS = high-quality downsampling
                img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)            
            
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": q},
                ],
            }])
        # print("messages"  , messages)  

        # Tokenize prompt
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        prompt_inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(device_q)

        # Store prompt lengths per example (count non-pad in input_ids)
        with torch.no_grad():
            prompt_lens = (prompt_inputs["input_ids"] != pad_id).sum(dim=1)

        # Generate with q
        with torch.no_grad():
            generated = model_p.generate(
                **prompt_inputs,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        # generated: (B, L_total)
        # Compute KL on teacher-forced prompt+generated
        # Make sure generated lives on device_q
        generated = generated.to(device_q)

        batch_kl = compute_path_kl_forced(
            model_q=model_q, model_p=model_p, processor=processor,
            prompt_inputs=prompt_inputs, generated_ids=generated,
            prompt_lens=prompt_lens, device_q=device_q, device_p=device_p
        )

        # Aggregate
        total_kl_sum += batch_kl.kl_sums.sum().item()
        total_tokens += batch_kl.token_counts.sum().item()

        # Minimal per-sample logging (optional: also save prompt/question)
        for j, ridx in enumerate(batch_idx):
            results_local.append({
                "row_index": int(ridx),
                "kl_sum": float(batch_kl.kl_sums[j].item()),
                "gen_tokens": int(batch_kl.token_counts[j].item()),
                "kl_per_token_nats": float(
                    (batch_kl.kl_sums[j] / max(1, batch_kl.token_counts[j])).item()
                ),
            })
            # print("results_local[-1]", results_local[-1])
        # Free between batches
        del prompt_inputs, generated
        torch.cuda.synchronize(device_q)
        torch.cuda.empty_cache()

    # Reduce across ranks
    # Pack local aggregates
    local_summary = {
        "kl_sum": total_kl_sum,
        "tok_count": total_tokens,
        "num_items": len(results_local),
        "rank": rank,
    }

    gathered = [None] * world_size
    if cpu_pg is not None:
        dist.all_gather_object(gathered, local_summary, group=cpu_pg)
    else:
        dist.all_gather_object(gathered, local_summary)

    # Root aggregates & write file
    if rank == args.main_rank:
        grand_kl_sum = sum(g["kl_sum"] for g in gathered)
        grand_tok = sum(g["tok_count"] for g in gathered)
        kl_per_tok_nats = grand_kl_sum / max(1, grand_tok)
        kl_per_tok_bits = kl_per_tok_nats / torch.log(torch.tensor(2.0)).item()

        print("\n===== KL(q||p) results on lmms-lab/COCO-Caption2017 =====")
        print(f"Total generated tokens: {grand_tok}")
        print(f"KL per token: {kl_per_tok_nats:.6f} nats/token  |  {kl_per_tok_bits:.6f} bits/token")

        # Gather per-example (optional, could be large)
        # gathered_lists = [None] * world_size
        # if cpu_pg is not None:
        #     dist.all_gather_object(gathered_lists, results_local, group=cpu_pg)
        # else:
        #     dist.all_gather_object(gathered_lists, results_local)
        # all_results = []
        # for lst in gathered_lists:
        #     all_results.extend(lst)


        
        with open(out_path, "w") as f:
            json.dump({
                "dataset": f"{args.dataset_name}:{args.dataset_split}",
                "model_q": args.model_path,
                "model_p": args.base_model_path,
                "do_sample": args.do_sample,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "kl_per_token_nats": kl_per_tok_nats,
                "kl_per_token_bits": kl_per_tok_bits,
                "total_tokens": grand_tok,
                # "results": all_results
            }, f, indent=2)
        print(f"Results saved to {out_path}")
        print("=========================================================")

    dist.barrier()


if __name__ == "__main__":
    main()