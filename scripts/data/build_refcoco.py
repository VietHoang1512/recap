#!/usr/bin/env python3
"""Build the RefCOCO grounding replay set used in the hybrid setting.

Merges the REC jsonl files (refcoco / refcoco+ / refcocog train splits, as distributed with
VLM-R1's `rec_jsons_processed`) into a single arrow dataset. Images are referenced by path
rather than embedded, so --image-root must stay valid at training time.

    python scripts/data/build_refcoco.py \\
        --data-files rec_jsons_processed/refcoco_train.jsonl \\
                     rec_jsons_processed/refcocop_train.jsonl \\
                     rec_jsons_processed/refcocog_train.jsonl \\
        --image-root $DATA_ROOT/RefCOCO \\
        --output-dir $DATA_ROOT/RefCOCO
"""
import argparse
import json
import os
from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm

def load_with_hf(data_files):
    """
    Uses 🤗datasets.load_dataset to read and merge multiple JSONL files
    and returns a DatasetDict with a 'train' split.
    """
    # load_dataset without `split` returns a DatasetDict
    ds_dict = load_dataset(
        "json",
        data_files={"train": data_files},
        keep_in_memory=False,
        cache_dir=None
    )
    return ds_dict  # this is a DatasetDict with only the 'train' split

def load_manually(data_files, image_root):
    """
    Fallback: manually read each line and build a DatasetDict.from_dict
    """
    all_data = []
    for path in data_files:
        print(f"Reading {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Lines"):
                item = json.loads(line)
                # pull out the first conversation as the problem
                item['problem'] = item['conversations'][0]['value']
                # second conversation as solution, cast to str if needed
                sol = item['conversations'][1]['value']
                item['solution'] = sol if isinstance(sol, str) else str(sol)
                # store image path rather than opening
                item['image_path'] = os.path.join(image_root, item['image'])
                # drop the raw conversations field
                del item['conversations']
                del item['image']
                all_data.append(item)
    # wrap into a single‐split DatasetDict
    return DatasetDict({"train": Dataset.from_list(all_data)})

def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple JSONL data files into one HF DatasetDict and save to disk."
    )
    parser.add_argument(
        "--data-files", "--data_files",
        dest="data_files",
        nargs="+",
        required=True,
        help="Paths to your input JSONL files"
    )
    parser.add_argument(
        "--image-root", "--image_root",
        dest="image_root",
        default=os.path.join(os.environ.get("DATA_ROOT", "./share_data"), "RefCOCO"),
        help="Directory the per-example `image` field is relative to. Baked into image_path, "
             "so it must still resolve at training time."
    )
    parser.add_argument(
        "--output-dir", "--output_dir",
        dest="output_dir",
        required=True,
        help="Directory where the merged dataset will be saved"
    )
    parser.add_argument(
        "--method",
        choices=["hf", "manual"],
        default="manual",
        help="Whether to use 🤗datasets.load_dataset or manual JSONL parsing"
    )

    args = parser.parse_args()

    if args.method == "hf":
        dataset_dict = load_with_hf(args.data_files)
    else:
        dataset_dict = load_manually(args.data_files, args.image_root)

    # sanity check
    num_examples = len(dataset_dict["train"])
    print(f"→ Loaded {num_examples} examples in split 'train'; saving to {args.output_dir!r}")

    # write to disk in Arrow format
    dataset_dict.save_to_disk(args.output_dir)
    print("✅ Done.")

if __name__ == "__main__":
    main()
