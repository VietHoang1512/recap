#!/usr/bin/env python3
"""Download the Hub-hosted datasets RECAP trains on and save them under $DATA_ROOT.

The names on the left match the keys in src/open_r1/dataset_info.json, so once this has run the
generated configs resolve without further edits.

    DATA_ROOT=/path/to/share_data python scripts/prepare_data.py            # everything
    python scripts/prepare_data.py --only sat-problems-dataset ViRL39K      # a subset
    python scripts/prepare_data.py --list

Set HF_TOKEN in your environment first if any of these are gated.

Two datasets are NOT downloadable and must be built locally -- see scripts/data/:
  LLaVA-OneVision-OCR-10k-128   scripts/data/build_ocr_replay.py
  RefCOCO                       scripts/data/build_refcoco.py
"""

import argparse
import os
import sys

from datasets import load_dataset

# dataset_info.json key (minus the `share_data/` prefix)  ->  Hub repo id
DATASETS = {
    # RLVR-only setting
    "lisa-problems-dataset": "yiqingliang/lisa-problems-dataset",
    "lisa-problems-dataset-test": "yiqingliang/lisa-problems-dataset-test",
    "sat-problems-dataset": "yiqingliang/sat-problems-dataset",
    "sat-problems-dataset-test": "yiqingliang/sat-problems-dataset-test",
    "sat-problems-dataset-mini": "yiqingliang/sat-problems-dataset-mini",
    "scienceqa-problems-dataset": "yiqingliang/scienceqa-problems-dataset",
    "scienceqa-problems-dataset-test": "yiqingliang/scienceqa-problems-dataset-test",
    "geoqav-problems-dataset": "yiqingliang/geoqav-problems-dataset",
    "ViRFT_COCO": "laolao77/ViRFT_COCO",
    # Hybrid setting
    "ThinkLite-VL-70k": "russwang/ThinkLite-VL-70k",
    "ThinkLite-VL-hard-11k": "russwang/ThinkLite-VL-hard-11k",
    "ViRL39K": "TIGER-Lab/ViRL39K",
    "RLAIF-V-Dataset": "openbmb/RLAIF-V-Dataset",
}

# Referenced by dataset_info.json but derived locally rather than pulled from the Hub.
LOCALLY_BUILT = {
    "LLaVA-OneVision-OCR-10k-128": "scripts/data/build_ocr_replay.py",
    "LLaVA-OneVision-OCR-5k": "scripts/data/build_ocr_replay.py --max-samples 5000 "
                              "--name LLaVA-OneVision-OCR-5k",
    "RefCOCO": "scripts/data/build_refcoco.py",
    "lisa-rescale-train": "lisa-problems-dataset with boxes rescaled to the resized image",
}

# dataset_info.json carries a few further entries that no config here uses. The OCR variants
# come from build_ocr_replay.py with a different --max-samples/--max-size/--name.


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "./share_data"),
                    help="where to save (default: $DATA_ROOT or ./share_data)")
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="download just these (default: all)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-download even if the target directory already exists")
    ap.add_argument("--list", action="store_true", help="print what is available and exit")
    args = ap.parse_args(argv)

    if args.list:
        print("From the Hub:")
        for name, repo in DATASETS.items():
            print(f"  {name:35} {repo}")
        print("\nBuilt locally:")
        for name, how in LOCALLY_BUILT.items():
            print(f"  {name:35} {how}")
        return 0

    selected = args.only or list(DATASETS)
    unknown = [n for n in selected if n not in DATASETS]
    if unknown:
        hints = [f"{n} (build locally: {LOCALLY_BUILT[n]})"
                 for n in unknown if n in LOCALLY_BUILT]
        print(f"unknown dataset(s): {', '.join(unknown)}", file=sys.stderr)
        for h in hints:
            print(f"  {h}", file=sys.stderr)
        return 1

    os.makedirs(args.data_root, exist_ok=True)
    for name in selected:
        repo_id = DATASETS[name]
        dest = os.path.join(args.data_root, name)
        if os.path.isdir(dest) and not args.overwrite:
            print(f"skip   {name}  (already at {dest})")
            continue
        print(f"fetch  {name}  <- {repo_id}")
        dataset = load_dataset(repo_id)          # picks up HF_TOKEN from the environment
        dataset.save_to_disk(dest)
        print(f"       saved to {dest}  ({ {k: len(v) for k, v in dataset.items()} })")

    print(f"\ndone. Point DATA_ROOT at {os.path.abspath(args.data_root)} when training.")
    if not args.only:
        print("Still to build locally:")
        for name, how in LOCALLY_BUILT.items():
            print(f"  {name:35} {how}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
