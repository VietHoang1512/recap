"""Build the LLaVA-OneVision OCR replay set used in the hybrid setting.

Concatenates the OCR configs of lmms-lab/LLaVA-OneVision-Data, caps each at --max-samples,
downsamples images to --max-size on the long edge (these are replay examples, so they do not
need full resolution), and saves to $DATA_ROOT/<name>.

    python scripts/data/build_ocr_replay.py                      # -> LLaVA-OneVision-OCR-10k-128
    python scripts/data/build_ocr_replay.py --max-samples 5000 --name LLaVA-OneVision-OCR-5k
"""

import argparse
import os

from datasets import (
    get_dataset_config_names,
    load_dataset,
    concatenate_datasets,
    DatasetDict,
)
from PIL import Image as PILImage

OCR_CONFIGS = [
    "iiit5k", "hme100k", "iam(cauldron)", "chrome_writting",
    "textcaps", "textocr(gpt4v)", "rendered_text(cauldron)", "k12_printing",
]


def resize_img(example, max_size=256, col="image"):
    img = example[col]           # PIL.Image
    w, h = img.size
    if max(w, h) <= max_size:    # already small enough
        return example
    scale = max_size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    example[col] = img.resize((new_w, new_h), resample=PILImage.BICUBIC)
    return example


def _mean_hw(ds, col="image"):
    h_sum = w_sum = 0
    n = 0
    for ex in ds:
        img = ex[col]
        w, h = img.size
        h_sum += h
        w_sum += w
        n += 1
    return (h_sum / n, w_sum / n, n)


def load_ocr_subsets_as_dict(
    dataset_name: str = "lmms-lab/LLaVA-OneVision-Data",
    split: str = "train",
    max_samples: int = 10000,
    seed: int = 42,
    max_size: int = 128,
    num_proc: int = 16,
) -> DatasetDict:
    all_configs = get_dataset_config_names(dataset_name)

    ocr_configs = [cfg for cfg in all_configs if cfg.lower() in OCR_CONFIGS]
    if not ocr_configs:
        raise ValueError("No OCR-related configs found for dataset.")

    ocr_datasets = []
    for cfg in ocr_configs:
        print(f"Loading config: {cfg}")
        ds = load_dataset(dataset_name, cfg, split=split)

        if len(ds) > max_samples:
            print(f"  → {len(ds):,} samples → sampling {max_samples:,}")
            ds = ds.shuffle(seed=seed).select(range(max_samples))
        else:
            print(f"  → {len(ds):,} samples (kept all)")

        ds = ds.map(
            resize_img,
            fn_kwargs={"max_size": max_size, "col": "image"},
            num_proc=num_proc,
            desc=f"Resizing to ≤{max_size}",
        )

        mean_h, mean_w, n = _mean_hw(ds, col="image")
        print(f"  mean image size (H×W) over {n:,} samples: {mean_h:.1f} × {mean_w:.1f}")

        ocr_datasets.append(ds)

    combined = concatenate_datasets(ocr_datasets)

    overall_h, overall_w, n_all = _mean_hw(combined, col="image")
    print(f"\nOverall mean image size (H×W) across ALL configs [{n_all:,}]: {overall_h:.1f} × {overall_w:.1f}")

    return DatasetDict({"train": combined})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-samples", type=int, default=10000,
                    help="cap per OCR config (default: 10000)")
    ap.add_argument("--max-size", type=int, default=128,
                    help="long-edge image size after downsampling (default: 128)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-proc", type=int, default=16)
    ap.add_argument("--name", default="LLaVA-OneVision-OCR-10k-128",
                    help="output name under $DATA_ROOT; must match dataset_info.json")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "./share_data"))
    args = ap.parse_args()

    ds_dict = load_ocr_subsets_as_dict(
        max_samples=args.max_samples,
        seed=args.seed,
        max_size=args.max_size,
        num_proc=args.num_proc,
    )
    print(ds_dict)
    out = os.path.join(args.data_root, args.name)
    ds_dict.save_to_disk(out)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()

