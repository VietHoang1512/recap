#!/usr/bin/env python3
"""Generate the training configs for every run reported in the RECAP paper.

Each run is `configs/template.yaml` with a small override dict merged on top. Run:

    python scripts/make_configs.py                 # -> configs/generated/
    python scripts/make_configs.py --list          # just print the run names
    python scripts/make_configs.py -o /tmp/cfgs    # write somewhere else

Output directories default to `$OUTPUT_ROOT/<run_name>`; set OUTPUT_ROOT (or pass
--output-root) to point at your checkpoint storage.
"""

import argparse
import copy
import os
import pathlib
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "configs" / "template.yaml"

# --------------------------------------------------------------------------------------------
# Setting 1 -- RLVR-only (paper Table 1).
#
# Qwen2.5-VL-3B over four RLVR domains, trained until the smallest domain is exhausted (328
# steps). Mirrors the MoDoMoDo protocol so the static-mixture baseline is directly comparable.
# --------------------------------------------------------------------------------------------
RLVR_DATASETS = [
    "share_data/lisa-rescale-train",
    "share_data/geoqav-problems-dataset",
    "share_data/sat-problems-dataset",
    "share_data/scienceqa-problems-dataset",
]

RLVR_BASE = {
    "model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 1,
    "dataset_names": list(RLVR_DATASETS),
    "reward_funcs": ["iou_bbox", "accuracy_sat", "format_sat"],
    "reward_weights": [2.0, 2.0, 1.0],
}

# The static-mixture baselines interleave and stop when the smallest domain is exhausted, which
# the original runs pinned to 328 steps. The RECAP runs concatenate instead and take one epoch,
# so they carry no max_steps.
RLVR_INTERLEAVE_STEPS = 328

# Setting 2 -- Hybrid RLVR + SFT replay (paper Table 2).
#
# Qwen2.5-VL-7B on ThinkLite-VL-70k with RefCOCO (grounding) and LLaVA-OneVision OCR replayed
# to preserve perception. 500 steps, per-device batch 1 x accum 2 = effective 16 on 8 GPUs,
# G=4 (appendix "For the larger hybrid setting").
HYBRID_DATASETS = [
    "share_data/ThinkLite-VL-70k",
    "share_data/RefCOCO",
    "share_data/LLaVA-OneVision-OCR-10k-128",
]

HYBRID_BASE = {
    "model_name_or_path": "Qwen/Qwen2.5-VL-7B-Instruct",
    "max_steps": 500,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "gradient_checkpointing": True,
    "max_completion_length": 2048,
    "dataset_names": list(HYBRID_DATASETS),
    "reward_funcs": [
        "accuracy_thinklite",
        "iou",
        "format_think",
        "format_answer",
    ],
    "reward_weights": [2.0, 2.0, 1.0, 1.0],
}

# RECAP defaults from the appendix: W=10, T=5.0, alpha=0.5.
RECAP = {
    "normalize_loss": "dwa",
    "iteration_window": 10,
    "softmax_temp": 5.0,
    "convergence_instablity_tradeoff": 0.5,
    "mix_strategy": "concat",
}

# Baselines share "no reweighting"; they differ in how data is sampled.
NO_REWEIGHT = {"normalize_loss": "none"}


def _runs():
    """(group, name, overrides) for every generated config."""
    runs = []

    # ---------------- RLVR-only ----------------
    def rlvr(name, **over):
        runs.append(("rlvr_only", name, {**RLVR_BASE, **over}))

    rlvr("uniform", **NO_REWEIGHT, mix_strategy="interleave_under", mixed_sampler="uniform",
         max_steps=RLVR_INTERLEAVE_STEPS)

    # MoDoMoDo: the static mixture its surrogate model selected, over five domains
    # (ViRFT_COCO prepended at weight 0). Vision tower frozen, as in the original run.
    rlvr(
        "modomodo",
        **NO_REWEIGHT,
        dataset_names=["share_data/ViRFT_COCO"] + RLVR_DATASETS,
        mix_strategy="interleave_under",
        interleave_probs=[0.0, 0.3045, 0.2024, 0.2117, 0.2814],
        freeze_vision_modules=True,
        max_steps=RLVR_INTERLEAVE_STEPS,
    )

    rlvr("recap", **RECAP)

    # Ablations (paper Table "Benchmark performance in RLVR-only setting", appendix).
    # alpha: 1.0 = pure convergence rate, 0.0 = pure instability, -1 = annealed.
    for alpha in (0.25, 0.5, 0.75, 1.0, -1):
        tag = "adaptive" if alpha == -1 else str(alpha).replace(".", "p")
        extra = {}
        if alpha == -1:
            # The annealed schedule is alpha = 1 - global_step/max_steps
            # (mixed_trainer.py:1790), so max_steps must be finite -- the trainer asserts
            # 0 <= alpha <= 1 and the HF default of -1 would trip it on the first step.
            extra["max_steps"] = RLVR_INTERLEAVE_STEPS
        rlvr(f"ablation_alpha_{tag}",
             **{**RECAP, "convergence_instablity_tradeoff": alpha, **extra})
    for temp in (1.0, 5.0, 50.0):
        rlvr(f"ablation_temp_{str(temp).replace('.', 'p')}", **{**RECAP, "softmax_temp": temp})
    for window in (10, 25, 50):
        rlvr(f"ablation_window_{window}", **{**RECAP, "iteration_window": window})

    # ---------------- Hybrid ----------------
    def hybrid(name, **over):
        runs.append(("hybrid", name, {**HYBRID_BASE, **over}))

    # Reasoning-only: no replay at all, so the grounding rewards drop out too.
    hybrid(
        "reasoning_only",
        **NO_REWEIGHT,
        dataset_names=["share_data/ThinkLite-VL-70k"],
        reward_funcs=["accuracy_thinklite", "format_think"],
        reward_weights=[2.0, 1.0],
    )
    hybrid("propmix", **NO_REWEIGHT, mixed_sampler="default")
    hybrid("uniform", **NO_REWEIGHT, mixed_sampler="uniform")
    # Coreset: replay a size-limited slice of the general data (the 5k OCR shard).
    hybrid(
        "coreset",
        **NO_REWEIGHT,
        mixed_sampler="uniform",
        dataset_names=[
            "share_data/ThinkLite-VL-70k",
            "share_data/RefCOCO",
            "share_data/LLaVA-OneVision-OCR-5k",
        ],
    )
    # LwF: uniform sampling plus a reference-KL penalty.
    hybrid("lwf", **NO_REWEIGHT, mixed_sampler="uniform", beta=0.01)
    hybrid("recap", **RECAP)

    return runs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", type=pathlib.Path,
                    default=REPO_ROOT / "configs" / "generated",
                    help="directory to write configs into (default: configs/generated)")
    ap.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", "./outputs"),
                    help="checkpoint root; each run writes to <root>/<run_name> "
                         "(default: $OUTPUT_ROOT or ./outputs)")
    ap.add_argument("--list", action="store_true", help="print run names and exit")
    args = ap.parse_args(argv)

    runs = _runs()
    if args.list:
        for group, name, _ in runs:
            print(f"{group}/{name}")
        return 0

    template = yaml.safe_load(TEMPLATE.read_text())

    for group, name, overrides in runs:
        cfg = copy.deepcopy(template)
        cfg.update(overrides)
        run_name = f"{group}_{name}"
        cfg["run_name"] = run_name
        cfg["output_dir"] = f"{args.output_root.rstrip('/')}/{run_name}"

        dest = args.out / group / f"{name}.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))

    print(f"wrote {len(runs)} configs to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
