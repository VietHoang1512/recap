#!/usr/bin/env python3
"""Parse every generated config through the real TrlParser dataclasses.

Catches typo'd keys, type mismatches, and values the trainer would reject -- without needing a
GPU. Run after `python scripts/make_configs.py`:

    python scripts/validate_configs.py

Requires the training dependencies (trl, transformers). Without them it falls back to a
structural check of the RECAP knobs, which needs nothing but PyYAML.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

try:
    from trl import TrlParser

    from src.open_r1.configs import GRPOConfig, GRPOModelConfig, GRPOScriptArguments
    from src.open_r1.vllm_serve import vLLMScriptArguments
    HAVE_TRL = True
except ImportError as exc:                                        # noqa: BLE001
    print(f"note: {exc.name} not importable -- falling back to a YAML-only structural check.\n"
          f"      Install requirements.txt for the full dataclass round trip.\n")
    HAVE_TRL = False


def _check_values(dataset_names, reward_funcs, reward_weights, interleave_probs,
                  normalize_loss, alpha, temp, window, max_steps):
    """Shared invariants. Mirrors what the trainer asserts at mixed_trainer.py:524-534."""
    problems = []
    if normalize_loss not in ("dwa", "ema", "none"):
        problems.append(f"normalize_loss={normalize_loss!r}")
    if not (0.0 <= alpha <= 1.0) and alpha != -1:
        problems.append(f"convergence_instablity_tradeoff={alpha}")
    # The annealed schedule divides by max_steps; HF's default of -1 would break it.
    if alpha == -1 and (max_steps is None or max_steps <= 0):
        problems.append("annealed alpha (-1) needs a positive max_steps")
    if window < 1:
        problems.append(f"iteration_window={window}")
    if temp <= 0:
        problems.append(f"softmax_temp={temp}")
    if len(reward_funcs) != len(reward_weights):
        problems.append(f"{len(reward_funcs)} reward_funcs vs {len(reward_weights)} reward_weights")
    if interleave_probs is not None and len(interleave_probs) != len(dataset_names):
        problems.append(f"{len(interleave_probs)} interleave_probs vs "
                        f"{len(dataset_names)} datasets")
    return problems


def _parse(cfg):
    """(fields, error). Uses the real dataclasses when available, raw YAML otherwise."""
    if HAVE_TRL:
        try:
            parser = TrlParser(
                (GRPOScriptArguments, GRPOConfig, GRPOModelConfig, vLLMScriptArguments)
            )
            s, t, _m, _v = parser.parse_args_and_config(args=["--config", str(cfg)])
        except Exception as exc:                                  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"
        return dict(
            dataset_names=s.dataset_names, reward_funcs=s.reward_funcs,
            reward_weights=t.reward_weights, interleave_probs=s.interleave_probs,
            normalize_loss=t.normalize_loss, alpha=t.convergence_instablity_tradeoff,
            temp=t.softmax_temp, window=t.iteration_window, max_steps=t.max_steps,
        ), None

    import yaml
    d = yaml.safe_load(cfg.read_text())
    return dict(
        dataset_names=d.get("dataset_names", []), reward_funcs=d.get("reward_funcs", []),
        reward_weights=d.get("reward_weights", []), interleave_probs=d.get("interleave_probs"),
        normalize_loss=d.get("normalize_loss", "none"),
        alpha=d.get("convergence_instablity_tradeoff", 1.0),
        temp=d.get("softmax_temp", 1.0), window=d.get("iteration_window", 10),
        max_steps=d.get("max_steps"),
    ), None


def main():
    configs = sorted((REPO_ROOT / "configs" / "generated").rglob("*.yaml"))
    if not configs:
        print("no configs found -- run `python scripts/make_configs.py` first", file=sys.stderr)
        return 1

    failures = []
    for cfg in configs:
        rel = cfg.relative_to(REPO_ROOT)
        fields, err = _parse(cfg)
        if err:
            failures.append(rel)
            print(f"FAIL  {rel}\n      {err}")
            continue

        problems = _check_values(**fields)
        if problems:
            failures.append(rel)
            print(f"FAIL  {rel}\n      " + "\n      ".join(problems))
        else:
            mode = fields["normalize_loss"]
            extra = (f"a={fields['alpha']} T={fields['temp']} W={fields['window']}"
                     if mode == "dwa" else "")
            print(f"ok    {rel}  [{mode}] {extra}")

    print(f"\n{len(configs) - len(failures)}/{len(configs)} configs valid"
          f"{'' if HAVE_TRL else ' (structural check only)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

