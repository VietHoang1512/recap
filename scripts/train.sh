#!/usr/bin/env bash
# Launch a single RECAP training run.
#
#   bash scripts/train.sh configs/generated/rlvr_only/recap.yaml
#   NUM_DEVICES=4 bash scripts/train.sh configs/generated/hybrid/recap.yaml
#
# Credentials are read from the environment -- never commit them. Export whichever you need:
#   export HF_TOKEN=...          # only for gated models/datasets
#   export WANDB_API_KEY=...     # only if report_to includes wandb
set -euo pipefail

CONFIG="${1:?usage: bash scripts/train.sh <config.yaml>}"
[[ -f "$CONFIG" ]] || { echo "no such config: $CONFIG" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Where prepare_data.py wrote the datasets; dataset_info.json resolves ${DATA_ROOT} against this.
export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/share_data}"

NUM_DEVICES="${NUM_DEVICES:-$(nvidia-smi --list-gpus | wc -l)}"
MASTER_PORT="${MASTER_PORT:-12345}"
RUN_NAME="$(basename "$CONFIG" .yaml)"

export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export WANDB_PROJECT="${WANDB_PROJECT:-recap}"

# Per-sample reward traces, written by the reward functions in src/open_r1/rewards/.
# Separate from the trainer's own debug logging, which is controlled by the log level.
export DEBUG_MODE="${DEBUG_MODE:-false}"
export LOG_PATH="${LOG_PATH:-$REPO_ROOT/outputs/${RUN_NAME}_reward_log.txt}"
mkdir -p "$(dirname "$LOG_PATH")"

echo "config      : $CONFIG"
echo "data root   : $DATA_ROOT"
echo "gpus        : $NUM_DEVICES"

exec python3 -m torch.distributed.run \
    --nproc_per_node="$NUM_DEVICES" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="$MASTER_PORT" \
    --module src.open_r1.mix \
    --config "$CONFIG"
