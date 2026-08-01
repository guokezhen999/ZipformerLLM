#!/bin/bash
# 将 granary_yodas GRPO JSONL 打成 Lhotse Shar，输出到:
#   data/fbank/speechllm_yodas/shar/en2zh_grpo
#
# 用法（在项目根目录执行）:
#   bash local/prepare_shar_yodas_grpo.sh
#   bash local/prepare_shar_yodas_grpo.sh --shard-size 10000 --num-jobs 16
#   bash local/prepare_shar_yodas_grpo.sh 2>&1 | tee logs/prepare_shar_yodas_grpo.log

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "ERROR: please run with bash, do not source this script"
  return 1 2>/dev/null || exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---- options ----
shard_size=3000
num_jobs=32
backup_existing=1

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}) $*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shard-size)
      shard_size="$2"; shift 2 ;;
    --num-jobs)
      num_jobs="$2"; shift 2 ;;
    --no-backup)
      backup_existing=0; shift ;;
    -h|--help)
      echo "Usage: bash local/prepare_shar_yodas_grpo.sh [--shard-size N] [--num-jobs N] [--no-backup]"
      exit 0 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# 优先用 speechllm 环境里的 python
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate speechllm 2>/dev/null || true
fi
PYTHON="${PYTHON:-python}"

CUTS_DIR="${CUTS_DIR:-/nfs/guokezhen/ZipformerLLM/data/cuts}"
SHAR_ROOT="data/fbank/speechllm_yodas/shar"

jsonl_inputs=(
  "${CUTS_DIR}/granary_yodas_en2zh_grpo_train_selected_300h.jsonl"
)

shar_outputs=(
  "${SHAR_ROOT}/en2zh_grpo"
)

for f in "${jsonl_inputs[@]}"; do
  if [[ ! -f "$f" ]]; then
    log "ERROR: missing input: $f"
    exit 1
  fi
done

mkdir -p "$SHAR_ROOT" logs

input_csv=$(IFS=,; echo "${jsonl_inputs[*]}")
output_csv=$(IFS=,; echo "${shar_outputs[*]}")

extra_args=()
if (( backup_existing )); then
  extra_args+=(--backup-existing)
fi

log "=========================================="
log "prepare_shar_yodas_grpo"
log "  cwd: $ROOT_DIR"
log "  python: $($PYTHON -c 'import sys; print(sys.executable)')"
log "  shard_size=$shard_size  num_jobs=$num_jobs"
log "  inputs:"
for f in "${jsonl_inputs[@]}"; do log "    $f"; done
log "  outputs:"
for d in "${shar_outputs[@]}"; do log "    $d"; done
log "=========================================="

$PYTHON -u local/prepare_shar.py \
  --input "$input_csv" \
  --output "$output_csv" \
  --shard-size "$shard_size" \
  --num-jobs "$num_jobs" \
  "${extra_args[@]}"

log "=========================================="
log "Done. Shar dirs:"
for d in "${shar_outputs[@]}"; do
  if [[ -d "$d" ]]; then
    n_cuts=$(find "$d" -name 'cuts.*.jsonl.gz' 2>/dev/null | wc -l)
    n_tars=$(find "$d" -name 'features.*.tar' 2>/dev/null | wc -l)
    log "  $d  (cuts=$n_cuts, tars=$n_tars)"
  else
    log "  MISSING: $d"
  fi
done
log "=========================================="
log "Next (example, single shar dir):"
log "  python local/generate_shard_duration.py --shar-dir ${SHAR_ROOT}/en2zh_grpo --output ${SHAR_ROOT}/en2zh_grpo_shard_durations.json"
log "  python local/assign_shards_to_ranks.py --duration-file ${SHAR_ROOT}/en2zh_grpo_shard_durations.json --world-size 8 --output exp/shard_assignment_ws8.json"
log "=========================================="
