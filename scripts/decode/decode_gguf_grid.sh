#!/usr/bin/env bash
#
# Grid decode: ASR/AST × num_chunks ∈ {1,2,4} × precisions.
#
# Usage:
#   bash scripts/decode/decode_gguf_grid.sh
#   MAX_SAMPLES=100 PRECISIONS="f16 q8 q4_k_m" bash scripts/decode/decode_gguf_grid.sh
#
# Outputs:
#   exp/decode_gguf/grid/<precision>/<task>_c<chunks>.jsonl
#   logs/decode_gguf_grid_*.log
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck source=../export/paths.sh
source "$PROJECT_ROOT/scripts/export/paths.sh"

INPUT_FILE="${INPUT_FILE:-data/cuts/dev/granary_yodas_en2zh_grpo_dev_repacked.jsonl}"
OUT_ROOT="${OUT_ROOT:-exp/decode_gguf/grid}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
DEVICE="${DEVICE:-cuda}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"

# space-separated
PRECISIONS="${PRECISIONS:-f16 q8 q4_k_m}"
TASKS="${TASKS:-asr ast}"
CHUNKS="${CHUNKS:-1 2 4}"

mkdir -p "$OUT_ROOT" logs
LOG="logs/decode_gguf_grid_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "========== GGUF grid decode =========="
echo "Input:       $INPUT_FILE"
echo "Out root:    $OUT_ROOT"
echo "Max samples: $MAX_SAMPLES"
echo "Precisions:  $PRECISIONS"
echo "Tasks:       $TASKS"
echo "Chunks:      $CHUNKS"
echo "Device:      $DEVICE  n_gpu_layers=$N_GPU_LAYERS"
echo "Log:         $LOG"
echo "======================================"

n_done=0
n_total=0
for _p in $PRECISIONS; do
  for _t in $TASKS; do
    for _c in $CHUNKS; do
      n_total=$((n_total + 1))
    done
  done
done

for PRECISION in $PRECISIONS; do
  EXPORT_DIR="$(precision_dir "$PRECISION")"
  mkdir -p "$OUT_ROOT/$PRECISION"
  for TASK in $TASKS; do
    if [ "$TASK" = "asr" ]; then
      TARGET_LANG="English"
    else
      TARGET_LANG="Chinese"
    fi
    for NUM_CHUNKS in $CHUNKS; do
      n_done=$((n_done + 1))
      OUTPUT_FILE="$OUT_ROOT/$PRECISION/${TASK}_c${NUM_CHUNKS}.jsonl"
      echo ""
      echo ">>> [$n_done/$n_total] precision=$PRECISION task=$TASK chunks=$NUM_CHUNKS"
      echo "    -> $OUTPUT_FILE"
      if [ "${SKIP_EXISTING:-0}" = "1" ] && [ -f "$OUTPUT_FILE" ] && [ -s "$OUTPUT_FILE" ]; then
        n_lines=$(wc -l < "$OUTPUT_FILE" | tr -d ' ')
        if [ "$n_lines" -ge "$MAX_SAMPLES" ] || [ "$MAX_SAMPLES" = "0" ]; then
          echo "    SKIP existing ($n_lines lines)"
          continue
        fi
      fi
      PRECISION="$PRECISION" \
      EXPORT_DIR="$EXPORT_DIR" \
      INPUT_FILE="$INPUT_FILE" \
      OUTPUT_FILE="$OUTPUT_FILE" \
      TASK="$TASK" \
      TARGET_LANG="$TARGET_LANG" \
      NUM_CHUNKS="$NUM_CHUNKS" \
      MAX_SAMPLES="$MAX_SAMPLES" \
      DEVICE="$DEVICE" \
      N_GPU_LAYERS="$N_GPU_LAYERS" \
        bash "$PROJECT_ROOT/scripts/export/run_infer.sh" "$INPUT_FILE" "$OUTPUT_FILE"
    done
  done
done

echo ""
echo "GRID_DONE root=$OUT_ROOT log=$LOG"
ls -lh "$OUT_ROOT"/*/*.jsonl 2>/dev/null || true
