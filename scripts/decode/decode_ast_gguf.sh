#!/usr/bin/env bash
#
# Decode with exported ONNX encoder + GGUF LLM (wrapper around run_infer.sh).
#
# Usage:
#   bash scripts/decode/decode_ast_gguf.sh
#   PRECISION=q8 bash scripts/decode/decode_ast_gguf.sh
#   EXPORT_DIR=pretrained_models/export/q4_k_m bash scripts/decode/decode_ast_gguf.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# shellcheck source=../export/paths.sh
source "$PROJECT_ROOT/scripts/export/paths.sh"

PRECISION="${PRECISION:-f16}"
EXPORT_DIR="${EXPORT_DIR:-$(precision_dir "$PRECISION")}"
INPUT_FILE="${INPUT_FILE:-data/cuts/dev/granary_yodas_en2zh_grpo_dev_repacked.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-exp/decode_gguf/ast_dev_${PRECISION}.jsonl}"
TARGET_LANG="${TARGET_LANG:-Chinese}"
NUM_CHUNKS="${NUM_CHUNKS:-1}"
DEVICE="${DEVICE:-cuda}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"

export EXPORT_DIR INPUT_FILE OUTPUT_FILE TARGET_LANG NUM_CHUNKS DEVICE N_GPU_LAYERS PRECISION
exec bash "$PROJECT_ROOT/scripts/export/run_infer.sh" "$INPUT_FILE" "$OUTPUT_FILE"
