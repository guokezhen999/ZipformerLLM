#!/bin/bash
# Export GRPO/SFT LLM weights to HuggingFace format for vLLM / run_app.
#
# Uses speechllm/utils/export_sft_llm.py
#
# Usage:
#   bash scripts/app/export_llm.sh
#   CKPT=... BASE_MODEL=... OUTPUT=... bash scripts/app/export_llm.sh

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"

source /pfs/asr/miniconda3/etc/profile.d/conda.sh
conda activate zipformer_vllm

PYTHON="${PYTHON:-/pfs/asr/miniconda3/envs/zipformer_vllm/bin/python}"

CKPT="${CKPT:-exp/grpo_vllm_kl_0.05_lr_1e-6_comet_step_2000/checkpoints/best-step-1850-weights.pt}"
BASE_MODEL="${BASE_MODEL:-pretrained_models/Qwen3-0.6B}"
OUTPUT="${OUTPUT:-pretrained_models/grpo_vllm_kl_0.05_lr_1e-6_comet_step_2000_llm}"

echo "CKPT       : ${CKPT}"
echo "BASE_MODEL : ${BASE_MODEL}"
echo "OUTPUT     : ${OUTPUT}"

${PYTHON} speechllm/utils/export_sft_llm.py \
    --ckpt "${CKPT}" \
    --base_model "${BASE_MODEL}" \
    --output "${OUTPUT}"

echo "Done. Point SPEECHLLM_LLM_PATH / VLLM_MODEL_PATH to: ${OUTPUT}"
