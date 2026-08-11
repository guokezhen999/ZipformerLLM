#!/usr/bin/env bash
#
# Export GRPO checkpoint LLM weights to HuggingFace for the streaming app.
# Thin wrapper around scripts/export/export_llm.sh with app defaults.
#
# Usage:
#   bash scripts/app/export_llm.sh
#   CKPT=... OUTPUT=... bash scripts/app/export_llm.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CKPT="${CKPT:-exp/grpo_vllm_kl_0.05_lr_1e-6_comet_no_kl_step_2000/checkpoints/best-step-step=1950.pt}"
export BASE_MODEL="${BASE_MODEL:-pretrained_models/Qwen3-0.6B}"
export OUTPUT="${OUTPUT:-pretrained_models/grpo_vllm_kl_0.05_lr_1e-6_comet_no_kl_step_2000_llm}"

exec bash "$PROJECT_ROOT/scripts/export/export_llm.sh"
