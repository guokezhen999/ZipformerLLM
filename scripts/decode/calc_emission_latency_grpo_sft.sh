#!/bin/bash
# Compute streaming AST emission latency (FTL / writes / wait%) for
# GRPO best-step-1850 and SFT epoch-14 on MuST decode dirs.
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

GRPO_DIR=exp/grpo_vllm_kl_0.05_lr_1e-6_comet_step_2000/decode_ast_must/best-step-1850-weights
SFT_DIR=exp/sft_stage2_asr_ast_step_200k/decode_ast_must/epoch-epoch=14-step-step=194787

for d in "$GRPO_DIR" "$SFT_DIR"; do
    if [ ! -d "$d" ]; then
        echo "错误: 目录不存在: $d"
        exit 1
    fi
done

echo "========================================"
echo "GRPO: $GRPO_DIR"
echo "SFT : $SFT_DIR"
echo "========================================"

python3 speechllm/eval/calc_emission_latency.py "$GRPO_DIR" "$SFT_DIR"
