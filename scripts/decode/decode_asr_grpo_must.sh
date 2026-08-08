#!/bin/bash
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT="https://hf-mirror.com"

source /nfs/asr/guokezhen/miniconda/etc/profile.d/conda.sh
conda activate speechllm

# --- 参数配置区 ---
chunk=32
CONFIG=conf/grpo_vllm_kl_0.05_lr_1e-6_comet_no_kl_step_2000.json
MAX_NEW_TOKENS=200
TARGET_LANG="English"

# --- 要遍历的检查点文件名列表 (可包含 epoch 或 step 格式) ---
checkpoints=(
    "best-step-step=1950.pt"
)

# --- 要遍历的 num_chunk 列表 ---
num_chunks=(1 2 4)

# --- 要遍历的 punct_kv_mode 列表 ---
punct_kv_modes=(3)

# --- 要遍历的测试集 (名称 输入路径) ---
declare -A INPUT_FILES
INPUT_FILES["must_common"]="data/cuts/test/MuST_COMMON_en2zh_test_cuts.jsonl"
INPUT_FILES["must_he"]="data/cuts/test/MuST_HE_en2zh_test_cuts.jsonl"

ulimit -n 65536

for ckpt_name in "${checkpoints[@]}"; do
    tag="${ckpt_name%.pt}"
    DECODE_DIR=exp/grpo_vllm_kl_0.05_lr_1e-6_comet_no_kl_step_2000/decode_asr_must/${tag}
    CHECKPOINT="exp/grpo_vllm_kl_0.05_lr_1e-6_comet_no_kl_step_2000/checkpoints/${ckpt_name}"

    mkdir -p $DECODE_DIR

    for num_chunk in "${num_chunks[@]}"; do
        TOTAL_CHUNK=$((num_chunk * chunk))
        CACHE_DIR=$DECODE_DIR/tmp_${TOTAL_CHUNK}

        for punct_kv_mode in "${punct_kv_modes[@]}"; do
            for dataset in "${!INPUT_FILES[@]}"; do
                INPUT_FILE="${INPUT_FILES[$dataset]}"
                OUTPUT_FILE=$DECODE_DIR/decode_${dataset}_asr_${tag}_chunk_${TOTAL_CHUNK}_punc${punct_kv_mode}.jsonl
                WER_FILE=$DECODE_DIR/wer_${dataset}_${tag}_chunk_${TOTAL_CHUNK}_punc${punct_kv_mode}.txt

                echo "========================================"
                echo "Checkpoint: $ckpt_name | 数据集: $dataset | Total Chunk: $TOTAL_CHUNK | Punct KV Mode: $punct_kv_mode"
                echo "配置文件: $CONFIG"
                echo "模型路径: $CHECKPOINT"
                echo "输入文件: $INPUT_FILE"
                echo "输出文件: $OUTPUT_FILE"

                # 检查 checkpoint 是否存在
                if [ ! -e "$CHECKPOINT" ]; then
                    echo "警告: 检查点不存在，跳过: $CHECKPOINT"
                    continue
                fi

                # 推理
                if [ -f "$OUTPUT_FILE" ]; then
                    echo "已存在，跳过推理: $OUTPUT_FILE"
                else
                    python3 speechllm/eval/decode_asr_stream.py \
                        --config $CONFIG \
                        --checkpoint "$CHECKPOINT" \
                        --input_file "$INPUT_FILE" \
                        --output_file "$OUTPUT_FILE" \
                        --cache_dir "$CACHE_DIR" \
                        --lang "$TARGET_LANG" \
                        --num_gpus 8 \
                        --procs_per_gpu 3 \
                        --max_new_tokens $MAX_NEW_TOKENS \
                        --num_chunks $num_chunk \
                        --punct_kv_mode $punct_kv_mode
                fi

                # 计算 WER 分数
                echo "计算 WER 分数..."
                python3 speechllm/eval/calc_wer.py \
                    --ref $INPUT_FILE \
                    --pred $OUTPUT_FILE \
                    --output $WER_FILE \
                    --lang en

                echo "WER 分数已保存至: $WER_FILE"
            done
        done
    done
done

echo "========================================"
echo "全部 GRPO ASR 解码和评估完成！"
