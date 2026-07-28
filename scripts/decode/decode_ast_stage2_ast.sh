#!/bin/bash
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT="https://hf-mirror.com"

# --- 参数配置区 ---
step=90814
chunk=32
DECODE_DIR=exp/sft_stage2_ast_step_100k/decode_ast/step_${step}
CONFIG=conf/sft_stage2_ast.json
CHECKPOINT="exp/sft_stage2_ast_step_100k/checkpoints/epoch-epoch=13-step-step=90814.ckpt"
MAX_NEW_TOKENS=200
COMET_MODEL=pretrained_models/wmt22-comet-da/checkpoints/model.ckpt
TARGET_LANG="Chinese"

# --- 要遍历的 num_chunk 列表 ---
num_chunks=(1 2 4)

# --- 要遍历的测试集 (名称 输入路径) ---
declare -A INPUT_FILES
INPUT_FILES["yodas"]="data/cuts/dev/granary_yodas_en2zh_sft_dev_repacked.jsonl"

mkdir -p $DECODE_DIR
ulimit -n 65536

for num_chunk in "${num_chunks[@]}"; do
    TOTAL_CHUNK=$((num_chunk * chunk))
    CACHE_DIR=$DECODE_DIR/tmp_${TOTAL_CHUNK}

    for dataset in "${!INPUT_FILES[@]}"; do
        INPUT_FILE="${INPUT_FILES[$dataset]}"
        OUTPUT_FILE=$DECODE_DIR/decode_en2zh_${dataset}_ast_step_${step}_chunk_${TOTAL_CHUNK}_punc0.jsonl
        COMET_FILE=$DECODE_DIR/comet_en2zh_${dataset}_step_${step}_chunk_${TOTAL_CHUNK}_punc0.txt
        BLEU_FILE=$DECODE_DIR/bleu_en2zh_${dataset}_step_${step}_chunk_${TOTAL_CHUNK}_punc0.txt

        echo "========================================"
        echo "数据集: $dataset | Total Chunk: $TOTAL_CHUNK"
        echo "配置文件: $CONFIG"
        echo "模型路径: $CHECKPOINT"
        echo "输入文件: $INPUT_FILE"
        echo "输出文件: $OUTPUT_FILE"

        # 推理
        if [ -f "$OUTPUT_FILE" ]; then
            echo "已存在，跳过推理: $OUTPUT_FILE"
        else
            python3 speechllm/bin/decode_ast_stream.py \
                --config $CONFIG \
                --checkpoint "$CHECKPOINT" \
                --input_file "$INPUT_FILE" \
                --output_file "$OUTPUT_FILE" \
                --cache_dir "$CACHE_DIR" \
                --lang "$TARGET_LANG" \
                --num_gpus 8 \
                --procs_per_gpu 3 \
                --max_new_tokens $MAX_NEW_TOKENS \
                --num_chunks $num_chunk 
        fi

        # 计算 COMET 分数
        echo "计算 COMET 分数..."
        python3 speechllm/bin/calc_comet.py \
            --ref_file $INPUT_FILE \
            --hyp_file $OUTPUT_FILE \
            --output_file $COMET_FILE \
            --model $COMET_MODEL \
            --batch_size 64

        echo "COMET 分数已保存至: $COMET_FILE"

        # 计算 BLEU 分数
        echo "计算 BLEU 分数..."
        python3 speechllm/bin/calc_bleu.py \
            --ref_file $INPUT_FILE \
            --hyp_file $OUTPUT_FILE \
            --output_file $BLEU_FILE

        echo "BLEU 分数已保存至: $BLEU_FILE"
    done
done

echo "========================================"
echo "全部 AST 解码和评估完成！"
