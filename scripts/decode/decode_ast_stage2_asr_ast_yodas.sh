#!/bin/bash
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT="https://hf-mirror.com"

source /nfs/asr/guokezhen/miniconda/etc/profile.d/conda.sh
conda activate speechllm

# --- 参数配置区 ---
chunk=32
CONFIG=conf/sft_stage2_ast_step_100k.json
MAX_NEW_TOKENS=200
COMET_MODEL=pretrained_models/wmt22-comet-da/checkpoints/model.ckpt
TARGET_LANG="Chinese"

# --- 要遍历的检查点文件名列表 (可包含 epoch 或 step 格式) ---
checkpoints=(
    # "epoch-epoch=00-step-step=12983.ckpt"
    # "epoch-epoch=01-step-step=25974.ckpt"
    # "epoch-epoch=02-step-step=38967.ckpt"
    # "epoch-epoch=03-step-step=51936.ckpt"
    # "epoch-epoch=04-step-step=64906.ckpt"
    "epoch-epoch=05-step-step=77910.ckpt"
    "epoch-epoch=06-step-step=90887.ckpt"
    "epoch-epoch=07-step-step=103882.ckpt"
    "epoch-epoch=08-step-step=116880.ckpt"
    "epoch-epoch=09-step-step=129852.ckpt"
    "epoch-epoch=10-step-step=142834.ckpt"
    "epoch-epoch=11-step-step=155821.ckpt"
    "epoch-epoch=12-step-step=168803.ckpt"
    "epoch-epoch=13-step-step=181798.ckpt"
    "epoch-epoch=14-step-step=194787.ckpt"
    "epoch-epoch=15-step-step=200000.ckpt"
    # 如果想跑 step 检查点，也可以取消下面注释:
    # "step-step=20000.ckpt"
    # "step-step=40000.ckpt"
    # "step-step=60000.ckpt"
    # "step-step=80000.ckpt"
    # "step-step=100000.ckpt"
)

# --- 要遍历的 num_chunk 列表 ---
num_chunks=(1 2 4)

# --- 要遍历的测试集 (名称 输入路径) ---
declare -A INPUT_FILES
INPUT_FILES["yodas"]="data/cuts/dev/granary_yodas_en2zh_sft_dev_repacked.jsonl"

ulimit -n 65536

for ckpt_name in "${checkpoints[@]}"; do
    tag="${ckpt_name%.ckpt}"
    DECODE_DIR=exp/sft_stage2_asr_ast_step_200k/decode_ast_yodas/${tag}
    CHECKPOINT="exp/sft_stage2_asr_ast_step_200k/checkpoints/${ckpt_name}"

    mkdir -p $DECODE_DIR

    for num_chunk in "${num_chunks[@]}"; do
        TOTAL_CHUNK=$((num_chunk * chunk))
        CACHE_DIR=$DECODE_DIR/tmp_${TOTAL_CHUNK}

        for dataset in "${!INPUT_FILES[@]}"; do
            INPUT_FILE="${INPUT_FILES[$dataset]}"
            OUTPUT_FILE=$DECODE_DIR/decode_en2zh_${dataset}_ast_${tag}_chunk_${TOTAL_CHUNK}_punc0.jsonl
            COMET_FILE=$DECODE_DIR/comet_en2zh_${dataset}_${tag}_chunk_${TOTAL_CHUNK}_punc0.txt
            BLEU_FILE=$DECODE_DIR/bleu_en2zh_${dataset}_${tag}_chunk_${TOTAL_CHUNK}_punc0.txt

            echo "========================================"
            echo "Checkpoint: $ckpt_name | 数据集: $dataset | Total Chunk: $TOTAL_CHUNK"
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
                python3 speechllm/eval/decode_ast_stream.py \
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
            python3 speechllm/eval/calc_comet.py \
                --ref_file $INPUT_FILE \
                --hyp_file $OUTPUT_FILE \
                --output_file $COMET_FILE \
                --model $COMET_MODEL \
                --batch_size 64

            echo "COMET 分数已保存至: $COMET_FILE"

            # 计算 BLEU 分数
            echo "计算 BLEU 分数..."
            python3 speechllm/eval/calc_bleu.py \
                --ref_file $INPUT_FILE \
                --hyp_file $OUTPUT_FILE \
                --output_file $BLEU_FILE

            echo "BLEU 分数已保存至: $BLEU_FILE"
        done
    done
done

echo "========================================"
echo "全部 AST 解码和评估完成！"
