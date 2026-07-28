import os
import argparse
import json
import torch
import logging
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.utils.data import DataLoader
from lhotse.dataset import SimpleCutSampler
from lhotse import CutSet
from speechllm.model.model_stream import SpeechLLMStream
from speechllm.dataset import DatasetForStreamAST
from speechllm.utils import load_config
from addict import Dict

if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([Dict])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_args():
    parser = argparse.ArgumentParser(description="SpeechLLM Streaming AST 推理脚本 (多 GPU 多进程版 - 预切分优化)")
    parser.add_argument('--config', required=True, type=str, help='配置文件路径')
    parser.add_argument('--checkpoint', required=True, type=str, help='模型检查点路径 (.ckpt/.bin)')
    parser.add_argument('--input_file', required=True, type=str, help='输入数据集路径 (json.gz/jsonl.gz)')
    parser.add_argument('--output_file', required=True, type=str, help='结果保存路径 (jsonl)')
    parser.add_argument('--lang', type=str, default=None, help='强制覆盖目标翻译语言 (如 English, Chinese)，默认使用数据中自带的语言')
    parser.add_argument('--cache_dir', type=str, default='./tmp_splits', help='存放临时数据切分文件的目录')
    parser.add_argument('--num_gpus', type=int, default=1, help='使用的 GPU 数量')
    parser.add_argument('--procs_per_gpu', type=int, default=1, help='每个 GPU 运行的进程数')
    parser.add_argument('--gpu_ids', type=str, default=None, help='指定的 GPU ID 列表 (逗号分隔，如 "0,1,2")')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='生成的最大 token 数量')
    parser.add_argument('--num_chunks', type=int, default=1, help='解码等待的 encoder 的 chunk 数量')
    parser.add_argument('--punct_kv_mode', type=int, default=0, choices=[0, 1, 2],
                        help='遇到终结标点时的 KV Cache 处理策略: 0=不处理, 1=仅移除末尾标点的KV(默认), 2=清除除Prompt外的所有KV')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='重复惩罚系数，>1.0 时抑制重复 (如 1.2)')
    return parser.parse_args()

def worker(rank, world_size, args, temp_input_files):
    if args.gpu_ids:
        gpu_list = [int(x) for x in args.gpu_ids.split(',')]
        gpu_id = gpu_list[(rank // args.procs_per_gpu) % len(gpu_list)]
    else:
        gpu_id = rank // args.procs_per_gpu

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"[Rank {rank}] 进程启动，使用设备: {device}")

    config = load_config(args.config)
    model = SpeechLLMStream(config, device)
    model.load_checkpoint(args.checkpoint)
    model.eval()

    my_input_file = temp_input_files[rank]
    dataset = DatasetForStreamAST(
        manifest_paths=my_input_file,
        mode='test',
        config=config.data
    )
    dataset.set_return_ids(True)
    dataset.chunk_counts = [args.num_chunks]
    dataset.chunk_probs = [1.0]
    dataset.ast_chunk_counts = [args.num_chunks]
    dataset.ast_chunk_probs = [1.0]

    my_cuts = dataset.cuts
    logging.info(f"[Rank {rank}] 专属数据加载完成，共处理 {len(my_cuts)} 条数据")

    if len(my_cuts) == 0:
        logging.info(f"[Rank {rank}] 无需处理任何数据，直接退出。")
        return

    sampler = SimpleCutSampler(my_cuts, max_cuts=1, shuffle=False)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=0,
        pin_memory=True
    )

    rank_output = f"{args.output_file}.rank_{rank}"
    processed_ids = set()
    if os.path.exists(rank_output):
        with open(rank_output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if 'id' in item: processed_ids.add(item['id'])
                except: continue

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": model.llm_tokenizer.pad_token_id,
        "eos_token_id": model.token_W_id,
        "repetition_penalty": args.repetition_penalty,
    }

    with torch.no_grad():
        with open(rank_output, 'a', encoding='utf-8') as f_out:
            pbar = tqdm(dataloader, desc=f"Rank {rank}", position=rank, leave=True)
            for batch in dataloader:
                batch_features, audio_lengths, segments, prompts, batch_ids = batch

                batch_features = batch_features.to(device)
                audio_lengths = audio_lengths.to(device)

                new_segments = []
                for i in range(len(batch_ids)):
                    if batch_ids[i] in processed_ids:
                        new_segments.append(None)
                        continue
                    new_segments.append(segments[i])

                active_indices = [i for i, s in enumerate(new_segments) if s is not None]
                if not active_indices:
                    if rank == 0: pbar.update(1)
                    continue

                sub_features = batch_features[active_indices]
                sub_lengths = audio_lengths[active_indices]
                sub_prompts = [f"Translate the audio in {args.lang}: " if args.lang else prompts[i] for i in active_indices]
                sub_segments = [new_segments[i] for i in active_indices]
                sub_ids = [batch_ids[i] for i in active_indices]

                audio_embeds, valid_lengths = model.forward_audio(sub_features, sub_lengths)
                batch_generated_texts = model.generate(
                    audio_embeds=audio_embeds,
                    audio_lengths=valid_lengths,
                    prompts=sub_prompts,
                    segments=sub_segments,
                    generation_config=generation_config,
                    punct_kv_mode=args.punct_kv_mode
                )

                for i in range(len(sub_ids)):
                    # 构建每个 chunk 的详细输出
                    chunks_detail = []
                    for chunk_idx, chunk_text in enumerate(batch_generated_texts[i]):
                        seg_info = sub_segments[i][chunk_idx] if chunk_idx < len(sub_segments[i]) else {}
                        chunks_detail.append({
                            "chunk_idx": chunk_idx,
                            "start_idx": seg_info.get("start_idx", None),
                            "end_idx": seg_info.get("end_idx", None),
                            "text": chunk_text
                        })
                    res = {
                        "id": sub_ids[i],
                        "segments_text": batch_generated_texts[i],
                        "chunks": chunks_detail
                    }
                    f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
                    f_out.flush()

                if rank == 0: pbar.update(1)

def main():
    args = get_args()
    target_world_size = args.num_gpus * args.procs_per_gpu

    os.makedirs(args.cache_dir, exist_ok=True)
    logging.info(f"主进程开始预读取并切分数据集: {args.input_file}")

    cuts = CutSet.from_file(args.input_file)
    cuts = cuts.shuffle()
    total_cuts = len(cuts)
    logging.info(f"数据集总计 {total_cuts} 条，准备切分为 {target_world_size} 份存入 {args.cache_dir}...")

    split_cuts = cuts.split(num_splits=target_world_size, shuffle=False)
    actual_world_size = len(split_cuts)

    temp_input_files = []
    for rank, rank_cuts in enumerate(split_cuts):
        temp_file = os.path.join(args.cache_dir, f"temp_split_rank_{rank}_{os.path.basename(args.input_file)}")
        rank_cuts.to_file(temp_file)
        temp_input_files.append(temp_file)
        logging.info(f"已生成 Rank {rank} 的临时切片: {temp_file} (包含 {len(rank_cuts)} 条)")

    logging.info(f"数据切分完毕，启动 {actual_world_size} 个进程进行推理...")
    mp.spawn(worker, nprocs=actual_world_size, args=(actual_world_size, args, temp_input_files))

    logging.info("所有进程推理已完成，开始合并结果...")
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for rank in range(actual_world_size):
            rank_output = f"{args.output_file}.rank_{rank}"
            if os.path.exists(rank_output):
                with open(rank_output, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        f_out.write(line)
                os.remove(rank_output)

            temp_input = temp_input_files[rank]
            if os.path.exists(temp_input):
                os.remove(temp_input)

    logging.info(f"结果已成功合并至: {args.output_file}，并已清理所有临时切片文件。")

    try:
        os.rmdir(args.cache_dir)
        logging.info(f"缓存目录 {args.cache_dir} 已被移除。")
    except OSError:
        pass

if __name__ == "__main__":
    main()
