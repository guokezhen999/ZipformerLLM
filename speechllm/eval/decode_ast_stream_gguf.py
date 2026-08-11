#!/usr/bin/env python3
"""
ZipformerLLM streaming AST inference with stage-1 ONNX + stage-2 GGUF LLM.

Stage 1 (ONNX): audio fbank -> audio embeddings (llm_dim)
Stage 2 (GGUF): prompt + <A> + audio_embeds + </A> -> translated text

Usage:
    python speechllm/eval/decode_ast_stream_gguf.py \\
        --export_dir export/model/grpo_global_comet_no_kl_step_2000 \\
        --input_file data/cuts/dev/granary_yodas_en2zh_grpo_dev_repacked.jsonl \\
        --output_file exp/decode_gguf/output.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from addict import Dict as AddictDict
from lhotse.dataset import SimpleCutSampler
from torch.utils.data import DataLoader

from speechllm.dataset import DatasetForStreamASR, DatasetForStreamAST
from speechllm.inference.llama_gguf_decoder import LlamaGgufStreamDecoder
from speechllm.inference.onnx_streaming_encoder import OnnxStreamingEncoder
from speechllm.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_infer_config(export_dir: Path) -> AddictDict:
    infer_path = export_dir / "infer_config.json"
    if not infer_path.exists():
        raise FileNotFoundError(
            f"infer_config.json not found in {export_dir}. "
            "Re-run export: bash scripts/export/run_export.sh 1 2"
        )
    with open(infer_path, "r", encoding="utf-8") as f:
        return AddictDict(json.load(f))


def load_speechllm_meta(export_dir: Path) -> Dict:
    meta_path = export_dir / "speechllm_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"speechllm_meta.json not found in {export_dir}. Run stage 2/3 export first."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZipformerLLM streaming AST inference (ONNX + GGUF)")
    parser.add_argument("--export_dir", required=True, help="Export dir with ONNX + GGUF + metadata")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional training config override (default: export_dir/infer_config.json)",
    )
    parser.add_argument("--input_file", required=True, help="Input cuts json/jsonl(.gz)")
    parser.add_argument("--output_file", required=True, help="Output jsonl path")
    parser.add_argument("--lang", default=None, help="Override language name in prompt (English/Chinese/...)")
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "asr", "ast"],
        help="auto: use dataset prompt; asr/ast: force Transcribe/Translate prompt with --lang",
    )
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of encoder chunks to wait for")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit number of cuts (0=all)")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--punct_kv_mode", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--first_token_eos_threshold", type=float, default=1.0,
                        help="Suppress premature <W> on first decode token when < 1.0")
    parser.add_argument("--eos_penalty_only_last_chunk", action="store_true",
                        help="Apply first_token_eos_threshold only on the last segment")
    parser.add_argument("--n_ctx", type=int, default=8192)
    parser.add_argument(
        "--n_threads",
        type=int,
        default=0,
        help="llama.cpp threads; 0 = min(8, CPU count). Do not leave 0 if you previously meant all cores.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "gpu"],
        help="ONNX encoder device (cuda/gpu uses CUDA EP when available)",
    )
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=0,
        help="llama.cpp GPU layers; -1=all, 0=CPU-only",
    )
    parser.add_argument(
        "--quantize_embeds",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="INT8 round-trip for audio embeds + load int8 special tokens; -1=follow speechllm_meta",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    export_dir = Path(args.export_dir)
    llm_meta = load_speechllm_meta(export_dir)

    if args.config is not None:
        config = load_config(args.config)
    else:
        config = load_infer_config(export_dir)
    encoder = OnnxStreamingEncoder(str(export_dir), device=args.device)
    logger.info("ONNX providers=%s", encoder.providers)
    gguf_path = export_dir / llm_meta["gguf_file"]
    quantize_embeds = None if args.quantize_embeds < 0 else bool(args.quantize_embeds)
    decoder = LlamaGgufStreamDecoder(
        gguf_path=str(gguf_path),
        meta=llm_meta,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,  # 0 → decoder clamps to min(8, nproc)
        n_gpu_layers=args.n_gpu_layers,
        num_chunks=args.num_chunks,
        quantize_injected_embeds=quantize_embeds,
    )
    logger.info(
        "Embed quant: enabled=%s audio_mode=%s",
        decoder.quantize_injected_embeds,
        decoder.audio_embed_quant_mode,
    )

    # ASR uses source-text alignments; AST uses translation alignments.
    # task=auto keeps AST dataset (historical default for this entrypoint).
    use_asr = args.task == "asr"
    ds_cls = DatasetForStreamASR if use_asr else DatasetForStreamAST
    dataset = ds_cls(
        manifest_paths=args.input_file,
        mode="test",
        config=config.data,
    )
    dataset.set_return_ids(True)
    dataset.chunk_counts = [args.num_chunks]
    dataset.chunk_probs = [1.0]
    if hasattr(dataset, "ast_chunk_counts"):
        dataset.ast_chunk_counts = [args.num_chunks]
        dataset.ast_chunk_probs = [1.0]
    if args.max_samples and args.max_samples > 0:
        cuts = dataset.cuts
        if len(cuts) > args.max_samples:
            dataset.cuts = cuts.subset(cut_ids=list(cuts.ids)[: args.max_samples])
            logger.info("Limited to %d samples", args.max_samples)
    logger.info("Dataset=%s task=%s", ds_cls.__name__, args.task)

    sampler = SimpleCutSampler(dataset.cuts, max_cuts=1, shuffle=False)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=0)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    def make_prompt(dataset_prompt: str) -> str:
        if args.task == "asr":
            if args.lang:
                return f"Transcribe the audio in {args.lang}: "
            return dataset_prompt
        if args.task == "ast":
            if args.lang:
                return f"Translate the audio in {args.lang}: "
            return dataset_prompt
        # auto
        if args.lang and "Translate" in (dataset_prompt or ""):
            return f"Translate the audio in {args.lang}: "
        if args.lang and "Transcribe" in (dataset_prompt or ""):
            return f"Transcribe the audio in {args.lang}: "
        return dataset_prompt

    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for sample_i, batch in enumerate(dataloader):
            batch_features, audio_lengths, segments, prompts, batch_ids = batch
            prompt = make_prompt(prompts[0])
            sample_id = batch_ids[0]
            sample_segments = segments[0]
            logger.info(
                "[%d] start %s feats=%s n_seg=%d prompt=%r",
                sample_i,
                sample_id,
                tuple(batch_features.shape),
                len(sample_segments),
                prompt,
            )

            decoder.reset()
            # Overlapping windows (stride=decode_chunk_len, window=input_time_steps)
            # matching streaming_session / model_asr.forward_audio
            audio_embeds = encoder.encode_features(batch_features)
            logger.info("[%d] encoded embeds=%s", sample_i, audio_embeds.shape)

            sample_texts: List[str] = []
            for seg_idx, seg in enumerate(sample_segments):
                s_idx = max(0, min(seg["start_idx"], audio_embeds.shape[0]))
                e_idx = max(s_idx, min(seg["end_idx"], audio_embeds.shape[0]))
                audio_slice = audio_embeds[s_idx:e_idx]

                text = decoder.generate_chunk(
                    prompt=prompt,
                    audio_embeds=audio_slice,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    first_token_eos_threshold=args.first_token_eos_threshold,
                    punct_kv_mode=args.punct_kv_mode,
                    is_first_chunk=(seg_idx == 0),
                    is_last_chunk=(seg_idx == len(sample_segments) - 1),
                    eos_penalty_only_last_chunk=args.eos_penalty_only_last_chunk,
                    # Match streaming_session: clear KV on sentence-end / segment limit
                    clear_kv_on_sentence_end=True,
                )
                sample_texts.append(text)
                if (seg_idx + 1) % 10 == 0 or seg_idx + 1 == len(sample_segments):
                    logger.info(
                        "[%d] seg %d/%d done text=%r",
                        sample_i,
                        seg_idx + 1,
                        len(sample_segments),
                        text[:80] if text else "",
                    )

            chunks_detail = []
            ref_segments: List[str] = []
            for chunk_idx, chunk_text in enumerate(sample_texts):
                seg_info = sample_segments[chunk_idx] if chunk_idx < len(sample_segments) else {}
                ref_text = (seg_info.get("text") or "").strip()
                ref_segments.append(ref_text)
                chunks_detail.append(
                    {
                        "chunk_idx": chunk_idx,
                        "start_idx": seg_info.get("start_idx"),
                        "end_idx": seg_info.get("end_idx"),
                        "text": chunk_text,
                        "ref": ref_text,
                    }
                )

            result = {
                "id": sample_id,
                "task": args.task,
                "num_chunks": args.num_chunks,
                "prompt": prompt,
                "segments_text": sample_texts,
                "ref_segments": ref_segments,
                "chunks": chunks_detail,
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()
            logger.info("Decoded %s", sample_id)

    logger.info("Results saved to %s", args.output_file)


if __name__ == "__main__":
    main()
