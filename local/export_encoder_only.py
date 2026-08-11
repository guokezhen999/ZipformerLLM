#!/usr/bin/env python3
"""
从 icefall Zipformer ASR checkpoint 中仅导出 encoder / encoder_embed，
供开源或作为 SpeechLLM 的 audio encoder 初始化权重。

Usage:
    python local/export_encoder_only.py \
        --input pretrained_models/zipformer_base_en/iter-210000-avg-6.pt \
        --output pretrained_models/zipformer_base_en/encoder-iter-210000-avg-6.pt

导出格式与 icefall 一致（{"model": state_dict}），可直接被
speechllm.module.encoder.get_encoder() 加载（只读取 encoder.* /
encoder_embed.*）。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

ENCODER_PREFIXES = ("encoder.", "encoder_embed.")


def _select_state_dict(ckpt: dict) -> tuple[dict, str]:
    """Prefer model_avg (averaged ckpt), then model, else treat ckpt as state_dict."""
    if "model_avg" in ckpt and isinstance(ckpt["model_avg"], dict):
        return ckpt["model_avg"], "model_avg"
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"], "model"
    # Flat state_dict already
    if any(k.startswith(ENCODER_PREFIXES) for k in ckpt):
        return ckpt, "flat"
    raise KeyError(
        "checkpoint 中未找到 model_avg / model，且顶层也不是 encoder state_dict。"
        f" top keys={list(ckpt.keys())[:20]}"
    )


def extract_encoder_state_dict(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        if not k.startswith(ENCODER_PREFIXES):
            continue
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().contiguous()
        else:
            out[k] = v
    return out


def _nbytes(sd: dict) -> int:
    return sum(
        v.numel() * v.element_size() for v in sd.values() if torch.is_tensor(v)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Zipformer encoder(+embed) weights only."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="原始 icefall / Zipformer .pt checkpoint",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="导出的 encoder-only .pt 路径",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="直接保存 flat state_dict，而不是 {\"model\": ...}",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if not args.input.is_file():
        raise FileNotFoundError(f"input not found: {args.input}")

    logging.info(f"Loading {args.input} ({args.input.stat().st_size / 1024 / 1024:.1f} MiB)")
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"unexpected checkpoint type: {type(ckpt)}")

    state, source = _select_state_dict(ckpt)
    logging.info(f"Using state_dict source: {source} ({len(state)} keys)")

    encoder_sd = extract_encoder_state_dict(state)
    if not encoder_sd:
        sample = list(state.keys())[:10]
        raise RuntimeError(
            "未找到 encoder.* / encoder_embed.* keys。"
            f" sample keys={sample}"
        )

    skipped = len(state) - len(encoder_sd)
    logging.info(
        f"Kept {len(encoder_sd)} encoder keys "
        f"({_nbytes(encoder_sd) / 1024 / 1024:.1f} MiB tensors), "
        f"skipped {skipped} non-encoder keys"
    )

    payload = encoder_sd if args.flat else {"model": encoder_sd}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    logging.info(
        f"Saved -> {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)"
    )


if __name__ == "__main__":
    main()
