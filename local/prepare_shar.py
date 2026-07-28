#!/usr/bin/env python3
"""
将 Lhotse JSONL CutSet 转换为 Shar 格式（cuts.*.jsonl.gz + features.*.tar）。

Usage:
    python local/prepare_shar.py \
        --input data/cuts/train.jsonl \
        --output data/fbank/shar/train \
        --shard-size 3000 \
        --num-jobs 16 \
        --seed 42

    # 多组输入/输出（逗号分隔，一一对应）
    python local/prepare_shar.py \
        --input a.jsonl,b.jsonl \
        --output shar/a,shar/b \
        --shard-size 3000 \
        --num-jobs 16
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime
from typing import List


def _patch_lhotse_supervision_translation():
    """将 supervision 顶层的 translation 挪到 custom，兼容 AST jsonl。"""
    from lhotse import SupervisionSegment

    if getattr(SupervisionSegment.from_dict, "_speechllm_translation_patched", False):
        return

    _original = SupervisionSegment.from_dict

    @classmethod
    def _patched(cls, data: dict):
        data = dict(data)  # 避免原地修改影响后续逻辑
        if "translation" in data:
            custom = data.get("custom")
            if custom is None:
                custom = {}
            else:
                custom = dict(custom)
            custom["translation"] = data.pop("translation")
            data["custom"] = custom
        return _original(data)

    _patched._speechllm_translation_patched = True
    SupervisionSegment.from_dict = _patched


def _split_paths(arg: str) -> List[str]:
    return [p.strip() for p in arg.split(",") if p.strip()]


def convert_one(
    input_file: str,
    output_dir: str,
    shard_size: int,
    num_jobs: int,
    backup_existing: bool,
    seed: int,
) -> None:
    _patch_lhotse_supervision_translation()
    from lhotse import load_manifest_lazy

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"input file not found: {input_file}")

    if os.path.isdir(output_dir):
        if backup_existing:
            bak = f"{output_dir}_bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logging.info(f"Output dir exists, moving to: {bak}")
            shutil.move(output_dir, bak)
        else:
            raise FileExistsError(
                f"Output dir already exists: {output_dir}. "
                f"Pass --backup-existing to rename it, or remove it manually."
            )
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Loading: {input_file}")
    cuts = load_manifest_lazy(input_file)

    total = len(cuts)
    n_shards = (total + shard_size - 1) // shard_size
    logging.info(f"Total cuts: {total}, estimated shards: {n_shards}")

    logging.info(f"Materializing cut metadata and shuffling with seed={seed}...")
    t_shuf = time.time()
    cuts = cuts.to_eager().shuffle(rng=random.Random(seed))
    logging.info(f"Shuffle done in {time.time() - t_shuf:.1f}s")

    t0 = time.time()
    cuts.to_shar(
        output_dir,
        fields={"features": "lilcom"},
        shard_size=shard_size,
        num_jobs=num_jobs,
        verbose=True,
    )
    elapsed = time.time() - t0
    logging.info(f"Done in {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    logging.info(f"Output: {output_dir}")
    logging.info(f"  shards: {n_shards}")
    logging.info(f"  cuts per shard: ~{shard_size}")
    logging.info(f"  compression: lilcom")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Lhotse JSONL CutSet(s) to Shar format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL path(s), comma-separated",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output Shar dir(s), comma-separated, 1:1 with --input",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=3000,
        help="Number of cuts per shard (default: 3000)",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=16,
        help="Parallel jobs for to_shar (default: 16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling cuts before packing (default: 42)",
    )
    parser.add_argument(
        "--backup-existing",
        action="store_true",
        help="If output dir exists, rename it with a timestamp suffix",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    inputs = _split_paths(args.input)
    outputs = _split_paths(args.output)
    if len(inputs) != len(outputs):
        logging.error(
            f"--input ({len(inputs)}) and --output ({len(outputs)}) length mismatch"
        )
        sys.exit(1)

    success = 0
    failed = 0
    for i, (inp, out) in enumerate(zip(inputs, outputs), start=1):
        logging.info("=" * 60)
        logging.info(f"[{i}/{len(inputs)}] {inp} -> {out}")
        logging.info("=" * 60)
        try:
            convert_one(
                input_file=inp,
                output_dir=out,
                shard_size=args.shard_size,
                num_jobs=args.num_jobs,
                backup_existing=args.backup_existing,
                seed=args.seed,
            )
            success += 1
        except Exception:
            logging.exception(f"FAILED: {inp}")
            failed += 1

    logging.info("=" * 60)
    logging.info(f"Done. success={success}, failed={failed}, total={len(inputs)}")
    logging.info("=" * 60)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
