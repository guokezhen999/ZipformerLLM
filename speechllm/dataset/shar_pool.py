"""
Lhotse Shar 预分配（preassigned）工具。

每个 rank 使用 assignment JSON 中固定的 shard 列表；
每个 epoch 仅 shuffle 本 rank shard 顺序，再 CutSet.from_shar 懒加载。
DynamicBucketingSampler 使用 rank=0/world_size=1（shard 级 DDP 分区）。
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import random
import re
from typing import List, Optional, Sequence, Union

from lhotse import CutSet


def parse_shar_dirs(dirs_arg: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """解析逗号/分号分隔的 Shar 目录列表。"""
    if dirs_arg is None:
        return []
    if isinstance(dirs_arg, (list, tuple)):
        return [str(d).strip() for d in dirs_arg if str(d).strip()]
    return [p.strip() for p in re.split(r"[,;]", str(dirs_arg)) if p.strip()]


def load_shard_assignment(
    assignment_file: str,
    rank: int,
    shar_dirs: List[str],
) -> List[str]:
    """从预分配 JSON 加载当前 rank 的 shard 完整路径。"""
    with open(assignment_file, "r") as f:
        data = json.load(f)

    rank_key = str(rank)
    if "assignment" not in data or rank_key not in data["assignment"]:
        raise KeyError(f"Rank {rank} not found in assignment file: {assignment_file}")

    shard_basenames = data["assignment"][rank_key]
    shar_dir = data.get("shar_dir", "")

    path_map = {}
    for d in shar_dirs:
        for basename in shard_basenames:
            candidate = os.path.join(d, basename)
            if os.path.exists(candidate):
                path_map[basename] = candidate

    my_shards = []
    for basename in shard_basenames:
        if basename in path_map:
            my_shards.append(path_map[basename])
        else:
            fallback = os.path.join(shar_dir, basename)
            if os.path.exists(fallback):
                my_shards.append(fallback)
            else:
                logging.warning(f"[Preassign] Shard not found: {basename}")

    logging.info(
        f"[Preassign] Rank {rank}: loaded {len(my_shards)}/{len(shard_basenames)} shards "
        f"from {assignment_file}"
    )
    return my_shards


def count_cuts_in_shards(shard_jsonls: Sequence[str]) -> int:
    """统计 shard jsonl 中的 cut 数（只读文本，不读 tar）。"""
    total = 0
    for jp in shard_jsonls:
        try:
            with gzip.open(jp, "rt") as f:
                for _ in f:
                    total += 1
        except Exception as e:
            logging.warning(f"[Preassign] Failed to count cuts in {jp}: {e}")
    return total


def create_cuts_from_assigned_shards(
    shard_jsonls: List[str],
    epoch: int = 0,
    seed: int = 42,
    shuffle: bool = True,
) -> CutSet:
    """对本 rank 的预分配 shard 做 epoch shuffle，再构建 lazy CutSet。"""
    if not shard_jsonls:
        raise ValueError("No shards assigned to create CutSet")

    epoch_shards = list(shard_jsonls)
    if shuffle:
        rng = random.Random(seed + epoch)
        rng.shuffle(epoch_shards)

    shard_feats = [
        p.replace("cuts.", "features.").replace(".jsonl.gz", ".tar")
        for p in epoch_shards
    ]
    logging.info(
        f"[Preassign] Epoch {epoch}: building CutSet.from_shar with "
        f"{len(epoch_shards)} shards (shuffle={shuffle}, seed={seed + epoch})"
    )
    return CutSet.from_shar(
        fields={"cuts": epoch_shards, "features": shard_feats},
        shuffle_shards=False,
        split_for_dataloading=False,
    )


class SyncExhaustDataLoader:
    """包装 DataLoader：任一 rank 耗尽时，所有 rank 同步结束本 epoch。

    与 icefall shar_pool 训练循环一致，避免各 rank batch 数不同导致
    Lightning DDP 在 epoch 边界 allreduce 死等 / NCCL timeout。
    """

    def __init__(self, loader):
        self._loader = loader
        # Lightning / checkpoint 可能访问这些属性
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.num_workers = loader.num_workers
        self.pin_memory = getattr(loader, "pin_memory", False)
        self.drop_last = getattr(loader, "drop_last", False)
        self.sampler = getattr(loader, "sampler", None)
        self.batch_sampler = getattr(loader, "batch_sampler", None)
        self.collate_fn = getattr(loader, "collate_fn", None)
        self.generator = getattr(loader, "generator", None)
        self.prefetch_factor = getattr(loader, "prefetch_factor", 2)
        self.persistent_workers = getattr(loader, "persistent_workers", False)

    def __iter__(self):
        import torch
        import torch.distributed as dist

        it = iter(self._loader)
        use_dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        if use_dist and torch.cuda.is_available():
            sync_device = torch.device("cuda", torch.cuda.current_device())
        else:
            sync_device = torch.device("cpu")

        batch_idx = 0
        while True:
            exhausted = 0
            batch = None
            try:
                batch = next(it)
            except StopIteration:
                exhausted = 1

            if use_dist:
                flag = torch.tensor([exhausted], device=sync_device, dtype=torch.int32)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                if flag.item() > 0:
                    if exhausted:
                        logging.info(
                            f"[SyncExhaust] local exhausted after {batch_idx} batches, "
                            f"all ranks stopping epoch together"
                        )
                    else:
                        logging.info(
                            f"[SyncExhaust] peer exhausted after {batch_idx} batches, "
                            f"stopping epoch (discard local leftover batch)"
                        )
                    break
            elif exhausted:
                break

            yield batch
            batch_idx += 1

    def __len__(self):
        # DynamicBucketingSampler 常使 DataLoader.__len__==0；
        # 若这里返回 0，Lightning 会判定无 batch 直接结束 fit。
        # 抛 TypeError 让 Lightning 按 Iterable + max_steps 跑。
        try:
            n = len(self._loader)
        except TypeError as e:
            raise TypeError("SyncExhaustDataLoader has no length") from e
        if n == 0:
            raise TypeError("SyncExhaustDataLoader has no length")
        return n

    def __getattr__(self, name):
        return getattr(self._loader, name)
