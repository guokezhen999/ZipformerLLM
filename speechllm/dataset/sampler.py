from torch.utils.data import Sampler, Dataset
from typing import Dict
from lhotse.dataset import DynamicBucketingSampler, StatelessSampler
import logging
import json
import os
from pathlib import Path

def get_sampler(
    sampler_type: str,
    dataset: Dataset = None,
    config: Dict = None,
    shuffle: bool = True,
    rank: int = 0,
    world_size: int = 1
):
    if sampler_type == 'pre_sampled_sampler':
        return get_pre_sampled_sampler(dataset, config, shuffle, rank, world_size)
    elif sampler_type == 'dynamic_bucket_sampler':
        return get_dynamic_bucket_sampler(dataset, config, shuffle, rank, world_size)
    elif sampler_type == 'stateless_sampler':
        return get_stateless_sampler(config, rank, world_size)
    elif sampler_type == 'shar_pool_sampler':
        # shard 已在池子层面按 rank 分好，这里固定 rank=0/world_size=1
        return get_shar_pool_sampler(dataset, config, shuffle)
    else:
        raise ValueError(f"[Sampler] 未知的采样器类型: {sampler_type}")

class PreSampledBucketingSampler(Sampler):
    def __init__(self, pre_sampled_path: str, rank: int = 0, world_size: int = 1):
        """
        预采样采样器。仅支持从预先生成的包含完整元数据的 Batch JSONL 文件中读取数据。
        
        Args:
            pre_sampled_path: 预采样 JSON 文件所在的目录。
            rank: 当前进程的 Rank。
            world_size: 总进程数。
        """
        self.pre_sampled_path = pre_sampled_path
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self._effective_length = 0
        
        if not self.pre_sampled_path:
            raise ValueError("[Sampler] PreSampledBucketingSampler 必须提供 pre_sampled_path")
            
        if self.rank == 0:
            logging.info(f"[Sampler] 初始化预采样加载器 (Metadata 模式)，目录: {self.pre_sampled_path}")

    def _get_file_path(self):
        return os.path.join(self.pre_sampled_path, f"epoch_{self.epoch}_rank_{self.rank}.jsonl")

    def __iter__(self):
        file_path = self._get_file_path()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Sampler] 找不到预采样文件: {file_path}。请先运行 prepare_batches.py")
        
        if self.rank == 0:
            logging.info(f"[Sampler] 正在从 {file_path} 加载完整元数据 Batch (JSONL 模式)")
            
        from lhotse import CutSet
        with open(file_path, 'r') as f:
            count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                batch_dicts = json.loads(line)
                # 这里的 batch_dicts 是包含完整 Cut 元数据的列表
                yield CutSet.from_dicts(batch_dicts)
                count += 1
            self._effective_length = count

    def __len__(self):
        if self._effective_length > 0:
            return self._effective_length
        
        file_path = self._get_file_path()
        if os.path.exists(file_path):
            # 对于 JSONL，计算行数即为 Batch 数
            with open(file_path, 'r') as f:
                count = sum(1 for _ in f)
                self._effective_length = count
                return count
        return 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._effective_length = 0

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict.get("epoch", self.epoch)
        self.rank = state_dict.get("rank", self.rank)
        self.world_size = state_dict.get("world_size", self.world_size)
        self._effective_length = 0
       
def get_pre_sampled_sampler(
    dataset: Dataset, 
    config: Dict, 
    shuffle: bool = True, 
    rank: int = 0, 
    world_size: int = 1
):
    pre_sampled_path = config.get('pre_sampled_path', None)
    if not pre_sampled_path:
        raise ValueError("[Sampler] PreSampledBucketingSampler 必须提供 pre_sampled_path")
    logging.info(f"[Sampler] 使用预采样加载模式: {pre_sampled_path}, Rank={rank}, WorldSize={world_size}")
    return PreSampledBucketingSampler(
        pre_sampled_path=pre_sampled_path,
        rank=rank,
        world_size=world_size
    )

def get_dynamic_bucket_sampler(
    dataset: Dataset, 
    config: Dict, 
    shuffle: bool = True, 
    rank: int = 0, 
    world_size: int = 1
):
    max_duration = config.get('max_duration', 200.0)
    num_buckets = config.get('num_buckets', 30)
    max_batch_size = config.get('max_batch_size', None)
    logging.info(f"[Sampler] 使用 Lhotse 动态分桶模式: max_duration={max_duration}, "
                    f"max_batch_size={max_batch_size}, num_buckets={num_buckets}, "
                    f"Shuffle={shuffle}, Rank={rank}, WorldSize={world_size}")
                     
    buffer_size = num_buckets * 2000
    shuffle_buffer_size = num_buckets * 5000 if shuffle else None
        
    return DynamicBucketingSampler(
        dataset.cuts,
        max_duration=max_duration,
        shuffle=shuffle,
        num_buckets=num_buckets,
        buffer_size=buffer_size,
        shuffle_buffer_size=shuffle_buffer_size,
        drop_last=shuffle,
        rank=rank,
        world_size=world_size,
        max_cuts=max_batch_size
    )

def get_stateless_sampler(
    config: Dict,
    rank: int = 0,
    world_size: int = 1
):
    cuts_paths = config.get('cuts_data')
    index_path = config.get('sampler_index_path', None)
    if index_path is not None:
        index_path = Path(index_path)
    base_seed = config.get('seed', 42)
    max_duration = config.get('max_duration', 200.0)
    num_buckets = config.get('num_buckets', 30)
    max_batch_size = config.get('max_batch_size', None)

    logging.info(f"[Sampler] 使用 StatelessSampler 模式: max_duration={max_duration}, "
                    f"max_batch_size={max_batch_size}, num_buckets={num_buckets}, "
                    f"Rank={rank}, WorldSize={world_size}, seed={base_seed}")

    return StatelessSampler(
        cuts_paths=cuts_paths,
        index_path=index_path,
        base_seed=base_seed,
        max_duration=max_duration,
        max_cuts=max_batch_size,
        num_buckets=num_buckets,
    )


def get_shar_pool_sampler(
    dataset: Dataset,
    config: Dict,
    shuffle: bool = True,
):
    """
    Shar 打包数据 + shard 池子分配后的 DynamicBucketingSampler。

    关键：rank/world_size 固定为 0/1，因为每个进程拿到的 CutSet
    已经只包含本 rank 的 shard，不再做 cut 级分区。
    """
    if dataset is None or getattr(dataset, "cuts", None) is None:
        raise ValueError(
            "[Sampler] shar_pool_sampler 需要 dataset.cuts 已注入 "
            "（通常在 train_dataloader 中按 epoch 分配 shard 后 set_cuts）"
        )

    max_duration = config.get('max_duration', 200.0)
    num_buckets = config.get('num_buckets', 30)
    max_batch_size = config.get('max_batch_size', None)
    drop_last = config.get('drop_last', shuffle)

    buffer_size = config.get('buffer_size', num_buckets * 2000)
    shuffle_buffer_size = config.get(
        'shuffle_buffer_size',
        num_buckets * 5000 if shuffle else None,
    )

    logging.info(
        f"[Sampler] 使用 SharPool+DynamicBucketing 模式: max_duration={max_duration}, "
        f"max_batch_size={max_batch_size}, num_buckets={num_buckets}, "
        f"Shuffle={shuffle}, Rank=0, WorldSize=1 (shard-level DDP)"
    )

    return DynamicBucketingSampler(
        dataset.cuts,
        max_duration=max_duration,
        shuffle=shuffle,
        num_buckets=num_buckets,
        buffer_size=buffer_size,
        shuffle_buffer_size=shuffle_buffer_size,
        drop_last=drop_last,
        rank=0,
        world_size=1,
        max_cuts=max_batch_size,
    )

