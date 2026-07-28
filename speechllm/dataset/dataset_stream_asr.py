import torch
from typing import Dict, List, Optional, Union

from .base import DatasetStream


class DatasetForStreamASR(DatasetStream):
    """流式 ASR 数据集。"""

    def __init__(
        self,
        manifest_paths: Union[str, List[str]],
        mode: str = 'train',
        config: Optional[Dict] = None
    ):
        super().__init__(manifest_paths, mode, config)


class DatasetForStreamASRCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        batch_features, audio_lengths, targets, prompts = batch

        # 处理 Prompts
        tokenized_prompts = self.tokenizer(
            prompts,
            padding=True,
            return_tensors='pt',
            add_special_tokens=False
        )
        prompt_ids = tokenized_prompts.input_ids
        prompt_lens = tokenized_prompts.attention_mask.sum(dim=1)

        # 处理 Targets
        flat_texts = []
        for sample_segments in targets:
            for segment in sample_segments:
                flat_texts.append(segment['text'])

        if flat_texts:
            tokenized_segments = self.tokenizer(
                flat_texts,
                padding=False,
                add_special_tokens=False
            )
            seg_input_ids = tokenized_segments.input_ids
        else:
            seg_input_ids = []

        global_seg_idx = 0
        for sample_segments in targets:
            for segment in sample_segments:
                ids_tensor = torch.tensor(seg_input_ids[global_seg_idx], dtype=torch.long)
                segment['input_ids'] = ids_tensor
                segment['input_len'] = len(ids_tensor)
                global_seg_idx += 1

        return batch_features, audio_lengths, targets, prompt_ids, prompt_lens
