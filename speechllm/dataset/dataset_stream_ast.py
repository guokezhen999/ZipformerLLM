import logging
import random
import torch
from typing import Dict, List, Optional, Union, Any

from lhotse import CutSet

from .base import DatasetStream
from .utils import lang_map, UNKONOW_LANG

# --- Monkey-Patch Lhotse SupervisionSegment -------------------------------------
from lhotse import SupervisionSegment

_original_supervision_from_dict = SupervisionSegment.from_dict


@classmethod
def _patched_supervision_from_dict(cls, data: dict):
    if 'translation' in data:
        if 'custom' not in data or data['custom'] is None:
            data['custom'] = {}
        data['custom']['translation'] = data.pop('translation')
    return _original_supervision_from_dict(data)


SupervisionSegment.from_dict = _patched_supervision_from_dict
# --------------------------------------------------------------------------------


class DatasetForStreamAST(DatasetStream):
    """
    流式 AST/ASR 数据集。
    - 如果 cut 的 supervision 中包含翻译片段（translation），则做翻译任务，
      targets 中使用翻译文本，prompt 为 "translate the audio in {lang}: "。
    - 否则做识别任务，targets 中使用识别文本，prompt 为 "Transcribe the audio: "。
    """

    def __init__(
        self,
        manifest_paths: Union[str, List[str]],
        mode: str = 'train',
        config: Optional[Dict] = None
    ):
        super().__init__(manifest_paths, mode, config)
        self._init_ast_chunk_probabilities()

    def _init_ast_chunk_probabilities(self):
        """初始化 AST 任务的 Chunk 采样概率"""
        ast_chunks_config = (
            self.config.get("ast_decode_chunks", None)
            if self.mode == 'train' else None
        )
        if ast_chunks_config is not None:
            self.ast_chunk_counts = [int(k) for k in ast_chunks_config.keys()]
            ast_probs = list(ast_chunks_config.values())
            ast_total_prob = sum(ast_probs)
            self.ast_chunk_probs = [p / ast_total_prob for p in ast_probs]
        else:
            self.ast_chunk_counts = self.chunk_counts
            self.ast_chunk_probs = self.chunk_probs

    def set_ast_decode_chunk_num(self, chunk_num: int):
        """设置 AST 任务的 decode chunk 数量（验证时使用）"""
        self.ast_chunk_counts = [int(chunk_num)]
        self.ast_chunk_probs = [1.0]

    def _select_ast_chunk_count(self) -> int:
        """AST 任务的 chunk 采样"""
        return random.choices(self.ast_chunk_counts, weights=self.ast_chunk_probs, k=1)[0]

    def _extract_flat_translations(self, supervisions) -> List[Dict[str, Any]]:
        """提取翻译对齐数据，将其展平为具有绝对时间戳的一维列表。"""
        flat_items = []
        for idx, sup in enumerate(supervisions):
            sup_start = sup.start
            sup_duration = sup.duration

            translation_list = []
            if hasattr(sup, 'translation') and sup.translation is not None:
                translation_list = sup.translation
            elif hasattr(sup, 'custom') and sup.custom and 'translation' in sup.custom:
                translation_list = sup.custom['translation']

            alignment_data = []
            is_relative = True

            if isinstance(translation_list, list) and len(translation_list) > 0:
                trans_obj = translation_list[0]
                if isinstance(trans_obj, dict) and 'segments' in trans_obj:
                    alignment_data = trans_obj['segments']
                elif hasattr(trans_obj, 'segments'):
                    alignment_data = trans_obj.segments

            if not alignment_data:
                if isinstance(translation_list, list) and len(translation_list) > 0:
                    trans_obj = translation_list[0]
                    trans_text = (
                        trans_obj.get('text', "")
                        if isinstance(trans_obj, dict)
                        else getattr(trans_obj, 'text', "")
                    )
                    if trans_text:
                        alignment_data = [{
                            'symbol': trans_text,
                            'start': 0.0,
                            'duration': sup_duration,
                            'id': f"trans_{idx}_fallback"
                        }]

            for i, item in enumerate(alignment_data):
                is_dict = isinstance(item, dict)
                item_symbol = item['symbol'] if is_dict else item.symbol
                item_start = item['start'] if is_dict else item.start
                item_duration = item['duration'] if is_dict else item.duration

                if is_dict and 'id' in item:
                    item_id = item['id']
                else:
                    item_id = f"trans_{idx}_{i}_{item_start:.3f}"

                abs_start = sup_start + item_start if is_relative else item_start

                flat_items.append({
                    'id': item_id,
                    'symbol': item_symbol,
                    'start': abs_start,
                    'end': abs_start + item_duration,
                    'duration': item_duration
                })

        return flat_items

    def _has_translation(self, supervisions) -> bool:
        """检查 supervision 中是否包含翻译数据"""
        for sup in supervisions:
            translation_list = []
            if hasattr(sup, 'translation') and sup.translation is not None:
                translation_list = sup.translation
            elif hasattr(sup, 'custom') and sup.custom and 'translation' in sup.custom:
                translation_list = sup.custom['translation']

            if isinstance(translation_list, list) and len(translation_list) > 0:
                return True
        return False

    def _process_single_cut(self, cut, time_offset: float = 0.0) -> List[Dict[str, Any]]:
        """
        处理单个音频切片。
        如果包含翻译片段，则返回翻译文本；否则返回识别文本。
        """
        sorted_sups = sorted(cut.supervisions, key=lambda s: s.start)
        if not sorted_sups:
            return []

        use_translation = self._has_translation(sorted_sups)
        num_chunks = self._select_ast_chunk_count() if use_translation else self._select_chunk_count()
        chunk_duration_sec = self.stride * 0.01

        if use_translation:
            flat_alignments = self._extract_flat_translations(sorted_sups)
        else:
            flat_alignments = self._extract_flat_alignments(sorted_sups, time_offset=time_offset)

        cut_segments = []
        current_time = self.start_current_time
        audio_duration = cut.duration + time_offset

        cumulative_overlaps = {}
        assigned_item_ids = set()
        is_first_segment = True

        while current_time < audio_duration:
            if num_chunks == -1:
                segment_end_time = audio_duration
            else:
                segment_end_time = min(
                    current_time + (num_chunks * chunk_duration_sec), audio_duration
                )

            start_idx, end_idx = self._compute_segment_indices(current_time, segment_end_time)

            text_search_start = 0.0 if is_first_segment else current_time
            segment_words = self._assign_words_to_segment(
                flat_alignments, segment_end_time,
                assigned_item_ids
            )

            cut_segments.append({
                "text": self._format_text(segment_words),
                "start_idx": start_idx,
                "end_idx": end_idx,
            })

            current_time = segment_end_time
            is_first_segment = False

            if num_chunks == -1:
                break

        # 处理漏网之鱼
        unassigned_words = [
            item['symbol'] for item in flat_alignments
            if item['id'] not in assigned_item_ids
        ]
        if unassigned_words and cut_segments:
            last_text = cut_segments[-1]['text']
            combined_words = [last_text] + unassigned_words if last_text else unassigned_words
            cut_segments[-1]['text'] = self._format_text(combined_words)

        if cut_segments and not cut_segments[-1]['text']:
            cut_segments.pop()

        return cut_segments

    def _get_prompt(self, supervisions) -> str:
        """生成 Prompt 文本。翻译任务使用翻译 prompt，否则使用识别 prompt。"""
        first_sup = supervisions[0] if supervisions else None

        if first_sup:
            translation_list = []
            if hasattr(first_sup, 'translation') and first_sup.translation is not None:
                translation_list = first_sup.translation
            elif hasattr(first_sup, 'custom') and first_sup.custom and 'translation' in first_sup.custom:
                translation_list = first_sup.custom['translation']

            if isinstance(translation_list, list) and len(translation_list) > 0:
                trans_obj = translation_list[0]
                lang_str = (
                    trans_obj.get('lang', '<unk_lang>')
                    if isinstance(trans_obj, dict)
                    else getattr(trans_obj, 'lang', '<unk_lang>')
                )
                target_lang = lang_map.get(lang_str, UNKONOW_LANG)
                return f"Translate the audio in {target_lang}: "

        # 没有翻译数据，使用识别 prompt
        return super()._get_prompt(supervisions)


class DatasetForStreamASTCollate:
    """Collate 函数，输出格式与 DatasetForStreamASRCollate 一致。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        batch_features, audio_lengths, targets, prompts = batch

        tokenized_prompts = self.tokenizer(
            prompts,
            padding=True,
            return_tensors='pt',
            add_special_tokens=False
        )
        prompt_ids = tokenized_prompts.input_ids
        prompt_lens = tokenized_prompts.attention_mask.sum(dim=1)

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
