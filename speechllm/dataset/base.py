"""
基类模块：BasicDataset, DatasetNonStream, DatasetStream

继承关系:
    BasicDataset
    ├── DatasetNonStream  (非流式数据集基类)
    │   ├── DatasetForASR
    │   ├── DatasetForAST
    │   └── DatasetForASRMultiLang
    └── DatasetStream     (流式数据集基类)
        ├── DatasetForStreamASR
        ├── DatasetForStreamAST
        ├── DatasetForStreamASRAST
        └── DatasetForStreamASRLeftPad
"""

import logging
import random
import torch
from typing import Dict, List, Optional, Union, Any

from torch.utils.data import Dataset
from lhotse import CutSet
from lhotse.dataset import SpecAugment, CutMix, CutConcatenate
from lhotse.dataset.input_strategies import PrecomputedFeatures

from .utils import load_cuts, load_cuts_lazy, lang_map, UNKONOW_LANG

# 提取常量
NO_SPACE_PUNCTUATIONS = {
    ',', '.', '!', '?', ':', ';',
    '，', '。', '！', '？', '：', '；',
    ')', ']', '}', '%'
}


# =============================================================================
# BasicDataset: 所有数据集的公共基类
# =============================================================================
class BasicDataset(Dataset):
    """
    所有数据集的公共基类。
    负责: 数据加载、transforms 初始化、SpecAugment、特征提取、pad_to_stride。
    """

    def __init__(
        self,
        manifest_paths: Union[str, List[str]],
        mode: str = 'train',
        config: Optional[Dict] = None
    ):
        self.mode = mode
        self.config = config or {}
        self.return_ids = False
        self.processed_count = 0
        self._estimated_len = None

        # Handle list of paths
        if isinstance(manifest_paths, list):
            manifest_paths = ",".join(manifest_paths) if manifest_paths else ""

        # shar_pool preassigned：cuts 在每个 epoch 的 train_dataloader 里注入
        sampler_type = self.config.get("sampler_type", "")
        defer_cuts = (
            mode == "train"
            and sampler_type == "shar_pool_sampler"
            and bool(self.config.get("shard_assignment_file"))
        )

        if defer_cuts:
            self.cuts = None
            logging.info(
                "[Dataset] shar_pool_sampler (preassigned): defer CutSet loading until "
                "train_dataloader (shuffle assigned shards each epoch)"
            )
        else:
            if not manifest_paths:
                raise ValueError(
                    f"manifest_paths is required for mode={mode}, sampler_type={sampler_type}"
                )
            lazy_load = self.config.get("lazy_load", False)
            self.cuts = load_cuts(manifest_paths, shuffle=(mode == 'train'), lazy=lazy_load)

        # 模拟流式的填充
        self.pad_last_chunk = self.config.get("pad_last_chunk", False)
        self.stride = self.config.get("pad_chunk_size", 32)

        # 初始化 transforms 和特征提取
        self._init_transforms()
        self._init_input_strategy()
        self._init_input_transforms()

    def _init_transforms(self):
        """初始化数据增强 transforms (CutConcatenate, CutMix)"""
        self.transforms = []
        if self.mode == 'train':
            if self.config.get("concatenate_cuts", False):
                self.transforms.append(
                    CutConcatenate(
                        duration_factor=self.config.get("duration_factor", 1.0),
                        gap=self.config.get("gap", 1.0)
                    )
                )

            if self.config.get("enable_musan", False):
                noise_paths = self.config.get("noise_cut_paths", None)
                if noise_paths:
                    if isinstance(noise_paths, list):
                        noise_paths = ",".join(noise_paths)
                    cuts_musan = load_cuts_lazy(noise_paths)
                    add_noise_prob = self.config.get("add_noise_prob", 0.5)
                    snr_min = self.config.get("snr_min", 10)
                    snr_max = self.config.get("snr_max", 20)
                    self.transforms.append(
                        CutMix(cuts=cuts_musan, p=add_noise_prob, snr=(snr_min, snr_max), preserve_id=True)
                    )
                    logging.info(f"Noise augmentation enabled: prob={add_noise_prob}, snr=({snr_min}, {snr_max})")
                else:
                    logging.warning("enable_musan 为 True 但未提供 noise_cut_paths。跳过 MUSAN 增强。")

    def _init_input_strategy(self):
        """初始化输入策略 (特征提取)"""
        logging.info("使用 PrecomputedFeatures (读取预计算特征)")
        self.input_strategy = PrecomputedFeatures()

    def _init_input_transforms(self):
        """初始化输入变换 (SpecAugment)"""
        self.input_transforms = []
        if self.mode == 'train' and self.config.get("enable_spec_aug", False):
            time_warp_factor = self.config.get("spec_aug_time_warp_factor", 0)
            num_feature_masks = self.config.get("spec_aug_num_feature_masks", 2)
            features_mask_size = self.config.get("spec_aug_features_mask_size", 27)
            num_frame_masks = self.config.get("spec_aug_num_frame_masks", 2)
            frames_mask_size = self.config.get("spec_aug_frames_mask_size", 50)

            self.input_transforms.append(
                SpecAugment(
                    time_warp_factor=time_warp_factor,
                    num_feature_masks=num_feature_masks,
                    features_mask_size=features_mask_size,
                    num_frame_masks=num_frame_masks,
                    frames_mask_size=frames_mask_size
                )
            )

    def __len__(self):
        if self._estimated_len is not None:
            return self._estimated_len
        if self.cuts is None:
            return 0
        try:
            return len(self.cuts)
        except Exception:
            return 0

    def set_cuts(self, cuts: CutSet, estimated_len: Optional[int] = None):
        """注入 / 替换当前 CutSet（shar_pool 每 epoch 重新分配 shard 时使用）。"""
        self.cuts = cuts
        self._estimated_len = estimated_len

    def set_return_ids(self, value: bool):
        self.return_ids = value

    def _resolve_cuts(self, cut_ids) -> CutSet:
        """统一的 cut 解析逻辑"""
        if hasattr(cut_ids, 'subset'):
            return cut_ids
        elif hasattr(cut_ids, 'id') and not isinstance(cut_ids, str):
            return CutSet.from_cuts([cut_ids])
        else:
            if isinstance(cut_ids, str):
                cut_ids = [cut_ids]
            return self.cuts.subset(cut_ids=cut_ids)

    def _extract_features(self, cuts: CutSet):
        """提取特征、应用 transforms、pad、SpecAugment"""
        for t in self.transforms:
            cuts = t(cuts)

        features, features_lens = self.input_strategy(cuts)

        if self.pad_last_chunk:
            features, features_lens = self._pad_to_stride(features, features_lens)

        for t in self.input_transforms:
            features = t(features)

        return cuts, features, features_lens

    def _pad_to_stride(self, features: torch.Tensor, features_lens: torch.Tensor):
        """
        统一填充 Tensor 到 Batch 最大的对齐长度，并更新有效长度列表。
        """
        max_current_len = features.size(1)
        num_steps = (max_current_len + self.stride - 1) // self.stride
        new_target_len = num_steps * self.stride + 13

        if new_target_len > max_current_len:
            features = torch.nn.functional.pad(
                features, (0, 0, 0, new_target_len - max_current_len), "constant", 0
            )
        elif new_target_len < max_current_len:
            features = features[:, :new_target_len, :]

        features_lens = ((features_lens + self.stride - 1) // self.stride) * self.stride + 13
        return features, features_lens


# =============================================================================
# DatasetNonStream: 非流式数据集基类
# =============================================================================
class DatasetNonStream(BasicDataset):
    """
    非流式数据集基类。
    子类需要实现 _extract_texts_and_prompts(cuts) 方法。
    """

    def __init__(
        self,
        manifest_paths: Union[str, List[str]],
        mode: str = 'train',
        config: Optional[Dict] = None
    ):
        super().__init__(manifest_paths, mode, config)

    def _extract_texts_and_prompts(self, cuts: CutSet):
        """
        从 cuts 中提取文本和提示词。
        子类必须实现此方法。

        返回值取决于子类:
        - 无 prompt 的场景: 返回 (texts,)
        - 有 prompt 的场景: 返回 (texts, prompts)
        """
        raise NotImplementedError

    def __getitem__(self, cut_ids):
        cuts = self._resolve_cuts(cut_ids)
        self.processed_count += len(cuts)

        cuts, features, features_lens = self._extract_features(cuts)

        result = self._extract_texts_and_prompts(cuts)

        if self.return_ids:
            return (features, features_lens) + result + ([c.id for c in cuts],)
        return (features, features_lens) + result


# =============================================================================
# DatasetStream: 流式数据集基类
# =============================================================================
class DatasetStream(BasicDataset):
    """
    流式数据集基类。
    负责: chunk 采样、alignment 展平、单 cut 切块处理、prompt 生成。
    子类可以覆盖 _process_single_cut / _get_prompt 来定制行为。
    """

    def __init__(
        self,
        manifest_paths: Union[str, List[str]],
        mode: str = 'train',
        config: Optional[Dict] = None
    ):
        # 流式数据集默认 lazy_load=True, pad_last_chunk=True
        config = config or {}
        config.setdefault("lazy_load", True)
        config.setdefault("pad_last_chunk", True)
        super().__init__(manifest_paths, mode, config)

        # 流式特有的基础配置
        self.frame_shift = 0.01
        self.downsample_factor = 4 * self.config.get("pooling_factor", 4)
        self.frame_offset = 7
        self.start_current_time = self.config.get("start_current_time", 0.025)

        # 流式与解码块配置
        self.overlap_threshold = self.config.get("overlap_threshold", 0.02)
        self.overlap_ratio = self.config.get("overlap_ratio", 0.9)
        self.mask_lang_prob = self.config.get("mask_lang_prob", 0.5) if mode == 'train' else 0.0

        self._init_chunk_probabilities()

    def _init_input_transforms(self):
        """流式数据集的 SpecAugment 默认 time_warp_factor=0"""
        self.input_transforms = []
        if self.mode == 'train' and self.config.get("enable_spec_aug", False):
            time_warp_factor = self.config.get("spec_aug_time_warp_factor", 0)
            num_feature_masks = self.config.get("spec_aug_num_feature_masks", 2)
            features_mask_size = self.config.get("spec_aug_features_mask_size", 27)
            num_frame_masks = self.config.get("spec_aug_num_frame_masks", 2)
            frames_mask_size = self.config.get("spec_aug_frames_mask_size", 100)

            self.input_transforms.append(
                SpecAugment(
                    time_warp_factor=time_warp_factor,
                    num_feature_masks=num_feature_masks,
                    features_mask_size=features_mask_size,
                    num_frame_masks=num_frame_masks,
                    frames_mask_size=frames_mask_size
                )
            )

    def _init_chunk_probabilities(self):
        """初始化 Chunk 采样概率"""
        decode_chunks_config = (
            self.config.get("decode_chunks", {"1": 1.0})
            if self.mode == 'train'
            else {"1": 1.0}
        )
        self.chunk_counts = [int(k) for k in decode_chunks_config.keys()]
        probs = list(decode_chunks_config.values())
        total_prob = sum(probs)
        self.chunk_probs = [p / total_prob for p in probs]

    def set_decode_chunk_num(self, chunk_num: int):
        self.chunk_counts = [int(chunk_num)]
        self.chunk_probs = [1.0]

    def _select_chunk_count(self) -> int:
        return random.choices(self.chunk_counts, weights=self.chunk_probs, k=1)[0]

    def _seconds_to_connector_index(self, seconds: float) -> int:
        fbank_frame = int(round((seconds - self.start_current_time) / self.frame_shift))
        return (fbank_frame + self.frame_offset) // self.downsample_factor

    @staticmethod
    def _is_cjk_char(ch: str) -> bool:
        """判断字符是否为中日韩统一表意文字"""
        cp = ord(ch)
        return (
            (0x4E00 <= cp <= 0x9FFF)       # CJK Unified Ideographs
            or (0x3400 <= cp <= 0x4DBF)    # CJK Unified Ideographs Extension A
            or (0x20000 <= cp <= 0x2A6DF)  # CJK Unified Ideographs Extension B
            or (0x2A700 <= cp <= 0x2B73F)  # CJK Unified Ideographs Extension C
            or (0x2B740 <= cp <= 0x2B81F)  # CJK Unified Ideographs Extension D
            or (0xF900 <= cp <= 0xFAFF)    # CJK Compatibility Ideographs
            or (0x2F800 <= cp <= 0x2FA1F)  # CJK Compatibility Ideographs Supplement
        )

    @staticmethod
    def _format_text(words: List[str]) -> str:
        """根据标点符号和中文字符规则合并文本列表。
        - 当前 word 以标点开头：不加空格
        - 当前 word 以中文字符开头：不加空格
        - 前一个 word 以中文字符结尾：不加空格
        - 其余情况：加空格
        """
        if not words:
            return ""
        merged_text = words[0]
        for word in words[1:]:
            if not word:
                continue
            if word[0] in NO_SPACE_PUNCTUATIONS:
                merged_text += word
            elif DatasetStream._is_cjk_char(word[0]):
                merged_text += word
            elif merged_text and DatasetStream._is_cjk_char(merged_text[-1]):
                merged_text += word
            else:
                merged_text += " " + word
        return merged_text.strip()

    def _extract_flat_alignments(self, supervisions, time_offset: float = 0.0) -> List[Dict[str, Any]]:
        """
        将复杂的 supervision 对齐结构展平为一个绝对时间戳的一维列表。
        time_offset: 在所有时间戳上叠加的偏移量（用于左侧 pad 后的时间校正）。
        """
        flat_items = []
        for idx, sup in enumerate(supervisions):
            sup_start = sup.start + time_offset
            sup_duration = sup.duration

            alignment_data = []
            is_relative = False

            if hasattr(sup, 'alignment') and sup.alignment is not None:
                if isinstance(sup.alignment, dict) and 'word' in sup.alignment:
                    alignment_data = sup.alignment['word']
                    is_relative = True
                elif hasattr(sup.alignment, 'word'):
                    alignment_data = sup.alignment.word
                    is_relative = True
            elif hasattr(sup, 'custom') and sup.custom and 'alignment' in sup.custom:
                if 'word' in sup.custom['alignment']:
                    alignment_data = sup.custom['alignment']['word']
                    is_relative = True

            if not alignment_data:
                sup_text = getattr(sup, 'text', "").strip()
                if sup_text:
                    alignment_data = [{
                        'symbol': sup_text,
                        'start': sup_start,
                        'duration': sup_duration,
                        'id': f"{idx}_fallback"
                    }]
                    is_relative = False

            for i, item in enumerate(alignment_data):
                is_dict = isinstance(item, dict)
                item_symbol = item['symbol'] if is_dict else item.symbol
                item_start = item['start'] if is_dict else item.start
                item_duration = item['duration'] if is_dict else item.duration

                if is_dict and 'id' in item:
                    item_id = item['id']
                else:
                    item_id = f"{idx}_{i}_{item_start:.3f}"

                if is_relative:
                    abs_start = sup_start + item_start
                else:
                    abs_start = item_start + time_offset if item_start != sup_start else sup_start

                flat_items.append({
                    'id': item_id,
                    'symbol': item_symbol,
                    'start': abs_start,
                    'end': abs_start + item_duration,
                    'duration': item_duration
                })

        return flat_items

    def _assign_words_to_segment(
        self,
        flat_alignments: List[Dict[str, Any]],
        segment_end_time: float,
        assigned_item_ids: set
    ) -> List[str]:
        """
        将词分配到当前时间段，返回分配到的词列表。

        对每个词计算判定时间点:
          assign_time = max(start + duration * overlap_ratio, end - overlap_threshold)
        当 assign_time <= segment_end_time 时分配。
        设为 -1 表示禁用对应项。

        - 短词: ratio 项主导 (说了 80% 就分配)
        - 长片段: threshold 项主导 (结束前 0.4s 就分配)
        """
        segment_words = []
        for item in flat_alignments:
            if item['id'] in assigned_item_ids:
                continue

            candidates = []
            if self.overlap_ratio >= 0:
                candidates.append(item['start'] + item['duration'] * self.overlap_ratio)
            if self.overlap_threshold >= 0:
                candidates.append(item['end'] - self.overlap_threshold)

            assign_time = max(candidates) if candidates else item['end']

            if assign_time <= segment_end_time:
                segment_words.append(item['symbol'])
                assigned_item_ids.add(item['id'])

        return segment_words

    def _compute_segment_indices(self, current_time: float, segment_end_time: float):
        """计算 segment 的 start_idx 和 end_idx，并做 align_factor 对齐"""
        start_idx = self._seconds_to_connector_index(current_time)
        end_idx = self._seconds_to_connector_index(segment_end_time)

        pooling_factor = self.config.get("pooling_factor", 4)
        align_factor = 8 // pooling_factor
        token_count = end_idx - start_idx
        remainder = token_count % align_factor
        if remainder != 0:
            end_idx += (align_factor - remainder)
        elif token_count == 0:
            end_idx += align_factor

        return start_idx, end_idx

    def _process_single_cut(self, cut, time_offset: float = 0.0) -> List[Dict[str, Any]]:
        """处理单个音频切片的切块和文本分配逻辑"""
        sorted_sups = sorted(cut.supervisions, key=lambda s: s.start)
        if not sorted_sups:
            return []

        num_chunks = self._select_chunk_count()
        chunk_duration_sec = self.stride * 0.01

        flat_alignments = self._extract_flat_alignments(sorted_sups, time_offset=time_offset)

        cut_segments = []
        current_time = self.start_current_time
        audio_duration = cut.duration + time_offset

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

        # 移除末尾为空的段
        if cut_segments and not cut_segments[-1]['text']:
            cut_segments.pop()

        return cut_segments

    def _get_prompt(self, supervisions) -> str:
        """生成 Prompt 文本，子类可覆盖"""
        if self.mode == 'train' and random.random() > self.mask_lang_prob:
            first_sup = supervisions[0] if supervisions else None
            lang_str = getattr(first_sup, 'language', '<unk_lang>')
            lang = lang_map.get(lang_str, UNKONOW_LANG)
            return f"Transcribe the audio in {lang}: "
        return "Transcribe the audio: "

    def __getitem__(self, cut_ids):
        cuts = self._resolve_cuts(cut_ids)
        self.processed_count += len(cuts)

        cuts, features, features_lens = self._extract_features(cuts)

        targets = []
        prompts = []
        for cut in cuts:
            targets.append(self._process_single_cut(cut))
            sorted_sups = sorted(cut.supervisions, key=lambda s: s.start)
            prompts.append(self._get_prompt(sorted_sups))

        if self.return_ids:
            return features, features_lens, targets, prompts, [c.id for c in cuts]
        return features, features_lens, targets, prompts
