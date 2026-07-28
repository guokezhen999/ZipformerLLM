from functools import lru_cache
from lhotse import load_manifest, load_manifest_lazy, CutSet
from tqdm import tqdm
import logging
import re

@lru_cache()
def load_cuts(manifest_paths: str, shuffle: bool = False, lazy: bool = False) -> CutSet:
    paths = [path.strip() for path in re.split(r'[,;]', manifest_paths) if path.strip()]
    if not paths:
        raise ValueError("未提供有效的 manifest 路径。")

    mode_str = "延迟" if lazy else "完全"
    logging.info(f"正在从 {len(paths)} 个 manifest 文件{mode_str}加载 CutSet。")

    load_fn = load_manifest_lazy if lazy else load_manifest
    
    combined_cuts = None
    for path in tqdm(paths, desc=f"加载 manifest ({mode_str})"):
        cuts = load_fn(path)
        if combined_cuts is None:
            combined_cuts = cuts
        else:
            combined_cuts = combined_cuts + cuts
            
    # 可选地对合并后的 CutSet 进行打乱
    if shuffle:
        # 注意：Lhotse 的延迟打乱 (lazy shuffling) 非常高效
        logging.info("正在对 CutSet 进行打乱...")
        combined_cuts = combined_cuts.shuffle()

    return combined_cuts

# Keep the old name for compatibility if needed, but internally use the new flexible one
def load_cuts_lazy(manifest_paths: str, shuffle: bool = False) -> CutSet:
    return load_cuts(manifest_paths, shuffle=shuffle, lazy=True)

lang_map = {
    "es": "Spanish",
    "Spanish": "Spanish",
    "en": "English",
    "English": "English",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "Mandarin": "Chinese",
}

UNKONOW_LANG = "Unknown"