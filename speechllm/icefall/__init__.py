# Vendored icefall helpers used by SpeechLLM training/inference.
# Keep this package self-contained so training does not depend on
# PYTHONPATH pointing at an external /pfs/asr/icefall install.

from .utils import AttributeDict, make_pad_mask

__all__ = ["AttributeDict", "make_pad_mask"]
