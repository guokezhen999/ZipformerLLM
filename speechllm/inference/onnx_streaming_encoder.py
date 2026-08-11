"""ONNX streaming encoder+adapter wrapper for stage-1 inference."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class OnnxStreamingEncoder:
    """Run stage-1 encoder+adapter ONNX and accumulate audio embeddings.

    Feature windowing matches streaming_session / model_asr.forward_audio:
        overlapping windows of ``input_time_steps`` (e.g. 45) with stride
        ``decode_chunk_len`` (e.g. 32). Do **not** feed non-overlapping
        stride-sized chunks and right-pad with zeros — that drops right-context
        frames and breaks alignment with dataset segment indices.
    """

    def __init__(
        self,
        export_dir: str,
        init_states_path: Optional[str] = None,
        providers: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime") from exc

        export_path = Path(export_dir)
        meta_path = export_path / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {export_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        onnx_file = export_path / self.meta["onnx_file"]
        if not onnx_file.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_file}")

        available = ort.get_available_providers()
        if providers is None:
            want_gpu = str(device).lower() in ("cuda", "gpu")
            if want_gpu and "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        # Drop unavailable providers rather than failing hard.
        providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(onnx_file),
            providers=providers,
            sess_options=self._make_session_options(ort),
        )
        self.providers = list(self.session.get_providers())
        self.state_input_names = [inp.name for inp in self.session.get_inputs() if inp.name != "x"]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.feature_dim = int(self.meta["feature_dim"])
        self.llm_dim = int(self.meta["llm_dim"])
        self.decode_chunk_len = int(self.meta["decode_chunk_len"])
        self.input_time_steps = int(self.meta["input_time_steps"])
        self._states: Dict[str, np.ndarray] = {}
        self._embed_chunks: List[np.ndarray] = []

        npz_path = Path(init_states_path) if init_states_path else export_path / "init_states.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"ONNX init states not found: {npz_path}\n"
                "Re-run stage-1 export: bash scripts/export/run_export.sh 1 1\n"
                "Or pass --init_states_path to decode_ast_stream_gguf.py"
            )
        self._init_states_template = dict(np.load(npz_path))

    @staticmethod
    def _make_session_options(ort_module):
        """Cap ORT threads; unset env would otherwise use all cores."""
        import os

        opts = ort_module.SessionOptions()
        n = int(os.environ.get("ORT_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS") or 8)
        n = max(1, n)
        opts.intra_op_num_threads = n
        opts.inter_op_num_threads = 1
        return opts

    def reset(self) -> None:
        self._states = {k: v.copy() for k, v in self._init_states_template.items()}
        self._embed_chunks = []

    def feed_features(self, features: np.ndarray) -> np.ndarray:
        """
        Feed one streaming chunk of fbank features (already windowed to
        ``input_time_steps``, or shorter on the final step — then right-padded).

        Args:
            features: shape (T, feature_dim) or (1, T, feature_dim)
        Returns:
            chunk embeddings: shape (chunk_out_len, llm_dim)
        """
        if features.ndim == 2:
            features = features[np.newaxis, ...]
        batch_size = features.shape[0]

        if features.shape[2] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {features.shape[2]}")

        t = features.shape[1]
        if t < self.input_time_steps:
            pad = np.zeros((batch_size, self.input_time_steps - t, self.feature_dim), dtype=np.float32)
            features = np.concatenate([features, pad], axis=1)
        elif t > self.input_time_steps:
            features = features[:, : self.input_time_steps, :]

        ort_inputs = {"x": features.astype(np.float32)}
        ort_inputs.update(self._states)

        outputs = self.session.run(None, ort_inputs)
        embeds = outputs[0]
        self._embed_chunks.append(embeds[0])

        for name, tensor in zip(self.state_input_names, outputs[1:]):
            self._states[name] = tensor

        return embeds[0]

    def encode_features(self, features: np.ndarray, reset: bool = True) -> np.ndarray:
        """
        Encode a full utterance with overlapping streaming windows.

        Mirrors ``StreamingSpeechLLM.forward_audio`` /
        ``streaming_session``: stride = decode_chunk_len, window = input_time_steps.
        """
        feats = np.asarray(features, dtype=np.float32)
        if feats.ndim == 3:
            feats = feats[0]
        if feats.ndim != 2:
            raise ValueError(f"Expected features (T, F) or (1, T, F), got {feats.shape}")

        if reset:
            self.reset()

        t = feats.shape[0]
        stride = self.decode_chunk_len
        tail = self.input_time_steps
        num_steps = math.ceil(t / stride) if t > 0 else 0
        for i in range(num_steps):
            start = i * stride
            chunk = feats[start : start + tail]
            self.feed_features(chunk)

        return self.get_full_embeddings()

    def get_full_embeddings(self) -> np.ndarray:
        if not self._embed_chunks:
            return np.zeros((0, self.llm_dim), dtype=np.float32)
        return np.concatenate(self._embed_chunks, axis=0)
