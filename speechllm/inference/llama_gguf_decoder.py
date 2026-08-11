"""
llama.cpp GGUF decoder with SpeechLLM-style embedding input.

Mirrors speechllm/streaming_model/model_asr_stream.py and
speechllm/app/bin/streaming_session.py KV semantics:

  - Prompt (only when KV is empty) + <A> + audio chunks accumulate in KV
  - </A> is fed as a separate single-step prefill before autoregressive decode
  - Generated text tokens stay in KV for subsequent segments (via embd, not eval)
  - Sentence-ending punctuation clears the entire KV (streaming_session policy)
  - Optional punct_kv_mode truncates trailing punct before <W> (model_stream)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from llama_cpp import Llama
    from llama_cpp._internals import LlamaBatch
except ImportError as exc:
    raise ImportError(
        "llama-cpp-python is required for GGUF inference. Install with: pip install llama-cpp-python"
    ) from exc

logger = logging.getLogger(__name__)

_SENT_END = (".", "?", "!", "。", "？", "！")


class LlamaGgufStreamDecoder:
    """Streaming AST decoder backed by a GGUF model and embedding injection."""

    def __init__(
        self,
        gguf_path: str,
        meta: Dict,
        n_ctx: int = 8192,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        num_chunks: int = 1,
        quantize_injected_embeds: Optional[bool] = None,
        audio_embed_quant_mode: Optional[str] = None,
    ):
        self.meta = meta
        self.gguf_path = Path(gguf_path)
        self.token_a_id = int(meta["special_token_ids"]["<A>"])
        self.token_a_end_id = int(meta["special_token_ids"]["</A>"])
        self.token_w_id = int(meta["special_token_ids"]["<W>"])
        self.llm_dim = int(meta["llm_dim"])
        self.n_gpu_layers = int(n_gpu_layers)

        emb_q = meta.get("embedding_quantization") or (meta.get("quantization") or {}).get("embeddings") or {}
        if quantize_injected_embeds is None:
            quantize_injected_embeds = bool(emb_q.get("enabled", False))
        self.quantize_injected_embeds = bool(quantize_injected_embeds)
        self.audio_embed_quant_mode = (
            audio_embed_quant_mode
            or emb_q.get("audio_scale_mode")
            or "per_token"
        )
        self.audio_embed_scale = emb_q.get("audio_scale")  # optional fixed tensor scale

        self.llm = Llama(
            model_path=str(self.gguf_path),
            n_ctx=n_ctx,
            n_batch=n_batch,
            # llama.cpp: 0 means "use all cores"; clamp to a sane default instead.
            n_threads=(n_threads if n_threads and n_threads > 0 else min(8, os.cpu_count() or 8)),
            n_gpu_layers=self.n_gpu_layers,
            logits_all=False,
            verbose=verbose,
        )
        if self.n_gpu_layers != 0:
            logger.info("GGUF loaded with n_gpu_layers=%s", self.n_gpu_layers)
        if self.llm.n_embd() != self.llm_dim:
            raise ValueError(
                f"GGUF n_embd={self.llm.n_embd()} does not match speechllm_meta llm_dim={self.llm_dim}"
            )

        self.emb_a, self.emb_a_end = self._load_special_token_embeddings()
        self._token_embd: Optional[np.ndarray] = None
        self.n_past = 0
        self.prompt_kv_len: Optional[int] = None

        # Match streaming_session.LLMDecoder eviction policy
        # streaming_session: max_segments = 64 // num_chunks
        self.num_chunks = max(1, int(num_chunks))
        self.max_segments = max(1, 64 // self.num_chunks)
        self.segment_count = 0
        self.is_new_seg = True
        self.embed_chunks_count = 0

    def _load_token_embd_table(self) -> np.ndarray:
        if self._token_embd is not None:
            return self._token_embd

        try:
            import gguf
        except ImportError as exc:
            raise ImportError(
                "gguf is required to embed prompt tokens. Install with: pip install gguf"
            ) from exc

        reader = gguf.GGUFReader(str(self.gguf_path))
        for tensor in reader.tensors:
            if tensor.name.endswith("token_embd.weight"):
                # F16/F32: tensor.data is already numeric. Q8_0/etc need dequantize
                # for SpeechLLM prompt embd injection (batch.embd path).
                data = tensor.data
                qtype = getattr(tensor, "tensor_type", None)
                if qtype is not None and int(qtype) not in (
                    int(gguf.GGMLQuantizationType.F32),
                    int(gguf.GGMLQuantizationType.F16),
                    int(gguf.GGMLQuantizationType.BF16),
                ):
                    data = gguf.dequantize(data, qtype)
                self._token_embd = np.asarray(data, dtype=np.float32)
                return self._token_embd
        raise RuntimeError(f"token_embd.weight not found in {self.gguf_path}")

    def _load_from_gguf_token_embd(self) -> tuple[np.ndarray, np.ndarray]:
        table = self._load_token_embd_table()
        return table[self.token_a_id], table[self.token_a_end_id]

    def _load_special_token_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        # Prefer training input_patch (correct <A>/</A> for embd injection).
        # Legacy special_token_embeddings.npz may contain output_patch rows if the
        # HF export overwrote tied embed_tokens with lm_head patches.
        # INT8 export: special_token_input_patch.int8.npz (emb_* + emb_*_scale).
        from speechllm.inference.embed_quant import unpack_special_token_int8

        emb_q = self.meta.get("embedding_quantization") or {}
        candidates = [
            emb_q.get("special_token_file"),
            self.meta.get("special_token_input_patch_file", "special_token_input_patch.npz"),
            "special_token_input_patch.int8.npz",
            self.meta.get("special_token_embeddings_file", "special_token_embeddings.npz"),
        ]
        for npz_name in candidates:
            if not npz_name:
                continue
            npz_path = self.gguf_path.parent / npz_name
            if not npz_path.exists():
                continue
            data = np.load(npz_path)
            emb_a, emb_a_end = unpack_special_token_int8(data)
            is_int8 = "emb_a_scale" in data.files
            logger.info(
                "Loaded <A>/</A> embeddings from %s (%s)",
                npz_path,
                "int8+scale" if is_int8 else "float",
            )
            return emb_a, emb_a_end

        emb_a, emb_a_end = self._load_from_gguf_token_embd()
        logger.info(
            "Loaded <A>/</A> embeddings from GGUF token_embd (ids %d, %d)",
            self.token_a_id,
            self.token_a_end_id,
        )
        return emb_a, emb_a_end

    def _maybe_quantize_audio_embeds(self, audio_embeds: np.ndarray) -> np.ndarray:
        """INT8 round-trip for audio embeddings when embedding_quantization is enabled."""
        if not self.quantize_injected_embeds:
            return audio_embeds
        from speechllm.inference.embed_quant import quantize_dequantize

        mode = self.audio_embed_quant_mode
        scale = self.audio_embed_scale if mode == "tensor" else None
        return quantize_dequantize(audio_embeds, mode=mode, scale=scale)
    def reset(self) -> None:
        # llama-cpp-python Llama.reset() only clears memory for recurrent/hybrid
        # models; for standard transformers we must clear KV explicitly, otherwise
        # subsequent prefills at pos=0 collide and llama_decode returns -1.
        self.llm._ctx.kv_cache_clear()
        self.llm.n_tokens = 0
        self.llm._requires_eval = True
        self.n_past = 0
        self.prompt_kv_len = None
        self.segment_count = 0
        self.is_new_seg = True
        self.embed_chunks_count = 0

    def _embed_text(self, text: str) -> np.ndarray:
        token_ids = self.llm.tokenize(text.encode("utf-8"), add_bos=False)
        if not token_ids:
            return np.zeros((0, self.llm_dim), dtype=np.float32)
        table = self._load_token_embd_table()
        return table[np.asarray(token_ids, dtype=np.int64)]

    def _embed_token_id(self, token_id: int) -> np.ndarray:
        table = self._load_token_embd_table()
        return table[int(token_id)].reshape(1, -1)

    def _prefill_embeddings(self, embeddings: np.ndarray, want_logits: bool) -> None:
        """Append embedding vectors to KV cache; optionally compute logits on last token."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.size == 0:
            return
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[1] != self.llm_dim:
            raise ValueError(f"Expected embeddings shape (T, {self.llm_dim}), got {embeddings.shape}")

        n_tokens = embeddings.shape[0]
        # Chunk large prefills to respect llama n_batch
        n_batch = max(1, int(self.llm.n_batch))
        offset = 0
        while offset < n_tokens:
            cur = min(n_batch, n_tokens - offset)
            chunk = embeddings[offset : offset + cur]
            want = want_logits and (offset + cur == n_tokens)
            self._prefill_embeddings_batch(chunk, want_logits=want)
            offset += cur

    def _prefill_embeddings_batch(self, embeddings: np.ndarray, want_logits: bool) -> None:
        n_tokens = embeddings.shape[0]
        batch = LlamaBatch(n_tokens=n_tokens, embd=self.llm_dim, n_seq_max=1, verbose=False)
        b = batch.batch
        b.n_tokens = n_tokens

        # Contiguous copy into llama_batch.embd (avoids slow per-float ctypes stores)
        emb_view = np.ctypeslib.as_array(b.embd, shape=(n_tokens * self.llm_dim,))
        emb_view[:] = embeddings.reshape(-1)

        for i in range(n_tokens):
            b.pos[i] = self.n_past + i
            b.seq_id[i][0] = 0
            b.n_seq_id[i] = 1
            b.logits[i] = False
        if want_logits:
            b.logits[n_tokens - 1] = True

        rc = self.llm._ctx.decode(batch)
        if rc is not None and rc != 0:
            raise RuntimeError(
                f"llama_decode failed with code {rc} "
                f"(n_past={self.n_past}, n_tokens={n_tokens}, want_logits={want_logits}). "
                "Often caused by KV position collision after incomplete cache clear."
            )
        self.n_past += n_tokens
        self.llm.n_tokens = self.n_past

    def _decode_token(self, token_id: int) -> None:
        """Append one generated token via embedding injection (never llm.eval).

        llama_cpp.Llama.eval() always calls kv_cache_seq_rm(-1, n_tokens, -1) and
        uses the token-id batch path. Mixing that with custom embd prefills corrupts
        cross-segment KV continuity.
        """
        self._prefill_embeddings(self._embed_token_id(token_id), want_logits=True)

    def _sample_logits(self) -> np.ndarray:
        return np.ctypeslib.as_array(
            self.llm._ctx.get_logits(),
            shape=(self.llm.n_vocab(),),
        ).copy()

    def _truncate_kv(self, remove_len: int = 1) -> None:
        if remove_len <= 0 or self.n_past <= 0:
            return
        new_len = max(0, self.n_past - remove_len)
        self.llm._ctx.kv_cache_seq_rm(0, new_len, self.n_past)
        self.n_past = new_len
        self.llm.n_tokens = new_len

    def _maybe_evict_cache(self, text: str) -> str:
        """Mirror streaming_session.LLMDecoder._maybe_evict_cache."""
        self.segment_count += 1

        if text.rstrip().endswith(_SENT_END):
            text = text.rstrip() + " "
            logger.info(
                "Sentence end, clearing KV cache (segments=%d)", self.segment_count
            )
            self.reset()
            # reset() clears is_new_seg / counts; keep is_new_seg True for next prompt
            self.is_new_seg = True
            return text

        if self.segment_count >= self.max_segments:
            logger.info(
                "Segment limit reached (%d/%d), clearing KV cache",
                self.segment_count,
                self.max_segments,
            )
            self.reset()
            self.is_new_seg = True

        return text

    def _autoregress_after_a_end(
        self,
        max_new_tokens: int,
        repetition_penalty: float,
        punct_kv_mode: int,
        *,
        first_token_eos_threshold: float = 1.0,
        apply_first_token_eos_threshold: bool = True,
        first_token_eos_penalty: float = 1.0,
    ) -> str:
        generated_ids: List[int] = []

        for step in range(max_new_tokens):
            logits = self._sample_logits()

            if step == 0:
                if apply_first_token_eos_threshold and first_token_eos_threshold < 1.0:
                    probs = self._softmax(logits)
                    if (
                        probs[self.token_w_id] > first_token_eos_threshold
                        and int(np.argmax(logits)) == self.token_w_id
                    ):
                        logits[self.token_w_id] = -np.inf
                elif first_token_eos_penalty > 1.0:
                    logits[self.token_w_id] /= first_token_eos_penalty

            if repetition_penalty != 1.0 and generated_ids:
                for tok_id in generated_ids:
                    if logits[tok_id] < 0:
                        logits[tok_id] *= repetition_penalty
                    else:
                        logits[tok_id] /= repetition_penalty

            token_id = int(np.argmax(logits))
            if token_id == self.token_w_id:
                if punct_kv_mode != 0 and generated_ids:
                    prev = self.llm.detokenize([generated_ids[-1]])
                    if isinstance(prev, bytes):
                        prev = prev.decode("utf-8", errors="ignore")
                    if prev and str(prev).strip() and str(prev).strip()[-1] in _SENT_END:
                        if punct_kv_mode == 1:
                            self._truncate_kv(1)
                        elif punct_kv_mode == 2 and self.prompt_kv_len is not None:
                            if self.n_past > self.prompt_kv_len:
                                self._truncate_kv(self.n_past - self.prompt_kv_len)
                break

            generated_ids.append(token_id)
            self._decode_token(token_id)

        if self.prompt_kv_len is None:
            self.prompt_kv_len = self.n_past - len(generated_ids)
            if self.prompt_kv_len < 0:
                self.prompt_kv_len = 0

        text = self.llm.detokenize(generated_ids, special=True)
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        return text

    def feed_chunk(
        self,
        audio_chunk_embed: np.ndarray,
        past_key_values: Optional[object] = None,
        past_seq_len: int = 0,
        prompt: Optional[str] = None,
        is_new_segment: bool = True,
        is_segment_end: bool = False,
        max_new_tokens: int = 200,
        repetition_penalty: float = 1.0,
        first_token_eos_penalty: float = 1.0,
        punct_kv_mode: int = 1,
        clear_kv_on_sentence_end: bool = True,
        first_token_eos_threshold: float = 1.0,
        apply_first_token_eos_threshold: bool = False,
    ) -> tuple[str, object, int]:
        """
        Incremental streaming decode matching model_asr_stream.generate()
        + streaming_session.LLMDecoder.feed_chunk().

        Returns:
            (text, cache_handle, seq_len) — cache_handle is self for API compatibility.
        """
        del past_key_values, past_seq_len  # KV lives inside llama context

        audio_chunk_embed = np.asarray(audio_chunk_embed, dtype=np.float32)
        if audio_chunk_embed.ndim == 3:
            audio_chunk_embed = audio_chunk_embed[0]
        if audio_chunk_embed.ndim == 1:
            audio_chunk_embed = audio_chunk_embed.reshape(1, -1)
        audio_chunk_embed = self._maybe_quantize_audio_embeds(audio_chunk_embed)

        prefix_parts: List[np.ndarray] = []
        if is_new_segment:
            # Prompt only when KV is empty (first utterance or after eviction)
            if self.n_past == 0 and prompt is not None:
                prefix_parts.append(self._embed_text(prompt))
            prefix_parts.append(self.emb_a.reshape(1, -1))

        if audio_chunk_embed.shape[0] > 0:
            prefix_parts.append(audio_chunk_embed)

        if prefix_parts:
            self._prefill_embeddings(np.concatenate(prefix_parts, axis=0), want_logits=False)

        if not is_segment_end:
            return "", self, self.n_past

        self._prefill_embeddings(self.emb_a_end.reshape(1, -1), want_logits=True)
        text = self._autoregress_after_a_end(
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            punct_kv_mode=punct_kv_mode,
            first_token_eos_penalty=first_token_eos_penalty,
            first_token_eos_threshold=first_token_eos_threshold,
            apply_first_token_eos_threshold=apply_first_token_eos_threshold,
        )

        if clear_kv_on_sentence_end and text:
            text = self._maybe_evict_cache(text)

        return text, self, self.n_past

    def generate_chunk(
        self,
        prompt: str,
        audio_embeds: np.ndarray,
        max_new_tokens: int = 200,
        repetition_penalty: float = 1.0,
        first_token_eos_threshold: float = 1.0,
        punct_kv_mode: int = 1,
        is_first_chunk: bool = True,
        is_last_chunk: bool = True,
        eos_penalty_only_last_chunk: bool = False,
        clear_kv_on_sentence_end: bool = True,
    ) -> str:
        """
        Offline segment decode (decode_ast_stream_gguf.py).

        Each call is one streaming segment end. Prompt is (re)injected whenever
        KV is empty — including after sentence-end eviction.
        """
        del is_first_chunk  # prompt gated by n_past==0, matching streaming_session

        apply_penalty = (not eos_penalty_only_last_chunk) or is_last_chunk
        text, _, _ = self.feed_chunk(
            audio_chunk_embed=audio_embeds,
            prompt=prompt,
            is_new_segment=True,
            is_segment_end=True,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            punct_kv_mode=punct_kv_mode,
            clear_kv_on_sentence_end=clear_kv_on_sentence_end,
            first_token_eos_threshold=first_token_eos_threshold,
            apply_first_token_eos_threshold=apply_penalty,
            first_token_eos_penalty=1.0,
        )
        return text

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
