"""
History-embed streaming decoder that offloads autoregressive generation to shared vLLM.

Mirrors speechllm.app.bin.streaming_session.LLMDecoder segment / eviction policy,
but accumulates prompt_embeds and calls SharedVLLMClient only on segment end
(same pattern as GRPO generate_via_vllm).
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import torch

from speechllm.app.stream_asr_ast_vllm.vllm_client import SharedVLLMClient

logger = logging.getLogger(__name__)

_SENT_END = (".", "?", "!", "。", "？", "！")


class VLLMDecoder:
    """Single-task ASR/AST decoder backed by shared vLLM.

    Eviction policy (aligned with LLMDecoder):
      - Sentence-end punctuation → full clear of prompt + history
      - segment_count >= max_segments → keep prompt + last keep_segments
        audio/text records (keep_segments<=0 → full clear)
    """

    def __init__(
        self,
        model,
        vllm_client: SharedVLLMClient,
        task: str = "asr",
        lang: str = "Chinese",
        num_chunks: int = 1,
        max_new_tokens: int = 200,
        repetition_penalty: float = 1.0,
        repetition_penalty_window: int = 0,
        max_segments: int = None,
        keep_segments: int = 16,
    ):
        self.model = model
        self.vllm_client = vllm_client
        self.task = task
        self.lang = lang
        self.num_chunks = max(1, int(num_chunks))
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.repetition_penalty_window = repetition_penalty_window

        self.prompt_emb: Optional[torch.Tensor] = None
        # Completed segments: (audio_seg_with_A_Aend, text_emb_or_None)
        self.records: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
        self.pending_segment: List[torch.Tensor] = []
        self.is_new_seg = True
        self.embed_chunks_count = 0

        self.max_segments = int(max_segments) if max_segments is not None else max(1, 64 // self.num_chunks)
        self.keep_segments = int(keep_segments)
        self.segment_count = 0

        self._emb_A: Optional[torch.Tensor] = None
        self._emb_A_end: Optional[torch.Tensor] = None
        self._dtype = None
        # Sticky key so consecutive segments hit the same vLLM actor's prefix KV cache.
        self._route_key = f"{task}:{id(self)}"

    def _get_prompt(self) -> str:
        if self.task == "ast":
            return f"Translate the audio in {self.lang}: "
        if self.lang == "auto":
            return "Transcribe the audio: "
        return f"Transcribe the audio in {self.lang}: "

    def _get_embedder(self):
        llm = self.model.llm_model
        if hasattr(llm, "get_input_embeddings"):
            return llm.get_input_embeddings()
        if hasattr(llm, "model") and hasattr(llm.model, "embed_tokens"):
            return llm.model.embed_tokens
        return llm.model.model.embed_tokens

    @staticmethod
    def _as_2d(t: torch.Tensor) -> torch.Tensor:
        """Normalize embed tensor to (T, D)."""
        if t.dim() == 1:
            return t.unsqueeze(0)
        if t.dim() == 3:
            return t.reshape(-1, t.size(-1))
        if t.dim() == 2:
            return t
        raise ValueError(f"unexpected embed ndim={t.dim()} shape={tuple(t.shape)}")

    def _ensure_special_embeds(self, dtype: torch.dtype):
        if self._emb_A is not None and self._dtype == dtype:
            return
        embedder = self._get_embedder()
        device = next(embedder.parameters()).device
        if getattr(self.model, "use_lora", False) or getattr(self.model, "use_patch_for_generation", False):
            emb_A = self.model.special_token_input_patch[0].detach().to(device=device, dtype=dtype)
            emb_A_end = self.model.special_token_input_patch[1].detach().to(device=device, dtype=dtype)
        else:
            special_ids = torch.tensor(
                [self.model.token_A_id, self.model.token_A_end_id], device=device
            )
            with torch.no_grad():
                s_embs = embedder(special_ids).to(dtype=dtype)
            emb_A, emb_A_end = s_embs[0], s_embs[1]
        # Always (1, D) so they cat with audio (T, D)
        self._emb_A = self._as_2d(emb_A.detach().float().cpu())
        self._emb_A_end = self._as_2d(emb_A_end.detach().float().cpu())
        self._dtype = dtype

    def _make_prompt_embed(self) -> torch.Tensor:
        embedder = self._get_embedder()
        device = next(embedder.parameters()).device
        prompt = self._get_prompt()
        ids = self.model.llm_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        with torch.no_grad():
            emb = embedder(ids).squeeze(0)
        return self._as_2d(emb.detach().float().cpu())

    def _text_embed(self, text: str) -> Optional[torch.Tensor]:
        if not text:
            return None
        embedder = self._get_embedder()
        device = next(embedder.parameters()).device
        ids = self.model.llm_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        if ids.numel() == 0:
            return None
        with torch.no_grad():
            emb = embedder(ids).squeeze(0)
        return self._as_2d(emb.detach().float().cpu())

    def _cat_parts(self, extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        if self.prompt_emb is not None:
            parts.append(self._as_2d(self.prompt_emb))
        for seg, txt in self.records:
            parts.append(self._as_2d(seg))
            if txt is not None:
                parts.append(self._as_2d(txt))
        if extra is not None:
            parts.append(self._as_2d(extra))
        if not parts:
            raise RuntimeError("empty history embeds")
        return torch.cat(parts, dim=0)

    def _full_clear(self):
        self.prompt_emb = None
        self.records = []
        self.pending_segment = []
        self.segment_count = 0

    def _slide_history(self):
        """Keep prompt + last keep_segments text-producing records (and in-between)."""
        k = min(self.keep_segments, self.segment_count)
        text_idx = [i for i, (_, t) in enumerate(self.records) if t is not None]
        n = len(text_idx)
        if k <= 0 or n == 0:
            self._full_clear()
            self.is_new_seg = True
            return
        if k >= n:
            self.segment_count = n
            return

        first_keep = text_idx[-k]
        dropped = first_keep
        self.records = self.records[first_keep:]
        self.segment_count = k
        self.is_new_seg = True
        logger.info(
            f"  -> [{self.task.upper()}] Segment limit reached "
            f"(kept {k}/{n} text segments, dropped {dropped} leading records, "
            f"prompt kept={self.prompt_emb is not None})"
        )

    async def _maybe_evict_cache(self, out_text: str) -> str:
        self.segment_count += 1

        if out_text.rstrip().endswith(_SENT_END):
            out_text = out_text.rstrip() + " "
            logger.info(
                f"  -> [{self.task.upper()}] Sentence end, clearing history embeds "
                f"(segments={self.segment_count})"
            )
            self._full_clear()
            return out_text

        if self.segment_count >= self.max_segments:
            if self.keep_segments <= 0:
                logger.info(
                    f"  -> [{self.task.upper()}] Segment limit reached "
                    f"({self.segment_count}/{self.max_segments}), clearing history embeds"
                )
                self._full_clear()
                self.is_new_seg = True
            else:
                self._slide_history()

        return out_text

    def _append_audio_chunk(self, chunk_out: torch.Tensor):
        """Accumulate one encoder chunk into the current segment."""
        audio = self._as_2d(chunk_out.detach().float().cpu())
        self._ensure_special_embeds(audio.dtype)

        if self.is_new_seg:
            if self.prompt_emb is None:
                self.prompt_emb = self._make_prompt_embed()
            self.pending_segment = [self._emb_A, audio]
            self.is_new_seg = False
        else:
            if not self.pending_segment:
                self.pending_segment = [self._emb_A, audio]
            else:
                self.pending_segment.append(audio)

    async def _decode_segment(self) -> str:
        """Close current segment with </A>, call vLLM, append embeds."""
        if not self.pending_segment:
            logger.warning(f"  -> [{self.task.upper()}] finalize with empty pending_segment, skip")
            return ""
        self.pending_segment.append(self._emb_A_end)
        # Defensive: all parts must be (T, D)
        segment = torch.cat([self._as_2d(t) for t in self.pending_segment], dim=0)
        self.pending_segment = []

        full = self._cat_parts(extra=segment)
        t0 = time.time()
        out_text = await self.vllm_client.generate(full, route_key=self._route_key)
        cost = time.time() - t0
        logger.info(f"  -> [{self.task.upper()} DECODE] cost={cost:.3f}s output=\"{out_text}\"")

        text_emb = self._text_embed(out_text) if out_text else None
        self.records.append((segment, text_emb))

        if out_text:
            out_text = await self._maybe_evict_cache(out_text)
            return out_text
        return ""

    async def feed_chunk(self, chunk_out: torch.Tensor) -> str:
        self.embed_chunks_count += 1
        is_seg_end = self.embed_chunks_count >= self.num_chunks
        self._append_audio_chunk(chunk_out)

        if not is_seg_end:
            logger.info(
                f"  -> [{self.task.upper()} cache] buffered chunk "
                f"{self.embed_chunks_count}/{self.num_chunks}"
            )
            return ""

        self.embed_chunks_count = 0
        self.is_new_seg = True
        return await self._decode_segment()

    async def finalize_chunk(self, chunk_out: torch.Tensor, is_last: bool) -> str:
        self._append_audio_chunk(chunk_out)
        if not is_last:
            logger.info(f"  -> [{self.task.upper()} cache] finalize non-last chunk")
            return ""

        self.embed_chunks_count = 0
        self.is_new_seg = True
        t0 = time.time()
        text = await self._decode_segment()
        if text:
            logger.info(f"  -> [{self.task.upper()} FINAL] cost={time.time()-t0:.3f}s")
        return text

    def reset(self):
        self._full_clear()
        self.is_new_seg = True
        self.embed_chunks_count = 0
