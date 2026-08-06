"""
Shared streaming session components for all server variants.

Classes:
    LLMDecoder          — single-task LLM decoder with independent KV cache
    StreamingSessionBase — base class: FBank extraction, adapter loop, buffer management
"""
import logging
import math
import time
import numpy as np
import torch
import torchaudio

from speechllm.app.bin.vad import VADHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Sentence-ending punctuation (EN + ZH)
_SENT_END = ('.', '?', '!', '。', '？', '！')


def extract_fbank(waveform_np: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
    """Extract 80-dim FBank features from raw waveform."""
    waveform = torch.from_numpy(waveform_np).unsqueeze(0)
    return torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25.0,
        frame_shift=10.0,
        sample_frequency=sample_rate,
    )


# ---------------------------------------------------------------------------
# LLMDecoder
# ---------------------------------------------------------------------------
class LLMDecoder:
    """Single-task LLM decoder that maintains its own KV cache and generation state.

    KV cache eviction policy:
        Keep at most `max_segments` segments of history (48 / num_chunks).
        Eviction only happens when the current segment actually produces text output,
        so silent chunks never trigger a cache clear.
    """

    def __init__(self, model, task: str = "asr", lang: str = "Chinese", num_chunks: int = 1, max_new_tokens: int = 200, repetition_penalty: float = 1.0, repetition_penalty_window: int = 0):
        self.model = model
        self.task = task
        self.lang = lang
        self.num_chunks = num_chunks
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.repetition_penalty_window = repetition_penalty_window
        self.past_cache = None
        self.past_len = 0
        self.is_new_seg = True
        self.embed_chunks_count = 0

        # KV cache eviction: keep 48 adapter-chunks worth of history
        self.max_segments = 48 // num_chunks  # 1→48, 2→24, 4→12
        self.segment_count = 0  # how many segments with text output accumulated

    def _get_prompt(self) -> str:
        if self.task == "ast":
            return f"Translate the audio in {self.lang}: "
        # ASR task: use language-specific prompt unless lang is "auto"
        if self.lang == "auto":
            return "Transcribe the audio: "
        return f"Transcribe the audio in {self.lang}: "

    def _maybe_evict_cache(self, out_text: str) -> str:
        """Increment segment counter and evict KV cache if history exceeds max_segments.

        Only called when out_text is non-empty (current segment produced output).
        Returns (possibly modified) out_text.
        """
        self.segment_count += 1

        # Sentence-end punctuation always clears cache (original behaviour)
        if out_text.rstrip().endswith(_SENT_END):
            out_text = out_text.rstrip() + " "
            logging.info(f"  -> [{self.task.upper()}] Sentence end, clearing KV cache (segments={self.segment_count})")
            self.past_cache = None
            self.past_len = 0
            self.segment_count = 0
            return out_text

        # Evict when accumulated segments exceed the limit
        if self.segment_count >= self.max_segments:
            logging.info(f"  -> [{self.task.upper()}] Segment limit reached ({self.segment_count}/{self.max_segments}), clearing KV cache")
            self.past_cache = None
            self.past_len = 0
            self.segment_count = 0
            self.is_new_seg = True

        return out_text

    def feed_chunk(self, chunk_out: torch.Tensor) -> str:
        """Feed an encoded audio chunk. Returns generated text (non-empty only at segment end)."""
        self.embed_chunks_count += 1
        is_seg_end = (self.embed_chunks_count >= self.num_chunks)

        t_start = time.time()
        out_text, self.past_cache, self.past_len = self.model.generate(
            audio_chunk_embed=chunk_out,
            past_key_values=self.past_cache,
            past_seq_len=self.past_len,
            prompt=self._get_prompt() if self.is_new_seg else None,
            is_new_segment=self.is_new_seg,
            is_segment_end=is_seg_end,
            generation_config={"max_new_tokens": self.max_new_tokens, "eos_token_id": self.model.token_W_id, "repetition_penalty": self.repetition_penalty, "repetition_penalty_window": self.repetition_penalty_window},
            first_token_eos_penalty=1.0,
        )
        t_cost = time.time() - t_start

        action = "DECODE" if is_seg_end else "cache"
        logging.info(f"  -> [{self.task.upper()} {action}] cost={t_cost:.3f}s output=\"{out_text}\"")

        self.is_new_seg = False

        if is_seg_end:
            self.embed_chunks_count = 0
            self.is_new_seg = True
            if out_text:
                out_text = self._maybe_evict_cache(out_text)
                return out_text
        return ""

    def finalize_chunk(self, chunk_out: torch.Tensor, is_last: bool) -> str:
        """Finalize-phase: feed chunk, decode only on the last chunk."""
        t_start = time.time()
        out_text, self.past_cache, self.past_len = self.model.generate(
            audio_chunk_embed=chunk_out,
            past_key_values=self.past_cache,
            past_seq_len=self.past_len,
            prompt=self._get_prompt() if self.is_new_seg else None,
            is_new_segment=self.is_new_seg,
            is_segment_end=is_last,
            generation_config={"max_new_tokens": self.max_new_tokens, "eos_token_id": self.model.token_W_id, "repetition_penalty": self.repetition_penalty, "repetition_penalty_window": self.repetition_penalty_window},
            first_token_eos_penalty=1.0,
        )
        t_cost = time.time() - t_start

        action = "FINAL DECODE" if is_last else "cache"
        logging.info(f"  -> [{self.task.upper()} {action}] cost={t_cost:.3f}s output=\"{out_text}\"")

        self.is_new_seg = False

        if is_last and out_text:
            out_text = self._maybe_evict_cache(out_text)
            return out_text
        return ""

    def reset(self):
        self.past_cache = None
        self.past_len = 0
        self.is_new_seg = True
        self.embed_chunks_count = 0
        self.segment_count = 0


# ---------------------------------------------------------------------------
# StreamingSessionBase
# ---------------------------------------------------------------------------
class StreamingSessionBase:
    """
    Base class for streaming audio sessions.

    Handles: waveform buffering → incremental FBank extraction → streaming_adapter → buffer cleanup.
    Subclasses implement `_on_encoder_output()` and `_on_finalize_encoder_output()` to
    route the encoder output to one or more LLMDecoders.
    """

    def __init__(self, model, config, device, tag: str = "", vad_handler=None):
        self.model = model
        self.device = device
        self.tag = tag
        self.vad_handler = vad_handler

        if config is None or model is None:
            # model not loaded yet — set safe defaults, process_audio/finalize will no-op
            self.stride = 32
            self.tail_length = 45
            self.audio_states = None
        else:
            chunk_size = config.model.zipformer.chunk_size
            self.stride = chunk_size * 2
            self.tail_length = chunk_size * 2 + 7 + 2 * 3
            self.audio_states = model.get_init_states(1)

        self.waveform_buffer = np.array([], dtype=np.float32)
        self.fbank_buffer = None
        self.fbank_frame_offset = 0
        self.absolute_frame_pos = 0
        self.adapter_step = 0

        self.sample_rate = 16000
        self.samples_per_frame = int(self.sample_rate * 0.01)  # 160

        # Audio end-time (seconds) of the most recently encoded chunk.
        # Set before each _on_encoder_output / _on_finalize_encoder_output so
        # subclasses can attach a timestamp to generated text.
        self.chunk_end_time: float = 0.0

    # -- encoder helper --
    def _run_encoder(self, chunk_fbank: torch.Tensor) -> torch.Tensor:
        chunk_in = chunk_fbank.unsqueeze(0).to(self.device)
        with torch.no_grad():
            chunk_out, self.audio_states = self.model.streaming_adapter(chunk_in, self.audio_states)
        target_dtype = chunk_out.dtype
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'dtype'):
            target_dtype = self.model.backbone.dtype
        return chunk_out.to(target_dtype)

    # -- hooks for subclasses --
    def _on_encoder_output(self, chunk_out: torch.Tensor):
        """Called for each encoder output during process_audio. Override in subclass."""
        raise NotImplementedError

    def _on_finalize_encoder_output(self, chunk_out: torch.Tensor, is_last: bool):
        """Called for each encoder output during finalize. Override in subclass."""
        raise NotImplementedError

    # -- private pipeline helper --
    def _run_pipeline(self):
        """Run FBank extraction → streaming adapter loop on waveform_buffer.

        Called by process_audio() whenever new speech audio is available.
        Operates on self.waveform_buffer and produces encoder outputs via
        _on_encoder_output().
        Returns True if any encoder step ran, False otherwise.
        """
        if self.model is None:
            return False

        min_samples = (self.tail_length - 1) * self.samples_per_frame + int(self.sample_rate * 0.025)
        if len(self.waveform_buffer) < min_samples:
            return False

        ran = False
        self.fbank_buffer = extract_fbank(self.waveform_buffer)

        while self.fbank_buffer is not None and (self.fbank_buffer.size(0) - self.fbank_frame_offset) >= self.tail_length:
            start_frame = self.fbank_frame_offset
            chunk_fbank = self.fbank_buffer[start_frame:start_frame + self.tail_length, :]

            self.adapter_step += 1
            abs_s = self.absolute_frame_pos
            prefix = f"[{self.tag} " if self.tag else "["
            logging.info(f"{prefix}Encoder Step {self.adapter_step}] {abs_s/100:.2f}s - {(abs_s+self.tail_length)/100:.2f}s")

            chunk_out = self._run_encoder(chunk_fbank)
            self.chunk_end_time = round((self.absolute_frame_pos + self.tail_length) / 100.0, 2)
            self._on_encoder_output(chunk_out)

            self.fbank_frame_offset += self.stride
            self.absolute_frame_pos += self.stride
            ran = True

        # cleanup consumed waveform / fbank
        consumed_samples = self.fbank_frame_offset * self.samples_per_frame
        if consumed_samples > 0 and consumed_samples <= len(self.waveform_buffer):
            self.waveform_buffer = self.waveform_buffer[consumed_samples:]
            if self.fbank_buffer is not None and self.fbank_frame_offset > 0:
                self.fbank_buffer = self.fbank_buffer[self.fbank_frame_offset:]
            self.fbank_frame_offset = 0

        return ran

    # -- main audio processing loop --
    def process_audio(self, pcm_chunk: np.ndarray):
        """
        Receive PCM chunk, run incremental FBank → adapter pipeline.
        Calls `_on_encoder_output()` for each adapter output.

        When VAD is enabled, audio is gated through the VAD handler:
        only speech segments (with leading silence context) reach the encoder.
        On end-of-speech, a hard reset clears encoder/decoder state for the
        next utterance.
        """
        if self.model is None:
            return

        # --- Passthrough mode (no VAD) ---
        if self.vad_handler is None:
            self.waveform_buffer = np.concatenate([self.waveform_buffer, pcm_chunk])
            self._run_pipeline()
            return

        # --- VAD-gated mode ---
        self.vad_handler.process(pcm_chunk)

        while self.vad_handler.is_speech_active or self.vad_handler.pending_reset:
            # 1. Get speech audio from VAD, append to waveform_buffer
            speech_audio = self.vad_handler.get_unconsumed_audio()
            if len(speech_audio) > 0:
                self.waveform_buffer = np.concatenate([self.waveform_buffer, speech_audio])

            # 2. Run the FBank + encoder pipeline on whatever is buffered
            self._run_pipeline()

            # 3. Handle end-of-speech (pending_reset)
            if self.vad_handler.pending_reset:
                # Encode any remaining audio in waveform_buffer with zero-padding
                # (same pattern as finalize())
                if len(self.waveform_buffer) > 0:
                    self.fbank_buffer = extract_fbank(self.waveform_buffer)
                remaining_steps = []
                if self.fbank_buffer is not None and self.fbank_buffer.size(0) > self.fbank_frame_offset:
                    remaining_fbank = self.fbank_buffer[self.fbank_frame_offset:]
                    remaining_frames = remaining_fbank.size(0)
                    num_steps = math.ceil(remaining_frames / self.stride)
                    for i in range(num_steps):
                        start = i * self.stride
                        chunk_fbank = remaining_fbank[start:start + self.tail_length, :]
                        if chunk_fbank.size(0) < self.tail_length:
                            chunk_fbank = torch.nn.functional.pad(
                                chunk_fbank, (0, 0, 0, self.tail_length - chunk_fbank.size(0)),
                                "constant", 0,
                            )
                        remaining_steps.append(chunk_fbank)

                total = len(remaining_steps)
                for idx, chunk_fbank in enumerate(remaining_steps):
                    is_last = (idx == total - 1)
                    self.adapter_step += 1
                    prefix = f"[{self.tag} " if self.tag else "["
                    logging.info(f"{prefix}VAD Finalize Step {self.adapter_step}] {self.absolute_frame_pos/100:.2f}s last={is_last}")

                    chunk_out = self._run_encoder(chunk_fbank)
                    self.chunk_end_time = round((self.absolute_frame_pos + self.tail_length) / 100.0, 2)
                    self._on_finalize_encoder_output(chunk_out, is_last)
                    self.absolute_frame_pos += self.stride

                # Reset decoders (subclass hook)
                self._on_vad_reset()

                # Save overflow before hard reset
                overflow = self.vad_handler.drain_pending_reset()

                # Hard reset: encoder states, audio buffers, VAD
                self._hard_reset()

                # Re-inject overflow for next utterance (back-to-back handling)
                if len(overflow) > 0:
                    self.vad_handler.process(overflow)
                    continue
                else:
                    break
            else:
                break  # speech active but no pending reset — wait for more audio

    # -- VAD hooks --
    def _on_vad_reset(self):
        """Called when VAD detects end-of-speech, before _hard_reset().

        Subclasses should reset their decoders (KV cache, generation state).
        """
        pass

    def _hard_reset(self):
        """Reset encoder states, audio buffers, and VAD handler for next utterance."""
        if self.model is not None:
            self.audio_states = self.model.get_init_states(1)

        self.waveform_buffer = np.array([], dtype=np.float32)
        self.fbank_buffer = None
        self.fbank_frame_offset = 0
        self.absolute_frame_pos = 0
        self.adapter_step = 0

        if self.vad_handler is not None:
            self.vad_handler.reset()

    def finalize(self):
        """Process remaining buffered audio and run final decode."""
        if self.model is None:
            return

        # Drain remaining speech audio from VAD (if gated)
        if self.vad_handler is not None:
            if self.vad_handler.is_speech_active:
                speech_audio = self.vad_handler.get_unconsumed_audio()
                if len(speech_audio) > 0:
                    self.waveform_buffer = np.concatenate([self.waveform_buffer, speech_audio])

        if len(self.waveform_buffer) > 0:
            self.fbank_buffer = extract_fbank(self.waveform_buffer)

        remaining_steps = []
        if self.fbank_buffer is not None and self.fbank_buffer.size(0) > self.fbank_frame_offset:
            remaining_fbank = self.fbank_buffer[self.fbank_frame_offset:]
            remaining_frames = remaining_fbank.size(0)
            num_steps = math.ceil(remaining_frames / self.stride)
            for i in range(num_steps):
                start = i * self.stride
                chunk_fbank = remaining_fbank[start:start + self.tail_length, :]
                if chunk_fbank.size(0) < self.tail_length:
                    chunk_fbank = torch.nn.functional.pad(
                        chunk_fbank, (0, 0, 0, self.tail_length - chunk_fbank.size(0)), "constant", 0,
                    )
                remaining_steps.append(chunk_fbank)

        total = len(remaining_steps)
        for idx, chunk_fbank in enumerate(remaining_steps):
            is_last = (idx == total - 1)
            self.adapter_step += 1
            prefix = f"[{self.tag} " if self.tag else "["
            logging.info(f"{prefix}Finalize Step {self.adapter_step}] {self.absolute_frame_pos/100:.2f}s last={is_last}")

            chunk_out = self._run_encoder(chunk_fbank)
            self.chunk_end_time = round((self.absolute_frame_pos + self.tail_length) / 100.0, 2)
            self._on_finalize_encoder_output(chunk_out, is_last)
            self.absolute_frame_pos += self.stride

        # reset buffers
        self.waveform_buffer = np.array([], dtype=np.float32)
        self.fbank_buffer = None
        self.fbank_frame_offset = 0

        # Apply VAD-triggered reset if pending (e.g. user stopped recording mid-speech)
        if self.vad_handler is not None and self.vad_handler.pending_reset:
            self._on_vad_reset()
            overflow = self.vad_handler.drain_pending_reset()
            self._hard_reset()
            if len(overflow) > 0:
                self.vad_handler.process(overflow)
