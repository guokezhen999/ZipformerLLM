"""Voice Activity Detection (VAD) wrapper using silero-vad.

Encapsulates the VADIterator state machine, buffer management, and
speech-boundary detection.  VAD **gates** the encoder — audio only
reaches the encoder after speech is detected (with ``min_silence_duration_ms``
of leading context), and stops when speech ends.
"""

import logging
from typing import Optional

import numpy as np
import torch

try:
    from silero_vad import VADIterator, load_silero_vad
    _HAS_VAD = True
except ImportError:
    _HAS_VAD = False
    VADIterator = None
    logging.warning(
        "silero-vad not available; VAD will be disabled. "
        "Install with: pip install silero-vad"
    )


class VADHandler:
    """Silero-VAD state machine that gates the streaming encoder.

    Audio flows into :meth:`process`.  While speech is inactive the audio
    is buffered but not forwarded.  When VAD detects a speech start,
    ``is_speech_active`` becomes True and :meth:`get_unconsumed_audio`
    returns audio from ``speech_start - min_silence_duration_ms`` onward.

    When VAD detects end-of-speech, ``pending_reset`` is set and the
    caller should finalise the current segment, then call :meth:`reset`
    to prepare for the next utterance.
    """

    VAD_CHUNK_SIZE = 512  # samples per VAD forward call

    def __init__(
        self,
        vad_model,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 200,
        speech_pad_ms: int = 100,
    ):
        self._vad_model = vad_model
        self._sampling_rate = sampling_rate

        self._vad_iterator = VADIterator(
            vad_model,
            threshold=threshold,
            sampling_rate=sampling_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

        # How many samples of leading silence to include before speech start.
        self._min_silence_samples = int(sampling_rate * min_silence_duration_ms / 1000)

        # Internal buffer — accumulates ALL incoming audio.
        self._buffer = np.array([], dtype=np.float32)
        self._chunk_offset = 0  # next sample offset to feed to VADIterator
        self._speech_start: Optional[int] = None  # sample offset where speech began
        self._speech_end: Optional[int] = None    # sample offset where speech ended
        self._triggered = False  # True between 'start' and 'end' VAD events
        self._consumed_offset = 0  # samples already returned via get_unconsumed_audio()
        self._next_speech_start: Optional[int] = None  # 'start' detected while pending_reset

        # Public state
        self.is_speech_active = False
        self.pending_reset = False

        # Diagnostic
        self._chunk_count = 0
        self._log_interval = 100  # log every N chunks when idle

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray) -> bool:
        """Feed new audio samples and advance the VAD state machine.

        Args:
            audio: 1-D float32 PCM array.

        Returns:
            True if a speech→silence transition was just detected
            (i.e. ``pending_reset`` was set).
        """
        self._buffer = np.concatenate([self._buffer, audio])
        return self._run()

    def get_unconsumed_audio(self) -> np.ndarray:
        """Return audio that should be fed to the encoder.

        When ``is_speech_active``: returns new audio from
        ``(speech_start - min_silence)`` up to the end of the buffer.

        When ``pending_reset``: returns the final segment capped at
        ``speech_end`` (does NOT include post-speech silence).

        Returns an empty array if no speech audio is available.
        """
        if not self.is_speech_active and not self.pending_reset:
            return np.array([], dtype=np.float32)
        if self._speech_start is None:
            return np.array([], dtype=np.float32)

        # Start from (speech_start - min_silence), but not before already-consumed offset.
        start = max(self._consumed_offset,
                     self._speech_start - self._min_silence_samples)

        if self.pending_reset and self._speech_end is not None:
            end = min(self._speech_end, len(self._buffer))
        else:
            end = len(self._buffer)

        if start >= end:
            return np.array([], dtype=np.float32)

        audio = self._buffer[start:end].copy()
        self._consumed_offset = end
        return audio

    def reset(self) -> None:
        """Reset the VAD iterator and all internal state for a new utterance."""
        if self._vad_iterator is not None:
            self._vad_iterator.reset_states()
        self._buffer = np.array([], dtype=np.float32)
        self._chunk_offset = 0
        self._speech_start = None
        self._speech_end = None
        self._triggered = False
        self._consumed_offset = 0
        self._next_speech_start = None
        self.is_speech_active = False
        self.pending_reset = False

    def drain_pending_reset(self) -> np.ndarray:
        """Return audio that should survive a :meth:`reset` for the next utterance.

        When two utterances are close enough to be detected in the same
        ``process()`` call, the audio after ``_speech_end`` (which may
        contain the start of the next utterance) would be lost if we
        simply called :meth:`reset`.  Call this BEFORE :meth:`reset` to
        save the overflow, then feed it back via :meth:`process` after
        the reset.

        Returns an empty array when there is no overflow.
        """
        if self._speech_end is None:
            return np.array([], dtype=np.float32)

        # If we detected a next speech start, include leading silence
        # context for it (without duplicating already-consumed audio).
        if self._next_speech_start is not None:
            safe_start = int(
                max(self._speech_end,
                    self._next_speech_start - self._min_silence_samples)
            )
        else:
            safe_start = int(self._speech_end)

        if safe_start < len(self._buffer):
            overflow = self._buffer[safe_start:].copy()
            logging.info(
                f"[VAD] drain_pending_reset: {len(overflow)} samples "
                f"({len(overflow) / self._sampling_rate:.2f}s) overflow saved"
            )
            return overflow
        return np.array([], dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> bool:
        """Feed buffered audio to VADIterator in VAD_CHUNK_SIZE increments.

        On 'start': records speech start, sets ``is_speech_active = True``.
        On 'end':   records speech end, sets ``pending_reset = True``,
                    resets the VAD iterator (but NOT the buffer — the
                    caller drains it via :meth:`get_unconsumed_audio`
                    before calling :meth:`reset`).
        """
        triggered = False
        while (self._chunk_offset + self.VAD_CHUNK_SIZE
               <= len(self._buffer)):
            start = self._chunk_offset
            chunk = torch.from_numpy(
                self._buffer[start:start + self.VAD_CHUNK_SIZE]
            ).float()
            self._chunk_offset += self.VAD_CHUNK_SIZE
            self._chunk_count += 1

            result = self._vad_iterator(chunk)
            if result is None:
                # Periodic log while idle so the user knows VAD is running
                if self._chunk_count % self._log_interval == 0:
                    state = "speech active" if self._triggered else "listening"
                    logging.info(
                        f"[VAD] {state}: {self._chunk_count} chunks "
                        f"({self._chunk_count * self.VAD_CHUNK_SIZE / self._sampling_rate:.1f}s)"
                    )
                continue

            if 'start' in result:
                # When a 'start' fires while we already have a pending
                # reset (back-to-back utterances within one chunk), save
                # it for the next utterance — don't overwrite the current
                # _speech_start / _speech_end pair.
                if self.pending_reset:
                    self._next_speech_start = result['start']
                    logging.info(
                        f"[VAD] back-to-back speech start at sample "
                        f"{self._next_speech_start} (queued after pending reset)"
                    )
                else:
                    self._speech_start = result['start']
                    self._triggered = True
                    self.is_speech_active = True
                    logging.info(
                        f"[VAD] speech start at sample {self._speech_start}"
                    )

            if 'end' in result and self._triggered:
                self._speech_end = result['end']
                logging.info(
                    f"[VAD] speech end: [{self._speech_start}, {self._speech_end}] "
                    f"({(self._speech_end - self._speech_start) / self._sampling_rate:.2f}s)"
                )

                # Reset the VAD iterator so it is ready for the next
                # utterance, but keep the buffer so the caller can drain
                # the remaining speech audio.
                self._vad_iterator.reset_states()
                self._triggered = False
                self.is_speech_active = False
                self.pending_reset = True
                triggered = True

        return triggered


def has_vad() -> bool:
    """Return True if silero-vad is installed and importable."""
    return _HAS_VAD


def create_vad_model():
    """Load the silero-vad JIT model.  Returns None if not available."""
    if not _HAS_VAD:
        return None
    return load_silero_vad()
