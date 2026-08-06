"""
Streaming ASR+AST server with shared vLLM decode for multi-user concurrency.

Architecture:
  - Zipformer encoder / connector runs locally (shared weights, per-session states)
  - ASR and AST share one (or more) vLLM Ray actor(s) via micro-batched generate_batch
  - Multiple WebSocket clients are accepted concurrently

Requires pre-started Ray vLLM actors (see run_app_vllm.sh / start_vllm_ray_actors.py).
Environment: zipformer_vllm
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from speechllm.app.bin.model_loader import load_model_from_env
from speechllm.app.bin.streaming_session import StreamingSessionBase
from speechllm.app.bin.vad import VADHandler, has_vad, create_vad_model
from speechllm.app.stream_asr_ast_vllm.vllm_client import SharedVLLMClient
from speechllm.app.stream_asr_ast_vllm.vllm_decoder import VLLMDecoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

global_model = None
global_config = None
global_device = None
global_vllm_client: Optional[SharedVLLMClient] = None
global_encoder_lock: Optional[asyncio.Lock] = None
global_connection_sem: Optional[asyncio.Semaphore] = None

global_repetition_penalty = 1.0
global_repetition_penalty_window = 0
global_max_new_tokens = 200
global_ast_lang_options = ["Chinese", "English", "Japanese", "French", "German", "Spanish"]
global_asr_lang_options = ["auto", "Chinese", "English"]
global_vad_model = None
global_use_vad = False
global_vad_threshold = 0.5
global_vad_min_silence_duration_ms = 200
global_vad_speech_pad_ms = 100
global_active_connections = 0
global_keep_segments = 16
global_max_segments = None


def _option_html(options, selected_idx=0):
    parts = []
    for i, opt in enumerate(options):
        sel = " selected" if i == selected_idx else ""
        label = "Auto" if opt == "auto" else opt
        parts.append(f'<option value="{opt}"{sel}>{label}</option>')
    return "\n".join(parts)


def _offload_llm_keep_embedder(model):
    """Move LLM weights to CPU so encoder GPU memory coexists with vLLM."""
    if model is None or not hasattr(model, "llm_model"):
        return
    logging.info("Offloading llm_model to CPU (keeping embedder for prompt/special embeds)")
    model.llm_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_model, global_config, global_device, global_vllm_client
    global global_encoder_lock, global_connection_sem
    global global_repetition_penalty, global_repetition_penalty_window, global_max_new_tokens
    global global_ast_lang_options, global_asr_lang_options
    global global_vad_model, global_use_vad, global_vad_threshold
    global global_vad_min_silence_duration_ms, global_vad_speech_pad_ms
    global global_keep_segments, global_max_segments

    global_model, global_config, global_device = load_model_from_env()
    if global_model is not None and os.environ.get("SPEECHLLM_OFFLOAD_LLM", "1").lower() in (
        "1", "true", "yes",
    ):
        _offload_llm_keep_embedder(global_model)

    global_repetition_penalty = float(os.environ.get("SPEECHLLM_REPETITION_PENALTY", "1.0"))
    global_repetition_penalty_window = int(os.environ.get("SPEECHLLM_REPETITION_PENALTY_WINDOW", "0"))
    global_max_new_tokens = int(os.environ.get("SPEECHLLM_MAX_NEW_TOKENS", "200"))
    global_ast_lang_options = os.environ.get(
        "SPEECHLLM_AST_LANG_OPTIONS", "Chinese,English,Japanese,French,German,Spanish"
    ).split(",")
    global_asr_lang_options = os.environ.get(
        "SPEECHLLM_ASR_LANG_OPTIONS", "auto,Chinese,English"
    ).split(",")
    global_keep_segments = int(os.environ.get("SPEECHLLM_KEEP_SEGMENTS", "16"))
    _max_seg = os.environ.get("SPEECHLLM_MAX_SEGMENTS", "").strip()
    global_max_segments = int(_max_seg) if _max_seg else None
    logging.info(
        f"KV keep_segments={global_keep_segments}, "
        f"max_segments={global_max_segments or '64//num_chunks'}"
    )

    max_conn = int(os.environ.get("SPEECHLLM_MAX_CONNECTIONS", "16"))
    global_encoder_lock = asyncio.Lock()
    global_connection_sem = asyncio.Semaphore(max_conn)

    # Connect to shared vLLM actors
    if global_model is not None:
        stop_token_ids = [int(global_model.token_W_id)]
        eos_id = getattr(global_model.llm_tokenizer, "eos_token_id", None)
        if eos_id is not None and int(eos_id) not in stop_token_ids:
            stop_token_ids.append(int(eos_id))

        sampling_params = {
            "temperature": float(os.environ.get("SPEECHLLM_VLLM_TEMPERATURE", "0.0")),
            "top_p": 1.0,
            "max_tokens": global_max_new_tokens,
            "stop_token_ids": stop_token_ids,
            "skip_special_tokens": True,
            "repetition_penalty": 1.0,  # disabled; do not apply repetition penalty in vLLM
        }

        num_actors = int(os.environ.get("SPEECHLLM_VLLM_NUM_ACTORS", "1"))
        actor_prefix = os.environ.get("SPEECHLLM_VLLM_ACTOR_PREFIX", "vllm_actor")
        ray_address = os.environ.get("SPEECHLLM_RAY_ADDRESS", "auto")
        batch_timeout_ms = float(os.environ.get("SPEECHLLM_VLLM_BATCH_TIMEOUT_MS", "20"))
        max_batch = int(os.environ.get("SPEECHLLM_VLLM_MAX_BATCH", "32"))

        global_vllm_client = SharedVLLMClient.connect(
            num_actors=num_actors,
            actor_name_prefix=actor_prefix,
            ray_address=ray_address,
            sampling_params=sampling_params,
            batch_timeout_ms=batch_timeout_ms,
            max_batch_size=max_batch,
        )
        global_vllm_client.start()
        logging.info(
            f"Shared vLLM ready: actors={num_actors} prefix={actor_prefix} "
            f"sampling={sampling_params}"
        )
    else:
        logging.error("Model not loaded; WebSocket will reject connections")

    # VAD
    global_use_vad = os.environ.get("SPEECHLLM_USE_VAD", "").lower() in ("1", "true", "yes")
    if global_use_vad:
        if not has_vad():
            logging.warning("VAD enabled but silero-vad not installed. Disabling VAD.")
            global_use_vad = False
        else:
            global_vad_model = create_vad_model()
            global_vad_threshold = float(os.environ.get("SPEECHLLM_VAD_THRESHOLD", "0.5"))
            global_vad_min_silence_duration_ms = int(
                os.environ.get("SPEECHLLM_VAD_MIN_SILENCE_DURATION_MS", "200")
            )
            global_vad_speech_pad_ms = int(os.environ.get("SPEECHLLM_VAD_SPEECH_PAD_MS", "100"))
            logging.info(
                f"VAD enabled: threshold={global_vad_threshold}, "
                f"min_silence={global_vad_min_silence_duration_ms}ms, "
                f"speech_pad={global_vad_speech_pad_ms}ms"
            )
    else:
        logging.info("VAD disabled (set SPEECHLLM_USE_VAD=1 to enable)")

    logging.info(f"Max concurrent WebSocket connections: {max_conn}")
    yield

    if global_vllm_client is not None:
        await global_vllm_client.close()
    logging.info("Shutting down ASR/AST vLLM Stream Server")


app = FastAPI(lifespan=lifespan)
_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/")
async def get_index():
    with open(os.path.join(_static_dir, "index.html"), "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__AST_LANG_OPTIONS__", _option_html(global_ast_lang_options))
    html = html.replace("__ASR_LANG_OPTIONS__", _option_html(global_asr_lang_options))
    return HTMLResponse(content=html)


@app.get("/health")
async def health():
    return {
        "model_loaded": global_model is not None,
        "vllm_ready": global_vllm_client is not None,
        "active_connections": global_active_connections,
    }


class ParallelVLLMSession(StreamingSessionBase):
    """Shared encoder + dual VLLMDecoder (ASR / AST) with async decode flush."""

    def __init__(
        self,
        asr_num_chunks=1,
        ast_num_chunks=1,
        ast_lang="Chinese",
        asr_lang="auto",
        enable_asr=True,
        enable_ast=True,
        repetition_penalty=1.0,
        repetition_penalty_window=0,
        vad_handler=None,
    ):
        super().__init__(global_model, global_config, global_device, vad_handler=vad_handler)
        self.asr_decoder = (
            VLLMDecoder(
                global_model,
                global_vllm_client,
                "asr",
                asr_lang,
                asr_num_chunks,
                max_new_tokens=global_max_new_tokens,
                repetition_penalty=repetition_penalty,
                repetition_penalty_window=repetition_penalty_window,
                max_segments=global_max_segments,
                keep_segments=global_keep_segments,
            )
            if enable_asr
            else None
        )
        self.ast_decoder = (
            VLLMDecoder(
                global_model,
                global_vllm_client,
                "ast",
                ast_lang,
                ast_num_chunks,
                max_new_tokens=global_max_new_tokens,
                repetition_penalty=repetition_penalty,
                repetition_penalty_window=repetition_penalty_window,
                max_segments=global_max_segments,
                keep_segments=global_keep_segments,
            )
            if enable_ast
            else None
        )
        self.results = {"asr": "", "ast": ""}
        self._pending_feeds = []  # (chunk_out, is_finalize, is_last)
        self._need_decoder_reset = False

    def _on_encoder_output(self, chunk_out):
        self._pending_feeds.append((chunk_out.detach(), False, False))

    def _on_finalize_encoder_output(self, chunk_out, is_last):
        self._pending_feeds.append((chunk_out.detach(), True, is_last))

    def _on_vad_reset(self):
        # Defer reset until after pending finalize decodes are flushed.
        self._need_decoder_reset = True

    async def _flush_pending_decodes(self):
        """Run ASR/AST decoders for buffered encoder outputs (parallel per chunk)."""
        for chunk_out, is_finalize, is_last in self._pending_feeds:
            coros = []
            tasks = []
            if self.asr_decoder is not None:
                tasks.append("asr")
                if is_finalize:
                    coros.append(self.asr_decoder.finalize_chunk(chunk_out, is_last))
                else:
                    coros.append(self.asr_decoder.feed_chunk(chunk_out))
            if self.ast_decoder is not None:
                tasks.append("ast")
                if is_finalize:
                    coros.append(self.ast_decoder.finalize_chunk(chunk_out, is_last))
                else:
                    coros.append(self.ast_decoder.feed_chunk(chunk_out))
            if coros:
                texts = await asyncio.gather(*coros)
                for task, text in zip(tasks, texts):
                    if text:
                        self.results[task] += text
        self._pending_feeds = []
        if self._need_decoder_reset:
            if self.asr_decoder is not None:
                self.asr_decoder.reset()
            if self.ast_decoder is not None:
                self.ast_decoder.reset()
            self._need_decoder_reset = False

    async def process_audio_dict(self, pcm_chunk: np.ndarray) -> dict:
        if global_model is None or global_vllm_client is None:
            return {"asr": "Model not loaded.", "ast": "Model not loaded."}
        self.results = {"asr": "", "ast": ""}
        self._pending_feeds = []
        async with global_encoder_lock:
            await asyncio.to_thread(self.process_audio, pcm_chunk)
        await self._flush_pending_decodes()
        return self.results

    async def finalize_dict(self) -> dict:
        self.results = {"asr": "", "ast": ""}
        self._pending_feeds = []
        async with global_encoder_lock:
            await asyncio.to_thread(self.finalize)
        await self._flush_pending_decodes()
        return self.results


def _make_vad_handler():
    if global_use_vad and global_vad_model is not None:
        return VADHandler(
            global_vad_model,
            threshold=global_vad_threshold,
            sampling_rate=16000,
            min_silence_duration_ms=global_vad_min_silence_duration_ms,
            speech_pad_ms=global_vad_speech_pad_ms,
        )
    return None


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    global global_active_connections

    acquired = False
    if global_connection_sem is not None:
        try:
            await asyncio.wait_for(global_connection_sem.acquire(), timeout=0.05)
            acquired = True
        except asyncio.TimeoutError:
            await websocket.accept()
            await websocket.send_json({"type": "error", "text": "Server at capacity, try again later."})
            await websocket.close(code=1013)
            return

    await websocket.accept()
    global_active_connections += 1
    client = getattr(websocket, "client", None)
    logging.info(f"Client connected ({global_active_connections} active): {client}")

    try:
        if global_model is None or global_vllm_client is None:
            await websocket.send_json({"type": "error", "text": "Model / vLLM not loaded. Check server logs."})
            await websocket.close()
            return

        default_ast = global_ast_lang_options[0] if global_ast_lang_options else "Chinese"
        default_asr = global_asr_lang_options[0] if global_asr_lang_options else "auto"

        session = ParallelVLLMSession(
            ast_lang=default_ast,
            asr_lang=default_asr,
            repetition_penalty=global_repetition_penalty,
            repetition_penalty_window=global_repetition_penalty_window,
            vad_handler=_make_vad_handler(),
        )

        while True:
            try:
                message = await websocket.receive()
            except RuntimeError:
                break

            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "config":
                        session = ParallelVLLMSession(
                            asr_num_chunks=int(data.get("asr_num_chunks", 1)),
                            ast_num_chunks=int(data.get("ast_num_chunks", 1)),
                            ast_lang=data.get("lang", default_ast),
                            asr_lang=data.get("asr_lang", default_asr),
                            enable_asr=data.get("enable_asr", True),
                            enable_ast=data.get("enable_ast", True),
                            repetition_penalty=global_repetition_penalty,
                            repetition_penalty_window=global_repetition_penalty_window,
                            vad_handler=_make_vad_handler(),
                        )
                        logging.info(f"Config from {client}: {data}")
                    elif data.get("type") == "stop":
                        results = await session.finalize_dict()
                        for task in ("asr", "ast"):
                            if results[task]:
                                await websocket.send_json(
                                    {"type": "final", "task": task, "text": results[task]}
                                )
                except Exception as e:
                    logging.error(f"Error parsing text message: {e}", exc_info=True)

            elif "bytes" in message:
                audio_bytes = message["bytes"]
                logging.info(
                    f"[{client}] Received {len(audio_bytes)} bytes "
                    f"({len(audio_bytes)/4/16000:.3f}s)"
                )
                try:
                    y = np.frombuffer(audio_bytes, dtype=np.float32)
                    results = await session.process_audio_dict(y)
                    for task in ("asr", "ast"):
                        if results[task]:
                            await websocket.send_json(
                                {"type": "partial", "task": task, "text": results[task]}
                            )
                except Exception as e:
                    logging.error(f"Error handling audio bytes: {e}", exc_info=True)
                    try:
                        await websocket.send_json({"type": "error", "text": str(e)})
                    except Exception:
                        pass

    except WebSocketDisconnect:
        logging.info(f"WebSocket client disconnected: {client}")
    finally:
        global_active_connections = max(0, global_active_connections - 1)
        if acquired and global_connection_sem is not None:
            global_connection_sem.release()
        logging.info(f"Client cleaned up ({global_active_connections} active)")
