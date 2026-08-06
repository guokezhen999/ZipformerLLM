"""
Streaming ASR+AST parallel server — single model, shared encoder, dual decoders.
"""
import os
import json
import logging
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from speechllm.app.bin.model_loader import load_model_from_env
from speechllm.app.bin.streaming_session import LLMDecoder, StreamingSessionBase
from speechllm.app.bin.vad import VADHandler, has_vad, create_vad_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

global_model = None
global_config = None
global_device = None
global_repetition_penalty = 1.0
global_repetition_penalty_window = 0
global_ast_lang_options = ["Chinese", "English", "Japanese", "French", "German", "Spanish"]
global_asr_lang_options = ["auto", "Chinese", "English"]
global_vad_model = None
global_use_vad = False
global_vad_threshold = 0.5
global_vad_min_silence_duration_ms = 200
global_vad_speech_pad_ms = 100


def _option_html(options, selected_idx=0):
    """Generate <option> HTML for a list of language options."""
    parts = []
    for i, opt in enumerate(options):
        sel = ' selected' if i == selected_idx else ''
        label = 'Auto' if opt == 'auto' else opt
        parts.append(f'<option value="{opt}"{sel}>{label}</option>')
    return '\n'.join(parts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_model, global_config, global_device, global_repetition_penalty, global_repetition_penalty_window
    global global_ast_lang_options, global_asr_lang_options
    global global_vad_model, global_use_vad, global_vad_threshold
    global global_vad_min_silence_duration_ms, global_vad_speech_pad_ms
    global_model, global_config, global_device = load_model_from_env()
    global_repetition_penalty = float(os.environ.get("SPEECHLLM_REPETITION_PENALTY", "1.0"))
    global_repetition_penalty_window = int(os.environ.get("SPEECHLLM_REPETITION_PENALTY_WINDOW", "0"))
    global_ast_lang_options = os.environ.get("SPEECHLLM_AST_LANG_OPTIONS", "Chinese,English,Japanese,French,German,Spanish").split(",")
    global_asr_lang_options = os.environ.get("SPEECHLLM_ASR_LANG_OPTIONS", "auto,Chinese,English").split(",")
    logging.info(f"Repetition penalty: {global_repetition_penalty}, window: {global_repetition_penalty_window}")
    logging.info(f"AST lang options: {global_ast_lang_options}")
    logging.info(f"ASR lang options: {global_asr_lang_options}")

    # VAD initialization
    global_use_vad = os.environ.get("SPEECHLLM_USE_VAD", "").lower() in ("1", "true", "yes")
    if global_use_vad:
        if not has_vad():
            logging.warning("VAD enabled but silero-vad not installed. Disabling VAD.")
            global_use_vad = False
        else:
            global_vad_model = create_vad_model()
            global_vad_threshold = float(os.environ.get("SPEECHLLM_VAD_THRESHOLD", "0.5"))
            global_vad_min_silence_duration_ms = int(os.environ.get("SPEECHLLM_VAD_MIN_SILENCE_DURATION_MS", "200"))
            global_vad_speech_pad_ms = int(os.environ.get("SPEECHLLM_VAD_SPEECH_PAD_MS", "100"))
            logging.info(
                f"VAD enabled: threshold={global_vad_threshold}, "
                f"min_silence={global_vad_min_silence_duration_ms}ms, "
                f"speech_pad={global_vad_speech_pad_ms}ms"
            )
    else:
        logging.info("VAD disabled (set SPEECHLLM_USE_VAD=1 to enable)")

    yield
    logging.info("Shutting down ASR/AST Parallel Stream Server")


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


class ParallelSession(StreamingSessionBase):
    """Shared encoder, independent ASR and AST decoders."""

    def __init__(self, asr_num_chunks=1, ast_num_chunks=1, ast_lang="Chinese", asr_lang="auto",
                 enable_asr=True, enable_ast=True, repetition_penalty=1.0, repetition_penalty_window=0,
                 vad_handler=None):
        super().__init__(global_model, global_config, global_device, vad_handler=vad_handler)
        self.asr_decoder = LLMDecoder(global_model, "asr", asr_lang, asr_num_chunks, repetition_penalty=repetition_penalty, repetition_penalty_window=repetition_penalty_window) if enable_asr else None
        self.ast_decoder = LLMDecoder(global_model, "ast", ast_lang, ast_num_chunks, repetition_penalty=repetition_penalty, repetition_penalty_window=repetition_penalty_window) if enable_ast else None
        self.results = {"asr": "", "ast": ""}

    def _on_encoder_output(self, chunk_out):
        if self.asr_decoder:
            text = self.asr_decoder.feed_chunk(chunk_out)
            if text:
                self.results["asr"] += text
        if self.ast_decoder:
            text = self.ast_decoder.feed_chunk(chunk_out)
            if text:
                self.results["ast"] += text

    def _on_finalize_encoder_output(self, chunk_out, is_last):
        if self.asr_decoder:
            text = self.asr_decoder.finalize_chunk(chunk_out, is_last)
            if text:
                self.results["asr"] += text
        if self.ast_decoder:
            text = self.ast_decoder.finalize_chunk(chunk_out, is_last)
            if text:
                self.results["ast"] += text

    def _on_vad_reset(self):
        """VAD end-of-speech: reset both decoders."""
        if self.asr_decoder is not None:
            self.asr_decoder.reset()
        if self.ast_decoder is not None:
            self.ast_decoder.reset()

    def process_audio_dict(self, pcm_chunk: np.ndarray) -> dict:
        if global_model is None:
            return {"asr": "Model not loaded.", "ast": "Model not loaded."}
        self.results = {"asr": "", "ast": ""}
        self.process_audio(pcm_chunk)
        return self.results

    def finalize_dict(self) -> dict:
        self.results = {"asr": "", "ast": ""}
        self.finalize()
        return self.results


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if global_model is None:
        await websocket.send_json({"type": "error", "text": "Model not loaded. Check server logs."})
        await websocket.close()
        return

    # Default language: first option from the configured option lists
    default_ast = global_ast_lang_options[0] if global_ast_lang_options else "Chinese"
    default_asr = global_asr_lang_options[0] if global_asr_lang_options else "auto"

    def _make_vad_handler():
        """Create a fresh VADHandler if VAD is enabled, else return None."""
        if global_use_vad and global_vad_model is not None:
            return VADHandler(
                global_vad_model,
                threshold=global_vad_threshold,
                sampling_rate=16000,
                min_silence_duration_ms=global_vad_min_silence_duration_ms,
                speech_pad_ms=global_vad_speech_pad_ms,
            )
        return None

    session = ParallelSession(
        ast_lang=default_ast, asr_lang=default_asr,
        repetition_penalty=global_repetition_penalty, repetition_penalty_window=global_repetition_penalty_window,
        vad_handler=_make_vad_handler())

    try:
        while True:
            try:
                message = await websocket.receive()
            except RuntimeError:
                # Starlette >=1.0 raises RuntimeError when receive() is called after disconnect
                break

            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "config":
                        session = ParallelSession(
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
                        logging.info(f"Config: {data}")
                    elif data.get("type") == "stop":
                        results = session.finalize_dict()
                        for task in ("asr", "ast"):
                            if results[task]:
                                await websocket.send_json({"type": "final", "task": task, "text": results[task]})
                except Exception as e:
                    logging.error(f"Error parsing text message: {e}", exc_info=True)

            elif "bytes" in message:
                audio_bytes = message["bytes"]
                logging.info(f"Received {len(audio_bytes)} bytes ({len(audio_bytes)/4/16000:.3f}s)")
                try:
                    y = np.frombuffer(audio_bytes, dtype=np.float32)
                    results = session.process_audio_dict(y)
                    for task in ("asr", "ast"):
                        if results[task]:
                            await websocket.send_json({"type": "partial", "task": task, "text": results[task]})
                except Exception as e:
                    logging.error(f"Error handling audio bytes: {e}", exc_info=True)

    except WebSocketDisconnect:
        logging.info("WebSocket Client Disconnected.")
