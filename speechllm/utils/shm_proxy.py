"""共享内存代理服务器。

在独立进程中运行，接收训练端的轻量请求（只含 shm 路径 + 元信息），
从 /dev/shm 读取 embedding，转成 SGLang 需要的 JSON 格式并转发。

这样做的好处：
  1. 训练进程不再做 .tolist() + JSON 序列化（这是最大瓶颈）
  2. 代理进程有独立的 GIL，序列化不阻塞训练
  3. /dev/shm 是内存文件系统，写入速度接近 memcpy
"""

import json
import logging
import os
import signal
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

import numpy as np
import requests

from speechllm.utils.shm_transport import read_tensor_from_shm, cleanup_shm_file

logger = logging.getLogger(__name__)


class ShmProxyHandler(BaseHTTPRequestHandler):
    """处理来自训练端的请求，从 shm 读取 embedding 并转发给 SGLang。"""

    # 类变量，由 ShmProxyServer 设置
    sglang_url: str = "http://127.0.0.1:30000"
    use_orjson: bool = False

    def log_message(self, format, *args):
        """抑制默认的 access log，太吵了。"""
        pass

    def do_POST(self):
        if self.path == "/generate_shm":
            self._handle_generate_shm()
        elif self.path == "/health":
            self._handle_health()
        elif self.path == "/shutdown":
            self._handle_shutdown()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/health":
            self._handle_health()
        else:
            self.send_error(404)

    def _handle_health(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def _handle_shutdown(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"shutting_down"}')
        # 在另一个线程中关闭服务器
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def _handle_generate_shm(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            req = json.loads(body)

            shm_path = req["shm_path"]
            shape = req["shape"]
            dtype = req["dtype"]
            sampling_params = req["sampling_params"]

            # 从共享内存读取 embedding
            arr = read_tensor_from_shm(shm_path, shape, dtype)

            # 清理 shm 文件（已读入内存）
            cleanup_shm_file(shm_path)

            # 转成 SGLang 需要的格式并转发
            # 用 float32 确保精度，SGLang 内部会自行转换
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)

            embeds_list = arr.tolist()

            data = {
                "input_embeds": embeds_list,
                "sampling_params": sampling_params,
            }

            # 转发给 SGLang
            resp = requests.post(
                f"{self.sglang_url}/generate",
                json=data,
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()

            # 附加输入长度信息
            result["_input_len"] = shape[0]

            response_body = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        except Exception as e:
            logger.error(f"[ShmProxy] Error handling request: {e}", exc_info=True)
            error_body = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(error_body)))
            self.end_headers()
            self.wfile.write(error_body)


class ThreadedHTTPServer(HTTPServer):
    """支持多线程处理请求的 HTTP Server。"""
    allow_reuse_address = True
    request_queue_size = 256

    def process_request(self, request, client_address):
        """每个请求在独立线程中处理。"""
        t = threading.Thread(
            target=self._handle_request_thread,
            args=(request, client_address),
            daemon=True,
        )
        t.start()

    def _handle_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def start_shm_proxy(
    sglang_url: str = "http://127.0.0.1:30000",
    proxy_port: int = 30100,
) -> None:
    """启动共享内存代理服务器（阻塞，用于独立进程）。

    Args:
        sglang_url: SGLang server 的 URL
        proxy_port: 代理监听端口
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [ShmProxy] %(levelname)s %(message)s",
    )

    ShmProxyHandler.sglang_url = sglang_url

    server = ThreadedHTTPServer(("0.0.0.0", proxy_port), ShmProxyHandler)
    logger.info(
        f"ShmProxy started on port {proxy_port}, forwarding to {sglang_url}"
    )

    # 优雅退出
    def _signal_handler(sig, frame):
        logger.info("Received signal, shutting down...")
        server.shutdown()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        server.serve_forever()
    finally:
        server.server_close()
        logger.info("ShmProxy stopped.")


# 允许直接 python -m speechllm.utils.shm_proxy 启动
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SHM Proxy for SGLang")
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    parser.add_argument("--port", type=int, default=30100)
    args = parser.parse_args()

    start_shm_proxy(sglang_url=args.sglang_url, proxy_port=args.port)
