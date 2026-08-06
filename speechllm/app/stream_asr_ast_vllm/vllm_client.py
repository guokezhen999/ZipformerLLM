"""
Shared vLLM Ray client with micro-batching for multi-user streaming decode.

Multiple WebSocket sessions (and ASR/AST tasks within a session) submit
generate requests into a shared queue; a background worker flushes them
as batched `generate_batch` calls.

Prefix KV reuse:
  - Actors run with enable_prefix_caching=True (hashes EmbedsPrompt blocks).
  - Requests carry a route_key so consecutive segments of the same decoder
    stick to the same actor (required for multi-GPU cache hits).
  - History embeds are re-sent in full; vLLM reuses matching prefix blocks.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class _PendingRequest:
    embeds_list: list
    future: asyncio.Future
    enqueued_at: float
    route_key: str = ""


class SharedVLLMClient:
    """Async facade over one or more VLLMEmbedActor Ray actors."""

    def __init__(
        self,
        actors: List[Any],
        sampling_params: dict,
        batch_timeout_ms: float = 20.0,
        max_batch_size: int = 32,
        ray_timeout_s: float = 120.0,
    ):
        if not actors:
            raise ValueError("SharedVLLMClient requires at least one Ray actor")
        self.actors = actors
        self.sampling_params = sampling_params
        self.batch_timeout_s = max(0.0, batch_timeout_ms) / 1000.0
        self.max_batch_size = max(1, int(max_batch_size))
        self.ray_timeout_s = ray_timeout_s

        self._queue: asyncio.Queue = asyncio.Queue()
        self._rr = 0
        self._worker_task: Optional[asyncio.Task] = None
        self._closed = False

    @classmethod
    def connect(
        cls,
        num_actors: int,
        actor_name_prefix: str = "vllm_actor",
        ray_address: str = "auto",
        ray_namespace: str = "speechllm_vllm",
        wait_s: float = 300.0,
        **kwargs,
    ) -> "SharedVLLMClient":
        """Connect to pre-started named Ray actors (same convention as GRPO)."""
        import ray

        if not ray.is_initialized():
            ray.init(address=ray_address, ignore_reinit_error=True, namespace=ray_namespace)
            logger.info(f"Ray initialized (address={ray_address}, namespace={ray_namespace})")

        actors = []
        for i in range(num_actors):
            name = f"{actor_name_prefix}_{i}"
            deadline = time.time() + wait_s
            actor = None
            while time.time() < deadline:
                try:
                    actor = ray.get_actor(name, namespace=ray_namespace)
                    ok = ray.get(actor.health.remote(), timeout=10)
                    if ok:
                        logger.info(f"Connected to vLLM actor '{name}'")
                        break
                except Exception as e:
                    logger.debug(f"Waiting for actor '{name}': {e}")
                time.sleep(2)
            else:
                raise RuntimeError(f"vLLM actor '{name}' not ready within {wait_s}s")
            actors.append(actor)

        return cls(actors=actors, **kwargs)

    def start(self):
        if self._worker_task is None or self._worker_task.done():
            self._closed = False
            self._worker_task = asyncio.create_task(self._batch_worker(), name="vllm-batch-worker")
            logger.info(
                f"vLLM batch worker started "
                f"(actors={len(self.actors)}, timeout={self.batch_timeout_s*1000:.0f}ms, "
                f"max_batch={self.max_batch_size}, prefix_cache sticky routing on)"
            )

    async def close(self):
        self._closed = True
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    def _actor_index(self, route_key: str) -> int:
        """Sticky actor for a decoder/session so prefix KV can hit."""
        n = len(self.actors)
        if n == 1:
            return 0
        if route_key:
            # Stable within process; keeps one stream on one GPU.
            return hash(route_key) % n
        idx = self._rr % n
        self._rr += 1
        return idx

    async def generate(self, embeds: torch.Tensor, route_key: str = "") -> str:
        """Queue one prompt_embeds tensor (T, D) and await decoded text."""
        if self._closed:
            raise RuntimeError("SharedVLLMClient is closed")
        if self._worker_task is None or self._worker_task.done():
            self.start()

        if embeds.dim() == 3:
            embeds = embeds.squeeze(0)
        # float32 list is deterministic; actor rebuilds as bfloat16 for hashing.
        embeds_list = embeds.detach().float().cpu().tolist()

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put(
            _PendingRequest(embeds_list, fut, time.monotonic(), route_key=route_key or "")
        )
        return await fut

    @staticmethod
    def _normalize_results(raw) -> List[Dict[str, Any]]:
        """Accept legacy List[str] or List[dict] from generate_batch."""
        out = []
        for item in raw:
            if isinstance(item, dict):
                out.append(
                    {
                        "text": item.get("text") or "",
                        "num_cached_tokens": int(item.get("num_cached_tokens") or 0),
                        "prompt_len": int(item.get("prompt_len") or 0),
                    }
                )
            else:
                out.append({"text": item if item is not None else "", "num_cached_tokens": 0, "prompt_len": 0})
        return out

    async def _flush_actor_group(
        self,
        actor_idx: int,
        group: List[_PendingRequest],
        ray_mod,
    ):
        actor = self.actors[actor_idx]
        embeds_batch = [req.embeds_list for req in group]
        t0 = time.monotonic()
        try:
            ref = actor.generate_batch.remote(embeds_batch, self.sampling_params)
            timeout_s = self.ray_timeout_s

            def _ray_get():
                return ray_mod.get(ref, timeout=timeout_s)

            raw = await asyncio.to_thread(_ray_get)
            results = self._normalize_results(raw)
            cost = time.monotonic() - t0
            if len(results) != len(group):
                raise RuntimeError(
                    f"vLLM returned {len(results)} results for batch of {len(group)}"
                )

            cached_sum = sum(r["num_cached_tokens"] for r in results)
            prompt_sum = sum(r["prompt_len"] for r in results) or 1
            hit_pct = 100.0 * cached_sum / prompt_sum
            logger.info(
                f"vLLM actor={actor_idx} batch={len(group)} cost={cost:.3f}s "
                f"queue_wait={t0 - group[0].enqueued_at:.3f}s "
                f"prefix_cached={cached_sum}/{prompt_sum} ({hit_pct:.0f}%)"
            )
            for req, res in zip(group, results):
                if not req.future.done():
                    req.future.set_result(res["text"])
        except Exception as e:
            logger.error(
                f"vLLM batch failed (actor={actor_idx}, size={len(group)}): {e}",
                exc_info=True,
            )
            for req in group:
                if not req.future.done():
                    req.future.set_exception(e)

    async def _batch_worker(self):
        import ray

        while not self._closed:
            try:
                first: _PendingRequest = await self._queue.get()
            except asyncio.CancelledError:
                break

            batch = [first]
            # Collect more requests until timeout or max batch size.
            deadline = time.monotonic() + self.batch_timeout_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            groups: Dict[int, List[_PendingRequest]] = defaultdict(list)
            for req in batch:
                groups[self._actor_index(req.route_key)].append(req)

            await asyncio.gather(
                *[
                    self._flush_actor_group(actor_idx, group, ray)
                    for actor_idx, group in groups.items()
                ]
            )
