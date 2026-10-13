"""Attention-side dispatch and combine for a shared expert pool."""

from __future__ import annotations

import atexit
import os
import threading

from .config import ModelSpec, load_config
from .rendezvous import claim_client, wait_for_workers


class ExpertPlexClient:
    def __init__(self, config_path: str, client_id: int):
        import torch

        from . import native

        self.config = load_config(config_path)
        self.spec = ModelSpec.from_model_path(self.config.model_path)
        self.config.validate(self.spec)
        self.client = self.config.client(client_id)
        self.device = torch.cuda.current_device()
        if self.device != self.client.device:
            raise ValueError(
                f"ExpertPlex client {client_id} needs CUDA device "
                f"{self.client.device}, current device is {self.device}. "
                "All processes must use the same CUDA_VISIBLE_DEVICES ordering."
            )
        self.manifest, descriptors = wait_for_workers(self.config)
        claim_client(self.config, self.manifest, client_id)
        self.peers = []
        try:
            for peer in descriptors:
                self.peers.append(
                    native.RemoteBuffers(peer["buffers"], client_id, self.device)
                )
        except BaseException:
            self.close()
            raise
        self.native = native
        self.epoch = 0
        self.failed = False
        self.lock = threading.Lock()
        atexit.register(self.close)

    def close(self):
        for peer in self.peers:
            peer.close()
        self.peers.clear()

    def run(self, layer_id, hidden_states, topk_ids, topk_weights):
        import torch

        if hidden_states.numel() == 0:
            return hidden_states
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("ExpertPlex currently requires --disable-cuda-graph")
        # One in-flight operation per attention process. This also excludes
        # concurrent TBO subbatches from reusing this process's IPC slot.
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Concurrent forwards cannot share an ExpertPlex slot")
        try:
            if self.failed:
                raise RuntimeError(
                    "ExpertPlex slot failed; restart the whole deployment with "
                    "a fresh rendezvous directory before reusing its buffers"
                )
            if not 0 <= layer_id < self.spec.num_layers:
                raise ValueError(f"Invalid ExpertPlex layer: {layer_id}")
            rows, hidden = hidden_states.shape
            capacity = (
                self.config.max_prefill_tokens
                if self.client.phase == "prefill"
                else self.config.max_decode_tokens
            )
            if rows > capacity or hidden != self.spec.hidden_size:
                raise ValueError(
                    f"ExpertPlex input {tuple(hidden_states.shape)} exceeds "
                    f"{self.client.phase} capacity [{capacity}, {self.spec.hidden_size}]"
                )
            if hidden_states.dtype != torch.bfloat16:
                raise ValueError("ExpertPlex accepts BF16 hidden states only")
            if topk_ids.shape != (rows, self.spec.top_k):
                raise ValueError("ExpertPlex router output shape mismatch")
            if topk_weights.shape != topk_ids.shape:
                raise ValueError(
                    "ExpertPlex top-k IDs and weights must have equal shapes"
                )
            if self.epoch >= (1 << 30) - 1:
                raise RuntimeError("ExpertPlex epoch exhausted; restart the deployment")
            self.epoch += 1
            hidden_states = hidden_states.contiguous()
            topk_ids = topk_ids.to(dtype=torch.int32).contiguous()
            topk_weights = topk_weights.float().contiguous()
            row_map = torch.empty_like(topk_ids)
            output = torch.zeros(
                (rows, hidden), dtype=torch.float32, device=hidden_states.device
            )
            phase = 0 if self.client.phase == "prefill" else 1
            world = len(self.peers)
            # Dispatch all peers before waiting for any peer. Expert workers
            # never wait for each other, or for another attention client's slot.
            for rank, peer in enumerate(self.peers):
                self.native.dispatch(
                    peer,
                    hidden_states,
                    topk_ids,
                    row_map,
                    self.epoch,
                    layer_id,
                    phase,
                    rank,
                    world,
                )
            for rank, peer in enumerate(self.peers):
                self.native.combine(
                    peer,
                    output,
                    topk_ids,
                    topk_weights,
                    row_map,
                    self.epoch,
                    rank,
                    world,
                    timeout_ms=int(self.config.operation_timeout_s * 1000),
                )
            return output.to(dtype=hidden_states.dtype)
        except BaseException:
            # A partially dispatched operation must never be overwritten after
            # timeout or cancellation; restart rather than corrupt a new epoch.
            self.failed = True
            raise
        finally:
            self.lock.release()


_client = None
_client_lock = threading.Lock()


def get_client() -> ExpertPlexClient:
    global _client
    with _client_lock:
        if _client is None:
            path = os.environ.get("SGLANG_EXPERTPLEX_CONFIG")
            client_id = os.environ.get("SGLANG_EXPERTPLEX_CLIENT_ID")
            if path is None or client_id is None:
                raise RuntimeError("Missing ExpertPlex config or client ID environment")
            _client = ExpertPlexClient(path, int(client_id))
        return _client
