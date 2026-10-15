"""Attention-side dispatch and combine for a shared expert pool."""

from __future__ import annotations

import atexit
import os
import threading
from dataclasses import dataclass
from typing import Any

from .config import ModelSpec, load_config
from .rendezvous import claim_client, wait_for_workers


@dataclass
class PendingMoE:
    mailbox: int
    output: Any
    done: Any
    references: tuple
    finished: bool = False


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
        self.mailboxes = []
        try:
            for index in range(self.config.mailboxes_per_client):
                peers = []
                self.mailboxes.append(peers)
                slot = client_id * self.config.mailboxes_per_client + index
                for peer in descriptors:
                    mapping = native.RemoteBuffers(peer["buffers"], slot, self.device)
                    peers.append(mapping)
                    self.peers.append(mapping)
        except BaseException:
            self.close()
            raise
        self.native = native
        self.groups = (
            [native.RemoteGroup(peers) for peers in self.mailboxes]
            if self.config.prefill_preprocess
            or os.environ.get("SGLANG_EXPERTPLEX_GROUP_COMM") == "1"
            else []
        )
        self.epoch = 0
        self.failed = False
        self.lock = threading.Lock()
        self.streams = [torch.cuda.Stream(device=self.device) for _ in self.mailboxes]
        self.pending = [None for _ in self.mailboxes]
        atexit.register(self.close)

    def close(self):
        # The exporter must outlive all remote reads. Launcher shutdown orders
        # clients before experts; synchronize our own streams before IPC close.
        for stream in getattr(self, "streams", []):
            stream.synchronize()
        self.groups = []
        for peer in self.peers:
            peer.close()
        self.peers.clear()

    def run(self, layer_id, hidden_states, topk_ids, topk_weights):
        return self.finish(self.begin(layer_id, hidden_states, topk_ids, topk_weights))

    def begin(self, layer_id, hidden_states, topk_ids, topk_weights, mailbox=0):
        import torch

        if hidden_states.numel() == 0:
            return PendingMoE(-1, hidden_states, None, ())
        if not 0 <= mailbox < len(self.mailboxes):
            raise ValueError("TBO microbatch has no allocated ExpertPlex mailbox")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Concurrent host threads cannot enqueue ExpertPlex work")
        try:
            if self.pending[mailbox] is not None:
                raise RuntimeError(
                    "ExpertPlex mailbox reused before its result was consumed"
                )
            if self.failed:
                raise RuntimeError(
                    "ExpertPlex slot failed; restart the whole deployment with "
                    "a fresh rendezvous directory before reusing its buffers"
                )
            if not 0 <= layer_id < self.spec.num_layers:
                raise ValueError(f"Invalid ExpertPlex layer: {layer_id}")
            rows, hidden = hidden_states.shape
            capacity = (
                self.config.prefill_mailbox_capacity
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
            compute_stream = torch.cuda.current_stream(self.device)
            comm_stream = self.streams[mailbox]
            # This is an event dependency, not a CPU or device synchronization.
            comm_stream.wait_stream(compute_stream)
            phase = 0 if self.client.phase == "prefill" else 1
            peers = self.mailboxes[mailbox]
            world = len(peers)
            inputs = (hidden_states, topk_ids, topk_weights)
            for tensor in inputs:
                tensor.record_stream(comm_stream)
            with torch.cuda.stream(comm_stream):
                hidden_states = hidden_states.contiguous()
                topk_ids = topk_ids.to(dtype=torch.int32).contiguous()
                topk_weights = topk_weights.float().contiguous()
                row_map = torch.empty_like(topk_ids)
                output = torch.zeros(
                    (rows, hidden), dtype=torch.float32, device=hidden_states.device
                )
                if self.groups:
                    self.groups[mailbox].dispatch(
                        hidden_states, topk_ids, row_map, layer_id, phase, topk_weights
                    )
                else:
                    for rank, peer in enumerate(peers):
                        self.native.dispatch(
                            peer,
                            hidden_states,
                            topk_ids,
                            row_map,
                            0,
                            layer_id,
                            phase,
                            rank,
                            world,
                        )
                # WaitDone is a one-thread GPU kernel on COMM. COMPUTE remains
                # available to another microbatch until finish() adds its edge.
                if self.groups:
                    self.groups[mailbox].combine(
                        output,
                        topk_ids,
                        topk_weights,
                        row_map,
                        timeout_ms=int(self.config.operation_timeout_s * 1000),
                    )
                else:
                    for rank, peer in enumerate(peers):
                        self.native.combine(
                            peer,
                            output,
                            topk_ids,
                            topk_weights,
                            row_map,
                            0,
                            rank,
                            world,
                            timeout_ms=int(self.config.operation_timeout_s * 1000),
                            synchronize=False,
                        )
                output = output.to(dtype=hidden_states.dtype)
                done = torch.cuda.Event()
                done.record(comm_stream)
            pending = PendingMoE(
                mailbox,
                output,
                done,
                inputs + (hidden_states, topk_ids, topk_weights, row_map),
            )
            self.pending[mailbox] = pending
            return pending
        except BaseException:
            # A partially dispatched operation must never be overwritten after
            # timeout or cancellation; restart rather than corrupt a new epoch.
            self.failed = True
            raise
        finally:
            self.lock.release()

    def finish(self, pending):
        import torch

        if pending.finished:
            raise RuntimeError("ExpertPlex result consumed twice")
        if pending.mailbox < 0:
            pending.finished = True
            return pending.output
        if self.pending[pending.mailbox] is not pending:
            raise RuntimeError("ExpertPlex result does not own its mailbox")
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(pending.done)
        pending.output.record_stream(stream)
        self.pending[pending.mailbox] = None
        pending.finished = True
        return pending.output


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
