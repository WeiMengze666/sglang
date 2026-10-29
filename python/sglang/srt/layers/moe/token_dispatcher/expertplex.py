"""ExpertPlex transport through SGLang's dispatcher and TBO interfaces."""

from dataclasses import dataclass
from typing import Any

from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInputFormat,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutputChecker


@dataclass
class ExpertPlexDispatchOutput:
    hidden_states: Any
    pending: Any

    @property
    def format(self):
        return DispatchOutputFormat.EXPERTPLEX


@dataclass
class ExpertPlexCombineInput:
    pending: Any

    @property
    def format(self):
        return CombineInputFormat.EXPERTPLEX


@dataclass
class FollowerContribution:
    """Zero partial result; native MOE_OUTPUT reduction supplies the leader sum."""

    output: Any
    finished: bool = False


class ExpertPlexDispatcher(BaseDispatcher):
    def __init__(self, layer_id: int):
        super().__init__()
        from sglang.srt.runtime_context import get_parallel

        self._transport_owner = get_parallel().tp_rank == 0
        self.layer_id = layer_id
        self._dispatching = {}
        self._combining = {}
        self._active = {}
        self._mailbox = 0

    def dispatch(self, hidden_states, topk_output):
        from sglang.srt.expertplex.client import get_client

        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError("ExpertPlex requires standard TopK IDs and weights")
        if not self._transport_owner:
            import torch

            # The native FFN boundary still owes MOE_OUTPUT's TP sum. Return
            # one complete routed contribution on rank zero, zero on followers;
            # never publish or acquire the owner's exclusive mailbox lease here.
            pending = FollowerContribution(torch.zeros_like(hidden_states))
            return ExpertPlexDispatchOutput(hidden_states, pending)
        pending = get_client().begin(
            self.layer_id,
            hidden_states,
            topk_output.topk_ids,
            topk_output.topk_weights,
            mailbox=self._mailbox,
        )
        return ExpertPlexDispatchOutput(hidden_states, pending)

    def combine(self, combine_input):
        from sglang.srt.expertplex.client import get_client

        pending = combine_input.pending
        if isinstance(pending, FollowerContribution):
            if self._transport_owner or pending.finished:
                raise RuntimeError("Invalid or already consumed follower contribution")
            pending.finished = True
            return pending.output
        if not self._transport_owner:
            raise RuntimeError("Only the TP leader can consume a remote result")
        return get_client().finish(pending)

    def dispatch_a(self, hidden_states, topk_output, tbo_subbatch_index=None):
        slot = tbo_subbatch_index or 0
        if slot in self._active:
            raise RuntimeError("ExpertPlex TBO mailbox already has a dispatch")
        self._mailbox = slot
        try:
            # Use BaseDispatcher's public entry, including registered hooks.
            result = self.dispatch(hidden_states=hidden_states, topk_output=topk_output)
        finally:
            self._mailbox = 0
        self._dispatching[slot] = result
        self._active[slot] = result.pending

    def dispatch_b(self, tbo_subbatch_index=None):
        return self._dispatching.pop(tbo_subbatch_index or 0)

    def combine_a(self, combine_input, tbo_subbatch_index=None):
        slot = tbo_subbatch_index or 0
        if (
            self._active.get(slot) is not combine_input.pending
            or slot in self._combining
        ):
            raise RuntimeError("ExpertPlex TBO combine does not own its dispatch")
        self._combining[slot] = combine_input

    def combine_b(self, tbo_subbatch_index=None):
        slot = tbo_subbatch_index or 0
        result = self.combine(self._combining.pop(slot))
        del self._active[slot]
        return result
