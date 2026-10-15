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


class ExpertPlexDispatcher(BaseDispatcher):
    def __init__(self, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self._dispatching = {}
        self._combining = {}
        self._active = {}
        self._mailbox = 0

    def dispatch(self, hidden_states, topk_output):
        from sglang.srt.expertplex.client import get_client

        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError("ExpertPlex requires standard TopK IDs and weights")
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

        return get_client().finish(combine_input.pending)

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
