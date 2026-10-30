"""Attention-side MoE layer with remote expert weight ownership.

Routing stays in the model's native TopK path. Dispatch/combine follows the
same SGLang interface as local expert runners; the persistent E worker performs
expert computation between the two asynchronous operations.
"""

from torch import nn
from torch.compiler import disable

from sglang.srt.layers.moe.token_dispatcher.expertplex import (
    ExpertPlexCombineInput,
    ExpertPlexDispatcher,
)


class ExpertPlexMoE(nn.Module):
    def __init__(self, *, layer_id, num_experts, quant_config=None, **kwargs):
        super().__init__()
        if quant_config is not None:
            raise ValueError("ExpertPlex attention supports unquantized BF16 only")
        self.layer_id = layer_id
        self.num_experts = num_experts
        self.num_local_experts = 0
        self.dispatcher = ExpertPlexDispatcher(layer_id)

    @disable
    def forward(self, hidden_states, topk_output):
        dispatched = self.dispatcher.dispatch(hidden_states, topk_output)
        return self.dispatcher.combine(self.run_moe_core(dispatched))

    def run_moe_core(self, dispatch_output):
        # No local GEMM or expert weights: E has already been notified by
        # dispatch, and combine waits through the existing stream/event edge.
        return ExpertPlexCombineInput(dispatch_output.pending)
