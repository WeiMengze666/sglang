"""Reject capacity predictions that omit the shared expert's critical path."""

import unittest

from test_config_optimizer import config, optimizer


class SharedProfileGuardTests(unittest.TestCase):
    def test_routed_only_profile_cannot_predict_shared_model(self):
        model = config.ModelSpec(
            hidden_size=2048,
            intermediate_size=1024,
            num_experts=16,
            top_k=4,
            num_layers=2,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=128,
            vocab_size=1024,
            architecture="Qwen2MoeForCausalLM",
            shared_intermediate_size=1024,
            qkv_bias=True,
            use_qk_norm=False,
        )
        for mode in ("sequential", "sbo", "auto"):
            with self.assertRaisesRegex(
                optimizer.ProfileCoverageError, "shared-branch"
            ):
                optimizer.search(
                    config.default_config("/model"),
                    model,
                    None,
                    input_length=128,
                    average_output_length=32,
                    context_length=4096,
                    prefill_slo_s=1.0,
                    decode_slo_s=0.05,
                    kv_tokens_per_attention_gpu=8192,
                    overlap=mode,
                )


if __name__ == "__main__":
    unittest.main()
