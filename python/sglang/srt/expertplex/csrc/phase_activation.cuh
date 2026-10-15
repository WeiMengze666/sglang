#pragma once

// Preserve the two native SGLang BF16 activation paths. Contiguous prefill
// uses the CUDA activation kernel; masked decode uses Triton's exp2 lowering
// and div.full.f32 (pinned Triton 3.8 / SM103). Both round only the final
// SiLU*up result to BF16. Re-run check_activation.py when upgrading kernels.
namespace expertplex_activation {
__device__ __forceinline__ float silu_mul(float gate, float up, int phase) {
  if (phase == 0)
    return gate / (1.f + expf(-gate)) * up;
  float value;
  asm("{\n"
      " .reg .f32 exponent, exponential, denominator, activated;\n"
      " mul.f32 exponent, %1, 0fBFB8AA3B;\n"
      " ex2.approx.f32 exponential, exponent;\n"
      " add.f32 denominator, exponential, 0f3F800000;\n"
      " div.full.f32 activated, %1, denominator;\n"
      " mul.f32 %0, activated, %2;\n"
      "}\n"
      : "=f"(value)
      : "f"(gate), "f"(up));
  return value;
}
} // namespace expertplex_activation
