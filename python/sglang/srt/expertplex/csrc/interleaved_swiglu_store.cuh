// SPDX-License-Identifier: Apache-2.0
// Adapted from sgl-deep-gemm 0.2.2,
// deep_gemm/impls/sm100_bf16_mega_moe.cuh (SHA256
// 3193a4fa4e9fea886de6e29bcdead1a930b3af20dbbee565d7e3653b555dc24e).
// Changes: task-local token ranges, APK retirement barriers and phase math.
// Derived from pinned DeepGEMM sm100_bf16_mega_moe.cuh L1 epilogue.
// Retains its granularity-8 gate/up interleave, TMEM fragment pairing,
// BF16 boundary, two-stage STSM/TMA store and 128-thread warpgroup.
//
// Optional sm100_swiglu backend; component improvements do not establish
// an end-to-end serving speedup. The separate activation backend remains available.
#pragma once
namespace expertplex_sm100 {

template <typename Storage>
__device__ __forceinline__ void store_interleaved_swiglu(
    Storage &smem, uint32_t &store_stage, uint32_t tmem_base,
    int token_base, int interleaved_column_base, int effective_rows,
    int phase, int epilogue_warp, int lane,
    const cutlass::arch::ClusterTransactionBarrier *tmem_empty,
    const cute::TmaDescriptor *activation_map) {
  // Native BF16 swap-AB geometry: 128 accumulator rows per CTA become
  // 64 activation columns; dynamic token dimension is a multiple of 16.
  constexpr int StoreRows = 16, AtomRows = 8, OutputColumns = 64;
  const int stores = effective_rows / StoreRows;
  for (int block = 0; block < stores; ++block) {
    __nv_bfloat162 output[4];
#pragma unroll
    for (int atom = 0; atom < 2; ++atom) {
      const uint32_t address = tmem_base + block * StoreRows + atom * AtomRows;
      uint32_t values[8];
      cute::SM100_TMEM_LOAD_16dp256b1x::copy(
          address, values[0], values[1], values[2], values[3]);
      cute::SM100_TMEM_LOAD_16dp256b1x::copy(
          address | 0x00100000, values[4], values[5], values[6], values[7]);
      cutlass::arch::fence_view_async_tmem_load();
      if (block == stores - 1 && atom == 1) {
        deep_gemm::ptx::tcgen05_before_thread_sync();
        tmem_empty->arrive(0u);
      }
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        // Native fragment pairs (0,2), (1,3), (4,6), (5,7).
        const auto gate_bf16 = __float22bfloat162_rn(make_float2(
            __uint_as_float(values[4 * pair]),
            __uint_as_float(values[4 * pair + 1])));
        const auto up_bf16 = __float22bfloat162_rn(make_float2(
            __uint_as_float(values[4 * pair + 2]),
            __uint_as_float(values[4 * pair + 3])));
        const float2 gate = __bfloat1622float2(gate_bf16);
        const float2 up = __bfloat1622float2(up_bf16);
        output[2 * atom + pair] = __float22bfloat162_rn(make_float2(
            expertplex_activation::silu_mul(gate.x, up.x, phase),
            expertplex_activation::silu_mul(gate.y, up.y, phase)));
      }
    }
    // The task ring already protects TMEM; this two-entry ring protects
    // shared output storage until the prior asynchronous store consumes it.
    if (epilogue_warp == 0)
      cute::tma_store_wait<1>();
    cutlass::arch::NamedBarrier::sync(128, 0);
#pragma unroll
    for (int atom = 0; atom < 2; ++atom) {
      const int row = lane % 8;
      const int column = epilogue_warp * 2 + lane / 8;
      auto *destination = smem.cd[store_stage] +
          (atom * AtomRows + row) * OutputColumns + (column ^ row) * 8;
      deep_gemm::ptx::SM90_U32x2_STSM_T<__nv_bfloat162>::copy(
          output[2 * atom], output[2 * atom + 1], destination);
    }
    cute::tma_store_fence();
    cutlass::arch::NamedBarrier::sync(128, 0);
    if (epilogue_warp == 0 && cute::elect_one_sync()) {
      cute::SM90_TMA_STORE_2D::copy(
          activation_map, smem.cd[store_stage],
          interleaved_column_base / 2, token_base + block * StoreRows);
      cute::tma_store_arrive();
    }
    __syncwarp();
    store_stage = (store_stage + 1) % 2;
  }
}
}  // namespace expertplex_sm100
