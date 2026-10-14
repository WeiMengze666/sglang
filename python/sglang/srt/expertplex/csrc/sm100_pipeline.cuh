// Blackwell BF16 APK development backend.
// Uses the native DeepGEMM SM100 TMA / UMMA primitives and producer, MMA,
// epilogue warp roles. See PAPER_CONFORMANCE.md for outstanding validation.
// Included after the shared IPC protocol definitions in expertplex.cu.
#pragma once

#include <cute/atom/copy_traits_sm100.hpp>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace expertplex_sm100 {
using Barrier = cutlass::arch::ClusterTransactionBarrier;
using Element = cutlass::bfloat16_t;
constexpr int M = 64, N = 128, K = 64, Stages = 4, OutputStages = 2;
constexpr int Threads = 192;

struct alignas(128) Maps {
  CUtensorMap input[2];
  CUtensorMap activation[2];
};
struct alignas(128) WeightMaps {
  CUtensorMap gate, up, down;
};
struct alignas(16) Task {
  int slot, stage, layer, expert, row, column, rows, phase;
  U64 epoch;
};
struct Storage {
  alignas(1024) Element a[Stages][M * K];
  alignas(1024) Element g[Stages][N * K];
  alignas(1024) Element u[Stages][N * K];
  Barrier full[Stages], empty[Stages];
  Barrier task_full[OutputStages], task_empty[OutputStages];
  Barrier accum_full[OutputStages], accum_empty[OutputStages];
  Barrier decision_full[OutputStages], cluster_empty[OutputStages];
  Task decision[OutputStages];
  Task task[OutputStages];
  uint32_t tmem;
};

inline void encode_map(CUtensorMap &map, void *ptr, int inner, int outer,
                       int tile_outer) {
  uint64_t dims[2] = {static_cast<uint64_t>(inner),
                      static_cast<uint64_t>(outer)};
  uint64_t strides[1] = {static_cast<uint64_t>(inner) * 2};
  uint32_t box[2] = {K, static_cast<uint32_t>(tile_outer)};
  uint32_t element_strides[2] = {1, 1};
  auto status = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(status == CUDA_SUCCESS,
              "cuTensorMapEncodeTiled failed: ", static_cast<int>(status));
}

struct Context {
  Maps *maps;
  WeightMaps *weights;
  int *policy;
};

inline Context prepare(OwnedBuffers &owner, const BF16 *w13, const BF16 *w2,
                       int layers, int ctas) {
  auto b = owner.b;
  TORCH_CHECK(b.hidden % K == 0 && b.intermediate % K == 0,
              "SM100 native pipeline requires H and I divisible by 64");
  std::vector<Maps> maps(b.slots * b.experts);
  for (int s = 0; s < b.slots; ++s)
    for (int e = 0; e < b.experts; ++e) {
      size_t index = static_cast<size_t>(s) * b.experts + e;
      for (int p = 0; p < 2; ++p) {
        const int tile_m = p == 0 ? 64 : 32;
        size_t base = b.base_rows[s] + (b.packed[s] ? 0 : e * b.capacity[s]);
        int rows = b.packed[s] ? b.storage_rows[s] : b.capacity[s];
        encode_map(maps[index].input[p], b.input + base * b.hidden, b.hidden,
                   rows, tile_m);
        encode_map(maps[index].activation[p],
                   b.activation + base * b.intermediate, b.intermediate, rows,
                   tile_m);
      }
    }
  std::vector<WeightMaps> weights(layers * b.experts);
  for (int l = 0; l < layers; ++l)
    for (int e = 0; e < b.experts; ++e) {
      size_t index = static_cast<size_t>(l) * b.experts + e;
      auto gate =
          const_cast<BF16 *>(w13) + index * 2 * b.intermediate * b.hidden;
      auto down = const_cast<BF16 *>(w2) + index * b.hidden * b.intermediate;
      encode_map(weights[index].gate, gate, b.hidden, b.intermediate, N);
      encode_map(weights[index].up,
                 gate + static_cast<size_t>(b.intermediate) * b.hidden,
                 b.hidden, b.intermediate, N);
      encode_map(weights[index].down, down, b.intermediate, b.hidden, N);
    }
  Context result;
  result.maps = static_cast<Maps *>(owner.allocate(maps.size() * sizeof(Maps)));
  result.weights = static_cast<WeightMaps *>(
      owner.allocate(weights.size() * sizeof(WeightMaps)));
  result.policy = static_cast<int *>(owner.allocate(ctas * sizeof(int)));
  C10_CUDA_CHECK(cudaMemcpy(result.maps, maps.data(),
                            maps.size() * sizeof(Maps),
                            cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(result.weights, weights.data(),
                            weights.size() * sizeof(WeightMaps),
                            cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemset(result.policy, 0, ctas * sizeof(int)));
  return result;
}

template <int Cluster>
__device__ void initialize(DeviceBuffers b, int slot, int layers) {
  auto *c = b.control + static_cast<size_t>(slot) * kControlWords;
  auto *s = b.states + slot;
  U64 q = queue_load(&s->queue);
  if (((q >> 32) & 3) != IDLE)
    return;
  U64 epoch = system_load(c + READY);
  if (epoch <= (q >> 34) ||
      !queue_cas(&s->queue, q, queue_value(epoch, INITIALIZING)))
    return;
  c[START_NS] = now_ns();
  s->layer = static_cast<int>(uncached_load_u64(c + LAYER));
  s->phase = static_cast<int>(uncached_load_u64(c + PHASE));
  if (s->layer < 0 || s->layer >= layers)
    set_error(c, BAD_LAYER);
  if (s->phase < 0 || s->phase > 1)
    set_error(c, BAD_PHASE);
  const int tile_m = s->phase == 0 ? 64 : 32;
  U64 rows = 0;
  for (int e = 0; e < b.experts; ++e) {
    int count =
        uncached_load_int(b.counts + static_cast<size_t>(slot) * b.experts + e);
    if (count < 0 || count > b.capacity[slot])
      set_error(c, ROW_OVERFLOW);
    else
      rows += (count + tile_m - 1) / tile_m;
  }
  U64 gate = rows * ((b.intermediate + N * Cluster - 1) / (N * Cluster));
  U64 down = rows * ((b.hidden + N * Cluster - 1) / (N * Cluster));
  if (gate > UINT_MAX - 65536ULL || down > UINT_MAX - 65536ULL)
    set_error(c, ROW_OVERFLOW);
  s->gate_tiles = gate;
  s->down_tiles = down;
  s->completed = 0;
  __threadfence();
  if (system_load(c + ERROR) || gate == 0) {
    queue_store(&s->queue, queue_value(epoch, IDLE));
    c[FINISH_NS] = now_ns();
    system_store(c + DONE, epoch);
  } else
    queue_store(&s->queue, queue_value(epoch, GATE));
}

// Only CTA0 reads attention readiness words. Other CTAs consume device-scope
// policies; no SM is reserved for a separate receiver/polling kernel.
template <int Cluster>
__device__ void update_policy(DeviceBuffers b, Context context, int layers,
                              int decode_budget, float expected, int cap) {
  for (int s = 0; s < b.slots; ++s)
    initialize<Cluster>(b, s, layers);
  U64 footprint = 0;
  for (int i = 0; i < b.slots; ++i) {
    auto *s = b.states + i;
    auto stage = (queue_load(&s->queue) >> 32) & 3;
    if ((stage == GATE || stage == DOWN) && s->phase == 1)
      footprint +=
          stage == GATE
              ? s->gate_tiles /
                    ((b.intermediate + N * Cluster - 1) / (N * Cluster))
              : s->down_tiles / ((b.hidden + N * Cluster - 1) / (N * Cluster));
  }
  int budget =
      footprint
          ? min(cap, max(1, static_cast<int>(ceilf(
                                decode_budget * static_cast<float>(footprint) /
                                expected))))
          : 0;
  budget = min(cap, (budget + Cluster - 1) / Cluster * Cluster);
  for (int i = 0; i < gridDim.x / Cluster; ++i)
    cuda::atomic_ref<int, cuda::thread_scope_device>(context.policy[i])
        .store(i * Cluster < budget, cuda::memory_order_release);
}

template <int Cluster>
__device__ bool claim(DeviceBuffers b, int phase, int cursor, Task &task) {
  for (int i = 0; i < b.slots; ++i) {
    int slot = (cursor + i) % b.slots;
    auto *state = b.states + slot;
    U64 q = queue_load(&state->queue);
    int stage = (q >> 32) & 3;
    if ((stage != GATE && stage != DOWN) || state->phase != phase)
      continue;
    unsigned index = static_cast<unsigned>(q);
    unsigned total = stage == GATE ? state->gate_tiles : state->down_tiles;
    if (index >= total || !queue_cas(&state->queue, q, q + 1))
      continue;
    const int tile_m = phase == 0 ? 64 : 32;
    const int tile_n =
        ((stage == GATE ? b.intermediate : b.hidden) + N * Cluster - 1) /
        (N * Cluster);
    for (int e = 0; e < b.experts; ++e) {
      int rows = b.counts[static_cast<size_t>(slot) * b.experts + e];
      unsigned count = ((rows + tile_m - 1) / tile_m) * tile_n;
      if (index < count) {
        task = {slot,
                stage,
                state->layer,
                e,
                static_cast<int>(index / tile_n) * tile_m,
                static_cast<int>(index % tile_n) * N * Cluster,
                rows,
                phase,
                q >> 34};
        return true;
      }
      index -= count;
    }
    asm volatile("trap;");
  }
  return false;
}

template <int Cluster>
__global__ __launch_bounds__(Threads,
                             1) void executor(DeviceBuffers b, Context context,
                                              int layers, int decode_budget,
                                              float expected, int cap) {
  extern __shared__ __align__(1024) unsigned char workspace[];
  auto &s = *reinterpret_cast<Storage *>(workspace);
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int rank = blockIdx.x % Cluster;
  if (threadIdx.x == 32) {
    for (int i = 0; i < Stages; ++i) {
      s.full[i].init(1);
      s.empty[i].init(Cluster);
    }
    for (int i = 0; i < OutputStages; ++i) {
      s.task_full[i].init(1);
      s.task_empty[i].init(64);
      s.accum_full[i].init(1);
      s.accum_empty[i].init(64);
      s.decision_full[i].init(1);
      s.cluster_empty[i].init(64 * Cluster);
    }
    cutlass::arch::fence_barrier_init();
  }
  if (warp == 2)
    cute::TMEM::Allocator1Sm().allocate(512, &s.tmem);
  __syncthreads();
  if constexpr (Cluster > 1) {
    cute::cluster_arrive();
    cute::cluster_wait();
  }
  uint32_t stage = 0, phase = 0;
  auto advance = [&]() {
    stage = (stage + 1) % Stages;
    phase ^= stage == 0;
  };

  if (warp == 0 && lane == 0) {
    int cursor = blockIdx.x % b.slots;
    int previous_phase = -1;
    for (uint32_t sequence = 0;; ++sequence) {
      const int index = sequence % OutputStages,
                parity = (sequence / OutputStages) & 1;
      s.task_empty[index].wait(parity ^ 1);
      Task task{};
      task.slot = -1;
      if (rank == 0) {
        // A ring slot is reusable only after every participating CTA retires
        // its epilogue. The other output slot remains pipelined concurrently.
        s.cluster_empty[index].wait(parity ^ 1);
        while (task.slot < 0) {
          if (atomicAdd(b.stop, 0)) {
            task.slot = -2;
            break;
          }
          if (blockIdx.x == 0)
            update_policy<Cluster>(b, context, layers, decode_budget, expected,
                                   cap);
          int preferred = cuda::atomic_ref<int, cuda::thread_scope_device>(
                              context.policy[blockIdx.x / Cluster])
                              .load(cuda::memory_order_acquire);
          if (!claim<Cluster>(b, preferred, cursor, task))
            claim<Cluster>(b, 1 - preferred, cursor, task);
          if (task.slot < 0)
            __nanosleep(64);
        }
        // Drain at a phase switch only, not after every tile. No member can
        // independently claim an old-phase tile after this shared decision.
        if (sequence && task.slot >= 0 && task.phase != previous_phase) {
          const uint32_t previous = sequence - 1;
          s.cluster_empty[previous % OutputStages].wait(
              (previous / OutputStages) & 1);
        }
        for (int member = 1; member < Cluster; ++member)
          deep_gemm::ptx::st_async_cluster(&s.decision[index], task, member,
                                           s.decision_full[index]);
      } else {
        s.decision_full[index].arrive_and_expect_tx(sizeof(Task));
        s.decision_full[index].wait(parity);
        task = s.decision[index];
      }
      task.column += rank * N;
      s.task[index] = task;
      s.task_full[index].arrive();
      if (task.slot == -2)
        break;
      previous_phase = task.phase;
      cursor = (task.slot + 1) % b.slots;
      const auto &maps = context.maps[task.slot * b.experts + task.expert];
      const auto &weights =
          context.weights[task.layer * b.experts + task.expert];
      const bool gate = task.stage == GATE;
      const int reduction = gate ? b.hidden : b.intermediate;
      const int tile_m = task.phase == 0 ? 64 : 32;
      const int map_row =
          task.row + (b.packed[task.slot]
                          ? b.offsets[task.slot * b.experts + task.expert]
                          : 0);
      for (int k = 0; k < reduction; k += K, advance()) {
        s.empty[stage].wait(phase ^ 1);
        auto *a = gate ? &maps.input[task.phase] : &maps.activation[task.phase];
        auto *g = gate ? &weights.gate : &weights.down;
        if constexpr (Cluster > 1) {
          if (rank == 0)
            cute::SM90_TMA_LOAD_MULTICAST_2D::copy(
                a, reinterpret_cast<uint64_t *>(&s.full[stage]),
                (1 << Cluster) - 1,
                static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                s.a[stage], k, map_row);
        } else {
          if (task.phase == 0)
            deep_gemm::tma::copy<K, 64, 128, Element>(a, &s.full[stage],
                                                      s.a[stage], k, map_row);
          else
            deep_gemm::tma::copy<K, 32, 128, Element>(a, &s.full[stage],
                                                      s.a[stage], k, map_row);
        }
        deep_gemm::tma::copy<K, N, 128, Element>(g, &s.full[stage], s.g[stage],
                                                 k, task.column);
        if (gate)
          deep_gemm::tma::copy<K, N, 128, Element>(&weights.up, &s.full[stage],
                                                   s.u[stage], k, task.column);
        s.full[stage].arrive_and_expect_tx((tile_m + N * (gate ? 2 : 1)) * K *
                                           2);
      }
    }
  } else if (warp == 1) {
    auto descriptor = cute::UMMA::make_instr_desc<Element, Element, float, 128,
                                                  N, cute::UMMA::Major::K,
                                                  cute::UMMA::Major::K>();
    auto instruction = cute::UMMA::make_runtime_instr_desc(descriptor);
    for (uint32_t sequence = 0;; ++sequence) {
      const int index = sequence % OutputStages,
                parity = (sequence / OutputStages) & 1;
      s.task_full[index].wait(parity);
      Task task = s.task[index];
      if (task.slot == -2)
        break;
      s.accum_empty[index].wait(parity ^ 1);
      deep_gemm::ptx::tcgen05_after_thread_sync();
      const bool gate = task.stage == GATE;
      const int reduction = gate ? b.hidden : b.intermediate;
      for (int k = 0; k < reduction; k += K, advance()) {
        s.full[stage].wait(phase);
        deep_gemm::ptx::tcgen05_after_thread_sync();
        if (lane == 0) {
          auto a =
              deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, M, K,
                                                    128>(s.a[stage], 0, 0);
          auto g =
              deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, N, K,
                                                    128>(s.g[stage], 0, 0);
          auto u =
              deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, N, K,
                                                    128>(s.u[stage], 0, 0);
          const uint32_t a_base = a.lo, g_base = g.lo, u_base = u.lo;
#pragma unroll
          for (int inner = 0; inner < K; inner += 16) {
            a.lo = a_base + inner * 2 / 16;
            g.lo = g_base + inner * 2 / 16;
            u.lo = u_base + inner * 2 / 16;
            deep_gemm::ptx::SM100_MMA_F16BF16_SS::fma(a, g, index * 256,
                                                      k || inner, instruction);
            if (gate)
              deep_gemm::ptx::SM100_MMA_F16BF16_SS::fma(
                  a, u, index * 256 + 128, k || inner, instruction);
          }
        }
        __syncwarp();
        if constexpr (Cluster > 1)
          cutlass::arch::umma_arrive_multicast(
              reinterpret_cast<uint64_t *>(&s.empty[stage]),
              (1 << Cluster) - 1);
        else
          cutlass::arch::umma_arrive(
              reinterpret_cast<uint64_t *>(&s.empty[stage]));
        if (k + K == reduction)
          cutlass::arch::umma_arrive(
              reinterpret_cast<uint64_t *>(&s.accum_full[index]));
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    const int local_row = (warp - 4) * 32 + lane;
    for (uint32_t sequence = 0;; ++sequence) {
      const int index = sequence % OutputStages,
                parity = (sequence / OutputStages) & 1;
      s.task_full[index].wait(parity);
      Task task = s.task[index];
      if (task.slot == -2)
        break;
      s.accum_full[index].wait(parity);
      deep_gemm::ptx::tcgen05_after_thread_sync();
      const bool gate = task.stage == GATE;
      const int width = gate ? b.intermediate : b.hidden;
      const int row = task.row + local_row;
      const bool valid_row =
          row < task.rows && local_row < (task.phase == 0 ? 64 : 32);
      size_t base = (expert_base(b, task.slot, task.expert) + row) * width;
      BF16 *output = gate ? b.activation : b.output;
#pragma unroll
      for (int col = 0; col < N; col += 8) {
        uint32_t g[8], u[8];
        cute::SM100_TMEM_LOAD_32dp32b8x::copy(
            index * 256 + col, g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7]);
        if (gate)
          cute::SM100_TMEM_LOAD_32dp32b8x::copy(index * 256 + 128 + col, u[0],
                                                u[1], u[2], u[3], u[4], u[5],
                                                u[6], u[7]);
        cutlass::arch::fence_view_async_tmem_load();
        alignas(16) BF16 converted[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float value = __uint_as_float(g[j]);
          if (gate) {
            // Preserve native BF16 GEMM output rounding without materializing
            // the gate/up tensors in HBM. Fusion must not change this boundary.
            value = __bfloat162float(__float2bfloat16(value));
            float up =
                __bfloat162float(__float2bfloat16(__uint_as_float(u[j])));
            value = value / (1.f + expf(-value)) * up;
          }
          converted[j] = __float2bfloat16(value);
        }
        // H/I and tile columns are 64-element aligned. One 128-bit store
        // avoids eight narrow, strided transactions per epilogue thread.
        if (valid_row && task.column + col < width)
          *reinterpret_cast<uint4 *>(output + base + task.column + col) =
              *reinterpret_cast<const uint4 *>(converted);
      }
      deep_gemm::ptx::tcgen05_before_thread_sync();
      s.accum_empty[index].arrive();
      __threadfence();
      cutlass::arch::NamedBarrier::sync(64, 0);
      if (local_row == 0) {
        auto *state = b.states + task.slot;
        unsigned total = gate ? state->gate_tiles : state->down_tiles;
        if (atomicAdd(&state->completed, 1U) == total * Cluster - 1) {
          state->completed = 0;
          __threadfence();
          if (gate)
            queue_store(&state->queue, queue_value(task.epoch, DOWN));
          else {
            queue_store(&state->queue, queue_value(task.epoch, IDLE));
            b.control[static_cast<size_t>(task.slot) * kControlWords +
                      FINISH_NS] = now_ns();
            __threadfence_system();
            system_store(b.control +
                             static_cast<size_t>(task.slot) * kControlWords +
                             DONE,
                         task.epoch);
          }
        }
      }
      cutlass::arch::NamedBarrier::sync(64, 0);
      s.task_empty[index].arrive();
      s.cluster_empty[index].arrive(0, true);
    }
  }
  __syncthreads();
  if constexpr (Cluster > 1) {
    cute::cluster_arrive();
    cute::cluster_wait();
  }
  if (warp == 0)
    cute::TMEM::Allocator1Sm().free(0, 512);
}
} // namespace expertplex_sm100
