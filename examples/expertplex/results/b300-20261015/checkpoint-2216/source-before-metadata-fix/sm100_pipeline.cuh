// Blackwell BF16 APK development backend.
// Uses the native DeepGEMM SM100 TMA / UMMA primitives and producer, MMA,
// epilogue warp roles. See PAPER_CONFORMANCE.md for outstanding validation.
// Included after the shared IPC protocol definitions in expertplex.cu.
#pragma once

#include <cute/atom/copy_traits_sm100.hpp>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace expertplex_sm100 {
using Barrier = cutlass::arch::ClusterTransactionBarrier;
using Element = cutlass::bfloat16_t;
constexpr int M = 128, N = 128, K = 64, OutputStages = 2;
// Nsight Systems confirms the pinned native BF16 masked and contiguous
// Coder templates both use M=128 (including D128). Keep this startup constant.
constexpr int DecodeM = M;
constexpr int Threads = 128 + M;

struct alignas(128) Maps {
  CUtensorMap input[2];
  CUtensorMap activation[2];
  CUtensorMap output[2];
  CUtensorMap activation_store[2];
  CUtensorMap gateup_store[2];
};
struct alignas(128) WeightMaps {
  CUtensorMap gate, up, down;
};
struct alignas(16) Task {
  int slot, stage, layer, expert, row, column, rows, phase;
  U64 epoch;
};
template <int Cluster, bool Fused> struct Storage {
  static constexpr int Stages = Fused ? 4 : (Cluster == 2 ? 9 : 6);
  using cd_dtype = Element;
  alignas(1024) Element a[Stages][M / Cluster * K];
  alignas(1024) Element g[Stages][N * K];
  alignas(1024) Element u[Fused ? Stages : 1][Fused ? N * K : 1];
  alignas(1024) Element store[Fused ? 2 : 1][Fused ? M * 64 : 1];
  alignas(1024) Element cd[2][Fused ? 1 : 16 * N];
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

// Optional diagnostics are copied only after the executor has stopped.
// Each CTA owns its trace row; no tracing atomics or receiver kernels.
struct TraceRecord {
  U64 begin_ns, finish_ns, ready_seen_ns, epoch;
  U64 sequence, slot, phase, stage;
};
constexpr int QueueShards = 8;
struct alignas(128) QueueHead {
  U64 word;
};
struct Context {
  bool pipelined_stores = false;
  int benchmark_stage =
      0; // Diagnostic single-operation execution; 0 = full MoE.
  int column_group = 0;
  TraceRecord *trace = nullptr;
  int trace_capacity = 0;
  Maps *maps;
  WeightMaps *weights;
  int *policy;
  int *tile_prefix;
  QueueHead *queues;
};

inline Context prepare(OwnedBuffers &owner, const BF16 *w13, const BF16 *w2,
                       int layers, int ctas, int trace_capacity, int cluster,
                       bool fused) {
  auto b = owner.b;
  TORCH_CHECK(b.hidden % K == 0 && b.intermediate % K == 0,
              "SM100 native pipeline requires H and I divisible by 64");
  // Match native DeepGEMM: one tensor map per operand/slot, with layer
  // and expert folded into the outer coordinate. Per-expert maps overflow
  // the descriptor working set and are unnecessary for fixed-stride weights.
  std::vector<Maps> maps(b.slots);
  for (int slot = 0; slot < b.slots; ++slot) {
    for (int phase = 0; phase < 2; ++phase) {
      int tile_m = phase == 0 ? M : DecodeM;
      size_t base = b.base_rows[slot];
      int rows = b.storage_rows[slot];
      encode_map(maps[slot].input[phase], b.input + base * b.hidden, b.hidden,
                 rows, tile_m / cluster);
      encode_map(maps[slot].activation[phase],
                 b.activation + base * b.intermediate, b.intermediate, rows,
                 tile_m / cluster);
      encode_map(maps[slot].activation_store[phase],
                 b.activation + base * b.intermediate, b.intermediate, rows,
                 tile_m);
      encode_map(maps[slot].gateup_store[phase],
                 b.gateup + base * 2 * b.intermediate, 2 * b.intermediate, rows,
                 16);
      encode_map(maps[slot].output[phase], b.output + base * b.hidden, b.hidden,
                 rows, fused ? tile_m : 16);
    }
  }
  std::vector<WeightMaps> weights(1);
  TORCH_CHECK(static_cast<int64_t>(layers) * b.experts * 2 * b.intermediate <=
                      INT_MAX &&
                  static_cast<int64_t>(layers) * b.experts * b.hidden <=
                      INT_MAX,
              "Flattened weights exceed tensor-map coordinate range");
  encode_map(weights[0].gate, const_cast<BF16 *>(w13), b.hidden,
             layers * b.experts * 2 * b.intermediate, N);
  // Gate/up share the same flattened storage and descriptor.
  weights[0].up = weights[0].gate;
  encode_map(weights[0].down, const_cast<BF16 *>(w2), b.intermediate,
             layers * b.experts * b.hidden, N);
  Context result;
  // Native DeepGEMM's get_num_1d_blocks_per_group, with M=N=128 and
  // multicast on A. Convert its CTA columns to our cluster task columns.
  const int group_ctas = 16 + (ctas + 15) / 16 < 8 + (ctas + 7) / 8 ? 16 : 8;
  result.column_group = group_ctas / cluster;
  result.trace_capacity = trace_capacity;
  if (trace_capacity) {
    size_t bytes =
        static_cast<size_t>(ctas) * trace_capacity * sizeof(TraceRecord);
    result.trace = static_cast<TraceRecord *>(owner.allocate(bytes));
    C10_CUDA_CHECK(cudaMemset(result.trace, 0, bytes));
  }
  result.maps = static_cast<Maps *>(owner.allocate(maps.size() * sizeof(Maps)));
  result.weights = static_cast<WeightMaps *>(
      owner.allocate(weights.size() * sizeof(WeightMaps)));
  result.policy = static_cast<int *>(owner.allocate(ctas * sizeof(int)));
  result.tile_prefix =
      static_cast<int *>(owner.allocate(b.slots * 2 * b.experts * sizeof(int)));
  result.queues = static_cast<QueueHead *>(
      owner.allocate(b.slots * QueueShards * sizeof(QueueHead)));
  C10_CUDA_CHECK(
      cudaMemset(result.queues, 0, b.slots * QueueShards * sizeof(QueueHead)));
  C10_CUDA_CHECK(cudaMemcpy(result.maps, maps.data(),
                            maps.size() * sizeof(Maps),
                            cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(result.weights, weights.data(),
                            weights.size() * sizeof(WeightMaps),
                            cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemset(result.policy, 0, ctas * sizeof(int)));
  return result;
}

__device__ __forceinline__ void reset_queues(Context context, int slot,
                                             U64 epoch, Stage stage) {
  // Each shard owns indices shard + k * QueueShards. Keep the generation in
  // every head: an idle claimant from a retired operation cannot reserve a
  // tile in the next operation using stale layer/phase metadata.
  for (int shard = 0; shard < QueueShards; ++shard)
    queue_store(&context.queues[slot * QueueShards + shard].word,
                queue_value(epoch, stage, shard));
}

__device__ __forceinline__ int tile_rows(int phase, int stage) {
  return stage == ACTIVATE ? 32 : (phase == 0 ? M : DecodeM);
}

__device__ __forceinline__ bool active_stage(int stage) {
  return stage == GATE || stage == ACTIVATE || stage == DOWN ||
         stage >= SCATTER1;
}
template <bool Fused>
__device__ __forceinline__ int operation_columns(const DeviceBuffers &b,
                                                 int stage) {
  return stage == DOWN ? b.hidden
                       : b.intermediate * (stage == GATE && !Fused ? 2 : 1);
}
__device__ __forceinline__ unsigned operation_tiles(const SlotState *state,
                                                    int stage) {
  return stage == SCATTER1   ? state->scatter1_tiles
         : stage == SCATTER2 ? state->scatter2_tiles
         : stage == GATHER   ? state->gather_tiles
         : stage == GATE     ? state->gate_tiles
         : stage == ACTIVATE ? state->activation_tiles
                             : state->down_tiles;
}

#include "preprocessing.cuh"

template <int Cluster, bool Fused>
__device__ __forceinline__ void initialize(const DeviceBuffers &b, int slot,
                                           int layers, Context context) {
  auto *c = b.control + static_cast<size_t>(slot) * kControlWords;
  auto *s = b.states + slot;
  U64 q = queue_load(&s->queue);
  if (((q >> 32) & 7) != IDLE)
    return;
  U64 epoch = system_load(c + READY);
  if (epoch <= (q >> 35) ||
      !queue_cas(&s->queue, q, queue_value(epoch, INITIALIZING)))
    return;
  system_store(c + START_NS, now_ns());
  s->layer = static_cast<int>(uncached_load_u64(c + LAYER));
  s->phase = static_cast<int>(uncached_load_u64(c + PHASE));
  if (s->layer < 0 || s->layer >= layers)
    set_error(c, BAD_LAYER);
  if (s->phase < 0 || s->phase > 1)
    set_error(c, BAD_PHASE);
  if (b.prefill_preprocess && b.packed[slot]) {
    const int n = uncached_load_int(b.raw_n + slot);
    if (n < 0 || n > b.capacity[slot])
      set_error(c, ROW_OVERFLOW);
    if (s->phase != 0)
      set_error(c, BAD_PHASE);
    if (system_load(c + ERROR) || n == 0) {
      queue_store(&s->queue, queue_value(epoch, IDLE));
      c[FINISH_NS] = now_ns();
      system_store(c + DONE, epoch);
      return;
    }
    s->scatter1_tiles = (n * b.top_k + 256 * Cluster - 1) / (256 * Cluster);
    s->scatter2_tiles = (n * b.top_k + 32 * Cluster - 1) / (32 * Cluster);
    s->gather_tiles = (n + 4 * Cluster - 1) / (4 * Cluster);
    s->completed = 0;
    __threadfence();
    reset_queues(context, slot, epoch, SCATTER1);
    queue_store(&s->queue, queue_value(epoch, SCATTER1));
    return;
  }
  const int tile_m = s->phase == 0 ? M : DecodeM;
  U64 rows = 0, activation_rows = 0;
  for (int e = 0; e < b.experts; ++e) {
    int count =
        uncached_load_int(b.counts + static_cast<size_t>(slot) * b.experts + e);
    if (count < 0 || count > b.capacity[slot])
      set_error(c, ROW_OVERFLOW);
    else {
      rows += (count + tile_m - 1) / tile_m;
      activation_rows += (count + 31) / 32;
    }
    // Build once at admission, rather than serially loading all expert counts
    // on every tile claim. The operation queue's release publishes these
    // cumulative row-tile counts together with its fixed generation.
    context.tile_prefix[(slot * 2) * b.experts + e] = rows;
    context.tile_prefix[(slot * 2 + 1) * b.experts + e] = activation_rows;
  }
  U64 gate = rows * ((operation_columns<Fused>(b, GATE) + N * Cluster - 1) /
                     (N * Cluster));
  U64 down = rows * ((b.hidden + N * Cluster - 1) / (N * Cluster));
  if (gate > UINT_MAX - 65536ULL || down > UINT_MAX - 65536ULL)
    set_error(c, ROW_OVERFLOW);
  s->gate_tiles = gate;
  s->down_tiles = down;
  s->activation_tiles =
      activation_rows * ((b.intermediate + N * Cluster - 1) / (N * Cluster));
  s->completed = 0;
  __threadfence();
  if (system_load(c + ERROR) || gate == 0) {
    queue_store(&s->queue, queue_value(epoch, IDLE));
    c[FINISH_NS] = now_ns();
    system_store(c + DONE, epoch);
  } else {
    Stage first = context.benchmark_stage == DOWN ? DOWN : GATE;
    reset_queues(context, slot, epoch, first);
    queue_store(&s->queue, queue_value(epoch, first));
  }
}

// Only CTA0 reads attention readiness words. Other CTAs consume device-scope
// policies; no SM is reserved for a separate receiver/polling kernel.
template <int Cluster, bool Fused>
__device__ __forceinline__ void update_policy(const DeviceBuffers &b,
                                              Context context, int layers,
                                              int decode_budget, float expected,
                                              int cap, int &previous_budget) {
  for (int s = 0; s < b.slots; ++s)
    initialize<Cluster, Fused>(b, s, layers, context);
  U64 footprint = 0;
  for (int i = 0; i < b.slots; ++i) {
    auto *s = b.states + i;
    auto stage = (queue_load(&s->queue) >> 32) & 7;
    if (active_stage(stage) && s->phase == 1)
      // Equation 3 measures the routed GEMM row-tile footprint. The finer
      // activation scheduling unit must not inflate the same MoE batch's
      // estimated load merely because it is currently at the activation stage.
      footprint += s->gate_tiles /
                   ((operation_columns<Fused>(b, GATE) + N * Cluster - 1) /
                    (N * Cluster));
  }
  int budget =
      footprint
          ? min(cap, max(1, static_cast<int>(ceilf(
                                decode_budget * static_cast<float>(footprint) /
                                expected))))
          : 0;
  budget = min(cap, (budget + Cluster - 1) / Cluster * Cluster);
  // The device policy describes a phase preference, not a task generation.
  // Re-publish only when that preference changes; readiness/operation queues
  // are independently acquired on every check epoch.
  if (budget == previous_budget)
    return;
  previous_budget = budget;
  for (int i = 0; i < gridDim.x / Cluster; ++i)
    cuda::atomic_ref<int, cuda::thread_scope_device>(context.policy[i])
        .store(i * Cluster < budget, cuda::memory_order_release);
}

template <int Cluster, bool Fused>
__device__ __forceinline__ bool claim(const DeviceBuffers &b, Context context,
                                      int phase, int cursor, Task &task) {
  for (int i = 0; i < b.slots; ++i) {
    int slot = (cursor + i) % b.slots;
    auto *state = b.states + slot;
    U64 q = queue_load(&state->queue);
    int stage = (q >> 32) & 7;
    if (!active_stage(stage) || state->phase != phase)
      continue;
    const U64 generation = q & ~static_cast<U64>(UINT_MAX);
    unsigned total = operation_tiles(state, stage);
    unsigned index = 0;
    bool reserved = false;
    for (int probe = 0; probe < QueueShards && !reserved; ++probe) {
      int shard = (blockIdx.x / Cluster + probe) % QueueShards;
      auto &head = context.queues[slot * QueueShards + shard].word;
      U64 ticket = queue_load(&head);
      for (int attempt = 0; attempt < 4; ++attempt) {
        index = static_cast<unsigned>(ticket);
        if ((ticket & ~static_cast<U64>(UINT_MAX)) != generation ||
            (queue_load(&state->queue) & ~static_cast<U64>(UINT_MAX)) !=
                generation ||
            index >= total)
          break;
        if (cuda::atomic_ref<U64, cuda::thread_scope_device>(head)
                .compare_exchange_strong(ticket, ticket + QueueShards,
                                         cuda::memory_order_acquire,
                                         cuda::memory_order_relaxed)) {
          reserved = true;
          break;
        }
      }
    }
    if (!reserved ||
        (queue_load(&state->queue) & ~static_cast<U64>(UINT_MAX)) != generation)
      continue;
    if (data_stage(stage)) {
      task = {slot,
              stage,
              state->layer,
              0,
              static_cast<int>(index) * data_tile_rows(stage) * Cluster,
              0,
              uncached_load_int(b.raw_n + slot),
              phase,
              q >> 35};
      return true;
    }
    const int tile_m = tile_rows(phase, stage);
    const int tile_n =
        (operation_columns<Fused>(b, stage) + N * Cluster - 1) / (N * Cluster);
    unsigned row_tile = index / tile_n;
    unsigned column_tile = index % tile_n;
    const int *prefix =
        context.tile_prefix + (slot * 2 + (stage == ACTIVATE)) * b.experts;
    if (phase == 0 && stage != ACTIVATE) {
      // Match native contiguous GEMM's L2 traversal: a small N-column
      // group over all row tiles before moving to the next N group. Row-major
      // traversal rereads large expert weights after they leave L2.
      const unsigned rows = prefix[b.experts - 1];
      const unsigned group_size = rows * context.column_group;
      const unsigned first_column = index / group_size * context.column_group;
      const unsigned columns = min(static_cast<unsigned>(context.column_group),
                                   tile_n - first_column);
      const unsigned within = index % group_size;
      row_tile = within / columns;
      column_tile = first_column + within % columns;
    }
    int lo = 0, hi = b.experts;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (static_cast<unsigned>(prefix[mid]) <= row_tile)
        lo = mid + 1;
      else
        hi = mid;
    }
    if (lo >= b.experts)
      asm volatile("trap;");
    const unsigned previous = lo ? prefix[lo - 1] : 0;
    unsigned local_row = row_tile - previous;
    if (phase == 1 && stage != ACTIVATE) {
      // Native masked GEMM applies the same swizzle independently to each
      // expert's row range rather than the concatenated prefill range.
      const unsigned rows = prefix[lo] - previous;
      const unsigned within_expert = index - previous * tile_n;
      const unsigned group_size = rows * context.column_group;
      const unsigned first_column =
          within_expert / group_size * context.column_group;
      const unsigned columns = min(static_cast<unsigned>(context.column_group),
                                   tile_n - first_column);
      const unsigned within = within_expert % group_size;
      local_row = within / columns;
      column_tile = first_column + within % columns;
    }
    task = {slot,
            stage,
            state->layer,
            lo,
            static_cast<int>(local_row) * tile_m,
            static_cast<int>(column_tile) * N * Cluster,
            b.counts[static_cast<size_t>(slot) * b.experts + lo],
            phase,
            q >> 35};
    return true;
  }
  return false;
}

// Commit only after the tile's asynchronous global stores are complete.
// Called by the issuing epilogue lane; mailbox/ring release is collective.
template <int Cluster, bool Fused>
__device__ __forceinline__ void retire_task(const DeviceBuffers &b,
                                            Context context, const Task &task,
                                            uint32_t sequence) {
  auto *state = b.states + task.slot;
  unsigned total = operation_tiles(state, task.stage);
  if (atomicAdd(&state->completed, 1U) == total * Cluster - 1) {
    state->completed = 0;
    __threadfence();
    Stage next = IDLE;
    if (b.prefill_preprocess && b.packed[task.slot]) {
      if (task.stage == SCATTER1) {
        build_row_tiles<Cluster, Fused>(b, task.slot, context);
        next = SCATTER2;
      } else if (task.stage == SCATTER2) {
        next = state->gate_tiles ? GATE : GATHER;
      } else if (task.stage == GATE)
        next = ACTIVATE;
      else if (task.stage == ACTIVATE)
        next = DOWN;
      else if (task.stage == DOWN)
        next = GATHER;

    } else if (task.stage != DOWN && context.benchmark_stage == 0) {
      next = task.stage == GATE && !Fused ? ACTIVATE : DOWN;
    }
    if (system_load(b.control + task.slot * kControlWords + ERROR))
      next = IDLE;
    if (next != IDLE) {
      __threadfence();
      reset_queues(context, task.slot, task.epoch, next);
      queue_store(&state->queue, queue_value(task.epoch, next));
    } else {
      queue_store(&state->queue, queue_value(task.epoch, IDLE));
      b.control[static_cast<size_t>(task.slot) * kControlWords + FINISH_NS] =
          now_ns();
      __threadfence_system();
      system_store(b.control + static_cast<size_t>(task.slot) * kControlWords +
                       DONE,
                   task.epoch);
    }
  }
  if (context.trace && sequence < context.trace_capacity)
    context
        .trace[static_cast<size_t>(blockIdx.x) * context.trace_capacity +
               sequence]
        .finish_ns = now_ns();
}

template <int Cluster, int Hidden = 0, int Intermediate = 0, bool Fused = false>
__global__ __launch_bounds__(Threads, 1) void executor(
    const __grid_constant__ DeviceBuffers b, Context context, int layers,
    int decode_budget, float expected, int cap) {
  const int hidden = Hidden > 0 ? Hidden : b.hidden;
  const int intermediate = Intermediate > 0 ? Intermediate : b.intermediate;
  extern __shared__ __align__(1024) unsigned char workspace[];
  auto &s = *reinterpret_cast<Storage<Cluster, Fused> *>(workspace);
  constexpr int Stages = Storage<Cluster, Fused>::Stages;
  constexpr int TmemColumns = Fused ? 512 : OutputStages * M;
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int rank = blockIdx.x % Cluster;
  if (threadIdx.x == 0) {
    // Native DeepGEMM warms the fixed descriptors before its first tile.
    // These addresses never change while the persistent executor is alive.
    cute::prefetch_tma_descriptor(&context.weights[0].gate);
    cute::prefetch_tma_descriptor(&context.weights[0].up);
    cute::prefetch_tma_descriptor(&context.weights[0].down);
    for (int slot = 0; slot < b.slots; ++slot)
      for (int phase = 0; phase < 2; ++phase) {
        cute::prefetch_tma_descriptor(&context.maps[slot].input[phase]);
        cute::prefetch_tma_descriptor(&context.maps[slot].activation[phase]);
        cute::prefetch_tma_descriptor(&context.maps[slot].gateup_store[phase]);
        cute::prefetch_tma_descriptor(&context.maps[slot].output[phase]);
        if constexpr (Fused)
          cute::prefetch_tma_descriptor(
              &context.maps[slot].activation_store[phase]);
      }
  }
  if (threadIdx.x == 32) {
    for (int i = 0; i < Stages; ++i) {
      s.full[i].init(Cluster);
      s.empty[i].init(1);
    }
    for (int i = 0; i < OutputStages; ++i) {
      s.task_full[i].init(1);
      s.task_empty[i].init(M);
      s.accum_full[i].init(1);
      s.accum_empty[i].init(M * Cluster);
      s.decision_full[i].init(1);
      s.cluster_empty[i].init(M * Cluster);
    }
    cutlass::arch::fence_barrier_init();
  }
  using Allocator = cute::conditional_t<Cluster == 1, cute::TMEM::Allocator1Sm,
                                        cute::TMEM::Allocator2Sm>;
  if constexpr (Cluster > 1) {
    cute::cluster_arrive();
    cute::cluster_wait();
  }
  if (warp == 2)
    Allocator().allocate(TmemColumns, &s.tmem);
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
    int previous_budget = -1;
    bool previous_was_flush = false;
    Task deferred{};
    deferred.slot = -1;
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
        if (deferred.slot >= 0) {
          task = deferred;
          deferred.slot = -1;
        }
        while (task.slot == -1) {
          // Read-only polling must not serialize every idle cluster through
          // a read-modify-write on the same L2 atomic location.
          if (cuda::atomic_ref<int, cuda::thread_scope_device>(*b.stop).load(
                  cuda::memory_order_acquire)) {
            task.slot = -2;
            break;
          }
          if (blockIdx.x == 0)
            update_policy<Cluster, Fused>(b, context, layers, decode_budget,
                                          expected, cap, previous_budget);
          int preferred = cuda::atomic_ref<int, cuda::thread_scope_device>(
                              context.policy[blockIdx.x / Cluster])
                              .load(cuda::memory_order_acquire);
          if (!claim<Cluster, Fused>(b, context, preferred, cursor, task))
            claim<Cluster, Fused>(b, context, 1 - preferred, cursor, task);
          if (task.slot < 0) {
            // A final tile cannot wait for a successor that does not exist.
            // A control-only flush releases its pending store/ring ownership.
            if (context.pipelined_stores && sequence && !previous_was_flush) {
              task.slot = -3;
              task.stage = IDLE;
              break;
            }
            __nanosleep(64);
          }
        }
        // Drain at a phase switch only, not after every tile. No member can
        // independently claim an old-phase tile after this shared decision.
        if (context.pipelined_stores && sequence && task.slot >= 0 &&
            task.phase != previous_phase && !previous_was_flush) {
          deferred = task;
          task.slot = -3;
          task.stage = IDLE;
        }
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
      if (data_stage(task.stage))
        task.row += rank * data_tile_rows(task.stage);
      else
        task.column += rank * N;
      if (task.slot >= 0 && context.trace &&
          sequence < context.trace_capacity) {
        auto &record = context.trace[static_cast<size_t>(blockIdx.x) *
                                         context.trace_capacity +
                                     sequence];
        record = {now_ns(),
                  0,
                  b.control[static_cast<size_t>(task.slot) * kControlWords +
                            START_NS],
                  task.epoch,
                  sequence,
                  static_cast<U64>(task.slot),
                  static_cast<U64>(task.phase),
                  static_cast<U64>(task.stage)};
      }
      s.task[index] = task;
      s.task_full[index].arrive();
      if (task.slot == -2)
        break;
      previous_was_flush = task.slot == -3;
      if (task.slot == -3)
        continue;
      previous_phase = task.phase;
      cursor = (task.slot + 1) % b.slots;
      if (task.stage == ACTIVATE || data_stage(task.stage))
        continue;
      const auto &maps = context.maps[task.slot];
      const auto &weights = context.weights[0];
      const bool gate = task.stage == GATE;
      const int reduction = gate ? hidden : intermediate;
      const int tile_m = tile_rows(task.phase, task.stage);
      const int map_row =
          task.row + static_cast<int>(expert_base(b, task.slot, task.expert) -
                                      b.base_rows[task.slot]);
      for (int k = 0; k < reduction; k += K, advance()) {
        s.empty[stage].wait(phase ^ 1);
        auto *a = gate ? &maps.input[task.phase] : &maps.activation[task.phase];
        auto *g = gate ? &weights.gate : &weights.down;
        const int weight_row = (task.layer * b.experts + task.expert) *
                                   (gate ? 2 * intermediate : hidden) +
                               task.column;
        // Swap A/B: weight columns are the MMA M dimension and tokens are
        // N. Each CTA owns N weight columns and half of the activation tile.
        // Native 2-SM TMA completion signals the leader barrier only.
        if (task.phase == 0)
          deep_gemm::tma::copy<K, M / Cluster, 128, Element>(
              a, &s.full[stage], s.a[stage], k, map_row + rank * (M / Cluster),
              Cluster);
        else
          deep_gemm::tma::copy<K, DecodeM / Cluster, 128, Element>(
              a, &s.full[stage], s.a[stage], k,
              map_row + rank * (DecodeM / Cluster), Cluster);
        deep_gemm::tma::copy<K, N, 128, Element>(g, &s.full[stage], s.g[stage],
                                                 k, weight_row, Cluster);
        if (Fused && gate)
          deep_gemm::tma::copy<K, N, 128, Element>(
              &weights.up, &s.full[stage], s.u[stage], k,
              weight_row + intermediate, Cluster);
        if (rank == 0)
          s.full[stage].arrive_and_expect_tx(
              (tile_m / Cluster + N * (Fused && gate ? 2 : 1)) * K * 2 *
              Cluster);
        else
          s.full[stage].arrive(0u);
      }
    }
  } else if (warp == 1 && rank == 0) {
    auto prefill_descriptor =
        cute::UMMA::make_instr_desc<Element, Element, float, N * Cluster, M,
                                    cute::UMMA::Major::K,
                                    cute::UMMA::Major::K>();
    auto decode_descriptor =
        cute::UMMA::make_instr_desc<Element, Element, float, N * Cluster,
                                    DecodeM, cute::UMMA::Major::K,
                                    cute::UMMA::Major::K>();
    using Mma =
        cute::conditional_t<Cluster == 1, deep_gemm::ptx::SM100_MMA_F16BF16_SS,
                            deep_gemm::ptx::SM100_MMA_F16BF16_2x1SM_SS>;
    // Preserve the native MMA issue pipeline: stage descriptors live in warp
    // lanes and are selected with a shuffle. Rebuilding a shared-memory
    // descriptor in the issuing lane on every K step lengthens that pipeline.
    auto a =
        deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, M / Cluster,
                                              K, 128>(s.a[0], 0, 0);
    auto g =
        deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, N, K, 128>(
            s.g[0], 0, 0);
    auto u =
        deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, N, K, 128>(
            Fused ? s.u[0] : s.g[0], 0, 0);
    const uint32_t a_stage_lo = a.lo + lane * sizeof(s.a[0]) / 16;
    const uint32_t g_stage_lo = g.lo + lane * sizeof(s.g[0]) / 16;
    const uint32_t u_stage_lo =
        u.lo + lane * (Fused ? sizeof(s.u[0]) : sizeof(s.g[0])) / 16;
    for (uint32_t sequence = 0;; ++sequence) {
      const int index = sequence % OutputStages,
                parity = (sequence / OutputStages) & 1;
      s.task_full[index].wait(parity);
      Task task = s.task[index];
      if (task.slot == -2)
        break;
      s.accum_empty[index].wait(parity ^ 1);
      deep_gemm::ptx::tcgen05_after_thread_sync();
      if (task.slot == -3 || task.stage == ACTIVATE || data_stage(task.stage)) {
        if (lane == 0)
          for (int member = 0; member < Cluster; ++member)
            s.accum_full[index].arrive(member, true);
        continue;
      }
      const bool gate = task.stage == GATE;
      const auto instruction = cute::UMMA::make_runtime_instr_desc(
          task.phase == 0 ? prefill_descriptor : decode_descriptor);
      const int reduction = gate ? hidden : intermediate;
      for (int k = 0; k < reduction; k += K, advance()) {
        s.full[stage].wait(phase);
        deep_gemm::ptx::tcgen05_after_thread_sync();
        const uint32_t a_base = __shfl_sync(0xffffffff, a_stage_lo, stage);
        const uint32_t g_base = __shfl_sync(0xffffffff, g_stage_lo, stage);
        const uint32_t u_base = __shfl_sync(0xffffffff, u_stage_lo, stage);
        if (lane == 0) {
#pragma unroll
          for (int inner = 0; inner < K; inner += 16) {
            a.lo = a_base + inner * 2 / 16;
            g.lo = g_base + inner * 2 / 16;
            u.lo = u_base + inner * 2 / 16;
            Mma::fma(g, a, index * (Fused ? 256 : 128), k || inner,
                     instruction);
            if (Fused && gate)
              Mma::fma(u, a, index * 256 + 128, k || inner, instruction);
          }
        }
        __syncwarp();
        if constexpr (Cluster > 1)
          cutlass::arch::umma_arrive_multicast_2x1SM(
              reinterpret_cast<uint64_t *>(&s.empty[stage]),
              (1 << Cluster) - 1);
        else
          cutlass::arch::umma_arrive(
              reinterpret_cast<uint64_t *>(&s.empty[stage]));
        if (k + K == reduction) {
          if constexpr (Cluster > 1)
            cutlass::arch::umma_arrive_multicast_2x1SM(
                reinterpret_cast<uint64_t *>(&s.accum_full[index]),
                (1 << Cluster) - 1);
          else
            cutlass::arch::umma_arrive(
                reinterpret_cast<uint64_t *>(&s.accum_full[index]));
        }
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    const int local_row = (warp - 4) * 32 + lane;
    uint32_t tma_store_stage = 0;
    Task pending{};
    pending.slot = -1;
    uint32_t pending_sequence = 0;
    auto release_task = [&](const Task &completed,
                            uint32_t completed_sequence) {
      if (local_row == 0 && completed.slot >= 0)
        retire_task<Cluster, Fused>(b, context, completed, completed_sequence);
      cutlass::arch::NamedBarrier::sync(M, 0);
      s.task_empty[completed_sequence % OutputStages].arrive();
      s.cluster_empty[completed_sequence % OutputStages].arrive(0, true);
    };
    auto drain_pending = [&]() {
      // Native epilogue elects a deterministic issuing lane from the first
      // full warp. Wait in that entire warp, without assuming the lane is 0.
      if (local_row < 32) {
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
      }
      cutlass::arch::NamedBarrier::sync(M, 0);
      if (pending.slot >= 0) {
        release_task(pending, pending_sequence);
        pending.slot = -1;
      }
    };
    for (uint32_t sequence = 0;; ++sequence) {
      const int index = sequence % OutputStages,
                parity = (sequence / OutputStages) & 1;
      s.task_full[index].wait(parity);
      Task task = s.task[index];
      if (task.slot == -2) {
        drain_pending();
        break;
      }
      s.accum_full[index].wait(parity);
      if (task.slot == -3) {
        drain_pending();
        s.accum_empty[index].arrive(0u);
        release_task(task, sequence);
        continue;
      }
      deep_gemm::ptx::tcgen05_after_thread_sync();
      const bool gate = task.stage == GATE;
      const int tile_m = tile_rows(task.phase, task.stage);
      const auto &maps = context.maps[task.slot];
      const int map_row =
          task.row + static_cast<int>(expert_base(b, task.slot, task.expert) -
                                      b.base_rows[task.slot]);
      if (data_stage(task.stage)) {
        process_data_tile<Hidden>(b, task, local_row);
      } else if (task.stage == ACTIVATE) {
        const size_t base = expert_base(b, task.slot, task.expert);
        const int column = task.column + local_row;
        if (column < intermediate) {
          // Four rows in flight, matching the native activation software
          // pipeline. Each CTA covers 4096 BF16 elements, independent of the
          // total routed batch; no operation-length loop is introduced.
          for (int token = 0; token < 32; token += 4) {
            float gate_values[4], up_values[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              size_t offset =
                  (base + task.row + token + j) * (2 * intermediate) + column;
              bool valid = task.row + token + j < task.rows;
              gate_values[j] = valid ? __bfloat162float(b.gateup[offset]) : 0.f;
              up_values[j] =
                  valid ? __bfloat162float(b.gateup[offset + intermediate])
                        : 0.f;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              float gate_value = gate_values[j];
              BF16 value = __float2bfloat16(
                  gate_value / (1.f + expf(-gate_value)) * up_values[j]);
              b.activation[(base + task.row + token + j) * intermediate +
                           column] = value;
            }
          }
        }
        __threadfence();
        asm volatile("fence.proxy.async.global;" ::: "memory");
      } else if constexpr (!Fused) {
        auto *map =
            gate ? &maps.gateup_store[task.phase] : &maps.output[task.phase];
        deep_gemm::epilogue::sm100_store_cd_swap_ab<
            M, N, 16, N, 128, 2, M, 0, deep_gemm::GemmType::MGroupedContiguous,
            false>(s, tma_store_stage, index * 128, map_row, task.column, 0,
                   false, tile_m, warp - 4, lane,
                   deep_gemm::epilogue::transform::EpilogueIdentity{}, false,
                   &s.accum_empty[index], &s.accum_empty[index],
                   *reinterpret_cast<const cute::TmaDescriptor *>(map));
      } else {
#pragma unroll
        for (int col = 0; col < tile_m; col += 8) {
          uint32_t g[8], u[8];
          cute::SM100_TMEM_LOAD_32dp32b8x::copy(
              index * (Fused ? 256 : 128) + col, g[0], g[1], g[2], g[3], g[4],
              g[5], g[6], g[7]);
          if (Fused && gate)
            cute::SM100_TMEM_LOAD_32dp32b8x::copy(index * 256 + 128 + col, u[0],
                                                  u[1], u[2], u[3], u[4], u[5],
                                                  u[6], u[7]);
          cutlass::arch::fence_view_async_tmem_load();
          alignas(16) BF16 converted[8];
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            float value = __uint_as_float(g[j]);
            if (Fused && gate) {
              // Preserve native BF16 GEMM output rounding without materializing
              // the gate/up tensors in HBM. Fusion must not change this
              // boundary.
              value = __bfloat162float(__float2bfloat16(value));
              float up =
                  __bfloat162float(__float2bfloat16(__uint_as_float(u[j])));
              value = value / (1.f + expf(-value)) * up;
            }
            converted[j] = __float2bfloat16(value);
          }
          // Transpose weight-row/token-column accumulators into the native
          // token-row/weight-column swizzled output layout. All CTAs use the
          // same token rows and write their own 128 output columns.
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const int token_row = col + j;
            const int store_index = local_row / 64;
            const int column = (local_row % 64) ^ ((token_row & 7) * 8);
            s.store[store_index][token_row * 64 + column] =
                task.row + token_row < task.rows ? Element(converted[j])
                                                 : Element(0.f);
          }
        }
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(M, 0);
        if (local_row == 0) {
          auto *map = gate ? (Fused ? &maps.activation_store[task.phase]
                                    : &maps.gateup_store[task.phase])
                           : &maps.output[task.phase];
#pragma unroll
          for (int part = 0; part < 2; ++part) {
            cute::SM90_TMA_STORE_2D::copy(map, s.store[part],
                                          task.column + part * 64, map_row);
            cute::tma_store_arrive();
          }
        }
      }
      deep_gemm::ptx::tcgen05_before_thread_sync();
      if (Fused || task.stage == ACTIVATE || data_stage(task.stage))
        s.accum_empty[index].arrive(0u);
      if (context.pipelined_stores && !Fused && task.stage != ACTIVATE &&
          !data_stage(task.stage)) {
        if (pending.slot >= 0) {
          if (local_row < 32) {
            // Native epilogue commits eight store groups per M128 tile.
            // Retain this tile's final two groups while retiring all older
            // writes. Do not use .read: DONE requires global completion.
            asm volatile("cp.async.bulk.wait_group 2;" ::: "memory");
            asm volatile("fence.proxy.async.global;" ::: "memory");
            __threadfence();
          }
          cutlass::arch::NamedBarrier::sync(M, 0);
          release_task(pending, pending_sequence);
        }
        pending = task;
        pending_sequence = sequence;
      } else {
        drain_pending();
        release_task(task, sequence);
      }
    }
  }
  __syncthreads();
  if constexpr (Cluster > 1) {
    cute::cluster_arrive();
    cute::cluster_wait();
  }
  if (warp == 0)
    Allocator().free(0, TmemColumns);
}
} // namespace expertplex_sm100
