// E-side normal-mode preprocessing/postprocessing templates. Included inside
// expertplex_sm100 after Task/Context and scheduler helper declarations.
#pragma once

__device__ __forceinline__ bool data_stage(int stage) {
  return stage == SCATTER1 || stage == SCATTER2 || stage == GATHER;
}
__device__ __forceinline__ int data_tile_rows(int stage) {
  return stage == SCATTER1 ? 256 : stage == SCATTER2 ? 32 : 4;
}
__device__ __forceinline__ uint4 load_vector_cg(const BF16 *pointer) {
  uint4 result;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w)
               : "l"(pointer)
               : "memory");
  return result;
}

template <int Cluster, bool Fused>
__device__ __forceinline__ void build_row_tiles(const DeviceBuffers &b,
                                                int slot, Context context) {
  auto *state = b.states + slot;
  int rows = 0, activation_rows = 0, offset = 0;
  for (int e = 0; e < b.experts; ++e) {
    int count = uncached_load_int(b.counts + slot * b.experts + e);
    if (count < 0 || count > b.capacity[slot]) {
      set_error(b.control + slot * kControlWords, ROW_OVERFLOW);
      count = 0;
    }
    b.offsets[slot * b.experts + e] = offset;
    offset += (count + 127) / 128 * 128;
    rows += (count + M - 1) / M;
    activation_rows += (count + 31) / 32;
    context.tile_prefix[slot * 2 * b.experts + e] = rows;
    context.tile_prefix[(slot * 2 + 1) * b.experts + e] = activation_rows;
  }
  if (offset > b.storage_rows[slot])
    set_error(b.control + slot * kControlWords, ROW_OVERFLOW);
  state->gate_tiles =
      rows *
      ((operation_columns<Fused>(b, GATE) + N * Cluster - 1) / (N * Cluster));
  state->down_tiles = rows * ((b.hidden + N * Cluster - 1) / (N * Cluster));
  state->activation_tiles =
      activation_rows * ((b.intermediate + N * Cluster - 1) / (N * Cluster));
}

// All loops have a startup-fixed tile/H/K bound, independent of batch length.
// Every epilogue writer fences before the operation completion counter.
template <int Hidden>
__device__ __forceinline__ void
process_data_tile(const DeviceBuffers &b, const Task &task, int thread) {
  const int hidden = Hidden > 0 ? Hidden : b.hidden;
  const int slot = task.slot, k = b.top_k;
  const size_t raw_base = b.raw_base[slot];
  auto *control = b.control + slot * kControlWords;
  if (task.stage == SCATTER1) {
    for (int i = thread; i < 256; i += 128) {
      int route = task.row + i;
      if (route >= task.rows * k)
        continue;
      const size_t index = raw_base * k + route;
      const int id = uncached_load_int(b.raw_ids + index);
      int row = -1;
      if (id < -1 || id >= b.experts * b.expert_world)
        set_error(control, BAD_ROUTE);
      else if (id >= 0 && id % b.expert_world == b.expert_rank) {
        row = atomicAdd(b.counts + slot * b.experts + id / b.expert_world, 1);
        if (row >= b.capacity[slot])
          set_error(control, ROW_OVERFLOW);
      }
      b.raw_map[index] = row;
    }
  } else if (task.stage == SCATTER2) {
    // Stage all 32 routes in warp lanes once. Reloading identical route
    // metadata in every copying thread serializes these memory-bound tiles.
    const int lane = thread & 31;
    const int route = task.row + lane;
    U64 destination = ~0ULL, source = 0;
    if (route < task.rows * k) {
      const size_t index = raw_base * k + route;
      const int row = uncached_load_int(b.raw_map + index);
      if (row >= 0 && row < b.capacity[slot]) {
        const int id = uncached_load_int(b.raw_ids + index);
        destination =
            (expert_base(b, slot, id / b.expert_world) + row) * hidden;
        source = (raw_base + route / k) * hidden;
      }
    }
    for (int i = 0; i < 32; ++i) {
      const U64 dst = __shfl_sync(0xffffffff, destination, i);
      const U64 src = __shfl_sync(0xffffffff, source, i);
      if (dst == ~0ULL)
        continue;
      if constexpr (Hidden > 0 && Hidden % 1024 == 0) {
        // Fixed native route tile: overlap independent H-vector loads before
        // consuming them. Tile geometry and dispatch order are unchanged.
        uint4 values[Hidden / 1024];
#pragma unroll
        for (int segment = 0; segment < Hidden / 1024; ++segment)
          values[segment] =
              load_vector_cg(b.raw_input + src + (thread + segment * 128) * 8);
#pragma unroll
        for (int segment = 0; segment < Hidden / 1024; ++segment)
          reinterpret_cast<uint4 *>(b.input + dst)[thread + segment * 128] =
              values[segment];
      } else {
        for (int h = thread; h < hidden / 8; h += 128)
          reinterpret_cast<uint4 *>(b.input + dst)[h] =
              load_vector_cg(b.raw_input + src + h * 8);
      }
    }
  } else {
    for (int i = 0; i < 4; ++i) {
      const int token = task.row + i;
      if (token >= task.rows)
        break;
      const int lane = thread & 31;
      U64 cached_source = ~0ULL;
      float cached_weight = 0;
      if (k <= 32 && lane < k) {
        const size_t index = (raw_base + token) * k + lane;
        const int row = uncached_load_int(b.raw_map + index);
        if (row >= 0 && row < b.capacity[slot]) {
          const int id = uncached_load_int(b.raw_ids + index);
          cached_source =
              (expert_base(b, slot, id / b.expert_world) + row) * hidden;
          cached_weight = __int_as_float(uncached_load_int(
              reinterpret_cast<const int *>(b.raw_weights + index)));
        }
      }
      for (int offset = 0; offset < hidden; offset += 128 * 8) {
        const int h = offset + thread * 8;
        float sum[8] = {};
        if (k == 8) {
          // Coder top-8: issue independent route loads before weighted sums.
          // Accumulation remains route-ordered FP32, with one BF16 rounding
          // at the native EP gather boundary below.
          uint4 vectors[8];
          float weights[8];
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const U64 source = __shfl_sync(0xffffffff, cached_source, j);
            weights[j] = __shfl_sync(0xffffffff, cached_weight, j);
            vectors[j] = source != ~0ULL && h < hidden
                             ? load_vector_cg(b.output + source + h)
                             : uint4{0, 0, 0, 0};
          }
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const uint32_t words[4] = {vectors[j].x, vectors[j].y, vectors[j].z,
                                       vectors[j].w};
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
              sum[pair * 2] +=
                  weights[j] *
                  __bfloat162float(__ushort_as_bfloat16(words[pair] & 65535));
              sum[pair * 2 + 1] +=
                  weights[j] *
                  __bfloat162float(__ushort_as_bfloat16(words[pair] >> 16));
            }
          }
        } else
          for (int j = 0; j < k; ++j) {
            U64 source;
            float weight;
            if (k <= 32) {
              source = __shfl_sync(0xffffffff, cached_source, j);
              weight = __shfl_sync(0xffffffff, cached_weight, j);
              if (source == ~0ULL)
                continue;
              source += h;
            } else {
              const size_t index = (raw_base + token) * k + j;
              const int row = uncached_load_int(b.raw_map + index);
              if (row < 0 || row >= b.capacity[slot])
                continue;
              const int id = uncached_load_int(b.raw_ids + index);
              weight = __int_as_float(uncached_load_int(
                  reinterpret_cast<const int *>(b.raw_weights + index)));
              source =
                  (expert_base(b, slot, id / b.expert_world) + row) * hidden +
                  h;
            }
            if (h >= hidden)
              continue;
            const uint4 vector = load_vector_cg(b.output + source);
            const uint32_t words[4] = {vector.x, vector.y, vector.z, vector.w};
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
              sum[pair * 2] +=
                  weight *
                  __bfloat162float(__ushort_as_bfloat16(words[pair] & 65535));
              sum[pair * 2 + 1] +=
                  weight *
                  __bfloat162float(__ushort_as_bfloat16(words[pair] >> 16));
            }
          }
        if (h >= hidden)
          continue;
        // Native normal-mode EP gather materializes BF16 before combine.
        alignas(16) BF16 result[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
          result[j] = __float2bfloat16(sum[j]);
        *reinterpret_cast<uint4 *>(b.raw_output + (raw_base + token) * hidden +
                                   h) = *reinterpret_cast<uint4 *>(result);
      }
    }
  }
  __threadfence();
  asm volatile("fence.proxy.async.global;" ::: "memory");
}
