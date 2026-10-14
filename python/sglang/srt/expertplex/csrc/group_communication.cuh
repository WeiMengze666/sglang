// Attention-initiated transfers for an entire expert group. Templates and
// imported pointers are fixed before graph capture. No E-side receive kernel.
#pragma once

struct GroupBuffers {
  DeviceBuffers local[8], peer[8];
  int *status[8];
  int world, slot;
};

__global__ void group_clear(const GroupBuffers *g, int layer, int phase) {
  clear_slot_body(g->local[blockIdx.x], g->slot, 0, layer, phase);
}
template <typename Id>
__global__ void group_count(const GroupBuffers *g, const Id *ids, int *row_map,
                            int routes) {
  count_routes_body(g->local[blockIdx.y], g->slot, ids, row_map, routes,
                    blockIdx.y, g->world);
}
__global__ void group_pack(const GroupBuffers *g) {
  pack_offsets_body(g->local[blockIdx.x], g->slot);
}
template <typename Id>
__global__ void group_dispatch_rows(const GroupBuffers *g, const BF16 *x,
                                    const Id *ids, int *rows, int n, int k) {
  const Id expert = ids[blockIdx.x];
  if (expert == -1) {
    if (threadIdx.x == 0)
      rows[blockIdx.x] = -1;
    return;
  }
  if (expert < 0 || expert >= g->local[0].experts * g->world) {
    if (threadIdx.x == 0)
      for (int rank = 0; rank < g->world; ++rank)
        set_error(g->peer[rank].control + g->slot * kControlWords, BAD_ROUTE);
    return;
  }
  const int rank = expert % g->world;
  dispatch_rows_body(g->local[rank], g->slot, x, ids, rows, n, k, rank,
                     g->world);
}
// Normal-mode dispatch lands unique token rows and routing metadata in the
// fixed E-side receive buffer. The persistent executor performs the local
// scatter/gather; no separate receiver kernel or reserved expert SM is used.
template <typename Id>
__global__ void group_dispatch_raw(const GroupBuffers *g, const BF16 *input,
                                   const Id *ids, const float *weights, int n) {
  const int rank = blockIdx.y, token = blockIdx.x;
  const auto &b = g->peer[rank];
  const int slot = g->slot;
  if (system_load(b.control + slot * kControlWords + ERROR))
    return;
  const size_t row = b.raw_base[slot] + token;
  bool needed = false;
  for (int j = 0; j < b.top_k; ++j) {
    const Id id = ids[token * b.top_k + j];
    if (id < -1 || id >= static_cast<Id>(b.experts) * g->world) {
      if (threadIdx.x == 0)
        set_error(b.control + slot * kControlWords, BAD_ROUTE);
      return;
    }
    needed |= id >= 0 && id % g->world == rank;
    if (threadIdx.x == 0) {
      b.raw_ids[row * b.top_k + j] = static_cast<int>(id);
      b.raw_weights[row * b.top_k + j] = weights[token * b.top_k + j];
    }
  }
  if (needed) {
    auto *dst = reinterpret_cast<uint4 *>(b.raw_input + row * b.hidden);
    auto *src = reinterpret_cast<const uint4 *>(
        input + static_cast<size_t>(token) * b.hidden);
    for (int h = threadIdx.x; h < b.hidden / 8; h += blockDim.x)
      dst[h] = src[h];
  }
  __threadfence_system();
}
__global__ void group_raw_size(const GroupBuffers *g, int n) {
  const auto &b = g->peer[blockIdx.x];
  b.raw_n[g->slot] = n;
  __threadfence_system();
}
__global__ void group_combine_raw(const GroupBuffers *g, float *output, int n) {
  const int hidden = g->peer[0].hidden;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < static_cast<size_t>(n) * hidden;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    float value = 0;
    for (int rank = 0; rank < g->world; ++rank) {
      if (*g->status[rank] != OK)
        continue;
      const auto &b = g->peer[rank];
      value += __bfloat162float(
          uncached_load(b.raw_output + b.raw_base[g->slot] * hidden + i));
    }
    output[i] = value;
  }
}
__global__ void group_publish(const GroupBuffers *g, bool publish) {
  const int rank = blockIdx.x;
  push_metadata_body(g->peer[rank], g->local[rank], g->slot);
  // All metadata writers fence their peer writes before the leader publishes
  // READY. Payload writers completed in the preceding stream-ordered kernel.
  __syncthreads();
  if (publish && threadIdx.x == 0) {
    auto *c = g->peer[rank].control + g->slot * kControlWords;
    if (!system_load(c + ERROR))
      system_store(c + READY, uncached_load_u64(c + PENDING));
  }
}
__global__ void group_wait(const GroupBuffers *g, U64 timeout_ns) {
  // Exactly one attention-side thread on COMM; no reserved expert SM.
  const U64 start = now_ns();
  for (int rank = 0; rank < g->world; ++rank) {
    const U64 elapsed = now_ns() - start;
    wait_slot_body(g->peer[rank], g->slot, 0,
                   elapsed < timeout_ns ? timeout_ns - elapsed : 0,
                   g->status[rank]);
  }
}
template <typename Id>
__global__ void group_combine_rows(const GroupBuffers *g, float *output,
                                   const Id *ids, const float *weights,
                                   const int *rows, int n, int k) {
  // Preserve the reference rank-major FP32 accumulation order exactly.
  for (int rank = 0; rank < g->world; ++rank)
    combine_rows_body(g->local[rank], g->slot, output, ids, weights, rows, n, k,
                      rank, g->world, g->status[rank]);
}

class RemoteGroup {
public:
  std::vector<std::shared_ptr<RemoteBuffers>> peers;
  GroupBuffers *gpu = nullptr;
  int device, slot, world;
  bool packed, preprocess;

  explicit RemoteGroup(std::vector<std::shared_ptr<RemoteBuffers>> mappings)
      : peers(std::move(mappings)) {
    TORCH_CHECK(!peers.empty() && peers.size() <= 8, "Invalid expert group");
    const auto &first = *peers[0];
    TORCH_CHECK(!first.closed, "Closed expert peer");
    device = first.device;
    slot = first.slot;
    world = peers.size();
    packed = first.b.packed[slot];
    preprocess = packed && first.b.prefill_preprocess;
    GroupBuffers descriptor{};
    descriptor.world = world;
    descriptor.slot = slot;
    for (int rank = 0; rank < world; ++rank) {
      auto &peer = *peers[rank];
      TORCH_CHECK(!peer.closed && peer.device == device && peer.slot == slot &&
                      peer.b.experts == first.b.experts &&
                      peer.b.hidden == first.b.hidden &&
                      peer.b.capacity[slot] == first.b.capacity[slot] &&
                      peer.b.packed[slot] == packed,
                  "Expert group geometry differs");
      TORCH_CHECK(peer.b.prefill_preprocess == first.b.prefill_preprocess &&
                      (!preprocess || (peer.b.expert_rank == rank &&
                                       peer.b.expert_world == world &&
                                       peer.b.top_k == first.b.top_k)),
                  "Preprocessing expert group rank mismatch");
      descriptor.local[rank] = peer.attention_metadata();
      descriptor.peer[rank] = peer.b;
      descriptor.status[rank] = peer.status;
    }
    c10::cuda::CUDAGuard guard(device);
    C10_CUDA_CHECK(cudaMalloc(&gpu, sizeof(GroupBuffers)));
    try {
      C10_CUDA_CHECK(cudaMemcpy(gpu, &descriptor, sizeof(descriptor),
                                cudaMemcpyHostToDevice));
    } catch (...) {
      cudaFree(gpu);
      gpu = nullptr;
      throw;
    }
  }
  ~RemoteGroup() {
    if (gpu) {
      int old = 0;
      cudaGetDevice(&old);
      cudaSetDevice(device);
      cudaFree(gpu);
      cudaSetDevice(old);
    }
  }
  void check_open() const {
    for (const auto &peer : peers)
      TORCH_CHECK(!peer->closed, "Expert group contains a closed IPC mapping");
  }
  void dispatch(torch::Tensor hidden, torch::Tensor ids, torch::Tensor rows,
                int layer, int phase, std::optional<torch::Tensor> weights,
                bool publish) {
    check_open();
    auto &first = *peers[0];
    check_tensor(hidden, first, at::kBFloat16, 2, "hidden");
    check_routes(ids, rows, first, 0, world);
    TORCH_CHECK(hidden.size(0) == ids.size(0) &&
                    hidden.size(1) == first.b.hidden && layer >= 0 &&
                    (phase == 0 || phase == 1),
                "Invalid group dispatch");
    c10::cuda::CUDAGuard guard(device);
    auto stream = at::cuda::getCurrentCUDAStream(device);
    if (preprocess) {
      TORCH_CHECK(ids.size(0) <= first.b.capacity[slot] && phase == 0 &&
                      ids.size(1) == first.b.top_k && weights.has_value(),
                  "Normal preprocessing requires prefill routes and weights");
      check_tensor(*weights, first, at::kFloat, 2, "topk_weights");
      TORCH_CHECK(weights->sizes() == ids.sizes(),
                  "Routing weights shape mismatch");
    }
    group_clear<<<world, 32, 0, stream>>>(gpu, layer, phase);
    if (preprocess) {
      group_raw_size<<<world, 1, 0, stream>>>(gpu, ids.size(0));
      if (ids.size(0)) {
        dim3 grid(ids.size(0), world);
        auto *x =
            reinterpret_cast<const BF16 *>(hidden.data_ptr<at::BFloat16>());
        if (ids.scalar_type() == at::kInt)
          group_dispatch_raw<<<grid, 128, 0, stream>>>(
              gpu, x, ids.data_ptr<int>(), weights->data_ptr<float>(),
              ids.size(0));
        else
          group_dispatch_raw<<<grid, 128, 0, stream>>>(
              gpu, x, ids.data_ptr<int64_t>(), weights->data_ptr<float>(),
              ids.size(0));
      }
      group_publish<<<world, 128, 0, stream>>>(gpu, publish);
      C10_CUDA_CHECK(cudaGetLastError());
      return;
    }
    const int routes = ids.numel();
    if (routes) {
      if (packed) {
        dim3 grid(std::min(256, (routes + 255) / 256), world);
        if (ids.scalar_type() == at::kInt)
          group_count<<<grid, 256, 0, stream>>>(gpu, ids.data_ptr<int>(),
                                                rows.data_ptr<int>(), routes);
        else
          group_count<<<grid, 256, 0, stream>>>(gpu, ids.data_ptr<int64_t>(),
                                                rows.data_ptr<int>(), routes);
        group_pack<<<world, 1, 0, stream>>>(gpu);
      }
      auto *x = reinterpret_cast<const BF16 *>(hidden.data_ptr<at::BFloat16>());
      if (ids.scalar_type() == at::kInt)
        group_dispatch_rows<<<routes, 128, 0, stream>>>(
            gpu, x, ids.data_ptr<int>(), rows.data_ptr<int>(), ids.size(0),
            ids.size(1));
      else
        group_dispatch_rows<<<routes, 128, 0, stream>>>(
            gpu, x, ids.data_ptr<int64_t>(), rows.data_ptr<int>(), ids.size(0),
            ids.size(1));
    }
    group_publish<<<world, 128, 0, stream>>>(gpu, publish);
    C10_CUDA_CHECK(cudaGetLastError());
  }
  void combine(torch::Tensor output, torch::Tensor ids, torch::Tensor weights,
               torch::Tensor rows, int timeout_ms) {
    check_open();
    auto &first = *peers[0];
    check_tensor(output, first, at::kFloat, 2, "output");
    check_routes(ids, rows, first, 0, world);
    check_tensor(weights, first, at::kFloat, 2, "topk_weights");
    TORCH_CHECK(output.size(0) == ids.size(0) &&
                    output.size(1) == first.b.hidden &&
                    weights.sizes() == ids.sizes() && timeout_ms > 0 &&
                    timeout_ms <= 3600000,
                "Invalid group combine");
    c10::cuda::CUDAGuard guard(device);
    auto stream = at::cuda::getCurrentCUDAStream(device);
    group_wait<<<1, 1, 0, stream>>>(gpu,
                                    static_cast<U64>(timeout_ms) * 1000000ULL);
    int blocks = std::min<int64_t>(65535, (output.numel() / 8 + 255) / 256);
    if (preprocess) {
      if (blocks)
        group_combine_raw<<<blocks, 256, 0, stream>>>(
            gpu, output.data_ptr<float>(), ids.size(0));
      C10_CUDA_CHECK(cudaGetLastError());
      return;
    }
    if (blocks) {
      if (ids.scalar_type() == at::kInt)
        group_combine_rows<<<blocks, 256, 0, stream>>>(
            gpu, output.data_ptr<float>(), ids.data_ptr<int>(),
            weights.data_ptr<float>(), rows.data_ptr<int>(), ids.size(0),
            ids.size(1));
      else
        group_combine_rows<<<blocks, 256, 0, stream>>>(
            gpu, output.data_ptr<float>(), ids.data_ptr<int64_t>(),
            weights.data_ptr<float>(), rows.data_ptr<int>(), ids.size(0),
            ids.size(1));
    }
    C10_CUDA_CHECK(cudaGetLastError());
  }
};
