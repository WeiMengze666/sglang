// Diagnostic-only access to prebuilt operation buffers. Production clients use
// the IPC dispatch/combine protocol above and never call these entrypoints.
#pragma once

// Communication-only fixture: copy the dispatched inputs to final output
// buffers between (never inside) the dispatch/combine timed regions. There is
// no persistent executor or receiver polling kernel during either transfer.
__global__ void benchmark_echo_done(DeviceBuffers b) {
  int slot = threadIdx.x;
  if (slot < b.slots) {
    auto *c = b.control + slot * kControlWords;
    __threadfence_system();
    system_store(c + DONE, system_load(c + READY));
  }
}
__global__ void benchmark_echo_raw(DeviceBuffers b) {
  const int slot = blockIdx.y;
  const int n = uncached_load_int(b.raw_n + slot);
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < static_cast<size_t>(n) * b.hidden;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t token = b.raw_base[slot] + i / b.hidden;
    const float value = __bfloat162float(
        uncached_load(b.raw_input + b.raw_base[slot] * b.hidden + i));
    float sum = 0;
    for (int j = 0; j < b.top_k; ++j) {
      const size_t index = token * b.top_k + j;
      const int id = uncached_load_int(b.raw_ids + index);
      if (id >= 0 && id % b.expert_world == b.expert_rank)
        sum += value * __int_as_float(uncached_load_int(
                           reinterpret_cast<const int *>(b.raw_weights + index)));
    }
    b.raw_output[b.raw_base[slot] * b.hidden + i] = __float2bfloat16(sum);
  }
  __threadfence_system();
}
static void benchmark_echo(std::shared_ptr<OwnedBuffers> owner) {
  TORCH_CHECK(!owner->started, "Communication echo cannot run beside an executor");
  c10::cuda::CUDAGuard guard(owner->device);
  auto stream = at::cuda::getCurrentCUDAStream(owner->device);
  auto b = owner->b;
  if (b.prefill_preprocess) {
    // Identity-only stand-in for persistent scatter/GEMM/gather, outside both
    // timed intervals. It never serves as a receive/polling communication SM.
    benchmark_echo_raw<<<dim3(128, b.slots), 256, 0, stream>>>(b);
  } else {
    size_t rows = b.base_rows[b.slots - 1] + b.storage_rows[b.slots - 1];
    C10_CUDA_CHECK(cudaMemcpyAsync(b.output, b.input, rows * b.hidden * sizeof(BF16),
                                 cudaMemcpyDeviceToDevice, stream));
  }
  benchmark_echo_done<<<1, 32, 0, stream>>>(b);
  C10_CUDA_CHECK(cudaGetLastError());
}

static py::dict benchmark_views(std::shared_ptr<OwnedBuffers> owner, int slot) {
  TORCH_CHECK(!owner->started, "Prepare diagnostic tensors before executor start");
  TORCH_CHECK(slot >= 0 && slot < owner->b.slots, "Invalid diagnostic slot");
  auto b = owner->b;
  auto options = torch::TensorOptions().device(torch::kCUDA, owner->device);
  auto keep_alive = [owner](void *) {};
  auto matrix = [&](BF16 *ptr, int width) {
    return torch::from_blob(ptr + b.base_rows[slot] * width,
                            {b.storage_rows[slot], width}, keep_alive,
                            options.dtype(torch::kBFloat16));
  };
  py::dict result;
  result["input"] = matrix(b.input, b.hidden);
  result["activation"] = matrix(b.activation, b.intermediate);
  result["gateup"] = matrix(b.gateup, 2 * b.intermediate);
  result["output"] = matrix(b.output, b.hidden);
  result["counts"] = torch::from_blob(
      b.counts + slot * b.experts, {b.experts}, keep_alive, options.dtype(torch::kInt32));
  result["offsets"] = torch::from_blob(
      b.offsets + slot * b.experts, {b.experts}, keep_alive, options.dtype(torch::kInt32));
  return result;
}

__device__ bool benchmark_publish_once(U64 *p, U64 *d, U64 delay_ns,
                                      bool run_p, bool run_d, U64 *stamps,
                                      U64 *published_epochs = nullptr) {
  U64 *controls[2] = {p, d};
  bool enabled[2] = {run_p, run_d};
  U64 epochs[2] = {0, 0};
  for (int phase = 0; phase < 2; ++phase) {
    auto *c = controls[phase];
    if (!enabled[phase])
      continue;
    auto pending = system_load(c + PENDING);
    if (system_load(c + DONE) != pending || system_load(c + ERROR) ||
        pending == kMaxEpoch) {
      stamps[4 + phase] = 1;
      return false;
    }
    c[LAYER] = 0;
    c[PHASE] = phase;
    c[START_NS] = c[FINISH_NS] = 0;
    epochs[phase] = pending + 1;
    if (published_epochs)
      published_epochs[phase] = epochs[phase];
    system_store(c + PENDING, epochs[phase]);
  }
  __threadfence_system();
  stamps[0] = now_ns();
  if (run_p)
    system_store(p + READY, epochs[0]);
  while (now_ns() - stamps[0] < delay_ns) {}
  stamps[1] = now_ns();
  if (run_d)
    system_store(d + READY, epochs[1]);
  const U64 limit = stamps[1] + 5000000000ULL;
  bool finished[2] = {!run_p, !run_d};
  while ((!finished[0] || !finished[1]) && now_ns() < limit) {
    for (int phase = 0; phase < 2; ++phase) {
      if (!finished[phase] && system_load(controls[phase] + DONE) == epochs[phase]) {
        stamps[2 + phase] = now_ns();
        stamps[4 + phase] = system_load(controls[phase] + ERROR);
        finished[phase] = true;
      }
    }
  }
  for (int phase = 0; phase < 2; ++phase)
    if (!finished[phase])
      stamps[4 + phase] = TIMEOUT;
  return finished[0] && finished[1] && !stamps[4] && !stamps[5];
}

__global__ void benchmark_publish_kernel(U64 *p, U64 *d, U64 delay_ns,
                                         bool run_p, bool run_d, U64 *stamps) {
  benchmark_publish_once(p, d, delay_ns, run_p, run_d, stamps);
}

// One attention thread repeatedly publishes fully retired generations. This
// diagnostic removes per-sample CPU synchronization/launch gaps from profiler
// ranges; it is not the serving arrival process or a transport benchmark.
__global__ void benchmark_publish_batch_kernel(U64 *p, U64 *d, U64 delay_ns,
                                               bool run_p, bool run_d,
                                               U64 *stamps, int iterations) {
  for (int i = 0; i < iterations; ++i) {
    U64 *row = stamps + static_cast<size_t>(i) * 8;
    if (!benchmark_publish_once(p, d, delay_ns, run_p, run_d, row, row + 6))
      return; // Never overwrite a failed or timed-out generation.
  }
}

static torch::Tensor benchmark_publish(std::shared_ptr<OwnedBuffers> owner,
                                       int attention_device, bool run_p,
                                       bool run_d, int delay_us,
                                       int iterations = 0) {
  TORCH_CHECK(owner->active && owner->b.slots >= 2,
              "Diagnostic publication requires a live executor with at least two slots");
  TORCH_CHECK(attention_device != owner->device && delay_us >= 0 &&
                  delay_us <= 1000000 && (run_p || run_d) &&
                  iterations >= 0 && iterations <= 100000,
              "Invalid diagnostic publication arguments");
  c10::cuda::CUDAGuard guard(attention_device);
  int accessible = 0, atomics = 0;
  C10_CUDA_CHECK(cudaDeviceCanAccessPeer(&accessible, attention_device, owner->device));
  C10_CUDA_CHECK(cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported,
                                         attention_device, owner->device));
  TORCH_CHECK(accessible && atomics, "Benchmark requires NVLink peer atomics");
  auto status = cudaDeviceEnablePeerAccess(owner->device, 0);
  if (status == cudaErrorPeerAccessAlreadyEnabled)
    cudaGetLastError();
  else
    C10_CUDA_CHECK(status);
  auto options = torch::TensorOptions().dtype(torch::kInt64)
                     .device(torch::kCUDA, attention_device);
  auto stamps = iterations ? torch::zeros({iterations, 8}, options)
                           : torch::zeros({6}, options);
  auto stream = at::cuda::getCurrentCUDAStream(attention_device);
  if (iterations)
    benchmark_publish_batch_kernel<<<1, 1, 0, stream>>>(
        owner->b.control, owner->b.control + kControlWords,
        static_cast<U64>(delay_us) * 1000, run_p, run_d,
        reinterpret_cast<U64 *>(stamps.data_ptr<int64_t>()), iterations);
  else
    benchmark_publish_kernel<<<1, 1, 0, stream>>>(
        owner->b.control, owner->b.control + kControlWords,
        static_cast<U64>(delay_us) * 1000, run_p, run_d,
        reinterpret_cast<U64 *>(stamps.data_ptr<int64_t>()));
  C10_CUDA_CHECK(cudaGetLastError());
  return stamps;
}
