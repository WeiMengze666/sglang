// Benchmark-only GPU-timed release, separate from the serving implementation.
// A control GPU publishes two flags exactly delay_ns apart and observes E's
// completion flags. No polling CTA is launched on the measured expert GPU.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <torch/extension.h>

using U64 = unsigned long long;
__device__ U64 timer() {
  U64 value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}
__device__ void publish(int *ptr) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(ptr), "r"(1)
               : "memory");
}
__device__ int observe(int *ptr) {
  int value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(ptr)
               : "memory");
  return value;
}
__global__ void release_and_ack(int *flags, U64 *stamps, U64 delay_ns) {
  const U64 start = timer();
  stamps[0] = start;
  publish(flags);
  while (timer() - start < delay_ns) {
  }
  stamps[1] = timer();
  publish(flags + 1);
  U64 p = 0, d = 0;
  while ((!p || !d) && timer() - start < 5000000000ULL) {
    if (!p && observe(flags + 2))
      p = timer();
    if (!d && observe(flags + 3))
      d = timer();
  }
  stamps[2] = p;
  stamps[3] = d;
}
void run(torch::Tensor flags, torch::Tensor stamps, int control_device,
         U64 delay_ns) {
  TORCH_CHECK(flags.is_cuda() && flags.scalar_type() == torch::kInt32 &&
              flags.numel() == 4);
  TORCH_CHECK(stamps.is_cuda() && stamps.get_device() == control_device &&
              stamps.scalar_type() == torch::kInt64 && stamps.numel() == 4);
  TORCH_CHECK(flags.get_device() != control_device);
  c10::cuda::CUDAGuard guard(control_device);
  auto status = cudaDeviceEnablePeerAccess(flags.get_device(), 0);
  if (status == cudaErrorPeerAccessAlreadyEnabled)
    cudaGetLastError();
  else
    C10_CUDA_CHECK(status);
  release_and_ack<<<1, 1, 0, at::cuda::getCurrentCUDAStream(control_device)>>>(
      flags.data_ptr<int>(),
      reinterpret_cast<U64 *>(stamps.data_ptr<int64_t>()), delay_ns);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void wait_flag(torch::Tensor flags, int index, uint64_t stream) {
  c10::cuda::CUDAGuard guard(flags.device());
  auto status = cuStreamWaitValue32(
      reinterpret_cast<CUstream>(stream),
      reinterpret_cast<CUdeviceptr>(flags.data_ptr<int>() + index), 1,
      CU_STREAM_WAIT_VALUE_EQ);
  TORCH_CHECK(status == CUDA_SUCCESS,
              "cuStreamWaitValue32 failed: ", int(status));
}
void done(torch::Tensor flags, int index, uint64_t stream) {
  c10::cuda::CUDAGuard guard(flags.device());
  auto status = cuStreamWriteValue32(
      reinterpret_cast<CUstream>(stream),
      reinterpret_cast<CUdeviceptr>(flags.data_ptr<int>() + index), 1, 0);
  TORCH_CHECK(status == CUDA_SUCCESS,
              "cuStreamWriteValue32 failed: ", int(status));
}
torch::Tensor allocate_flags(int device) {
  c10::cuda::CUDAGuard guard(device);
  void *ptr = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&ptr, 4 * sizeof(int)));
  return torch::from_blob(
      ptr, {4},
      [device](void *p) {
        c10::cuda::CUDAGuard guard(device);
        cudaFree(p);
      },
      torch::TensorOptions().device(torch::kCUDA, device).dtype(torch::kInt32));
}
pybind11::bytes export_flags(torch::Tensor flags) {
  c10::cuda::CUDAGuard guard(flags.device());
  cudaIpcMemHandle_t handle;
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&handle, flags.data_ptr()));
  return pybind11::bytes(reinterpret_cast<char *>(&handle), sizeof(handle));
}
torch::Tensor import_flags(pybind11::bytes raw, int device) {
  c10::cuda::CUDAGuard guard(device);
  std::string bytes = raw;
  TORCH_CHECK(bytes.size() == sizeof(cudaIpcMemHandle_t));
  cudaIpcMemHandle_t handle;
  memcpy(&handle, bytes.data(), sizeof(handle));
  void *ptr = nullptr;
  C10_CUDA_CHECK(
      cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
  return torch::from_blob(
      ptr, {4},
      [device](void *p) {
        c10::cuda::CUDAGuard guard(device);
        cudaIpcCloseMemHandle(p);
      },
      torch::TensorOptions().device(torch::kCUDA, device).dtype(torch::kInt32));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run);
  m.def("wait_flag", &wait_flag);
  m.def("done", &done);
  m.def("allocate_flags", &allocate_flags);
  m.def("export_flags", &export_flags);
  m.def("import_flags", &import_flags);
}
