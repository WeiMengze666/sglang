// Isolated test adapter for the production activation helper.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include "phase_activation.cuh"

__global__ void activate(const __nv_bfloat16 *x, __nv_bfloat16 *y, int count,
                         int columns, int phase) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const int row = i / columns, h = i % columns;
    const float gate = __bfloat162float(x[row * columns * 2 + h]);
    const float up = __bfloat162float(x[row * columns * 2 + columns + h]);
    y[i] = __float2bfloat16(expertplex_activation::silu_mul(gate, up, phase));
  }
}

void run(torch::Tensor x, torch::Tensor y, int phase) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda() && x.device() == y.device() &&
                  x.is_contiguous() && y.is_contiguous(),
              "Use contiguous tensors on the same CUDA device");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
                  y.scalar_type() == at::kBFloat16 && x.numel() == y.numel() * 2 &&
                  y.dim() > 0 && y.size(-1) > 0 && (phase == 0 || phase == 1),
              "Invalid activation fixture");
  activate<<<(y.numel() + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()),
      reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), y.numel(), y.size(-1), phase);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run); }
