// Experimental ExpertPlex B300 backend. Independent CTAs (cluster size one).
// No TMA / asynchronous matrix operation spans a task scheduling boundary.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;
using U64 = unsigned long long;
using BF16 = __nv_bfloat16;
constexpr int kControlWords = 8;
enum Control { READY = 0, DONE = 1, LAYER = 2, PHASE = 3, ERROR = 4 };
enum Error {
  OK = 0,
  BUSY = 1,
  BAD_EPOCH = 2,
  BAD_ROUTE = 3,
  ROW_OVERFLOW = 4,
  BAD_LAYER = 5,
  BAD_PHASE = 6,
  TIMEOUT = 7,
  STOPPED = 8
};
enum Stage { IDLE = 0, GATE = 1, DOWN = 2, INITIALIZING = 3 };
constexpr U64 kMaxEpoch = (1ULL << 30) - 1;

struct SlotState {
  U64 queue; // epoch[63:34], stage[33:32], next task[31:0]
  unsigned int completed;
  unsigned int gate_tiles;
  unsigned int down_tiles;
  int phase;
  int layer;
};
static_assert(sizeof(SlotState) == 32,
              "Update the Python memory estimator for changed state layout");

struct DeviceBuffers {
  BF16 *input, *activation, *output;
  int *counts;
  U64 *control;
  SlotState *states;
  int *stop;
  int slots, experts, rows, hidden, intermediate;
};

static std::string encode_handle(void *pointer) {
  cudaIpcMemHandle_t handle;
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&handle, pointer));
  const unsigned char *bytes = reinterpret_cast<const unsigned char *>(&handle);
  constexpr char hex[] = "0123456789abcdef";
  std::string result(sizeof(handle) * 2, '0');
  for (size_t i = 0; i < sizeof(handle); ++i) {
    result[2 * i] = hex[bytes[i] >> 4];
    result[2 * i + 1] = hex[bytes[i] & 15];
  }
  return result;
}

static void *open_handle(const std::string &encoded) {
  cudaIpcMemHandle_t handle;
  TORCH_CHECK(encoded.size() == sizeof(handle) * 2,
              "Malformed CUDA IPC handle");
  auto digit = [](char c) -> int {
    if (c >= '0' && c <= '9')
      return c - '0';
    if (c >= 'a' && c <= 'f')
      return c - 'a' + 10;
    TORCH_CHECK(false, "Malformed hexadecimal CUDA IPC handle");
  };
  auto *bytes = reinterpret_cast<unsigned char *>(&handle);
  for (size_t i = 0; i < sizeof(handle); ++i)
    bytes[i] = (digit(encoded[2 * i]) << 4) | digit(encoded[2 * i + 1]);
  void *pointer = nullptr;
  C10_CUDA_CHECK(
      cudaIpcOpenMemHandle(&pointer, handle, cudaIpcMemLazyEnablePeerAccess));
  return pointer;
}

static size_t checked_product(std::initializer_list<int64_t> dims) {
  size_t result = 1;
  for (int64_t d : dims) {
    TORCH_CHECK(d > 0, "ExpertPlex dimensions must be positive");
    TORCH_CHECK(static_cast<uint64_t>(d) <= SIZE_MAX / result,
                "Allocation size overflow");
    result *= static_cast<size_t>(d);
  }
  return result;
}

class OwnedBuffers {
public:
  DeviceBuffers b{};
  int device;
  bool active = false;
  bool started = false;
  std::vector<void *> allocations;
  std::string input_handle, output_handle, counts_handle, control_handle;

  OwnedBuffers(int slots, int experts, int rows, int hidden, int intermediate,
               int dev)
      : device(dev) {
    TORCH_CHECK(slots > 0 && experts > 0 && rows > 0 && hidden > 0 &&
                    intermediate > 0,
                "All buffer dimensions must be positive");
    TORCH_CHECK(rows <= INT_MAX - 16 && intermediate <= INT_MAX / 2,
                "Dimensions exceed the native indexing range");
    TORCH_CHECK(
        hidden % 16 == 0 && intermediate % 16 == 0,
        "WMMA backend requires hidden and intermediate multiples of 16");
    TORCH_CHECK(static_cast<int64_t>(slots) * experts <= INT_MAX,
                "Too many slots/experts");
    c10::cuda::CUDAGuard guard(device);
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    TORCH_CHECK(prop.major == 10 && prop.minor == 3,
                "This experimental extension targets B300 (sm_103)");
    b.slots = slots;
    b.experts = experts;
    b.rows = rows;
    b.hidden = hidden;
    b.intermediate = intermediate;
    try {
      b.input = static_cast<BF16 *>(
          allocate(checked_product({slots, experts, rows, hidden, 2})));
      b.activation = static_cast<BF16 *>(
          allocate(checked_product({slots, experts, rows, intermediate, 2})));
      b.output = static_cast<BF16 *>(
          allocate(checked_product({slots, experts, rows, hidden, 2})));
      b.counts =
          static_cast<int *>(allocate(checked_product({slots, experts, 4})));
      b.control = static_cast<U64 *>(
          allocate(checked_product({slots, kControlWords, 8})));
      b.states = static_cast<SlotState *>(
          allocate(checked_product({slots, sizeof(SlotState)})));
      b.stop = static_cast<int *>(allocate(sizeof(int)));
      C10_CUDA_CHECK(
          cudaMemset(b.counts, 0, checked_product({slots, experts, 4})));
      C10_CUDA_CHECK(
          cudaMemset(b.control, 0, checked_product({slots, kControlWords, 8})));
      C10_CUDA_CHECK(
          cudaMemset(b.states, 0, checked_product({slots, sizeof(SlotState)})));
      C10_CUDA_CHECK(cudaMemset(b.stop, 0, sizeof(int)));
      C10_CUDA_CHECK(cudaStreamSynchronize(nullptr));
      // Cache handles before launching an indefinitely running kernel. No CUDA
      // allocation/export API is needed to publish the readiness descriptor.
      input_handle = encode_handle(b.input);
      output_handle = encode_handle(b.output);
      counts_handle = encode_handle(b.counts);
      control_handle = encode_handle(b.control);
    } catch (...) {
      release();
      throw;
    }
  }

  ~OwnedBuffers() { release(); }
  void *allocate(size_t bytes) {
    void *p = nullptr;
    C10_CUDA_CHECK(cudaMalloc(&p, bytes));
    allocations.push_back(p);
    return p;
  }
  void release() noexcept {
    if (allocations.empty())
      return;
    int old = 0;
    cudaGetDevice(&old);
    cudaSetDevice(device);
    for (void *p : allocations)
      cudaFree(p);
    allocations.clear();
    cudaSetDevice(old);
  }
  py::dict descriptor() const {
    py::dict d;
    d["version"] = 1;
    d["device"] = device;
    d["slot_count"] = b.slots;
    d["local_experts"] = b.experts;
    d["max_rows"] = b.rows;
    d["hidden"] = b.hidden;
    d["intermediate"] = b.intermediate;
    d["input_handle"] = input_handle;
    d["output_handle"] = output_handle;
    d["counts_handle"] = counts_handle;
    d["control_handle"] = control_handle;
    return d;
  }
};

class RemoteBuffers {
public:
  DeviceBuffers b{};
  int device, slot;
  std::vector<void *> mappings;
  int *status = nullptr;
  int *host_status = nullptr;
  bool closed = false;

  RemoteBuffers(py::dict d, int slot_id, int dev) : device(dev), slot(slot_id) {
    TORCH_CHECK(py::cast<int>(d["version"]) == 1,
                "Unsupported IPC descriptor version");
    b.slots = py::cast<int>(d["slot_count"]);
    b.experts = py::cast<int>(d["local_experts"]);
    b.rows = py::cast<int>(d["max_rows"]);
    b.hidden = py::cast<int>(d["hidden"]);
    b.intermediate = py::cast<int>(d["intermediate"]);
    TORCH_CHECK(slot >= 0 && slot < b.slots, "Invalid attention slot");
    checked_product({b.slots, b.experts, b.rows, b.hidden});
    TORCH_CHECK(b.hidden % 16 == 0 && b.intermediate > 0,
                "Malformed IPC dimensions");
    int owner_device = py::cast<int>(d["device"]);
    TORCH_CHECK(device != owner_device,
                "Attention and expert must use distinct GPUs with the same "
                "CUDA_VISIBLE_DEVICES ordering");
    c10::cuda::CUDAGuard guard(device);
    int accessible = 0, atomics = 0;
    C10_CUDA_CHECK(cudaDeviceCanAccessPeer(&accessible, device, owner_device));
    C10_CUDA_CHECK(cudaDeviceGetP2PAttribute(
        &atomics, cudaDevP2PAttrNativeAtomicSupported, device, owner_device));
    TORCH_CHECK(accessible && atomics,
                "ExpertPlex requires peer access and native peer atomics "
                "(single-node NVLink)");
    try {
      b.input =
          static_cast<BF16 *>(map(py::cast<std::string>(d["input_handle"])));
      b.output =
          static_cast<BF16 *>(map(py::cast<std::string>(d["output_handle"])));
      b.counts =
          static_cast<int *>(map(py::cast<std::string>(d["counts_handle"])));
      b.control =
          static_cast<U64 *>(map(py::cast<std::string>(d["control_handle"])));
      C10_CUDA_CHECK(cudaMalloc(&status, sizeof(int)));
      C10_CUDA_CHECK(
          cudaHostAlloc(&host_status, sizeof(int), cudaHostAllocDefault));
    } catch (...) {
      close_noexcept();
      throw;
    }
  }
  ~RemoteBuffers() { close_noexcept(); }
  void *map(const std::string &h) {
    void *p = open_handle(h);
    mappings.push_back(p);
    return p;
  }
  void close() {
    if (closed)
      return;
    c10::cuda::CUDAGuard guard(device);
    // The last combine synchronizes its stream. Closing an outstanding dispatch
    // is a caller lifecycle error; explicitly synchronize before unmapping.
    C10_CUDA_CHECK(
        cudaStreamSynchronize(at::cuda::getCurrentCUDAStream(device)));
    close_noexcept();
  }
  void close_noexcept() noexcept {
    if (closed)
      return;
    int old = 0;
    cudaGetDevice(&old);
    cudaSetDevice(device);
    for (void *p : mappings)
      cudaIpcCloseMemHandle(p);
    mappings.clear();
    if (status)
      cudaFree(status);
    if (host_status)
      cudaFreeHost(host_status);
    status = nullptr;
    host_status = nullptr;
    closed = true;
    cudaSetDevice(old);
  }
};

// Use explicit acquire/release scopes: ordinary legacy CUDA atomics are relaxed
// and do not by themselves publish all the preceding peer-buffer writes.
__device__ __forceinline__ U64 system_load(U64 *p) {
  return cuda::atomic_ref<U64, cuda::thread_scope_system>(*p).load(
      cuda::memory_order_acquire);
}
__device__ __forceinline__ void system_store(U64 *p, U64 value) {
  cuda::atomic_ref<U64, cuda::thread_scope_system>(*p).store(
      value, cuda::memory_order_release);
}
__device__ __forceinline__ void set_error(U64 *c, Error code) {
  U64 expected = 0;
  cuda::atomic_ref<U64, cuda::thread_scope_system>(c[ERROR])
      .compare_exchange_strong(expected, static_cast<U64>(code),
                               cuda::memory_order_acq_rel);
}
__device__ __forceinline__ U64 queue_load(U64 *p) {
  return cuda::atomic_ref<U64, cuda::thread_scope_device>(*p).load(
      cuda::memory_order_acquire);
}
__device__ __forceinline__ bool queue_cas(U64 *p, U64 expected, U64 desired) {
  return cuda::atomic_ref<U64, cuda::thread_scope_device>(*p)
      .compare_exchange_strong(expected, desired, cuda::memory_order_acq_rel);
}
__device__ __forceinline__ void queue_store(U64 *p, U64 value) {
  cuda::atomic_ref<U64, cuda::thread_scope_device>(*p).store(
      value, cuda::memory_order_release);
}
__device__ __forceinline__ BF16 uncached_load(const BF16 *p) {
  // Slots are reused. Bypass the reader's L1 so another GPU's writes cannot
  // leave a stale cached line from the preceding invocation.
  unsigned short bits;
  asm volatile("ld.global.cg.b16 %0, [%1];" : "=h"(bits) : "l"(p) : "memory");
  return __ushort_as_bfloat16(bits);
}
__device__ __forceinline__ int uncached_load_int(const int *p) {
  int value;
  asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
  return value;
}
__device__ __forceinline__ U64 uncached_load_u64(const U64 *p) {
  U64 value;
  asm volatile("ld.global.cg.b64 %0, [%1];" : "=l"(value) : "l"(p) : "memory");
  return value;
}
__device__ __forceinline__ U64 queue_value(U64 epoch, Stage stage,
                                           unsigned task = 0) {
  return (epoch << 34) | (static_cast<U64>(stage) << 32) | task;
}
__device__ __forceinline__ U64 now_ns() {
  U64 value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__global__ void clear_slot(DeviceBuffers b, int slot, U64 epoch, int layer,
                           int phase) {
  U64 *c = b.control + static_cast<size_t>(slot) * kControlWords;
  if (threadIdx.x == 0) {
    U64 ready = system_load(c + READY), done = system_load(c + DONE);
    if (ready != done)
      set_error(c, BUSY);
    else if (epoch <= ready)
      set_error(c, BAD_EPOCH);
    else {
      system_store(c + ERROR, 0ULL);
      c[LAYER] = layer;
      c[PHASE] = phase;
    }
  }
  __syncthreads();
  if (system_load(c + ERROR))
    return;
  for (int e = threadIdx.x; e < b.experts; e += blockDim.x)
    b.counts[static_cast<size_t>(slot) * b.experts + e] = 0;
  __threadfence_system();
}

template <typename Id>
__global__ void dispatch_rows(DeviceBuffers b, int slot, const BF16 *hidden,
                              const Id *ids, int *row_map, int n, int k,
                              int rank, int world) {
  U64 *c = b.control + static_cast<size_t>(slot) * kControlWords;
  if (system_load(c + ERROR))
    return;
  int route = blockIdx.x;
  Id routed_expert = ids[route];
  if (routed_expert < 0 ||
      routed_expert >= static_cast<Id>(b.experts) * world) {
    if (threadIdx.x == 0)
      set_error(c, BAD_ROUTE);
    return;
  }
  int expert = static_cast<int>(routed_expert);
  if (expert % world != rank)
    return;
  int local = expert / world;
  __shared__ int row;
  if (threadIdx.x == 0) {
    row = atomicAdd_system(
        b.counts + static_cast<size_t>(slot) * b.experts + local, 1);
    row_map[route] = row;
    if (row >= b.rows)
      set_error(c, ROW_OVERFLOW);
  }
  __syncthreads();
  if (row >= b.rows)
    return;
  size_t base =
      ((static_cast<size_t>(slot) * b.experts + local) * b.rows + row) *
      b.hidden;
  int token = route / k;
  for (int h = threadIdx.x; h < b.hidden; h += blockDim.x)
    b.input[base + h] = hidden[static_cast<size_t>(token) * b.hidden + h];
  // Every writer, rather than only the publishing thread, fences its peer data.
  __threadfence_system();
}

__global__ void publish_slot(DeviceBuffers b, int slot, U64 epoch) {
  U64 *c = b.control + static_cast<size_t>(slot) * kControlWords;
  // A failed clear must not replace the READY epoch of an active invocation.
  // Errors are fail-closed; the worker/client report them and must restart.
  if (system_load(c + ERROR))
    return;
  __threadfence_system();
  system_store(c + READY, epoch);
}

__global__ void wait_slot(DeviceBuffers b, int slot, U64 epoch, U64 timeout_ns,
                          int *status) {
  U64 *c = b.control + static_cast<size_t>(slot) * kControlWords;
  U64 started = now_ns();
  while (true) {
    U64 error = system_load(c + ERROR);
    if (error) {
      *status = static_cast<int>(error);
      return;
    }
    if (system_load(c + DONE) == epoch) {
      __threadfence_system();
      *status = OK;
      return;
    }
    if (now_ns() - started >= timeout_ns) {
      set_error(c, TIMEOUT);
      *status = TIMEOUT;
      return;
    }
    __nanosleep(128);
  }
}

template <typename Id>
__global__ void combine_rows(DeviceBuffers b, int slot, float *output,
                             const Id *ids, const float *weights,
                             const int *row_map, int n, int k, int rank,
                             int world, const int *status) {
  if (*status != OK)
    return;
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < static_cast<size_t>(n) * b.hidden; i += stride) {
    int token = i / b.hidden, h = i % b.hidden;
    float sum = 0;
    for (int j = 0; j < k; ++j) {
      size_t route = static_cast<size_t>(token) * k + j;
      int expert = static_cast<int>(ids[route]);
      if (expert % world != rank)
        continue;
      int row = row_map[route];
      if (row < 0 || row >= b.rows)
        continue; // dispatch errors surfaced by wait.
      size_t base =
          ((static_cast<size_t>(slot) * b.experts + expert / world) * b.rows +
           row) *
          b.hidden;
      sum +=
          weights[route] * __bfloat162float(uncached_load(b.output + base + h));
    }
    output[i] += sum;
  }
}

// One warp per CTA keeps the reference matrix path simple. Dynamic shared
// memory reserves enough resources to admit exactly one CTA per SM. Budgeting
// CTAs then corresponds to budgeting SMs, without relying on block-index-to-SM
// placement. Matrix arithmetic uses BF16 tensor cores with FP32 accumulation.
__device__ void compute_tile(DeviceBuffers b, const BF16 *w13, const BF16 *w2,
                             int slot, int layer, Stage stage, unsigned task,
                             BF16 *sa, BF16 *sg, BF16 *su, float *cg,
                             float *cu) {
  int ntile = (stage == GATE ? b.intermediate : b.hidden) / 16;
  int expert = 0, mtile = 0, column = 0;
  for (; expert < b.experts; ++expert) {
    int count = uncached_load_int(
        b.counts + static_cast<size_t>(slot) * b.experts + expert);
    unsigned tiles =
        static_cast<unsigned>((static_cast<U64>(count) + 15) / 16 * ntile);
    if (task < tiles) {
      mtile = task / ntile;
      column = task % ntile;
      break;
    }
    task -= tiles;
  }
  if (expert == b.experts)
    return;
  int count = uncached_load_int(b.counts +
                                static_cast<size_t>(slot) * b.experts + expert);
  int width = stage == GATE ? b.hidden : b.intermediate;
  int output_width = stage == GATE ? b.intermediate : b.hidden;
  const BF16 *input =
      (stage == GATE ? b.input : b.activation) +
      ((static_cast<size_t>(slot) * b.experts + expert) * b.rows) * width;
  const BF16 *weight =
      stage == GATE ? w13 + ((static_cast<size_t>(layer) * b.experts + expert) *
                             2 * b.intermediate) *
                                b.hidden
                    : w2 + ((static_cast<size_t>(layer) * b.experts + expert) *
                            b.hidden) *
                               b.intermediate;
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, BF16, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, BF16, wmma::col_major> g, u;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> gate, up;
  wmma::fill_fragment(gate, 0.0f);
  wmma::fill_fragment(up, 0.0f);
  for (int offset = 0; offset < width; offset += 16) {
    for (int i = threadIdx.x; i < 256; i += 32) {
      int m = i / 16, inner = i % 16, row = mtile * 16 + m;
      sa[i] = row < count
                  ? uncached_load(input + static_cast<size_t>(row) * width +
                                  offset + inner)
                  : __float2bfloat16(0.0f);
      // Column-major KxN view of row-major [N,K] checkpoint weights.
      int col = i / 16, kk = i % 16;
      sg[i] =
          weight[static_cast<size_t>(column * 16 + col) * width + offset + kk];
      if (stage == GATE)
        su[i] = weight[static_cast<size_t>(b.intermediate + column * 16 + col) *
                           width +
                       offset + kk];
    }
    __syncthreads();
    wmma::load_matrix_sync(a, sa, 16);
    wmma::load_matrix_sync(g, sg, 16);
    wmma::mma_sync(gate, a, g, gate);
    if (stage == GATE) {
      wmma::load_matrix_sync(u, su, 16);
      wmma::mma_sync(up, a, u, up);
    }
    __syncthreads();
  }
  wmma::store_matrix_sync(cg, gate, 16, wmma::mem_row_major);
  if (stage == GATE)
    wmma::store_matrix_sync(cu, up, 16, wmma::mem_row_major);
  __syncthreads();
  BF16 *destination =
      (stage == GATE ? b.activation : b.output) +
      ((static_cast<size_t>(slot) * b.experts + expert) * b.rows) *
          output_width;
  for (int i = threadIdx.x; i < 256; i += 32) {
    int row = mtile * 16 + i / 16, col = column * 16 + i % 16;
    if (row < count) {
      float value = cg[i];
      if (stage == GATE)
        value = (value / (1.0f + expf(-value))) * cu[i];
      destination[static_cast<size_t>(row) * output_width + col] =
          __float2bfloat16(value);
    }
  }
  // The next claimant can reuse shared memory only after all matrix operations
  // and stores above have completed. There is no asynchronous pipeline to
  // drain.
  __syncthreads();
}

__device__ void initialize_slot(DeviceBuffers b, int slot, int layers) {
  U64 *c = b.control + static_cast<size_t>(slot) * kControlWords;
  SlotState *s = b.states + slot;
  U64 q = queue_load(&s->queue);
  if (((q >> 32) & 3) != IDLE)
    return;
  U64 epoch = system_load(c + READY);
  if (epoch <= (q >> 34))
    return;
  if (!queue_cas(&s->queue, q, queue_value(epoch, INITIALIZING)))
    return;
  s->layer = static_cast<int>(uncached_load_u64(c + LAYER));
  s->phase = static_cast<int>(uncached_load_u64(c + PHASE));
  if (s->layer < 0 || s->layer >= layers)
    set_error(c, BAD_LAYER);
  if (s->phase < 0 || s->phase > 1)
    set_error(c, BAD_PHASE);
  U64 rows = 0;
  for (int e = 0; e < b.experts; ++e) {
    int count =
        uncached_load_int(b.counts + static_cast<size_t>(slot) * b.experts + e);
    if (count < 0 || count > b.rows)
      set_error(c, ROW_OVERFLOW);
    else
      rows += (static_cast<U64>(count) + 15) / 16;
  }
  U64 gate_tiles = rows * (b.intermediate / 16),
      down_tiles = rows * (b.hidden / 16);
  // Low queue bits carry the task counter. Leave headroom for one outstanding
  // overshoot per CTA; rejecting impossibly large requests avoids carry into
  // stage.
  if (gate_tiles > UINT_MAX - 65536ULL || down_tiles > UINT_MAX - 65536ULL)
    set_error(c, ROW_OVERFLOW);
  s->gate_tiles = static_cast<unsigned>(gate_tiles);
  s->down_tiles = static_cast<unsigned>(down_tiles);
  s->completed = 0;
  __threadfence();
  if (system_load(c + ERROR) || gate_tiles == 0) {
    queue_store(&s->queue, queue_value(epoch, IDLE));
    __threadfence_system();
    system_store(c + DONE, epoch);
  } else
    queue_store(&s->queue, queue_value(epoch, GATE));
}

__device__ bool claim_slot(DeviceBuffers b, int phase, int cursor, int &slot,
                           Stage &stage, unsigned &task, U64 &epoch) {
  for (int i = 0; i < b.slots; ++i) {
    int candidate = (cursor + i) % b.slots;
    SlotState *s = b.states + candidate;
    if (system_load(b.control + static_cast<size_t>(candidate) * kControlWords +
                    ERROR))
      continue;
    U64 q = queue_load(&s->queue);
    Stage st = static_cast<Stage>((q >> 32) & 3);
    if ((st != GATE && st != DOWN) || s->phase != phase)
      continue;
    unsigned t = static_cast<unsigned>(q);
    unsigned total = st == GATE ? s->gate_tiles : s->down_tiles;
    if (t >= total)
      continue;
    if (!queue_cas(&s->queue, q, q + 1))
      continue;
    slot = candidate;
    stage = st;
    task = t;
    epoch = q >> 34;
    return true;
  }
  return false;
}

__global__ void persistent_executor(DeviceBuffers b, const BF16 *w13,
                                    const BF16 *w2, int layers,
                                    int decode_budget, float expected_tiles,
                                    int max_decode_ctas) {
  __shared__ __align__(32) BF16 sa[256], sg[256], su[256];
  __shared__ __align__(32) float cg[256], cu[256];
  __shared__ int chosen_slot, chosen_stage;
  __shared__ unsigned chosen_task;
  __shared__ U64 chosen_epoch;
  int cursor = blockIdx.x % b.slots;
  while (true) {
    if (threadIdx.x == 0) {
      chosen_slot = -1;
      if (atomicAdd(b.stop, 0)) {
        for (int s = 0; s < b.slots; ++s) {
          U64 *c = b.control + static_cast<size_t>(s) * kControlWords;
          if (system_load(c + READY) != system_load(c + DONE))
            set_error(c, STOPPED);
        }
        chosen_slot = -2;
      } else {
        for (int i = 0; i < b.slots; ++i)
          initialize_slot(b, (cursor + i) % b.slots, layers);
        // Eq. 3/4 counts routed row tiles, independent of the GEMM's output
        // column width. Gate/up and down have the same online row footprint.
        // Cap the allocation to leave a prefill CTA; idle CTAs can borrow.
        U64 decode_tiles = 0;
        for (int i = 0; i < b.slots; ++i) {
          SlotState *s = b.states + i;
          U64 q = queue_load(&s->queue);
          int st = (q >> 32) & 3;
          if ((st == GATE || st == DOWN) && s->phase == 1)
            decode_tiles += st == GATE ? s->gate_tiles / (b.intermediate / 16)
                                       : s->down_tiles / (b.hidden / 16);
        }
        float requested = ceilf(
            decode_budget * static_cast<float>(decode_tiles) / expected_tiles);
        int budget = decode_tiles == 0 ? 0
                     : requested >= max_decode_ctas
                         ? max_decode_ctas
                         : max(1, static_cast<int>(requested));
        int preferred = blockIdx.x < budget ? 1 : 0;
        Stage stage = IDLE;
        unsigned task = 0;
        U64 epoch = 0;
        int slot = -1;
        if (!claim_slot(b, preferred, cursor, slot, stage, task, epoch))
          claim_slot(b, 1 - preferred, cursor, slot, stage, task, epoch);
        chosen_slot = slot;
        chosen_stage = stage;
        chosen_task = task;
        chosen_epoch = epoch;
      }
    }
    __syncthreads();
    if (chosen_slot == -2)
      break;
    if (chosen_slot < 0) {
      __nanosleep(256);
      continue;
    }
    int slot = chosen_slot;
    Stage stage = static_cast<Stage>(chosen_stage);
    SlotState *state = b.states + slot;
    compute_tile(b, w13, w2, slot, state->layer, stage, chosen_task, sa, sg, su,
                 cg, cu);
    // Every lane flushes stores before lane zero signals tile completion.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned total = stage == GATE ? state->gate_tiles : state->down_tiles;
      if (atomicAdd(&state->completed, 1U) == total - 1) {
        state->completed = 0;
        __threadfence();
        if (stage == GATE)
          queue_store(&state->queue, queue_value(chosen_epoch, DOWN));
        else {
          // Publish idle state before done; the client cannot reuse this slot
          // until it observes done. Queue generations reject stale claim CASes.
          queue_store(&state->queue, queue_value(chosen_epoch, IDLE));
          __threadfence_system();
          system_store(b.control + static_cast<size_t>(slot) * kControlWords +
                           DONE,
                       chosen_epoch);
        }
      }
    }
    __syncthreads();
    cursor = (slot + 1) % b.slots;
  }
}

class PersistentExecutor {
public:
  std::shared_ptr<OwnedBuffers> owner;
  torch::Tensor w13, w2;
  cudaStream_t stream = nullptr, signal_stream = nullptr;
  int *host_stop = nullptr;
  U64 *host_control = nullptr;
  bool running = false;

  PersistentExecutor(std::shared_ptr<OwnedBuffers> buffers,
                     torch::Tensor gateup, torch::Tensor down,
                     int decode_budget, double expected_tiles,
                     double max_decode_fraction, int requested_ctas)
      : owner(std::move(buffers)), w13(std::move(gateup)), w2(std::move(down)) {
    TORCH_CHECK(!owner->active && !owner->started,
                "A buffer set admits one executor lifetime; allocate fresh "
                "buffers to restart");
    check_weight(w13, 4, "w13");
    check_weight(w2, 4, "w2");
    DeviceBuffers b = owner->b;
    TORCH_CHECK(w13.size(1) == b.experts && w13.size(2) == 2 * b.intermediate &&
                    w13.size(3) == b.hidden,
                "w13 must be [layers,local_experts,2*intermediate,hidden]");
    TORCH_CHECK(w2.size(0) == w13.size(0) && w2.size(1) == b.experts &&
                    w2.size(2) == b.hidden && w2.size(3) == b.intermediate,
                "w2 must be [layers,local_experts,hidden,intermediate]");
    TORCH_CHECK(w13.size(0) > 0 && w13.size(0) <= INT_MAX,
                "Invalid layer count");
    TORCH_CHECK(decode_budget > 0 &&
                    expected_tiles >= std::numeric_limits<float>::min() &&
                    expected_tiles <= std::numeric_limits<float>::max() &&
                    std::isfinite(expected_tiles),
                "Decode budget and expected tile count must be positive");
    TORCH_CHECK(
        max_decode_fraction > 0 && max_decode_fraction < 1 &&
            std::isfinite(max_decode_fraction),
        "max_decode_sm_fraction must lie strictly between zero and one");
    c10::cuda::CUDAGuard guard(owner->device);
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, owner->device));
    int ctas = requested_ctas == 0 ? prop.multiProcessorCount : requested_ctas;
    TORCH_CHECK(ctas >= 2 && ctas <= prop.multiProcessorCount,
                "CTA count must be between 2 and the number of SMs");
    int max_decode_ctas = std::min(
        ctas - 1, static_cast<int>(std::floor(ctas * max_decode_fraction)));
    TORCH_CHECK(max_decode_ctas >= 1,
                "Decode cap must admit at least one decode CTA");
    cudaFuncAttributes attrs;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, persistent_executor));
    int optin = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, owner->device));
    int reserve = prop.sharedMemPerMultiprocessor / 2 + 256;
    TORCH_CHECK(reserve + attrs.sharedSizeBytes <= static_cast<size_t>(optin),
                "Cannot reserve one persistent CTA per SM on this device");
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        persistent_executor, cudaFuncAttributeMaxDynamicSharedMemorySize,
        reserve));
    int resident = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, persistent_executor, 32, reserve));
    TORCH_CHECK(
        resident == 1,
        "Persistent scheduling requires exactly one resident CTA per SM");
    try {
      C10_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      C10_CUDA_CHECK(
          cudaStreamCreateWithFlags(&signal_stream, cudaStreamNonBlocking));
      C10_CUDA_CHECK(
          cudaHostAlloc(&host_stop, sizeof(int), cudaHostAllocDefault));
      C10_CUDA_CHECK(cudaHostAlloc(&host_control,
                                   b.slots * kControlWords * sizeof(U64),
                                   cudaHostAllocDefault));
      *host_stop = 0;
      // Weights may have been copied asynchronously on PyTorch's stream.
      cudaEvent_t ready;
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
      C10_CUDA_CHECK(cudaEventRecord(
          ready, at::cuda::getCurrentCUDAStream(owner->device)));
      C10_CUDA_CHECK(cudaStreamWaitEvent(stream, ready, 0));
      C10_CUDA_CHECK(cudaEventDestroy(ready));
      persistent_executor<<<ctas, 32, reserve, stream>>>(
          b, reinterpret_cast<const BF16 *>(w13.data_ptr<at::BFloat16>()),
          reinterpret_cast<const BF16 *>(w2.data_ptr<at::BFloat16>()),
          static_cast<int>(w13.size(0)), decode_budget,
          static_cast<float>(expected_tiles), max_decode_ctas);
      C10_CUDA_CHECK(cudaGetLastError());
      owner->active = true;
      owner->started = true;
      running = true;
    } catch (...) {
      release_noexcept();
      throw;
    }
  }
  ~PersistentExecutor() { release_noexcept(); }
  void check_weight(const torch::Tensor &t, int dims, const char *name) {
    TORCH_CHECK(t.is_cuda() && t.device().index() == owner->device &&
                    t.is_contiguous() && t.scalar_type() == at::kBFloat16 &&
                    t.dim() == dims,
                name, " must be a contiguous BF16 tensor on the expert GPU");
  }
  void stop() {
    if (!running)
      return;
    c10::cuda::CUDAGuard guard(owner->device);
    // A host-to-device copy uses a copy engine; a stop kernel could deadlock
    // behind an executor occupying every SM.
    *host_stop = 1;
    C10_CUDA_CHECK(cudaMemcpyAsync(owner->b.stop, host_stop, sizeof(int),
                                   cudaMemcpyHostToDevice, signal_stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(signal_stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    running = false;
    owner->active = false;
  }
  std::vector<int> check_error() {
    c10::cuda::CUDAGuard guard(owner->device);
    if (running) {
      cudaError_t result = cudaStreamQuery(stream);
      TORCH_CHECK(
          result == cudaErrorNotReady || result == cudaSuccess,
          "Persistent CUDA kernel failed: ", cudaGetErrorString(result));
      TORCH_CHECK(result == cudaErrorNotReady,
                  "Persistent CUDA kernel stopped unexpectedly");
    }
    C10_CUDA_CHECK(cudaMemcpyAsync(host_control, owner->b.control,
                                   owner->b.slots * kControlWords * sizeof(U64),
                                   cudaMemcpyDeviceToHost, signal_stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(signal_stream));
    std::vector<int> errors;
    for (int s = 0; s < owner->b.slots; ++s)
      errors.push_back(host_control[s * kControlWords + ERROR]);
    return errors;
  }
  void release_noexcept() noexcept {
    if (!stream && !signal_stream && !host_stop && !host_control)
      return;
    int old = 0;
    cudaGetDevice(&old);
    cudaSetDevice(owner->device);
    if (running) {
      *host_stop = 1;
      cudaMemcpyAsync(owner->b.stop, host_stop, sizeof(int),
                      cudaMemcpyHostToDevice, signal_stream);
      cudaStreamSynchronize(signal_stream);
      cudaStreamSynchronize(stream);
      running = false;
      owner->active = false;
    }
    if (stream)
      cudaStreamDestroy(stream);
    if (signal_stream)
      cudaStreamDestroy(signal_stream);
    if (host_stop)
      cudaFreeHost(host_stop);
    if (host_control)
      cudaFreeHost(host_control);
    stream = nullptr;
    signal_stream = nullptr;
    host_stop = nullptr;
    host_control = nullptr;
    cudaSetDevice(old);
  }
};

static void check_tensor(const torch::Tensor &t, const RemoteBuffers &r,
                         at::ScalarType dtype, int dim, const char *name) {
  TORCH_CHECK(!r.closed, "IPC mapping has been closed");
  TORCH_CHECK(t.is_cuda() && t.device().index() == r.device &&
                  t.is_contiguous() && t.scalar_type() == dtype &&
                  t.dim() == dim,
              name, " has wrong device, dtype, rank, or contiguity");
}
static void check_routes(const torch::Tensor &ids, const torch::Tensor &rows,
                         const RemoteBuffers &r, int rank, int world) {
  TORCH_CHECK(ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong,
              "topk_ids must be int32 or int64");
  check_tensor(ids, r, ids.scalar_type(), 2, "topk_ids");
  check_tensor(rows, r, at::kInt, 2, "row_map");
  TORCH_CHECK(ids.sizes() == rows.sizes(), "Routing/row_map shape mismatch");
  TORCH_CHECK(world > 0 && rank >= 0 && rank < world &&
                  static_cast<int64_t>(r.b.experts) * world <= INT_MAX,
              "Invalid expert rank/world");
  TORCH_CHECK(ids.size(0) <= INT_MAX && ids.size(1) > 0 &&
                  ids.numel() <= INT_MAX,
              "Invalid routing shape");
}

static void dispatch(std::shared_ptr<RemoteBuffers> r, torch::Tensor hidden,
                     torch::Tensor ids, torch::Tensor rows, int64_t epoch,
                     int layer, int phase, int rank, int world) {
  check_tensor(hidden, *r, at::kBFloat16, 2, "hidden");
  check_routes(ids, rows, *r, rank, world);
  TORCH_CHECK(hidden.size(0) == ids.size(0) && hidden.size(1) == r->b.hidden,
              "Hidden/routing shape mismatch");
  TORCH_CHECK(epoch > 0 && static_cast<U64>(epoch) <= kMaxEpoch,
              "Epoch must be in [1,2**30)");
  TORCH_CHECK(layer >= 0 && phase >= 0 && phase <= 1, "Invalid layer/phase");
  c10::cuda::CUDAGuard guard(r->device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(r->device);
  clear_slot<<<1, 32, 0, stream>>>(r->b, r->slot, epoch, layer, phase);
  int routes = ids.numel();
  if (routes > 0) {
    if (ids.scalar_type() == at::kInt)
      dispatch_rows<<<routes, 128, 0, stream>>>(
          r->b, r->slot,
          reinterpret_cast<const BF16 *>(hidden.data_ptr<at::BFloat16>()),
          ids.data_ptr<int>(), rows.data_ptr<int>(), ids.size(0), ids.size(1),
          rank, world);
    else
      dispatch_rows<<<routes, 128, 0, stream>>>(
          r->b, r->slot,
          reinterpret_cast<const BF16 *>(hidden.data_ptr<at::BFloat16>()),
          ids.data_ptr<int64_t>(), rows.data_ptr<int>(), ids.size(0),
          ids.size(1), rank, world);
  }
  publish_slot<<<1, 1, 0, stream>>>(r->b, r->slot, epoch);
  C10_CUDA_CHECK(cudaGetLastError());
}

static void combine(std::shared_ptr<RemoteBuffers> r, torch::Tensor output,
                    torch::Tensor ids, torch::Tensor weights,
                    torch::Tensor rows, int64_t epoch, int rank, int world,
                    int timeout_ms) {
  check_tensor(output, *r, at::kFloat, 2, "output");
  check_routes(ids, rows, *r, rank, world);
  check_tensor(weights, *r, at::kFloat, 2, "topk_weights");
  TORCH_CHECK(output.size(0) == ids.size(0) && output.size(1) == r->b.hidden &&
                  weights.sizes() == ids.sizes(),
              "Output/routing shape mismatch");
  TORCH_CHECK(epoch > 0 && static_cast<U64>(epoch) <= kMaxEpoch &&
                  timeout_ms > 0 && timeout_ms <= 3600000,
              "Invalid epoch or timeout (1..3600000 ms)");
  c10::cuda::CUDAGuard guard(r->device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(r->device);
  wait_slot<<<1, 1, 0, stream>>>(r->b, r->slot, epoch,
                                 static_cast<U64>(timeout_ms) * 1000000ULL,
                                 r->status);
  int blocks = std::min<int64_t>(65535, (output.numel() + 255) / 256);
  if (blocks > 0) {
    if (ids.scalar_type() == at::kInt)
      combine_rows<<<blocks, 256, 0, stream>>>(
          r->b, r->slot, output.data_ptr<float>(), ids.data_ptr<int>(),
          weights.data_ptr<float>(), rows.data_ptr<int>(), ids.size(0),
          ids.size(1), rank, world, r->status);
    else
      combine_rows<<<blocks, 256, 0, stream>>>(
          r->b, r->slot, output.data_ptr<float>(), ids.data_ptr<int64_t>(),
          weights.data_ptr<float>(), rows.data_ptr<int>(), ids.size(0),
          ids.size(1), rank, world, r->status);
  }
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaMemcpyAsync(r->host_status, r->status, sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  TORCH_CHECK(*r->host_status == OK, "ExpertPlex request failed (native code ",
              *r->host_status,
              "; 1=busy, 2=epoch, 3=route, 4=capacity, 5=layer, 6=phase, "
              "7=timeout, 8=stopped)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<OwnedBuffers, std::shared_ptr<OwnedBuffers>>(m, "OwnedBuffers")
      .def(py::init<int, int, int, int, int, int>())
      .def("descriptor", &OwnedBuffers::descriptor);
  py::class_<RemoteBuffers, std::shared_ptr<RemoteBuffers>>(m, "RemoteBuffers")
      .def(py::init<py::dict, int, int>())
      .def("close", &RemoteBuffers::close);
  py::class_<PersistentExecutor, std::shared_ptr<PersistentExecutor>>(
      m, "PersistentExecutor")
      .def(py::init<std::shared_ptr<OwnedBuffers>, torch::Tensor, torch::Tensor,
                    int, double, double, int>())
      .def("stop", &PersistentExecutor::stop,
           py::call_guard<py::gil_scoped_release>())
      .def("check_error", &PersistentExecutor::check_error,
           py::call_guard<py::gil_scoped_release>());
  m.def("dispatch", &dispatch);
  m.def("combine", &combine, py::call_guard<py::gil_scoped_release>());
}
