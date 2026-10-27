// ExpertPlex B300 IPC protocol, reference WMMA and native SM100 executors.
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
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;
using U64 = unsigned long long;
using BF16 = __nv_bfloat16;
constexpr int kControlWords = 8;
constexpr int kMaxSlots = 32;
enum Control {
  READY = 0,
  DONE = 1,
  LAYER = 2,
  PHASE = 3,
  ERROR = 4,
  PENDING = 5,
  START_NS = 6,
  FINISH_NS = 7
};
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
enum Stage {
  IDLE = 0,
  GATE = 1,
  DOWN = 2,
  ACTIVATE = 3,
  INITIALIZING = 4,
  SCATTER1 = 5,
  SCATTER2 = 6,
  GATHER = 7
};
constexpr U64 kMaxEpoch = (1ULL << 29) - 1;

struct SlotState {
  U64 queue; // epoch[63:35], stage[34:32], next task[31:0]
  unsigned int completed;
  unsigned int gate_tiles;
  unsigned int down_tiles;
  unsigned int activation_tiles;
  // Keep per-operation bounds immutable for the entire invocation. An idle
  // claimant may still hold the retiring stage generation while its last CTA
  // publishes the next one; reusing a tile bound here admits phantom tasks.
  unsigned int scatter1_tiles, scatter2_tiles, gather_tiles;
  int phase;
  int layer;
};
static_assert(sizeof(SlotState) == 48,
              "Update the Python memory estimator for changed state layout");

struct DeviceBuffers {
  BF16 *input, *activation, *output, *gateup;
  int *counts, *offsets;
  BF16 *raw_input, *raw_output;
  int *raw_ids, *raw_map, *raw_n;
  float *raw_weights;
  size_t raw_base[kMaxSlots];
  int top_k, expert_rank, expert_world, prefill_preprocess;
  U64 *control;
  SlotState *states;
  int *stop;
  int slots, experts, rows, hidden, intermediate;
  int capacity[kMaxSlots], packed[kMaxSlots], storage_rows[kMaxSlots];
  size_t base_rows[kMaxSlots];
};

__device__ __forceinline__ int uncached_load_int(const int *p);

__device__ __forceinline__ size_t expert_base(const DeviceBuffers &b, int slot,
                                              int expert) {
  return b.base_rows[slot] +
         (b.packed[slot]
              ? uncached_load_int(b.offsets + slot * b.experts + expert)
              : static_cast<size_t>(expert) *
                    ((b.capacity[slot] + 127) / 128 * 128));
}

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
  std::string input_handle, output_handle, counts_handle, control_handle,
      offsets_handle;
  std::string raw_input_handle, raw_output_handle, raw_ids_handle,
      raw_weights_handle, raw_n_handle;

  OwnedBuffers(int slots, int experts, int rows, int hidden, int intermediate,
               int dev, std::vector<int> capacities, std::vector<int> packed,
               int top_k, bool prefill_preprocess, int expert_rank,
               int expert_world)
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
    TORCH_CHECK(expert_world > 0 && expert_rank >= 0 &&
                    expert_rank < expert_world,
                "Invalid expert rank");
    TORCH_CHECK(!prefill_preprocess || top_k > 0,
                "Preprocessing requires top_k");
    b.top_k = top_k;
    b.expert_rank = expert_rank;
    b.expert_world = expert_world;
    b.prefill_preprocess = prefill_preprocess;
    b.slots = slots;
    b.experts = experts;
    b.rows = rows;
    b.hidden = hidden;
    b.intermediate = intermediate;
    TORCH_CHECK(slots <= kMaxSlots, "Too many mailboxes");
    if (capacities.empty())
      capacities.assign(slots, rows);
    if (packed.empty())
      packed.assign(slots, 0);
    TORCH_CHECK(capacities.size() == slots && packed.size() == slots,
                "Mailbox layout length mismatch");
    size_t total_rows = 0, raw_rows = 0;
    for (int s = 0; s < slots; ++s) {
      TORCH_CHECK(capacities[s] > 0 && capacities[s] <= rows &&
                      (packed[s] == 0 || packed[s] == 1),
                  "Invalid mailbox layout");
      TORCH_CHECK(!packed[s] || top_k > 0, "Packed layout requires top_k");
      int64_t storage =
          packed[s]
              ? static_cast<int64_t>(capacities[s]) * std::min(top_k, experts) +
                    127LL * experts
              : static_cast<int64_t>((capacities[s] + 127) / 128 * 128) *
                    experts;
      // Descriptor dimensions must admit a full 128-row zero-padded tile.
      storage = (storage + 127) / 128 * 128;
      TORCH_CHECK(storage <= INT_MAX, "Mailbox exceeds TMA row indexing range");
      b.capacity[s] = capacities[s];
      b.packed[s] = packed[s];
      b.storage_rows[s] = storage;
      b.base_rows[s] = total_rows;
      total_rows += storage;
      b.raw_base[s] = raw_rows;
      if (prefill_preprocess && packed[s])
        raw_rows += capacities[s];
    }
    try {
      if (raw_rows) {
        b.raw_input = static_cast<BF16 *>(allocate(
            checked_product({static_cast<int64_t>(raw_rows), hidden, 2})));
        b.raw_output = static_cast<BF16 *>(allocate(
            checked_product({static_cast<int64_t>(raw_rows), hidden, 2})));
        b.raw_ids = static_cast<int *>(allocate(
            checked_product({static_cast<int64_t>(raw_rows), top_k, 4})));
        b.raw_map = static_cast<int *>(allocate(
            checked_product({static_cast<int64_t>(raw_rows), top_k, 4})));
        b.raw_weights = static_cast<float *>(allocate(
            checked_product({static_cast<int64_t>(raw_rows), top_k, 4})));
        b.raw_n = static_cast<int *>(allocate(checked_product({slots, 4})));
        raw_input_handle = encode_handle(b.raw_input);
        raw_output_handle = encode_handle(b.raw_output);
        raw_ids_handle = encode_handle(b.raw_ids);
        raw_weights_handle = encode_handle(b.raw_weights);
        raw_n_handle = encode_handle(b.raw_n);
      }
      b.input = static_cast<BF16 *>(allocate(
          checked_product({static_cast<int64_t>(total_rows), hidden, 2})));
      b.gateup = static_cast<BF16 *>(allocate(checked_product(
          {static_cast<int64_t>(total_rows), intermediate, 4})));
      b.activation = static_cast<BF16 *>(allocate(checked_product(
          {static_cast<int64_t>(total_rows), intermediate, 2})));
      b.output = static_cast<BF16 *>(allocate(
          checked_product({static_cast<int64_t>(total_rows), hidden, 2})));
      b.counts =
          static_cast<int *>(allocate(checked_product({slots, experts, 4})));
      b.offsets =
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
      offsets_handle = encode_handle(b.offsets);
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
    d["version"] = 5;
    d["prefill_preprocess"] = bool(b.prefill_preprocess);
    d["top_k"] = b.top_k;
    d["expert_rank"] = b.expert_rank;
    d["expert_world"] = b.expert_world;
    d["raw_base"] = std::vector<size_t>(b.raw_base, b.raw_base + b.slots);
    if (b.raw_input) {
      d["raw_input_handle"] = raw_input_handle;
      d["raw_output_handle"] = raw_output_handle;
      d["raw_ids_handle"] = raw_ids_handle;
      d["raw_weights_handle"] = raw_weights_handle;
      d["raw_n_handle"] = raw_n_handle;
    }
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
    d["offsets_handle"] = offsets_handle;
    d["capacities"] = std::vector<int>(b.capacity, b.capacity + b.slots);
    d["packed"] = std::vector<int>(b.packed, b.packed + b.slots);
    d["storage_rows"] =
        std::vector<int>(b.storage_rows, b.storage_rows + b.slots);
    d["base_rows"] = std::vector<size_t>(b.base_rows, b.base_rows + b.slots);
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
  int *local_counts = nullptr, *local_offsets = nullptr;
  bool closed = false;

  RemoteBuffers(py::dict d, int slot_id, int dev) : device(dev), slot(slot_id) {
    TORCH_CHECK(py::cast<int>(d["version"]) == 5,
                "Unsupported IPC descriptor version");
    b.slots = py::cast<int>(d["slot_count"]);
    b.experts = py::cast<int>(d["local_experts"]);
    b.rows = py::cast<int>(d["max_rows"]);
    b.hidden = py::cast<int>(d["hidden"]);
    b.intermediate = py::cast<int>(d["intermediate"]);
    TORCH_CHECK(b.slots > 0 && b.slots <= kMaxSlots, "Invalid slot count");
    b.prefill_preprocess = py::cast<bool>(d["prefill_preprocess"]);
    b.top_k = py::cast<int>(d["top_k"]);
    b.expert_rank = py::cast<int>(d["expert_rank"]);
    b.expert_world = py::cast<int>(d["expert_world"]);
    auto raw_bases = py::cast<std::vector<size_t>>(d["raw_base"]);
    TORCH_CHECK(raw_bases.size() == b.slots, "Malformed raw mailbox layout");
    std::copy(raw_bases.begin(), raw_bases.end(), b.raw_base);
    auto capacities = py::cast<std::vector<int>>(d["capacities"]);
    auto packed = py::cast<std::vector<int>>(d["packed"]);
    auto storage = py::cast<std::vector<int>>(d["storage_rows"]);
    auto bases = py::cast<std::vector<size_t>>(d["base_rows"]);
    TORCH_CHECK(capacities.size() == b.slots && packed.size() == b.slots &&
                    storage.size() == b.slots && bases.size() == b.slots,
                "Malformed mailbox layout");
    std::copy(capacities.begin(), capacities.end(), b.capacity);
    std::copy(packed.begin(), packed.end(), b.packed);
    std::copy(storage.begin(), storage.end(), b.storage_rows);
    std::copy(bases.begin(), bases.end(), b.base_rows);
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
      if (b.prefill_preprocess && b.packed[slot]) {
        b.raw_input = static_cast<BF16 *>(
            map(py::cast<std::string>(d["raw_input_handle"])));
        b.raw_output = static_cast<BF16 *>(
            map(py::cast<std::string>(d["raw_output_handle"])));
        b.raw_ids =
            static_cast<int *>(map(py::cast<std::string>(d["raw_ids_handle"])));
        b.raw_weights = static_cast<float *>(
            map(py::cast<std::string>(d["raw_weights_handle"])));
        b.raw_n =
            static_cast<int *>(map(py::cast<std::string>(d["raw_n_handle"])));
      }
      b.input =
          static_cast<BF16 *>(map(py::cast<std::string>(d["input_handle"])));
      b.output =
          static_cast<BF16 *>(map(py::cast<std::string>(d["output_handle"])));
      b.counts =
          static_cast<int *>(map(py::cast<std::string>(d["counts_handle"])));
      b.offsets =
          static_cast<int *>(map(py::cast<std::string>(d["offsets_handle"])));
      b.control =
          static_cast<U64 *>(map(py::cast<std::string>(d["control_handle"])));
      C10_CUDA_CHECK(cudaMalloc(&status, sizeof(int)));
      const size_t metadata_bytes = b.slots * b.experts * sizeof(int);
      C10_CUDA_CHECK(cudaMalloc(&local_counts, metadata_bytes));
      C10_CUDA_CHECK(cudaMalloc(&local_offsets, metadata_bytes));
      C10_CUDA_CHECK(cudaMemset(local_offsets, 0, metadata_bytes));
      C10_CUDA_CHECK(
          cudaHostAlloc(&host_status, sizeof(int), cudaHostAllocDefault));
    } catch (...) {
      close_noexcept();
      throw;
    }
  }
  ~RemoteBuffers() { close_noexcept(); }
  DeviceBuffers attention_metadata() const {
    auto transfer = b;
    transfer.counts = local_counts;
    transfer.offsets = local_offsets;
    return transfer;
  }
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
    if (local_counts)
      cudaFree(local_counts);
    if (local_offsets)
      cudaFree(local_offsets);
    if (host_status)
      cudaFreeHost(host_status);
    status = nullptr;
    local_counts = local_offsets = nullptr;
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
  return (epoch << 35) | (static_cast<U64>(stage) << 32) | task;
}
__device__ __forceinline__ U64 now_ns() {
  U64 value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ void clear_slot_body(const DeviceBuffers &b,
                                                int slot, U64 epoch, int layer,
                                                int phase) {
  U64 *c = b.control + static_cast<size_t>(slot) * kControlWords;
  if (threadIdx.x == 0) {
    U64 ready = system_load(c + READY), done = system_load(c + DONE);
    if (epoch == 0)
      epoch = ready + 1; // Recomputed on every CUDA Graph replay.
    if (system_load(c + ERROR)) {
      // Keep a failed slot poisoned until the deployment is restarted.
    } else if (ready != done)
      set_error(c, BUSY);
    else if (epoch <= ready || epoch > kMaxEpoch)
      set_error(c, BAD_EPOCH);
    else {
      c[LAYER] = layer;
      c[PHASE] = phase;
      c[PENDING] = epoch;
      c[START_NS] = 0;
      c[FINISH_NS] = 0;
    }
  }
  __syncthreads();
  if (system_load(c + ERROR))
    return;
  for (int e = threadIdx.x; e < b.experts; e += blockDim.x)
    b.counts[static_cast<size_t>(slot) * b.experts + e] = 0;
  __threadfence_system();
}
__global__ void clear_slot(const __grid_constant__ DeviceBuffers b, int slot,
                           U64 epoch, int layer, int phase) {
  clear_slot_body(b, slot, epoch, layer, phase);
}

template <typename Id>
__device__ __forceinline__ void
count_routes_body(const DeviceBuffers &b, int slot, const Id *ids, int *row_map,
                  int routes, int rank, int world) {
  auto *c = b.control + static_cast<size_t>(slot) * kControlWords;
  if (system_load(c + ERROR))
    return;
  for (int route = blockIdx.x * blockDim.x + threadIdx.x; route < routes;
       route += gridDim.x * blockDim.x) {
    Id expert = ids[route];
    // SGLang marks CUDA Graph/TBO padding routes with -1.
    if (expert == -1) {
      if (rank == 0)
        row_map[route] = -1;
      continue;
    }
    if (expert < 0 || expert >= static_cast<Id>(b.experts) * world) {
      set_error(c, BAD_ROUTE);
      continue;
    }
    if (expert % world != rank)
      continue;
    int row = atomicAdd(b.counts + slot * b.experts + expert / world, 1);
    row_map[route] = row;
    if (row >= b.capacity[slot])
      set_error(c, ROW_OVERFLOW);
  }
  __threadfence_system();
}
template <typename Id>
__global__ void count_routes(const __grid_constant__ DeviceBuffers b, int slot,
                             const Id *ids, int *row_map, int routes, int rank,
                             int world) {
  count_routes_body(b, slot, ids, row_map, routes, rank, world);
}

// Runs on the attention GPU, before payload copies. No expert receive kernel.
__device__ __forceinline__ void pack_offsets_body(const DeviceBuffers &b,
                                                  int slot) {
  int offset = 0;
  for (int e = 0; e < b.experts; ++e) {
    b.offsets[slot * b.experts + e] = offset;
    int count = uncached_load_int(b.counts + slot * b.experts + e);
    offset += (count + 127) / 128 * 128;
  }
  if (offset > b.storage_rows[slot])
    set_error(b.control + slot * kControlWords, ROW_OVERFLOW);
  __threadfence_system();
}
__global__ void pack_offsets(const __grid_constant__ DeviceBuffers b,
                             int slot) {
  pack_offsets_body(b, slot);
}

template <typename Id>
__device__ __forceinline__ void
dispatch_rows_body(const DeviceBuffers &b, int slot, const BF16 *hidden,
                   const Id *ids, int *row_map, int n, int k, int rank,
                   int world) {
  U64 *c = b.control + static_cast<size_t>(slot) * kControlWords;
  int route = blockIdx.x;
  Id routed_expert = ids[route];
  if (routed_expert == -1) {
    if (threadIdx.x == 0)
      row_map[route] = -1;
    return;
  }
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
    row =
        system_load(c + ERROR) ? -1
        : b.packed[slot]
            ? row_map[route]
            : atomicAdd(
                  b.counts + static_cast<size_t>(slot) * b.experts + local, 1);
    row_map[route] = row;
    if (row >= b.capacity[slot])
      set_error(c, ROW_OVERFLOW);
  }
  __syncthreads();
  if (row < 0 || row >= b.capacity[slot])
    return;
  size_t base = (expert_base(b, slot, local) + row) * b.hidden;
  int token = route / k;
  auto *destination = reinterpret_cast<uint4 *>(b.input + base);
  auto *source = reinterpret_cast<const uint4 *>(
      hidden + static_cast<size_t>(token) * b.hidden);
  for (int h = threadIdx.x; h < b.hidden / 8; h += blockDim.x)
    destination[h] = source[h];
  // Every writer, rather than only the publishing thread, fences its peer data.
  __threadfence_system();
}
template <typename Id>
__global__ void dispatch_rows(const __grid_constant__ DeviceBuffers b, int slot,
                              const BF16 *hidden, const Id *ids, int *row_map,
                              int n, int k, int rank, int world) {
  dispatch_rows_body(b, slot, hidden, ids, row_map, n, k, rank, world);
}

// A mailbox has exactly one attention writer. Route counting and prefix sums
// therefore use attention-local atomics; only the final metadata crosses
// NVLink. Payloads already land in the final expert compute buffers.
__device__ __forceinline__ void push_metadata_body(const DeviceBuffers &peer,
                                                   const DeviceBuffers &local,
                                                   int slot) {
  if (system_load(peer.control + slot * kControlWords + ERROR))
    return;
  for (int e = threadIdx.x; e < peer.experts; e += blockDim.x) {
    int index = slot * peer.experts + e;
    peer.counts[index] = local.counts[index];
    if (peer.packed[slot])
      peer.offsets[index] = local.offsets[index];
  }
  __threadfence_system();
}
__global__ void push_metadata(const __grid_constant__ DeviceBuffers peer,
                              const __grid_constant__ DeviceBuffers local,
                              int slot) {
  push_metadata_body(peer, local, slot);
}

__global__ void publish_slot(const __grid_constant__ DeviceBuffers b, int slot,
                             U64 epoch) {
  U64 *c = b.control + static_cast<size_t>(slot) * kControlWords;
  // A failed clear must not replace the READY epoch of an active invocation.
  // Errors are fail-closed; the worker/client report them and must restart.
  if (system_load(c + ERROR))
    return;
  __threadfence_system();
  system_store(c + READY, epoch ? epoch : uncached_load_u64(c + PENDING));
}

__device__ __forceinline__ void wait_slot_body(const DeviceBuffers &b, int slot,
                                               U64 epoch, U64 timeout_ns,
                                               int *status) {
  U64 *c = b.control + static_cast<size_t>(slot) * kControlWords;
  if (epoch == 0)
    epoch = uncached_load_u64(c + PENDING);
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
__global__ void wait_slot(const __grid_constant__ DeviceBuffers b, int slot,
                          U64 epoch, U64 timeout_ns, int *status) {
  wait_slot_body(b, slot, epoch, timeout_ns, status);
}

template <typename Id>
__device__ __forceinline__ void
combine_rows_body(const DeviceBuffers &b, int slot, float *output,
                  const Id *ids, const float *weights, const int *row_map,
                  int n, int k, int rank, int world, const int *status) {
  if (*status != OK)
    return;
  const int vectors = b.hidden / 8;
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < static_cast<size_t>(n) * vectors; i += stride) {
    int token = i / vectors, h = (i % vectors) * 8;
    float sum[8] = {};
    for (int j = 0; j < k; ++j) {
      size_t route = static_cast<size_t>(token) * k + j;
      int expert = static_cast<int>(ids[route]);
      if (expert < 0 || expert >= b.experts * world || expert % world != rank)
        continue;
      int row = row_map[route];
      if (row < 0 || row >= b.capacity[slot])
        continue;
      size_t base = (expert_base(b, slot, expert / world) + row) * b.hidden + h;
      uint32_t words[4];
      asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]),
                     "=r"(words[3])
                   : "l"(b.output + base)
                   : "memory");
      float weight = weights[route];
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        float low = __bfloat162float(__ushort_as_bfloat16(words[pair] & 65535));
        float high = __bfloat162float(__ushort_as_bfloat16(words[pair] >> 16));
        sum[2 * pair] += weight * low;
        sum[2 * pair + 1] += weight * high;
      }
    }
    auto *destination = reinterpret_cast<float4 *>(output + i * 8);
    float4 first = destination[0], second = destination[1];
    first.x += sum[0];
    first.y += sum[1];
    first.z += sum[2];
    first.w += sum[3];
    second.x += sum[4];
    second.y += sum[5];
    second.z += sum[6];
    second.w += sum[7];
    destination[0] = first;
    destination[1] = second;
  }
}
template <typename Id>
__global__ void combine_rows(const __grid_constant__ DeviceBuffers b, int slot,
                             float *output, const Id *ids, const float *weights,
                             const int *row_map, int n, int k, int rank,
                             int world, const int *status) {
  combine_rows_body(b, slot, output, ids, weights, row_map, n, k, rank, world,
                    status);
}

// Four warp tensor-core tile, with 128-wide asynchronous global/shared copies.
// All copy groups and matrix accumulators are committed before yielding a tile.
__device__ __forceinline__ void copy16(BF16 *dst, const BF16 *src, bool valid) {
  unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  int bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address),
               "l"(src), "r"(bytes)
               : "memory");
}

template <int M>
__device__ void compute_tile_wide(DeviceBuffers b, const BF16 *w13,
                                  const BF16 *w2, int slot, int layer,
                                  Stage stage, unsigned task, BF16 *sa,
                                  BF16 *sg, BF16 *su, float *cg, float *cu) {
  constexpr int N = 128, K = 128, S = 144;
  int out = stage == GATE ? b.intermediate : b.hidden;
  int width = stage == GATE ? b.hidden : b.intermediate;
  int ntile = (out + N - 1) / N;
  int expert = 0, mtile = 0, column = 0, count = 0;
  for (; expert < b.experts; ++expert) {
    count = uncached_load_int(b.counts + static_cast<size_t>(slot) * b.experts +
                              expert);
    unsigned tiles =
        static_cast<unsigned>((static_cast<U64>(count) + M - 1) / M * ntile);
    if (task < tiles) {
      mtile = task / ntile;
      column = task % ntile;
      break;
    }
    task -= tiles;
  }
  if (expert == b.experts)
    return;
  const BF16 *input = (stage == GATE ? b.input : b.activation) +
                      expert_base(b, slot, expert) * width;
  const BF16 *weight =
      stage == GATE ? w13 + ((static_cast<size_t>(layer) * b.experts + expert) *
                             2 * b.intermediate) *
                                b.hidden
                    : w2 + ((static_cast<size_t>(layer) * b.experts + expert) *
                            b.hidden) *
                               b.intermediate;
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, BF16, wmma::row_major> a[M / 16];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, BF16, wmma::col_major> g, u;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> gate[M / 16][2],
      up[M / 16][2];
#pragma unroll
  for (int m = 0; m < M / 16; ++m) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      wmma::fill_fragment(gate[m][n], 0.f);
      wmma::fill_fragment(up[m][n], 0.f);
    }
  }
  int warp = threadIdx.x / 32;
  auto prefetch = [&](int offset, int buffer) {
    BF16 *aa = sa + buffer * 64 * S, *gg = sg + buffer * N * S,
         *uu = su + buffer * N * S;
    for (int v = threadIdx.x; v < M * K / 8; v += blockDim.x) {
      int row = v * 8 / K, kk = v * 8 % K;
      bool valid = mtile * M + row < count && offset + kk < width;
      copy16(aa + row * S + kk,
             input + static_cast<size_t>(valid ? mtile * M + row : 0) * width +
                 offset + kk,
             valid);
    }
    for (int v = threadIdx.x; v < N * K / 8; v += blockDim.x) {
      int col = v * 8 / K, kk = v * 8 % K;
      bool valid = column * N + col < out && offset + kk < width;
      const BF16 *src =
          weight + static_cast<size_t>(valid ? column * N + col : 0) * width +
          offset + kk;
      copy16(gg + col * S + kk, src, valid);
      if (stage == GATE)
        copy16(uu + col * S + kk,
               src + static_cast<size_t>(b.intermediate) * width, valid);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  prefetch(0, 0);
  for (int offset = 0, step = 0; offset < width; offset += K, ++step) {
    int buffer = step & 1;
    if (offset + K < width) {
      prefetch(offset + K, 1 - buffer);
      asm volatile("cp.async.wait_group 1;" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();
    BF16 *aa = sa + buffer * 64 * S, *gg = sg + buffer * N * S,
         *uu = su + buffer * N * S;
#pragma unroll
    for (int kk = 0; kk < K; kk += 16) {
#pragma unroll
      for (int m = 0; m < M / 16; ++m)
        wmma::load_matrix_sync(a[m], aa + m * 16 * S + kk, S);
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        wmma::load_matrix_sync(g, gg + (warp * 32 + n * 16) * S + kk, S);
        if (stage == GATE)
          wmma::load_matrix_sync(u, uu + (warp * 32 + n * 16) * S + kk, S);
#pragma unroll
        for (int m = 0; m < M / 16; ++m) {
          wmma::mma_sync(gate[m][n], a[m], g, gate[m][n]);
          if (stage == GATE)
            wmma::mma_sync(up[m][n], a[m], u, up[m][n]);
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int m = 0; m < M / 16; ++m) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      wmma::store_matrix_sync(cg + m * 16 * N + warp * 32 + n * 16, gate[m][n],
                              N, wmma::mem_row_major);
      if (stage == GATE)
        wmma::store_matrix_sync(cu + m * 16 * N + warp * 32 + n * 16, up[m][n],
                                N, wmma::mem_row_major);
    }
  }
  __syncthreads();
  BF16 *destination = (stage == GATE ? b.activation : b.output) +
                      expert_base(b, slot, expert) * out;
  for (int i = threadIdx.x; i < M * N; i += blockDim.x) {
    int row = mtile * M + i / N, col = column * N + i % N;
    if (row < count && col < out) {
      float value = cg[i];
      if (stage == GATE) {
        value = __bfloat162float(__float2bfloat16(value));
        float up = __bfloat162float(__float2bfloat16(cu[i]));
        value = (value / (1.f + expf(-value))) * up;
      }
      destination[static_cast<size_t>(row) * out + col] =
          __float2bfloat16(value);
    }
  }
  __syncthreads();
}

__device__ void initialize_slot(DeviceBuffers b, int slot, int layers) {
  U64 *c = b.control + static_cast<size_t>(slot) * kControlWords;
  SlotState *s = b.states + slot;
  U64 q = queue_load(&s->queue);
  if (((q >> 32) & 7) != IDLE)
    return;
  U64 epoch = system_load(c + READY);
  if (epoch <= (q >> 35))
    return;
  if (!queue_cas(&s->queue, q, queue_value(epoch, INITIALIZING)))
    return;
  system_store(c + START_NS, now_ns());
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
    if (count < 0 || count > b.capacity[slot])
      set_error(c, ROW_OVERFLOW);
    else
      rows += (static_cast<U64>(count) + (s->phase == 0 ? 63 : 15)) /
              (s->phase == 0 ? 64 : 16);
  }
  U64 gate_tiles = rows * ((b.intermediate + 127) / 128),
      down_tiles = rows * ((b.hidden + 127) / 128);
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
    c[FINISH_NS] = now_ns();
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
    Stage st = static_cast<Stage>((q >> 32) & 7);
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
    epoch = q >> 35;
    return true;
  }
  return false;
}

__global__ void persistent_executor(const __grid_constant__ DeviceBuffers b,
                                    const BF16 *w13, const BF16 *w2, int layers,
                                    int decode_budget, float expected_tiles,
                                    int max_decode_ctas) {
  extern __shared__ __align__(128) unsigned char workspace[];
  BF16 *sa = reinterpret_cast<BF16 *>(workspace);
  BF16 *sg = sa + 2 * 64 * 144;
  BF16 *su = sg + 2 * 128 * 144;
  float *cg = reinterpret_cast<float *>(workspace);
  float *cu = cg + 64 * 128;
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
          int st = (q >> 32) & 7;
          if ((st == GATE || st == DOWN) && s->phase == 1)
            decode_tiles += st == GATE
                                ? s->gate_tiles / ((b.intermediate + 127) / 128)
                                : s->down_tiles / ((b.hidden + 127) / 128);
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
    if (state->phase == 0)
      compute_tile_wide<64>(b, w13, w2, slot, state->layer, stage, chosen_task,
                            sa, sg, su, cg, cu);
    else
      compute_tile_wide<16>(b, w13, w2, slot, state->layer, stage, chosen_task,
                            sa, sg, su, cg, cu);
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
          b.control[static_cast<size_t>(slot) * kControlWords + FINISH_NS] =
              now_ns();
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

#include "sm100_pipeline.cuh"

class PersistentExecutor {
public:
  std::shared_ptr<OwnedBuffers> owner;
  torch::Tensor w13, w2;
  cudaStream_t stream = nullptr, signal_stream = nullptr;
  int *host_stop = nullptr;
  U64 *host_control = nullptr;
  bool running = false;
  cudaFuncAttributes attributes{};
  int launch_ctas = 0, launch_threads = 0, dynamic_shared = 0, cluster_size = 1;
  std::string backend_name;
  expertplex_sm100::Context native_context{};

  PersistentExecutor(std::shared_ptr<OwnedBuffers> buffers,
                     torch::Tensor gateup, torch::Tensor down,
                     int decode_budget, double expected_tiles,
                     double max_decode_fraction, int requested_ctas,
                     const std::string &backend, int trace_capacity,
                     int benchmark_stage)
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
    TORCH_CHECK(!owner->b.prefill_preprocess ||
                    (backend == "sm100" && benchmark_stage == 0),
                "Expert-side preprocessing requires the complete native SM100 "
                "pipeline");
    const bool fused = backend == "sm100_fused";
    const bool sm100 = backend == "sm100" || backend == "sm100_cta1" || fused;
    const int cluster = backend == "sm100_cta1" ? 1 : 2;
    TORCH_CHECK(backend == "wmma" || sm100, "Unknown ExpertPlex kernel");
    TORCH_CHECK(benchmark_stage == 0 ||
                    (sm100 && !fused &&
                     (benchmark_stage == GATE || benchmark_stage == DOWN)),
                "Single-operation benchmarks require an unfused SM100 kernel");
    TORCH_CHECK(ctas % cluster == 0 && ctas >= 2 * cluster,
                "CTA count must fit at least two complete clusters");
    max_decode_ctas =
        std::min(ctas - cluster, max_decode_ctas / cluster * cluster);
    TORCH_CHECK(max_decode_ctas >= cluster, "Decode cap must fit one cluster");
    // Specialize the two target models at compile time, retaining a generic
    // shape only for small synthetic debugging checkpoints.
    auto with_native_kernel = [&](auto &&fn) {
      if (fused) {
        if (b.hidden == 6144 && b.intermediate == 2560)
          return fn(expertplex_sm100::executor<2, 6144, 2560, true>);
        if (b.hidden == 2048 && b.intermediate == 768)
          return fn(expertplex_sm100::executor<2, 2048, 768, true>);
        return fn(expertplex_sm100::executor<2, 0, 0, true>);
      }
      if (b.hidden == 6144 && b.intermediate == 2560)
        return cluster == 2 ? fn(expertplex_sm100::executor<2, 6144, 2560>)
                            : fn(expertplex_sm100::executor<1, 6144, 2560>);
      if (b.hidden == 2048 && b.intermediate == 768)
        return cluster == 2 ? fn(expertplex_sm100::executor<2, 2048, 768>)
                            : fn(expertplex_sm100::executor<1, 2048, 768>);
      return cluster == 2 ? fn(expertplex_sm100::executor<2>)
                          : fn(expertplex_sm100::executor<1>);
    };
    const void *kernel =
        sm100 ? with_native_kernel(
                    [](auto f) { return reinterpret_cast<const void *>(f); })
              : reinterpret_cast<const void *>(persistent_executor);
    const int threads = sm100 ? expertplex_sm100::Threads : 128;
    TORCH_CHECK(trace_capacity >= 0 && trace_capacity <= 65536,
                "trace_capacity must be in [0, 65536]");
    TORCH_CHECK(sm100 || trace_capacity == 0,
                "Tracing requires the SM100 backend");
    if (sm100)
      native_context = expertplex_sm100::prepare(
          *owner, reinterpret_cast<const BF16 *>(w13.data_ptr<at::BFloat16>()),
          reinterpret_cast<const BF16 *>(w2.data_ptr<at::BFloat16>()),
          w13.size(0), ctas, trace_capacity, cluster, fused);
    native_context.benchmark_stage = benchmark_stage;
    const char *pipeline_stores =
        std::getenv("SGLANG_EXPERTPLEX_PIPELINED_STORES");
    // Keep the complete SM100 backend's native epilogue pipeline enabled.
    // Explicit 0 is a diagnostic ablation; other backends retain their own path.
    native_context.pipelined_stores = pipeline_stores
        ? std::string(pipeline_stores) == "1"
        : backend == "sm100";
    TORCH_CHECK(
        !native_context.pipelined_stores || backend == "sm100",
        "Pipelined store retirement requires the complete SM100 backend");
    cudaFuncAttributes attrs;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, kernel));
    int optin = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, owner->device));
    int required = 198656;
    if (sm100) {
      if (fused)
        required = sizeof(expertplex_sm100::Storage<2, true>);
      else
        required = cluster == 2 ? sizeof(expertplex_sm100::Storage<2, false>)
                                : sizeof(expertplex_sm100::Storage<1, false>);
    }
    int reserve = std::max(
        required, static_cast<int>(prop.sharedMemPerMultiprocessor / 2 + 256 -
                                   attrs.sharedSizeBytes));
    TORCH_CHECK(reserve + attrs.sharedSizeBytes <= static_cast<size_t>(optin),
                "Cannot reserve one persistent CTA per SM on this device");
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, reserve));
    int resident = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, kernel, threads, reserve));
    TORCH_CHECK(
        resident == 1,
        "Persistent scheduling requires exactly one resident CTA per SM");
    attributes = attrs;
    launch_ctas = ctas;
    launch_threads = threads;
    dynamic_shared = reserve;
    cluster_size = cluster;
    backend_name = backend;
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
      if (sm100) {
        cudaLaunchConfig_t launch{};
        launch.gridDim = dim3(ctas);
        launch.blockDim = dim3(threads);
        launch.dynamicSmemBytes = reserve;
        launch.stream = stream;
        cudaLaunchAttribute attr{};
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {static_cast<unsigned>(cluster), 1, 1};
        launch.attrs = &attr;
        launch.numAttrs = 1;
        C10_CUDA_CHECK(with_native_kernel([&](auto f) {
          return cudaLaunchKernelEx(
              &launch, f, b, native_context, static_cast<int>(w13.size(0)),
              decode_budget, static_cast<float>(expected_tiles),
              max_decode_ctas);
        }));
      } else
        persistent_executor<<<ctas, 128, reserve, stream>>>(
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
  py::dict describe() const {
    py::dict d;
    d["trace_capacity_per_cta"] = native_context.trace_capacity;
    if (!running && native_context.trace_capacity) {
      c10::cuda::CUDAGuard guard(owner->device);
      std::vector<U64> counts(launch_ctas);
      C10_CUDA_CHECK(cudaMemcpy(counts.data(), native_context.trace_sequence_counts,
                                counts.size() * sizeof(U64),
                                cudaMemcpyDeviceToHost));
      d["trace_sequence_counts"] = counts;
    }
    d["backend"] = backend_name;
    d["benchmark_stage"] = native_context.benchmark_stage;
    d["pipelined_stores"] = native_context.pipelined_stores;
    d["ctas"] = launch_ctas;
    d["cluster_size"] = cluster_size;
    d["prefill_preprocess"] = bool(owner->b.prefill_preprocess);
    d["operation_templates"] =
        owner->b.prefill_preprocess
            ? "scatter1,scatter2,gate_up,activation,down,gather"
        : backend_name == "sm100_fused" || backend_name == "wmma"
            ? "fused_gate_up_activation,down"
            : "gate_up,activation,down";
    d["loading_stages"] =
        backend_name == "sm100_fused" ? 4 : (cluster_size == 2 ? 9 : 6);
    d["threads_per_cta"] = launch_threads;
    d["registers_per_thread"] = attributes.numRegs;
    d["local_bytes_per_thread"] = attributes.localSizeBytes;
    d["dynamic_shared_bytes_per_cta"] = dynamic_shared;
    d["static_shared_bytes_per_cta"] = attributes.sharedSizeBytes;
    d["hidden"] = owner->b.hidden;
    d["intermediate"] = owner->b.intermediate;
    d["prefill_tile_rows"] = backend_name == "wmma" ? 64 : expertplex_sm100::M;
    d["decode_tile_rows"] =
        backend_name == "wmma" ? 16 : expertplex_sm100::DecodeM;
    return d;
  }
  torch::Tensor read_trace() {
    TORCH_CHECK(!running, "Stop the executor before reading diagnostic traces");
    c10::cuda::CUDAGuard guard(owner->device);
    auto result = torch::zeros(
        {launch_ctas, native_context.trace_capacity, 8},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    if (native_context.trace_capacity)
      C10_CUDA_CHECK(cudaMemcpy(result.data_ptr(), native_context.trace,
                                result.numel() * sizeof(U64),
                                cudaMemcpyDeviceToHost));
    return result;
  }
  torch::Tensor read_decision_trace() {
    TORCH_CHECK(!running, "Stop the executor before reading diagnostic traces");
    c10::cuda::CUDAGuard guard(owner->device);
    auto result = torch::zeros(
        {launch_ctas, native_context.trace_capacity, 8},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    if (native_context.trace_capacity)
      C10_CUDA_CHECK(cudaMemcpy(result.data_ptr(), native_context.decision_trace,
                                result.numel() * sizeof(U64),
                                cudaMemcpyDeviceToHost));
    return result;
  }
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
                     int layer, int phase, int rank, int world, bool publish) {
  check_tensor(hidden, *r, at::kBFloat16, 2, "hidden");
  check_routes(ids, rows, *r, rank, world);
  TORCH_CHECK(!(r->b.prefill_preprocess && r->b.packed[r->slot]),
              "Expert-side preprocessing requires RemoteGroup dispatch with "
              "routing weights");
  TORCH_CHECK(hidden.size(0) == ids.size(0) && hidden.size(1) == r->b.hidden,
              "Hidden/routing shape mismatch");
  TORCH_CHECK(
      epoch >= 0 && static_cast<U64>(epoch) <= kMaxEpoch,
      "Epoch must be in [0,2**29); zero selects a device-generated epoch");
  TORCH_CHECK(layer >= 0 && phase >= 0 && phase <= 1, "Invalid layer/phase");
  c10::cuda::CUDAGuard guard(r->device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(r->device);
  auto transfer = r->attention_metadata();
  clear_slot<<<1, 32, 0, stream>>>(transfer, r->slot, epoch, layer, phase);
  int routes = ids.numel();
  if (routes > 0) {
    if (r->b.packed[r->slot]) {
      int blocks = std::min(256, (routes + 255) / 256);
      if (ids.scalar_type() == at::kInt)
        count_routes<<<blocks, 256, 0, stream>>>(
            transfer, r->slot, ids.data_ptr<int>(), rows.data_ptr<int>(),
            routes, rank, world);
      else
        count_routes<<<blocks, 256, 0, stream>>>(
            transfer, r->slot, ids.data_ptr<int64_t>(), rows.data_ptr<int>(),
            routes, rank, world);
      pack_offsets<<<1, 1, 0, stream>>>(transfer, r->slot);
    }
    if (ids.scalar_type() == at::kInt)
      dispatch_rows<<<routes, 128, 0, stream>>>(
          transfer, r->slot,
          reinterpret_cast<const BF16 *>(hidden.data_ptr<at::BFloat16>()),
          ids.data_ptr<int>(), rows.data_ptr<int>(), ids.size(0), ids.size(1),
          rank, world);
    else
      dispatch_rows<<<routes, 128, 0, stream>>>(
          transfer, r->slot,
          reinterpret_cast<const BF16 *>(hidden.data_ptr<at::BFloat16>()),
          ids.data_ptr<int64_t>(), rows.data_ptr<int>(), ids.size(0),
          ids.size(1), rank, world);
  }
  push_metadata<<<1, 128, 0, stream>>>(r->b, transfer, r->slot);
  if (publish)
    publish_slot<<<1, 1, 0, stream>>>(r->b, r->slot, epoch);
  C10_CUDA_CHECK(cudaGetLastError());
}

// Diagnostic-only release of pre-staged payloads at a GPU-timed interval.
// Both timestamps use the attention GPU timer; never subtract them from E's
// timer.
__global__ void publish_pair_kernel(U64 *p, U64 *d, U64 delay_ns, U64 *stamps) {
  if (system_load(p + ERROR) || system_load(d + ERROR))
    return;
  __threadfence_system();
  stamps[0] = now_ns();
  system_store(p + READY, uncached_load_u64(p + PENDING));
  while (now_ns() - stamps[0] < delay_ns) {
  }
  stamps[1] = now_ns();
  system_store(d + READY, uncached_load_u64(d + PENDING));
  U64 limit = stamps[1] + 2000000000ULL;
  while ((!stamps[2] || !stamps[3]) && now_ns() < limit) {
    if (!stamps[2] && system_load(p + START_NS))
      stamps[2] = now_ns();
    if (!stamps[3] && system_load(d + START_NS))
      stamps[3] = now_ns();
  }
}
static torch::Tensor publish_pair(std::shared_ptr<RemoteBuffers> p,
                                  std::shared_ptr<RemoteBuffers> d,
                                  int delay_us) {
  TORCH_CHECK(!p->closed && !d->closed && p->device == d->device &&
                  p->slot != d->slot && delay_us >= 0 && delay_us <= 1000000,
              "Invalid diagnostic publication pair");
  c10::cuda::CUDAGuard guard(p->device);
  auto stamps = torch::zeros({4}, torch::TensorOptions()
                                      .dtype(torch::kInt64)
                                      .device(torch::kCUDA, p->device));
  publish_pair_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream(p->device)>>>(
      p->b.control + p->slot * kControlWords,
      d->b.control + d->slot * kControlWords, static_cast<U64>(delay_us) * 1000,
      reinterpret_cast<U64 *>(stamps.data_ptr<int64_t>()));
  C10_CUDA_CHECK(cudaGetLastError());
  return stamps;
}

static void combine(std::shared_ptr<RemoteBuffers> r, torch::Tensor output,
                    torch::Tensor ids, torch::Tensor weights,
                    torch::Tensor rows, int64_t epoch, int rank, int world,
                    int timeout_ms, bool synchronize) {
  check_tensor(output, *r, at::kFloat, 2, "output");
  check_routes(ids, rows, *r, rank, world);
  check_tensor(weights, *r, at::kFloat, 2, "topk_weights");
  TORCH_CHECK(output.size(0) == ids.size(0) && output.size(1) == r->b.hidden &&
                  weights.sizes() == ids.sizes(),
              "Output/routing shape mismatch");
  TORCH_CHECK(epoch >= 0 && static_cast<U64>(epoch) <= kMaxEpoch &&
                  timeout_ms > 0 && timeout_ms <= 3600000,
              "Invalid epoch or timeout (1..3600000 ms)");
  c10::cuda::CUDAGuard guard(r->device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(r->device);
  wait_slot<<<1, 1, 0, stream>>>(r->b, r->slot, epoch,
                                 static_cast<U64>(timeout_ms) * 1000000ULL,
                                 r->status);
  auto transfer = r->attention_metadata();
  int blocks = std::min<int64_t>(65535, (output.numel() / 8 + 255) / 256);
  if (blocks > 0) {
    if (ids.scalar_type() == at::kInt)
      combine_rows<<<blocks, 256, 0, stream>>>(
          transfer, r->slot, output.data_ptr<float>(), ids.data_ptr<int>(),
          weights.data_ptr<float>(), rows.data_ptr<int>(), ids.size(0),
          ids.size(1), rank, world, r->status);
    else
      combine_rows<<<blocks, 256, 0, stream>>>(
          transfer, r->slot, output.data_ptr<float>(), ids.data_ptr<int64_t>(),
          weights.data_ptr<float>(), rows.data_ptr<int>(), ids.size(0),
          ids.size(1), rank, world, r->status);
  }
  C10_CUDA_CHECK(cudaGetLastError());
  if (!synchronize)
    return; // Stream-ordered result; worker monitors persistent protocol
            // errors.
  C10_CUDA_CHECK(cudaMemcpyAsync(r->host_status, r->status, sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  TORCH_CHECK(*r->host_status == OK, "ExpertPlex request failed (native code ",
              *r->host_status,
              "; 1=busy, 2=epoch, 3=route, 4=capacity, 5=layer, 6=phase, "
              "7=timeout, 8=stopped)");
}

__global__ void read_timing_kernel(const __grid_constant__ DeviceBuffers b,
                                   int slot, int64_t *output) {
  auto *c = b.control + static_cast<size_t>(slot) * kControlWords;
  output[0] = system_load(c + DONE);
  output[1] = system_load(c + READY);
  output[2] = uncached_load_u64(c + START_NS);
  output[3] = uncached_load_u64(c + FINISH_NS);
}

static torch::Tensor read_timing(std::shared_ptr<RemoteBuffers> r) {
  TORCH_CHECK(!r->closed, "IPC mapping has been closed");
  c10::cuda::CUDAGuard guard(r->device);
  auto result = torch::empty({4}, torch::TensorOptions()
                                      .dtype(torch::kInt64)
                                      .device(torch::kCUDA, r->device));
  read_timing_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream(r->device)>>>(
      r->b, r->slot, result.data_ptr<int64_t>());
  C10_CUDA_CHECK(cudaGetLastError());
  return result;
}

#include "benchmark.cuh"
#include "group_communication.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<OwnedBuffers, std::shared_ptr<OwnedBuffers>>(m, "OwnedBuffers")
      .def(py::init<int, int, int, int, int, int, std::vector<int>,
                    std::vector<int>, int, bool, int, int>())
      .def("descriptor", &OwnedBuffers::descriptor);
  py::class_<RemoteBuffers, std::shared_ptr<RemoteBuffers>>(m, "RemoteBuffers")
      .def(py::init<py::dict, int, int>())
      .def("close", &RemoteBuffers::close);
  py::class_<PersistentExecutor, std::shared_ptr<PersistentExecutor>>(
      m, "PersistentExecutor")
      .def(py::init<std::shared_ptr<OwnedBuffers>, torch::Tensor, torch::Tensor,
                    int, double, double, int, const std::string &, int, int>())
      .def("stop", &PersistentExecutor::stop,
           py::call_guard<py::gil_scoped_release>())
      .def("describe", &PersistentExecutor::describe)
      .def("read_trace", &PersistentExecutor::read_trace)
      .def("read_decision_trace", &PersistentExecutor::read_decision_trace)
      .def("check_error", &PersistentExecutor::check_error,
           py::call_guard<py::gil_scoped_release>());
  py::class_<RemoteGroup, std::shared_ptr<RemoteGroup>>(m, "RemoteGroup")
      .def(py::init<std::vector<std::shared_ptr<RemoteBuffers>>>())
      .def("dispatch", &RemoteGroup::dispatch, py::arg("hidden"),
           py::arg("ids"), py::arg("rows"), py::arg("layer"), py::arg("phase"),
           py::arg("weights") = py::none(), py::arg("publish") = true)
      .def("combine", &RemoteGroup::combine);
  m.def("dispatch", &dispatch);
  m.def("read_timing", &read_timing);
  m.def("publish_pair", &publish_pair);
  m.def("combine", &combine, py::call_guard<py::gil_scoped_release>());
  m.def("benchmark_views", &benchmark_views);
  m.def("benchmark_publish", &benchmark_publish);
  m.def("benchmark_echo", &benchmark_echo);
}
