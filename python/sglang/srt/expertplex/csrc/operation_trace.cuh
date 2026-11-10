// Optional bounded per-operation observations, read only after executor stop.
// One admission writer per slot. Full buffers retain their earliest records.
#pragma once
struct alignas(64) OperationTraceRecord {
  U64 epoch, layer, phase, start_ns, finish_ns;
  U64 gate_tiles, activation_tiles, down_tiles;
};
static_assert(sizeof(OperationTraceRecord) == 8 * sizeof(U64));

__device__ __forceinline__ OperationTraceRecord *
operation_record(Context context, int slot) {
  const U64 count = context.operation_trace_counts[slot];
  if (!count || count > static_cast<U64>(context.operation_trace_capacity))
    return nullptr;
  return context.operation_trace +
         static_cast<size_t>(slot) * context.operation_trace_capacity + count -
         1;
}

__device__ __forceinline__ void operation_begin(Context context, int slot,
                                                U64 epoch, int layer, int phase,
                                                U64 start_ns) {
  if (!context.operation_trace_capacity)
    return;
  // Admission is serialized by the existing slot generation protocol. Count
  // every admission, including overflow, so a stopped reader detects
  // truncation.
  ++context.operation_trace_counts[slot];
  auto *record = operation_record(context, slot);
  if (record)
    *record = {epoch,
               static_cast<U64>(layer),
               static_cast<U64>(phase),
               start_ns,
               0,
               0,
               0,
               0};
}

__device__ __forceinline__ void operation_finish(Context context, int slot,
                                                 U64 epoch, U64 finish_ns,
                                                 const SlotState *state) {
  if (!context.operation_trace_capacity)
    return;
  auto *record = operation_record(context, slot);
  if (!record)
    return;
  record->gate_tiles = state ? state->gate_tiles : 0;
  record->activation_tiles = state ? state->activation_tiles : 0;
  record->down_tiles = state ? state->down_tiles : 0;
  record->finish_ns = finish_ns;
  // Existing DONE publication orders these writes. No live host polling.
}
