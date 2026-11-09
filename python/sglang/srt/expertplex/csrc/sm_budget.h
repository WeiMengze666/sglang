#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define EXPERTPLEX_BUDGET_INLINE __host__ __device__ __forceinline__
#else
#define EXPERTPLEX_BUDGET_INLINE inline
#endif

namespace expertplex {

// Eq. 4: q is a preference under contention; the caller still lends idle
// clusters to the other phase. Inputs are validated by PersistentExecutor.
// Saturate in floating point before converting: a positive finite expected
// footprint can produce a request above INT_MAX, or even +infinity.
template <int Cluster>
EXPERTPLEX_BUDGET_INLINE int adaptive_decode_budget(
    int decode_budget, uint64_t footprint, float expected, int cap) {
  static_assert(Cluster > 0, "A CTA cluster must have at least one member");
  if (footprint == 0)
    return 0;
  const float requested =
      ceilf(decode_budget * static_cast<float>(footprint) / expected);
  if (requested >= static_cast<float>(cap))
    return cap;
  const int whole = requested < 1.f ? 1 : static_cast<int>(requested);
  const int remainder = whole % Cluster;
  const int padding = remainder == 0 ? 0 : Cluster - remainder;
  // Check the cap before addition as well, avoiding integer overflow even
  // when used with a large diagnostic cap. Production caps are SM counts.
  return whole > cap - padding ? cap : whole + padding;
}

} // namespace expertplex

#undef EXPERTPLEX_BUDGET_INLINE
