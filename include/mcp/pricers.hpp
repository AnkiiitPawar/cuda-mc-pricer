#pragma once

#include <cstdint>

#include "mcp/model.hpp"

namespace mcp {

// Y = discounted call payoff and X = discounted put payoff on the same paths
// (for European options); Y = arithmetic-average Asian call, X = geometric-average
// Asian call (for Asian options). Exact GBM sampling: no discretisation error.

namespace cpu {
Moments european(const Market& m, double strike, std::uint64_t paths, std::uint64_t seed);
Moments asian(const Market& m, double strike, int fixings, std::uint64_t paths, std::uint64_t seed);
}  // namespace cpu

namespace gpu {
// True if a CUDA device is present and usable. Always false in a CPU-only build.
bool available();
Moments european(const Market& m, double strike, std::uint64_t paths, std::uint64_t seed);
Moments asian(const Market& m, double strike, int fixings, std::uint64_t paths, std::uint64_t seed);
}  // namespace gpu

}  // namespace mcp
