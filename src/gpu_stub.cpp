// CPU-only build: the GPU entry points exist but report that no device is available.

#include <stdexcept>

#include "mcp/pricers.hpp"

namespace mcp::gpu {

bool available() { return false; }

Moments european(const Market&, double, std::uint64_t, std::uint64_t) {
    throw std::runtime_error("built without CUDA");
}

Moments asian(const Market&, double, int, std::uint64_t, std::uint64_t) {
    throw std::runtime_error("built without CUDA");
}

}  // namespace mcp::gpu
