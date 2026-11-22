// GPU Monte Carlo: one Philox random stream per thread (cuRAND device API),
// paths split across threads, then a two-level reduction: a tree reduction in
// shared memory per block, and a final sum of the per-block partials on the host.

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "mcp/pricers.hpp"

namespace mcp::gpu {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 1024;

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

struct Sums {
    double y, yy, x, xx, xy;
};

// Tree reduction of each thread's sums in shared memory; thread 0 writes the block total.
__device__ void block_reduce(Sums mine, Sums* out) {
    __shared__ Sums shared[kThreads];
    shared[threadIdx.x] = mine;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            Sums& a = shared[threadIdx.x];
            const Sums& b = shared[threadIdx.x + stride];
            a.y += b.y; a.yy += b.yy; a.x += b.x; a.xx += b.xx; a.xy += b.xy;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = shared[0];
}

__device__ void accumulate(Sums& s, double y, double x) {
    s.y += y; s.yy += y * y; s.x += x; s.xx += x * x; s.xy += x * y;
}

__global__ void european_kernel(Market m, double k, unsigned long long paths, unsigned long long seed, Sums* out) {
    const unsigned long long tid = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, tid, 0, &rng);  // independent subsequence per thread
    const double drift = (m.rate - m.dividend - 0.5 * m.vol * m.vol) * m.expiry;
    const double diffusion = m.vol * sqrt(m.expiry);
    const double disc = exp(-m.rate * m.expiry);
    Sums s{0, 0, 0, 0, 0};
    for (unsigned long long p = tid; p < paths; p += stride) {
        const double st = m.spot * exp(drift + diffusion * curand_normal_double(&rng));
        accumulate(s, disc * fmax(st - k, 0.0), disc * fmax(k - st, 0.0));
    }
    block_reduce(s, out);
}

__global__ void asian_kernel(Market m, double k, int fixings, unsigned long long paths, unsigned long long seed,
                             Sums* out) {
    const unsigned long long tid = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, tid, 0, &rng);
    const double dt = m.expiry / fixings;
    const double drift = (m.rate - m.dividend - 0.5 * m.vol * m.vol) * dt;
    const double diffusion = m.vol * sqrt(dt);
    const double disc = exp(-m.rate * m.expiry);
    Sums s{0, 0, 0, 0, 0};
    for (unsigned long long p = tid; p < paths; p += stride) {
        double log_s = log(m.spot), sum = 0, sum_log = 0;
        for (int i = 0; i < fixings; ++i) {
            log_s += drift + diffusion * curand_normal_double(&rng);
            sum += exp(log_s);
            sum_log += log_s;
        }
        accumulate(s, disc * fmax(sum / fixings - k, 0.0), disc * fmax(exp(sum_log / fixings) - k, 0.0));
    }
    block_reduce(s, out);
}

template <class Launch>
Moments run(std::uint64_t paths, Launch&& launch) {
    const int blocks = static_cast<int>(std::min<std::uint64_t>(kMaxBlocks, (paths + kThreads - 1) / kThreads));
    Sums* d_out = nullptr;
    check(cudaMalloc(&d_out, blocks * sizeof(Sums)), "cudaMalloc");
    launch(blocks, d_out);
    check(cudaGetLastError(), "kernel launch");
    std::vector<Sums> partial(blocks);
    check(cudaMemcpy(partial.data(), d_out, blocks * sizeof(Sums), cudaMemcpyDeviceToHost), "cudaMemcpy");
    check(cudaFree(d_out), "cudaFree");
    Moments total;
    total.n = static_cast<double>(paths);
    for (const Sums& s : partial) {
        total.y += s.y; total.yy += s.yy; total.x += s.x; total.xx += s.xx; total.xy += s.xy;
    }
    return total;
}

}  // namespace

bool available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

Moments european(const Market& m, double k, std::uint64_t paths, std::uint64_t seed) {
    return run(paths, [&](int blocks, Sums* out) { european_kernel<<<blocks, kThreads>>>(m, k, paths, seed, out); });
}

Moments asian(const Market& m, double k, int fixings, std::uint64_t paths, std::uint64_t seed) {
    return run(paths, [&](int blocks, Sums* out) { asian_kernel<<<blocks, kThreads>>>(m, k, fixings, paths, seed, out); });
}

}  // namespace mcp::gpu
