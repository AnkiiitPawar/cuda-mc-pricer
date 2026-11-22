#include <algorithm>
#include <cmath>
#include <random>

#include "mcp/pricers.hpp"

namespace mcp::cpu {

Moments european(const Market& m, double k, std::uint64_t paths, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal;
    const double drift = (m.rate - m.dividend - 0.5 * m.vol * m.vol) * m.expiry;
    const double diffusion = m.vol * std::sqrt(m.expiry);
    const double disc = std::exp(-m.rate * m.expiry);
    Moments s;
    for (std::uint64_t p = 0; p < paths; ++p) {
        const double st = m.spot * std::exp(drift + diffusion * normal(rng));
        s.add(disc * std::max(st - k, 0.0), disc * std::max(k - st, 0.0));
    }
    return s;
}

Moments asian(const Market& m, double k, int fixings, std::uint64_t paths, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal;
    const double dt = m.expiry / fixings;
    const double drift = (m.rate - m.dividend - 0.5 * m.vol * m.vol) * dt;
    const double diffusion = m.vol * std::sqrt(dt);
    const double disc = std::exp(-m.rate * m.expiry);
    Moments s;
    for (std::uint64_t p = 0; p < paths; ++p) {
        double log_s = std::log(m.spot), sum = 0, sum_log = 0;
        for (int i = 0; i < fixings; ++i) {
            log_s += drift + diffusion * normal(rng);
            sum += std::exp(log_s);
            sum_log += log_s;
        }
        const double arithmetic = sum / fixings;
        const double geometric = std::exp(sum_log / fixings);
        s.add(disc * std::max(arithmetic - k, 0.0), disc * std::max(geometric - k, 0.0));
    }
    return s;
}

}  // namespace mcp::cpu
