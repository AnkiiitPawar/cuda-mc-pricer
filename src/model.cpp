#include "mcp/model.hpp"

#include <cmath>

namespace mcp {
namespace {
double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }
}  // namespace

double bs_call(const Market& m, double k) {
    const double sd = m.vol * std::sqrt(m.expiry);
    const double d1 = (std::log(m.spot / k) + (m.rate - m.dividend) * m.expiry) / sd + 0.5 * sd;
    return m.spot * std::exp(-m.dividend * m.expiry) * norm_cdf(d1) - k * std::exp(-m.rate * m.expiry) * norm_cdf(d1 - sd);
}

double bs_put(const Market& m, double k) {
    // put-call parity
    return bs_call(m, k) - m.spot * std::exp(-m.dividend * m.expiry) + k * std::exp(-m.rate * m.expiry);
}

double geometric_asian_call(const Market& m, double k, int n) {
    const double nn = static_cast<double>(n);
    const double mu = std::log(m.spot) + (m.rate - m.dividend - 0.5 * m.vol * m.vol) * m.expiry * (nn + 1) / (2 * nn);
    const double var = m.vol * m.vol * m.expiry * (nn + 1) * (2 * nn + 1) / (6 * nn * nn);
    const double sd = std::sqrt(var);
    const double d1 = (mu - std::log(k) + var) / sd;
    const double d2 = d1 - sd;
    return std::exp(-m.rate * m.expiry) * (std::exp(mu + 0.5 * var) * norm_cdf(d1) - k * norm_cdf(d2));
}

Estimate plain(const Moments& s) {
    const double mean = s.y / s.n;
    const double var = (s.yy / s.n - mean * mean) * s.n / (s.n - 1);
    return {mean, std::sqrt(var / s.n)};
}

Estimate plain_control(const Moments& s) {
    const double mean = s.x / s.n;
    const double var = (s.xx / s.n - mean * mean) * s.n / (s.n - 1);
    return {mean, std::sqrt(var / s.n)};
}

Estimate control_variate(const Moments& s, double exact) {
    const double my = s.y / s.n, mx = s.x / s.n;
    const double var_y = s.yy / s.n - my * my;
    const double var_x = s.xx / s.n - mx * mx;
    const double cov = s.xy / s.n - mx * my;
    const double b = cov / var_x;
    const double residual = (var_y - cov * cov / var_x) * s.n / (s.n - 2);
    return {my - b * (mx - exact), std::sqrt(residual / s.n)};
}

}  // namespace mcp
