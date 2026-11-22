// mc_pricer                 price on CPU (and GPU if available), print a comparison table
// mc_pricer --convergence F write CPU convergence data (paths, error, standard errors) to CSV F

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "mcp/model.hpp"
#include "mcp/pricers.hpp"

using namespace mcp;
using Clock = std::chrono::steady_clock;

template <class F>
static double seconds(F&& f) {
    const auto t0 = Clock::now();
    f();
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static int convergence(const char* path) {
    const Market m{100.0, 0.05, 0.02, 0.25, 1.0};
    const double k = 105.0;
    const int fixings = 12;
    const double exact_call = bs_call(m, k), exact_geo = geometric_asian_call(m, k, fixings);
    std::FILE* f = std::fopen(path, "w");
    if (!f) return 1;
    std::fprintf(f, "paths,euro_abs_error,euro_se,asian_plain_se,asian_cv_se\n");
    for (std::uint64_t paths = 1000; paths <= 2'048'000; paths *= 2) {
        const Estimate e = plain(cpu::european(m, k, paths, paths));
        const Moments a = cpu::asian(m, k, fixings, paths, paths + 1);
        std::fprintf(f, "%llu,%.6e,%.6e,%.6e,%.6e\n", static_cast<unsigned long long>(paths),
                     std::abs(e.price - exact_call), e.std_error, plain(a).std_error,
                     control_variate(a, exact_geo).std_error);
    }
    std::fclose(f);
    std::printf("wrote %s\n", path);
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--convergence") == 0) return convergence(argv[2]);

    const Market m{100.0, 0.05, 0.02, 0.25, 1.0};
    const double k = 105.0;
    const int fixings = 12;
    const std::uint64_t cpu_paths = 1'000'000, gpu_paths = 100'000'000;
    const double exact_geo = geometric_asian_call(m, k, fixings);

    std::printf("S=%.0f K=%.0f r=%.2f q=%.2f vol=%.2f T=%.1f, Asian with %d monthly fixings\n\n", m.spot, k, m.rate,
                m.dividend, m.vol, m.expiry, fixings);
    std::printf("Black-Scholes call        %.6f\n", bs_call(m, k));
    std::printf("geometric Asian (exact)   %.6f\n\n", exact_geo);

    Moments ce, ca;
    const double t_ce = seconds([&] { ce = cpu::european(m, k, cpu_paths, 1); });
    const double t_ca = seconds([&] { ca = cpu::asian(m, k, fixings, cpu_paths, 2); });
    std::printf("CPU, %llu paths\n", static_cast<unsigned long long>(cpu_paths));
    std::printf("  European call          %.6f +/- %.6f   %.3f s\n", plain(ce).price, plain(ce).std_error, t_ce);
    std::printf("  Asian call, plain      %.6f +/- %.6f   %.3f s\n", plain(ca).price, plain(ca).std_error, t_ca);
    std::printf("  Asian call, control    %.6f +/- %.6f\n", control_variate(ca, exact_geo).price,
                control_variate(ca, exact_geo).std_error);

    if (!gpu::available()) {
        std::printf("\nno CUDA device found: GPU results skipped\n");
        return 0;
    }
    Moments ge, ga;
    gpu::european(m, k, 1'000'000, 0);  // warm-up: context creation and module load
    const double t_ge = seconds([&] { ge = gpu::european(m, k, gpu_paths, 3); });
    const double t_ga = seconds([&] { ga = gpu::asian(m, k, fixings, gpu_paths, 4); });
    std::printf("\nGPU, %llu paths\n", static_cast<unsigned long long>(gpu_paths));
    std::printf("  European call          %.6f +/- %.6f   %.3f s  (%.0fx CPU paths/s)\n", plain(ge).price,
                plain(ge).std_error, t_ge, (gpu_paths / t_ge) / (cpu_paths / t_ce));
    std::printf("  Asian call, control    %.6f +/- %.6f   %.3f s  (%.0fx CPU paths/s)\n",
                control_variate(ga, exact_geo).price, control_variate(ga, exact_geo).std_error, t_ga,
                (gpu_paths / t_ga) / (cpu_paths / t_ca));
    return 0;
}
