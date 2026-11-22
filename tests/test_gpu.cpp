// GPU tests. Without a CUDA device they exit with 77, which CTest reports as "skipped".

#include <cmath>
#include <cstdio>

#include "mcp/model.hpp"
#include "mcp/pricers.hpp"
#include "test_util.hpp"

using namespace mcp;
using testing::check;

int main() {
    if (!gpu::available()) {
        std::printf("no CUDA device: skipping GPU tests\n");
        return 77;
    }
    const Market m{100.0, 0.05, 0.02, 0.25, 1.0};
    const double k = 105.0;
    {
        const Moments s = gpu::european(m, k, 4'000'000, 11);
        const Estimate call = plain(s), put = plain_control(s);
        check(std::abs(call.price - bs_call(m, k)) < 4 * call.std_error, "GPU European call within 4 SE of Black-Scholes");
        check(std::abs(put.price - bs_put(m, k)) < 4 * put.std_error, "GPU European put within 4 SE of Black-Scholes");
    }
    {
        const int fixings = 12;
        const double exact_geo = geometric_asian_call(m, k, fixings);
        const Moments g = gpu::asian(m, k, fixings, 2'000'000, 12);
        const Moments c = cpu::asian(m, k, fixings, 200'000, 13);
        const Estimate geo = plain_control(g);
        check(std::abs(geo.price - exact_geo) < 4 * geo.std_error, "GPU geometric Asian matches the closed form");
        const Estimate gcv = control_variate(g, exact_geo), ccv = control_variate(c, exact_geo);
        const double se = std::sqrt(gcv.std_error * gcv.std_error + ccv.std_error * ccv.std_error);
        check(std::abs(gcv.price - ccv.price) < 4 * se, "GPU and CPU arithmetic Asian prices agree");
    }
    std::printf("%d failure(s)\n", testing::failures);
    return testing::failures == 0 ? 0 : 1;
}
