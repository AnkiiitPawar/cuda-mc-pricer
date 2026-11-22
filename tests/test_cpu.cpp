#include <cmath>
#include <cstdio>

#include "mcp/model.hpp"
#include "mcp/pricers.hpp"
#include "test_util.hpp"

using namespace mcp;
using testing::check;

int main() {
    const Market m{100.0, 0.05, 0.02, 0.25, 1.0};
    const double k = 105.0;

    {
        const Moments s = cpu::european(m, k, 400'000, 1);
        const Estimate call = plain(s), put = plain_control(s);
        check(std::abs(call.price - bs_call(m, k)) < 4 * call.std_error, "European call within 4 SE of Black-Scholes");
        check(std::abs(put.price - bs_put(m, k)) < 4 * put.std_error, "European put within 4 SE of Black-Scholes");
    }
    {
        const Moments small = cpu::european(m, k, 100'000, 2);
        const Moments large = cpu::european(m, k, 400'000, 3);
        const double ratio = plain(large).std_error / plain(small).std_error;
        check(std::abs(ratio - 0.5) < 0.05, "standard error falls as 1/sqrt(paths)");
    }
    {
        const int fixings = 12;
        const Moments s = cpu::asian(m, k, fixings, 200'000, 4);
        const double exact_geo = geometric_asian_call(m, k, fixings);
        const Estimate geo = plain_control(s);
        check(std::abs(geo.price - exact_geo) < 4 * geo.std_error,
              "geometric Asian Monte Carlo matches its closed form");

        const Estimate raw = plain(s), cv = control_variate(s, exact_geo);
        check(std::abs(raw.price - cv.price) < 4 * raw.std_error, "control variate agrees with the plain estimate");
        const double reduction = (raw.std_error * raw.std_error) / (cv.std_error * cv.std_error);
        std::printf("  variance reduction factor: %.0fx\n", reduction);
        check(reduction > 20, "geometric control variate cuts variance by more than 20x");
        check(raw.price >= geo.price, "arithmetic Asian is worth at least the geometric (AM >= GM)");
    }
    {
        const int fixings = 1;  // one fixing at expiry: an Asian option is a European option
        check(std::abs(geometric_asian_call(m, k, fixings) - bs_call(m, k)) < 1e-12,
              "one-fixing geometric Asian equals Black-Scholes");
    }
    std::printf("%d failure(s)\n", testing::failures);
    return testing::failures == 0 ? 0 : 1;
}
