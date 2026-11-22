#pragma once

#include <cstdint>

namespace mcp {

struct Market {
    double spot = 100.0;
    double rate = 0.05;      // continuously compounded
    double dividend = 0.0;   // continuous yield
    double vol = 0.2;
    double expiry = 1.0;     // years
};

struct Estimate {
    double price = 0;
    double std_error = 0;
};

// Black-Scholes closed form.
double bs_call(const Market& m, double strike);
double bs_put(const Market& m, double strike);

// Geometric-average Asian call with `fixings` equally spaced observations t_i = iT/n.
// ln G is normal with mean ln S + (r - q - sigma^2/2) T (n+1)/(2n)
// and variance sigma^2 T (n+1)(2n+1)/(6 n^2), so it has an exact Black-Scholes-like price.
double geometric_asian_call(const Market& m, double strike, int fixings);

// Running sums of a payoff Y and a control X, accumulated path by path (CPU)
// or reduced across threads (GPU). Everything below works from these six numbers.
struct Moments {
    double n = 0, y = 0, yy = 0, x = 0, xx = 0, xy = 0;

    void add(double Y, double X) {
        n += 1;
        y += Y;
        yy += Y * Y;
        x += X;
        xx += X * X;
        xy += X * Y;
    }
    Moments& operator+=(const Moments& o) {
        n += o.n; y += o.y; yy += o.yy; x += o.x; xx += o.xx; xy += o.xy;
        return *this;
    }
};

// Plain estimate of E[Y].
Estimate plain(const Moments& s);
// Plain estimate of E[X].
Estimate plain_control(const Moments& s);
// Control-variate estimate: mean(Y) - b (mean(X) - E[X]), b = Cov(X,Y)/Var(X).
// The variance shrinks by a factor 1 / (1 - corr(X,Y)^2).
Estimate control_variate(const Moments& s, double exact_control_mean);

}  // namespace mcp
