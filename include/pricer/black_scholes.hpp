// Closed-form Black–Scholes–Merton pricing, Greeks and implied volatility
// for European options on an asset paying a continuous dividend yield q.
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pricer {

enum class OptionType { Call, Put };

struct EuropeanOption {
    double spot;      // S: current price of the underlying
    double strike;    // K
    double rate;      // r: continuously compounded risk-free rate
    double dividend;  // q: continuous dividend yield (0 for none)
    double vol;       // sigma: annualised volatility
    double expiry;    // T: time to expiry in years
    OptionType type;
};

struct Greeks {
    double price;
    double delta;  // dV/dS
    double gamma;  // d2V/dS2
    double vega;   // dV/dsigma, per 1.00 of vol (divide by 100 for per vol point)
    double theta;  // dV/dt, per year (divide by 365 for per day)
    double rho;    // dV/dr, per 1.00 of rate
};

inline constexpr double kPi = 3.14159265358979323846;

inline double norm_pdf(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * kPi); }

// Standard normal CDF via the complementary error function (accurate in both tails).
inline double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

inline void validate(const EuropeanOption& o) {
    if (!(o.spot > 0.0)) throw std::invalid_argument("spot must be positive");
    if (!(o.strike > 0.0)) throw std::invalid_argument("strike must be positive");
    if (!(o.vol > 0.0)) throw std::invalid_argument("vol must be positive");
    if (!(o.expiry > 0.0)) throw std::invalid_argument("expiry must be positive");
}

namespace detail {
struct D {
    double d1, d2, sqrt_t, disc_r, disc_q;
};

inline D compute_d(const EuropeanOption& o) {
    const double sqrt_t = std::sqrt(o.expiry);
    const double sig_sqrt_t = o.vol * sqrt_t;
    const double d1 =
        (std::log(o.spot / o.strike) + (o.rate - o.dividend + 0.5 * o.vol * o.vol) * o.expiry) /
        sig_sqrt_t;
    return {d1, d1 - sig_sqrt_t, sqrt_t, std::exp(-o.rate * o.expiry),
            std::exp(-o.dividend * o.expiry)};
}
}  // namespace detail

inline double bs_price(const EuropeanOption& o) {
    validate(o);
    const auto d = detail::compute_d(o);
    const double fwd_s = o.spot * d.disc_q;  // S e^{-qT}
    const double pv_k = o.strike * d.disc_r;  // K e^{-rT}
    if (o.type == OptionType::Call) return fwd_s * norm_cdf(d.d1) - pv_k * norm_cdf(d.d2);
    return pv_k * norm_cdf(-d.d2) - fwd_s * norm_cdf(-d.d1);
}

inline Greeks bs_greeks(const EuropeanOption& o) {
    validate(o);
    const auto d = detail::compute_d(o);
    const double fwd_s = o.spot * d.disc_q;
    const double pv_k = o.strike * d.disc_r;
    const double pdf_d1 = norm_pdf(d.d1);

    Greeks g{};
    // Gamma and vega are the same for calls and puts (put-call parity is linear in S, sigma-free).
    g.gamma = d.disc_q * pdf_d1 / (o.spot * o.vol * d.sqrt_t);
    g.vega = fwd_s * pdf_d1 * d.sqrt_t;
    const double decay = -fwd_s * pdf_d1 * o.vol / (2.0 * d.sqrt_t);

    if (o.type == OptionType::Call) {
        const double n1 = norm_cdf(d.d1), n2 = norm_cdf(d.d2);
        g.price = fwd_s * n1 - pv_k * n2;
        g.delta = d.disc_q * n1;
        g.theta = decay - o.rate * pv_k * n2 + o.dividend * fwd_s * n1;
        g.rho = o.expiry * pv_k * n2;
    } else {
        const double n1 = norm_cdf(-d.d1), n2 = norm_cdf(-d.d2);
        g.price = pv_k * n2 - fwd_s * n1;
        g.delta = -d.disc_q * n1;
        g.theta = decay + o.rate * pv_k * n2 - o.dividend * fwd_s * n1;
        g.rho = -o.expiry * pv_k * n2;
    }
    return g;
}

// Implied volatility: Newton–Raphson on vega, falling back to bisection whenever a
// Newton step would leave the current bracket. Always converges for arbitrage-free prices.
inline double implied_vol(double target_price, EuropeanOption o, double tol = 1e-10,
                          int max_iter = 200) {
    o.vol = 1.0;  // placeholder so validate() passes
    validate(o);
    const double fwd_s = o.spot * std::exp(-o.dividend * o.expiry);
    const double pv_k = o.strike * std::exp(-o.rate * o.expiry);
    const double lower = o.type == OptionType::Call ? std::max(0.0, fwd_s - pv_k)
                                                    : std::max(0.0, pv_k - fwd_s);
    const double upper = o.type == OptionType::Call ? fwd_s : pv_k;
    if (!(target_price > lower && target_price < upper))
        throw std::domain_error("price is outside no-arbitrage bounds");

    double lo = 1e-6, hi = 10.0, sigma = 0.2;
    for (int i = 0; i < max_iter; ++i) {
        o.vol = sigma;
        const Greeks g = bs_greeks(o);
        const double diff = g.price - target_price;
        if (std::abs(diff) < tol) return sigma;
        (diff > 0.0 ? hi : lo) = sigma;  // price is increasing in vol
        const double newton = sigma - diff / g.vega;
        sigma = (g.vega > 1e-14 && newton > lo && newton < hi) ? newton : 0.5 * (lo + hi);
        if (hi - lo < 1e-15) return sigma;
    }
    throw std::runtime_error("implied_vol did not converge");
}

}  // namespace pricer
