// One-factor Hull–White short-rate model, fitted exactly to an initial discount curve.
//
//   r(t) = x(t) + phi(t),   dx = -a x dt + sigma dW,   x(0) = 0
//
// phi(t) is chosen so the model reproduces today's market discount factors P^M(0, T) exactly
// (Brigo & Mercurio, Interest Rate Models, ch. 3.3). Because x is Gaussian, bonds, bond options,
// caplets and caps all have closed forms, and the pair (x(T), integral of x) can be simulated
// exactly in one step, so the Monte Carlo has no time-discretisation error to hide behind.
#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <stdexcept>

#include "black_scholes.hpp"
#include "monte_carlo.hpp"

namespace pricer {

// Initial discount curve P^M(0, T). Nelson–Siegel zero rates give a smooth, realistic shape:
//   z(T) = b0 + b1 (1 - e^{-T/tau}) / (T/tau) + b2 [(1 - e^{-T/tau}) / (T/tau) - e^{-T/tau}]
struct NelsonSiegel {
    double b0, b1, b2, tau;
    double zero_rate(double t) const {
        if (t < 1e-10) return b0 + b1;  // limit as t -> 0
        const double x = t / tau, e = std::exp(-x), f = (1.0 - e) / x;
        return b0 + b1 * f + b2 * (f - e);
    }
    double discount(double t) const { return std::exp(-zero_rate(t) * t); }
};

class HullWhite {
public:
    HullWhite(double mean_reversion, double vol, std::function<double(double)> discount_curve)
        : a_(mean_reversion), sigma_(vol), p_(std::move(discount_curve)) {
        if (!(a_ > 0.0) || !(sigma_ > 0.0)) throw std::invalid_argument("a and sigma must be positive");
    }

    double a() const { return a_; }
    double sigma() const { return sigma_; }
    double market_discount(double t) const { return t <= 0.0 ? 1.0 : p_(t); }

    // B(t,T) = (1 - e^{-a(T-t)}) / a : sensitivity of ln P(t,T) to x(t)
    double B(double t, double T) const { return (1.0 - std::exp(-a_ * (T - t))) / a_; }

    // V(t,T) = Var[ integral_t^T x(u) du | F_t ]
    double V(double t, double T) const {
        const double tau = T - t;
        return sigma_ * sigma_ / (a_ * a_) *
               (tau + 2.0 / a_ * std::exp(-a_ * tau) - 0.5 / a_ * std::exp(-2.0 * a_ * tau) - 1.5 / a_);
    }

    // Zero-coupon bond price at time t given the state x(t).
    double bond_price(double t, double T, double x) const {
        return market_discount(T) / market_discount(t) *
               std::exp(0.5 * (V(t, T) - V(0.0, T) + V(0.0, t)) - B(t, T) * x);
    }

    // European call / put at T1 on the zero-coupon bond maturing at T2, strike X.
    double bond_option(double T1, double T2, double X, OptionType type) const {
        const double p1 = market_discount(T1), p2 = market_discount(T2);
        const double sp = sigma_ * std::sqrt((1.0 - std::exp(-2.0 * a_ * T1)) / (2.0 * a_)) * B(T1, T2);
        const double h = std::log(p2 / (p1 * X)) / sp + 0.5 * sp;
        if (type == OptionType::Call) return p2 * norm_cdf(h) - X * p1 * norm_cdf(h - sp);
        return X * p1 * norm_cdf(sp - h) - p2 * norm_cdf(-h);
    }

    // Simple forward rate for [T1, T2].
    double forward_rate(double T1, double T2) const {
        return (market_discount(T1) / market_discount(T2) - 1.0) / (T2 - T1);
    }

    // Caplet on the rate fixing at T1, paid at T2, unit notional. Its payoff tau (L - K)^+ at T2
    // equals (1 + K tau) times a put on the T2 bond with strike 1/(1 + K tau), expiring at T1.
    double caplet(double T1, double T2, double K) const {
        const double tau = T2 - T1;
        return (1.0 + K * tau) * bond_option(T1, T2, 1.0 / (1.0 + K * tau), OptionType::Put);
    }

    // Cap = strip of caplets from `start` to `end` with accrual period `period`
    // (the first period, whose rate is already fixed, is excluded by market convention).
    double cap(double start, double end, double period, double K) const {
        double total = 0.0;
        for (double t = start + period; t + period <= end + 1e-9; t += period) total += caplet(t, t + period, K);
        return total;
    }

private:
    double a_, sigma_;
    std::function<double(double)> p_;
};

// Calibrate sigma (holding a fixed) to a market cap price by bisection: cap value is increasing
// in sigma, so the root is unique.
inline double calibrate_hw_sigma(double target_cap, double a, std::function<double(double)> curve,
                                 double start, double end, double period, double K) {
    double lo = 1e-5, hi = 0.2;
    auto price = [&](double s) { return HullWhite(a, s, curve).cap(start, end, period, K); };
    if (!(price(lo) < target_cap && target_cap < price(hi)))
        throw std::domain_error("target cap price not attainable for sigma in [1e-5, 0.2]");
    for (int i = 0; i < 200 && hi - lo > 1e-12; ++i) {
        const double mid = 0.5 * (lo + hi);
        (price(mid) < target_cap ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}

// ---------------------------------------------------------------- exact simulation

// Draws (x(T), I(T) = integral_0^T x du) jointly: both Gaussian with mean 0 and
//   Var x = sigma^2 (1 - e^{-2aT}) / (2a),  Var I = V(0,T),  Cov = sigma^2 (1 - e^{-aT})^2 / (2a^2).
// The stochastic discount factor is then D(T) = P^M(0,T) exp(-V(0,T)/2 - I(T)), so E[D(T)] = P^M(0,T).
class HullWhiteSampler {
public:
    HullWhiteSampler(const HullWhite& m, double T) {
        const double a = m.a(), s = m.sigma();
        sx_ = s * std::sqrt((1.0 - std::exp(-2.0 * a * T)) / (2.0 * a));
        si_ = std::sqrt(m.V(0.0, T));
        const double cov = s * s * std::pow(1.0 - std::exp(-a * T), 2) / (2.0 * a * a);
        rho_ = cov / (sx_ * si_);
        log_p_ = std::log(m.market_discount(T)) - 0.5 * m.V(0.0, T);
    }
    // z1, z2 independent N(0,1) -> state x(T) and discount factor D(T).
    void draw(double z1, double z2, double& x, double& discount) const {
        x = sx_ * z1;
        const double integral = si_ * (rho_ * z1 + std::sqrt(1.0 - rho_ * rho_) * z2);
        discount = std::exp(log_p_ - integral);
    }

private:
    double sx_, si_, rho_, log_p_;
};

// Monte Carlo zero-coupon bond: E[D(T)] should equal the market discount factor.
inline McResult mc_hw_zero_bond(const HullWhite& m, double T, const McConfig& cfg) {
    HullWhiteSampler sampler(m, T);
    std::mt19937_64 rng(cfg.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const std::size_t n = sample_count(cfg);
    PairStats stats;
    double x, d, x2, d2;
    for (std::size_t i = 0; i < n; ++i) {
        const double z1 = normal(rng), z2 = normal(rng);
        sampler.draw(z1, z2, x, d);
        if (cfg.antithetic) {
            sampler.draw(-z1, -z2, x2, d2);
            d = 0.5 * (d + d2);
        }
        stats.add(d, 0.0);
    }
    return summarise(stats, cfg.antithetic ? 2 * n : n);
}

// Monte Carlo caplet: simulate to the fixing date T1, read off the T2 bond price from the state,
// pay (1 + K tau)(X - P(T1,T2))^+ discounted with the simulated D(T1). Optional control variate:
// D(T1) P(T1,T2), whose expectation is exactly P^M(0, T2).
inline McResult mc_hw_caplet(const HullWhite& m, double T1, double T2, double K, const McConfig& cfg) {
    HullWhiteSampler sampler(m, T1);
    std::mt19937_64 rng(cfg.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const double tau = T2 - T1, strike = 1.0 / (1.0 + K * tau), scale = 1.0 + K * tau;
    const std::size_t n = sample_count(cfg);
    PairStats stats;
    auto sample = [&](double z1, double z2, double& y, double& cv) {
        double x, d;
        sampler.draw(z1, z2, x, d);
        const double bond = m.bond_price(T1, T2, x);
        y = d * scale * std::max(strike - bond, 0.0);
        cv = d * bond;
    };
    for (std::size_t i = 0; i < n; ++i) {
        const double z1 = normal(rng), z2 = normal(rng);
        double y, cv;
        sample(z1, z2, y, cv);
        if (cfg.antithetic) {
            double y2, cv2;
            sample(-z1, -z2, y2, cv2);
            y = 0.5 * (y + y2);
            cv = 0.5 * (cv + cv2);
        }
        stats.add(y, cv);
    }
    return summarise(stats, cfg.antithetic ? 2 * n : n, cfg.control_variate, m.market_discount(T2));
}

}  // namespace pricer
