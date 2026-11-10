// Monte Carlo pricing under geometric Brownian motion:
//   dS = (r - q) S dt + sigma S dW   =>   S_T = S_0 exp((r - q - sigma^2/2) T + sigma sqrt(T) Z)
//
// Variance reduction: antithetic variates (pair Z with -Z) and a control variate
// whose expectation is known exactly. Every estimate comes with its standard error.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "black_scholes.hpp"

namespace pricer {

struct McConfig {
    std::size_t paths = 100'000;  // number of simulated terminal prices / paths
    std::uint64_t seed = 42;
    bool antithetic = false;
    bool control_variate = false;
};

struct McResult {
    double price = 0.0;
    double std_error = 0.0;  // standard error of the estimate
    std::size_t paths = 0;
    double ci95_low() const { return price - 1.96 * std_error; }
    double ci95_high() const { return price + 1.96 * std_error; }
};

// Single-pass (Welford) running mean, variance and covariance of a pair (y, x).
// Numerically stable for millions of samples, and needs no storage.
class PairStats {
public:
    void add(double y, double x) {
        ++n_;
        const double dx = x - mean_x_;
        mean_x_ += dx / static_cast<double>(n_);
        const double dy = y - mean_y_;
        mean_y_ += dy / static_cast<double>(n_);
        m2x_ += dx * (x - mean_x_);
        m2y_ += dy * (y - mean_y_);
        cxy_ += dx * (y - mean_y_);
    }
    // Combine statistics from two independent streams (Chan, Golub & LeVeque), so threads can
    // each accumulate locally and merge at the end with no loss of accuracy.
    void merge(const PairStats& o) {
        if (o.n_ == 0) return;
        if (n_ == 0) {
            *this = o;
            return;
        }
        const double n = static_cast<double>(n_ + o.n_);
        const double na = static_cast<double>(n_), nb = static_cast<double>(o.n_);
        const double dx = o.mean_x_ - mean_x_, dy = o.mean_y_ - mean_y_;
        m2x_ += o.m2x_ + dx * dx * na * nb / n;
        m2y_ += o.m2y_ + dy * dy * na * nb / n;
        cxy_ += o.cxy_ + dx * dy * na * nb / n;
        mean_x_ += dx * nb / n;
        mean_y_ += dy * nb / n;
        n_ += o.n_;
    }
    std::size_t count() const { return n_; }
    double mean_y() const { return mean_y_; }
    double mean_x() const { return mean_x_; }
    double var_y() const { return n_ > 1 ? m2y_ / static_cast<double>(n_ - 1) : 0.0; }
    double var_x() const { return n_ > 1 ? m2x_ / static_cast<double>(n_ - 1) : 0.0; }
    double cov() const { return n_ > 1 ? cxy_ / static_cast<double>(n_ - 1) : 0.0; }

private:
    std::size_t n_ = 0;
    double mean_x_ = 0.0, mean_y_ = 0.0, m2x_ = 0.0, m2y_ = 0.0, cxy_ = 0.0;
};

// Turn accumulated samples into an estimate. With a control variate X of known mean E[X],
// the estimator is  Y - b (X - E[X])  with the variance-minimising b = Cov(Y,X) / Var(X);
// its variance is  Var(Y) (1 - rho^2).
inline McResult summarise(const PairStats& s, std::size_t paths, bool use_cv = false,
                          double x_expectation = 0.0) {
    McResult r;
    r.paths = paths;
    double mean = s.mean_y(), var = s.var_y();
    if (use_cv && s.var_x() > 0.0) {
        const double b = s.cov() / s.var_x();
        mean -= b * (s.mean_x() - x_expectation);
        var = std::max(0.0, var - b * s.cov());
    }
    r.price = mean;
    r.std_error = std::sqrt(var / static_cast<double>(s.count()));
    return r;
}

inline double payoff(OptionType type, double underlying, double strike) {
    return type == OptionType::Call ? std::max(underlying - strike, 0.0)
                                    : std::max(strike - underlying, 0.0);
}

inline std::size_t sample_count(const McConfig& cfg) {
    const std::size_t n = cfg.antithetic ? cfg.paths / 2 : cfg.paths;
    if (n < 2) throw std::invalid_argument("need at least 2 samples (4 paths if antithetic)");
    return n;
}

// ---------------------------------------------------------------- European option

// Control variate: the discounted terminal price e^{-rT} S_T, whose expectation under the
// risk-neutral measure is exactly S_0 e^{-qT}.
inline McResult mc_european(const EuropeanOption& o, const McConfig& cfg) {
    validate(o);
    std::mt19937_64 rng(cfg.seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    const double drift = (o.rate - o.dividend - 0.5 * o.vol * o.vol) * o.expiry;
    const double diffusion = o.vol * std::sqrt(o.expiry);
    const double disc = std::exp(-o.rate * o.expiry);
    const double cv_expectation = o.spot * std::exp(-o.dividend * o.expiry);

    const std::size_t n = sample_count(cfg);
    PairStats stats;
    for (std::size_t i = 0; i < n; ++i) {
        const double z = normal(rng);
        double st = o.spot * std::exp(drift + diffusion * z);
        double y = disc * payoff(o.type, st, o.strike);
        double x = disc * st;
        if (cfg.antithetic) {
            st = o.spot * std::exp(drift - diffusion * z);
            y = 0.5 * (y + disc * payoff(o.type, st, o.strike));
            x = 0.5 * (x + disc * st);
        }
        stats.add(y, x);
    }
    return summarise(stats, cfg.antithetic ? 2 * n : n, cfg.control_variate, cv_expectation);
}

// ---------------------------------------------------------------- Monte Carlo Greeks

struct McGreeks {
    McResult price;
    McResult delta;  // pathwise estimator
    McResult vega;   // pathwise estimator
    double gamma;    // central difference of pathwise delta, common random numbers
};

namespace detail {
// Pathwise (infinitesimal perturbation) estimators. For a call, payoff (S_T - K)^+ has
// derivative 1{S_T > K} * dS_T/dtheta almost everywhere, with
//   dS_T/dS_0   = S_T / S_0
//   dS_T/dsigma = S_T (ln(S_T/S_0) - (r - q + sigma^2/2) T) / sigma
inline void pathwise_sample(const EuropeanOption& o, double z, double& price, double& delta,
                            double& vega) {
    const double drift = (o.rate - o.dividend - 0.5 * o.vol * o.vol) * o.expiry;
    const double st = o.spot * std::exp(drift + o.vol * std::sqrt(o.expiry) * z);
    const double disc = std::exp(-o.rate * o.expiry);
    const bool call = o.type == OptionType::Call;
    const bool in_the_money = call ? st > o.strike : st < o.strike;
    const double sign = call ? 1.0 : -1.0;
    price = disc * payoff(o.type, st, o.strike);
    if (!in_the_money) {
        delta = vega = 0.0;
        return;
    }
    const double dst_dsigma =
        st * (std::log(st / o.spot) - (o.rate - o.dividend + 0.5 * o.vol * o.vol) * o.expiry) /
        o.vol;
    delta = sign * disc * st / o.spot;
    vega = sign * disc * dst_dsigma;
}

inline double pathwise_delta_mean(EuropeanOption o, double spot, const McConfig& cfg) {
    o.spot = spot;
    std::mt19937_64 rng(cfg.seed);  // same seed => common random numbers
    std::normal_distribution<double> normal(0.0, 1.0);
    const std::size_t n = sample_count(cfg);
    double sum = 0.0, p, d, v;
    for (std::size_t i = 0; i < n; ++i) {
        const double z = normal(rng);
        pathwise_sample(o, z, p, d, v);
        double total = d;
        if (cfg.antithetic) {
            pathwise_sample(o, -z, p, d, v);
            total = 0.5 * (total + d);
        }
        sum += total;
    }
    return sum / static_cast<double>(n);
}
}  // namespace detail

inline McGreeks mc_european_greeks(const EuropeanOption& o, const McConfig& cfg,
                                   double gamma_bump = 0.01) {
    validate(o);
    std::mt19937_64 rng(cfg.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const std::size_t n = sample_count(cfg);
    PairStats price_s, delta_s, vega_s;
    double p1, d1, v1, p2, d2, v2;
    for (std::size_t i = 0; i < n; ++i) {
        const double z = normal(rng);
        detail::pathwise_sample(o, z, p1, d1, v1);
        if (cfg.antithetic) {
            detail::pathwise_sample(o, -z, p2, d2, v2);
            p1 = 0.5 * (p1 + p2);
            d1 = 0.5 * (d1 + d2);
            v1 = 0.5 * (v1 + v2);
        }
        price_s.add(p1, 0.0);
        delta_s.add(d1, 0.0);
        vega_s.add(v1, 0.0);
    }
    const std::size_t paths = cfg.antithetic ? 2 * n : n;
    const double h = gamma_bump * o.spot;
    const double gamma = (detail::pathwise_delta_mean(o, o.spot + h, cfg) -
                          detail::pathwise_delta_mean(o, o.spot - h, cfg)) /
                         (2.0 * h);
    return {summarise(price_s, paths), summarise(delta_s, paths), summarise(vega_s, paths), gamma};
}

// ---------------------------------------------------------------- Asian option

// Average-price option on n equally spaced fixings t_i = i T / n, i = 1..n.
struct AsianOption {
    double spot, strike, rate, dividend, vol, expiry;
    int fixings;
    OptionType type;
};

inline void validate(const AsianOption& a) {
    validate(EuropeanOption{a.spot, a.strike, a.rate, a.dividend, a.vol, a.expiry, a.type});
    if (a.fixings < 1) throw std::invalid_argument("fixings must be at least 1");
}

// Exact price of the discretely monitored GEOMETRIC average option. ln G is normal with
//   mean  ln S_0 + (r - q - sigma^2/2) T (n+1)/(2n)
//   var   sigma^2 T (n+1)(2n+1)/(6n^2)
inline double geometric_asian_price(const AsianOption& a) {
    validate(a);
    const double n = a.fixings;
    const double mu = std::log(a.spot) +
                      (a.rate - a.dividend - 0.5 * a.vol * a.vol) * a.expiry * (n + 1.0) / (2.0 * n);
    const double var = a.vol * a.vol * a.expiry * (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n * n);
    const double sd = std::sqrt(var);
    const double fwd = std::exp(mu + 0.5 * var);  // E[G]
    const double d1 = (mu - std::log(a.strike) + var) / sd;
    const double d2 = d1 - sd;
    const double disc = std::exp(-a.rate * a.expiry);
    if (a.type == OptionType::Call) return disc * (fwd * norm_cdf(d1) - a.strike * norm_cdf(d2));
    return disc * (a.strike * norm_cdf(-d2) - fwd * norm_cdf(-d1));
}

struct AsianMcResult {
    McResult arithmetic_plain;  // no control variate
    McResult arithmetic_cv;     // geometric Asian as control variate
    McResult geometric_mc;      // simulated geometric price (checks the closed form)
    double geometric_exact;
    double correlation;  // between arithmetic and geometric payoffs
};

// The ARITHMETIC average has no closed form; the geometric one does and is almost perfectly
// correlated with it, which makes it a very strong control variate.
inline AsianMcResult mc_asian(const AsianOption& a, const McConfig& cfg) {
    validate(a);
    std::mt19937_64 rng(cfg.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const double dt = a.expiry / a.fixings;
    const double step_drift = (a.rate - a.dividend - 0.5 * a.vol * a.vol) * dt;
    const double step_vol = a.vol * std::sqrt(dt);
    const double disc = std::exp(-a.rate * a.expiry);
    const double log_s0 = std::log(a.spot);
    const double geo_exact = geometric_asian_price(a);

    const std::size_t n = sample_count(cfg);
    const int legs = cfg.antithetic ? 2 : 1;
    std::vector<double> z(static_cast<std::size_t>(a.fixings));
    PairStats arith_geo, geo_only;
    for (std::size_t i = 0; i < n; ++i) {
        for (auto& zi : z) zi = normal(rng);
        double y = 0.0, x = 0.0;
        for (int leg = 0; leg < legs; ++leg) {
            const double sign = leg == 0 ? 1.0 : -1.0;
            double log_s = log_s0, sum_s = 0.0, sum_log = 0.0;
            for (double zi : z) {
                log_s += step_drift + step_vol * sign * zi;  // exact GBM step
                sum_s += std::exp(log_s);
                sum_log += log_s;
            }
            y += disc * payoff(a.type, sum_s / a.fixings, a.strike);
            x += disc * payoff(a.type, std::exp(sum_log / a.fixings), a.strike);
        }
        y /= legs;
        x /= legs;
        arith_geo.add(y, x);
        geo_only.add(x, 0.0);
    }
    const std::size_t paths = cfg.antithetic ? 2 * n : n;
    AsianMcResult r;
    r.arithmetic_plain = summarise(arith_geo, paths);
    r.arithmetic_cv = summarise(arith_geo, paths, true, geo_exact);
    r.geometric_mc = summarise(geo_only, paths);
    r.geometric_exact = geo_exact;
    const double denom = std::sqrt(arith_geo.var_x() * arith_geo.var_y());
    r.correlation = denom > 0.0 ? arith_geo.cov() / denom : 0.0;
    return r;
}

}  // namespace pricer
