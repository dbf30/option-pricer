// American options: priced two independent ways so each checks the other.
//   1. Cox–Ross–Rubinstein binomial tree (deterministic, converges to the true price).
//   2. Longstaff–Schwartz least-squares Monte Carlo (regression estimate of the continuation
//      value; the method used in practice for high-dimensional early-exercise products).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "black_scholes.hpp"
#include "monte_carlo.hpp"

namespace pricer {

// ---------------------------------------------------------------- binomial tree

// CRR tree: up = e^{sigma sqrt(dt)}, down = 1/up, risk-neutral p = (e^{(r-q)dt} - down)/(up - down).
// Backward induction; at each node an American option takes max(continuation, exercise).
// O(steps^2) time, O(steps) memory.
inline double binomial_price(const EuropeanOption& o, int steps, bool american) {
    validate(o);
    if (steps < 1) throw std::invalid_argument("steps must be at least 1");
    const double dt = o.expiry / steps;
    const double up = std::exp(o.vol * std::sqrt(dt));
    const double down = 1.0 / up;
    const double p = (std::exp((o.rate - o.dividend) * dt) - down) / (up - down);
    if (!(p > 0.0 && p < 1.0))
        throw std::domain_error("too few steps: risk-neutral probability outside (0, 1)");
    const double disc = std::exp(-o.rate * dt);
    const double up2 = up * up;

    std::vector<double> v(static_cast<std::size_t>(steps) + 1);
    double s = o.spot * std::pow(down, steps);  // lowest terminal node
    for (int j = 0; j <= steps; ++j, s *= up2) v[j] = payoff(o.type, s, o.strike);

    for (int i = steps - 1; i >= 0; --i) {
        s = o.spot * std::pow(down, i);
        for (int j = 0; j <= i; ++j, s *= up2) {
            const double cont = disc * (p * v[j + 1] + (1.0 - p) * v[j]);
            v[j] = american ? std::max(cont, payoff(o.type, s, o.strike)) : cont;
        }
    }
    return v[0];
}

// ---------------------------------------------------------------- Longstaff–Schwartz

struct LsmConfig {
    std::size_t paths = 100'000;  // per pass (training and pricing each use this many)
    int exercise_dates = 50;      // equally spaced; the product priced is Bermudan
    int degree = 3;               // polynomial basis 1, x, ..., x^degree in x = S/K
    std::uint64_t seed = 42;
};

struct LsmResult {
    McResult price;          // out-of-sample estimate (a lower bound in expectation)
    double in_sample_price;  // fitted and priced on the same paths (can overfit)
    std::vector<double> boundary;  // estimated exercise boundary S*(t_k), NaN if never exercised
};

namespace detail {
// Solve the least-squares problem min ||A b - y|| via the normal equations with Gaussian
// elimination and partial pivoting. A has `cols` columns; small, so this is cheap and stable
// enough once x = S/K is scaled to O(1).
inline std::vector<double> least_squares(const std::vector<double>& ata, std::vector<double> aty,
                                         int cols) {
    std::vector<double> m(ata);
    for (int c = 0; c < cols; ++c) {
        int piv = c;
        for (int r = c + 1; r < cols; ++r)
            if (std::abs(m[r * cols + c]) > std::abs(m[piv * cols + c])) piv = r;
        if (std::abs(m[piv * cols + c]) < 1e-14) throw std::runtime_error("singular regression");
        if (piv != c) {
            for (int k = 0; k < cols; ++k) std::swap(m[c * cols + k], m[piv * cols + k]);
            std::swap(aty[c], aty[piv]);
        }
        for (int r = c + 1; r < cols; ++r) {
            const double f = m[r * cols + c] / m[c * cols + c];
            for (int k = c; k < cols; ++k) m[r * cols + k] -= f * m[c * cols + k];
            aty[r] -= f * aty[c];
        }
    }
    std::vector<double> b(cols);
    for (int r = cols - 1; r >= 0; --r) {
        double sum = aty[r];
        for (int k = r + 1; k < cols; ++k) sum -= m[r * cols + k] * b[k];
        b[r] = sum / m[r * cols + r];
    }
    return b;
}

inline double poly(const std::vector<double>& beta, double x) {
    double v = 0.0;
    for (auto it = beta.rbegin(); it != beta.rend(); ++it) v = v * x + *it;  // Horner
    return v;
}

// Simulate `n` paths (antithetic pairs) on the exercise grid; row-major [path][date].
inline std::vector<double> simulate_grid(const EuropeanOption& o, std::size_t n, int dates,
                                         std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const double dt = o.expiry / dates;
    const double drift = (o.rate - o.dividend - 0.5 * o.vol * o.vol) * dt;
    const double vol = o.vol * std::sqrt(dt);
    std::vector<double> s(n * dates);
    for (std::size_t p = 0; p + 1 < n; p += 2) {
        double a = std::log(o.spot), b = a;
        for (int k = 0; k < dates; ++k) {
            const double z = normal(rng);
            a += drift + vol * z;
            b += drift - vol * z;
            s[p * dates + k] = std::exp(a);
            s[(p + 1) * dates + k] = std::exp(b);
        }
    }
    return s;
}
}  // namespace detail

inline LsmResult lsm_american(const EuropeanOption& o, const LsmConfig& cfg) {
    validate(o);
    if (cfg.paths < 1000 || cfg.paths % 2) throw std::invalid_argument("paths must be even, >= 1000");
    if (cfg.exercise_dates < 2 || cfg.degree < 1 || cfg.degree > 6)
        throw std::invalid_argument("need >= 2 exercise dates and degree in 1..6");
    const int dates = cfg.exercise_dates, cols = cfg.degree + 1;
    const std::size_t n = cfg.paths;
    const double dt = o.expiry / dates;
    // disc[k] = e^{-r t_k}, with exercise dates t_k = (k + 1) dt for k = 0..dates-1
    std::vector<double> disc(dates + 1);
    for (int k = 0; k <= dates; ++k) disc[k] = std::exp(-o.rate * dt * k);

    // ---- pass 1: fit the continuation-value regressions on training paths
    const auto s = detail::simulate_grid(o, n, dates, cfg.seed);
    std::vector<double> cash(n);      // value of following the current policy, at date `when`
    std::vector<int> when(n, dates - 1);
    for (std::size_t p = 0; p < n; ++p) cash[p] = payoff(o.type, s[p * dates + dates - 1], o.strike);

    std::vector<std::vector<double>> beta(dates);
    LsmResult result;
    result.boundary.assign(dates, std::nan(""));
    result.boundary[dates - 1] = o.strike;
    std::vector<double> ata(cols * cols), aty(cols), basis(cols);
    for (int k = dates - 2; k >= 0; --k) {
        std::fill(ata.begin(), ata.end(), 0.0);
        std::fill(aty.begin(), aty.end(), 0.0);
        std::size_t itm = 0;
        for (std::size_t p = 0; p < n; ++p) {
            const double sk = s[p * dates + k];
            if (payoff(o.type, sk, o.strike) <= 0.0) continue;  // regress on in-the-money paths only
            ++itm;
            const double y = cash[p] * disc[when[p] - k];
            const double x = sk / o.strike;
            basis[0] = 1.0;
            for (int c = 1; c < cols; ++c) basis[c] = basis[c - 1] * x;
            for (int i = 0; i < cols; ++i) {
                aty[i] += basis[i] * y;
                for (int j = 0; j < cols; ++j) ata[i * cols + j] += basis[i] * basis[j];
            }
        }
        if (itm < static_cast<std::size_t>(4 * cols)) continue;  // too few points: never exercise
        beta[k] = detail::least_squares(ata, aty, cols);
        double boundary = std::nan("");
        for (std::size_t p = 0; p < n; ++p) {
            const double sk = s[p * dates + k];
            const double ex = payoff(o.type, sk, o.strike);
            if (ex > 0.0 && ex > detail::poly(beta[k], sk / o.strike)) {
                cash[p] = ex;
                when[p] = k;
                // put: boundary = highest spot exercised; call: lowest
                if (std::isnan(boundary) || (o.type == OptionType::Put ? sk > boundary : sk < boundary))
                    boundary = sk;
            }
        }
        result.boundary[k] = boundary;
    }
    double in_sample = 0.0;
    for (std::size_t p = 0; p < n; ++p) in_sample += cash[p] * disc[when[p] + 1];
    result.in_sample_price = in_sample / static_cast<double>(n);

    // ---- pass 2: apply the fitted policy to fresh, independent paths
    const auto s2 = detail::simulate_grid(o, n, dates, cfg.seed ^ 0x9E3779B97F4A7C15ULL);
    PairStats stats;
    for (std::size_t p = 0; p < n; p += 2) {
        double pair = 0.0;
        for (std::size_t q = p; q < p + 2; ++q) {
            double value = 0.0;
            for (int k = 0; k < dates; ++k) {
                const double sk = s2[q * dates + k];
                const double ex = payoff(o.type, sk, o.strike);
                const bool last = k == dates - 1;
                if (ex > 0.0 && (last || (!beta[k].empty() && ex > detail::poly(beta[k], sk / o.strike)))) {
                    value = ex * disc[k + 1];
                    break;
                }
            }
            pair += 0.5 * value;
        }
        stats.add(pair, 0.0);
    }
    result.price = summarise(stats, n);
    // Exercising immediately is always available to an American holder.
    result.price.price = std::max(result.price.price, payoff(o.type, o.spot, o.strike));
    return result;
}

}  // namespace pricer
