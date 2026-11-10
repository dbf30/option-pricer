// Speed: multi-threaded Monte Carlo and randomised quasi-Monte Carlo.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "black_scholes.hpp"
#include "monte_carlo.hpp"

namespace pricer {

// ---------------------------------------------------------------- multi-threaded Monte Carlo

// Each thread owns an independent Mersenne Twister stream seeded from (seed, thread index)
// through std::seed_seq, accumulates its own PairStats, and the results are merged exactly.
// No shared state, no locks; results are reproducible for a given (seed, threads).
inline McResult mc_european_parallel(const EuropeanOption& o, const McConfig& cfg, unsigned threads) {
    validate(o);
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t total = sample_count(cfg);
    const double drift = (o.rate - o.dividend - 0.5 * o.vol * o.vol) * o.expiry;
    const double diffusion = o.vol * std::sqrt(o.expiry);
    const double disc = std::exp(-o.rate * o.expiry);
    const double cv_expectation = o.spot * std::exp(-o.dividend * o.expiry);

    std::vector<PairStats> partial(threads);
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            const std::size_t begin = total * t / threads, end = total * (t + 1) / threads;
            std::seed_seq seq{static_cast<std::uint32_t>(cfg.seed), static_cast<std::uint32_t>(cfg.seed >> 32), t};
            std::mt19937_64 rng(seq);
            std::normal_distribution<double> normal(0.0, 1.0);
            PairStats local;
            for (std::size_t i = begin; i < end; ++i) {
                const double z = normal(rng);
                double st = o.spot * std::exp(drift + diffusion * z);
                double y = disc * payoff(o.type, st, o.strike), x = disc * st;
                if (cfg.antithetic) {
                    st = o.spot * std::exp(drift - diffusion * z);
                    y = 0.5 * (y + disc * payoff(o.type, st, o.strike));
                    x = 0.5 * (x + disc * st);
                }
                local.add(y, x);
            }
            partial[t] = local;  // each thread writes only its own slot
        });
    }
    for (auto& th : pool) th.join();
    PairStats all;
    for (const auto& p : partial) all.merge(p);
    return summarise(all, cfg.antithetic ? 2 * total : total, cfg.control_variate, cv_expectation);
}

// ---------------------------------------------------------------- quasi-Monte Carlo

// Inverse standard normal CDF: Acklam's rational approximation (relative error ~1e-9),
// polished with one Halley step against the exact CDF to near machine precision.
inline double inverse_norm_cdf(double p) {
    if (!(p > 0.0 && p < 1.0)) throw std::domain_error("inverse_norm_cdf needs 0 < p < 1");
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                               1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                               6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                               -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                               3.754408661907416e+00};
    const double lo = 0.02425, hi = 1.0 - lo;
    double x;
    if (p < lo) {
        const double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= hi) {
        const double q = p - 0.5, r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    const double e = norm_cdf(x) - p;  // Halley refinement
    const double u = e * std::sqrt(2.0 * kPi) * std::exp(0.5 * x * x);
    return x - u / (1.0 + 0.5 * x * u);
}

// Van der Corput sequence in base 2: reverse the bits of i. This is exactly the
// one-dimensional Sobol sequence, the most evenly spread set of points in [0, 1).
inline double van_der_corput(std::uint64_t i) {
    i = ((i >> 1) & 0x5555555555555555ULL) | ((i & 0x5555555555555555ULL) << 1);
    i = ((i >> 2) & 0x3333333333333333ULL) | ((i & 0x3333333333333333ULL) << 2);
    i = ((i >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((i & 0x0F0F0F0F0F0F0F0FULL) << 4);
    i = ((i >> 8) & 0x00FF00FF00FF00FFULL) | ((i & 0x00FF00FF00FF00FFULL) << 8);
    i = ((i >> 16) & 0x0000FFFF0000FFFFULL) | ((i & 0x0000FFFF0000FFFFULL) << 16);
    i = (i >> 32) | (i << 32);
    return static_cast<double>(i >> 11) * 0x1.0p-53;  // top 53 bits -> [0, 1)
}

// Randomised QMC: `shifts` independent random rotations (Cranley–Patterson, u -> frac(u + U))
// of the same low-discrepancy point set. Each rotation gives an unbiased estimate; their spread
// gives an honest standard error, which plain QMC cannot provide. Total paths = points * shifts.
inline McResult rqmc_european(const EuropeanOption& o, std::size_t points, int shifts, std::uint64_t seed) {
    validate(o);
    if (points < 1 || shifts < 2) throw std::invalid_argument("need >= 1 point and >= 2 shifts");
    const double drift = (o.rate - o.dividend - 0.5 * o.vol * o.vol) * o.expiry;
    const double diffusion = o.vol * std::sqrt(o.expiry);
    const double disc = std::exp(-o.rate * o.expiry);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    PairStats across_shifts;
    for (int s = 0; s < shifts; ++s) {
        const double shift = uniform(rng);
        double sum = 0.0;
        for (std::size_t i = 0; i < points; ++i) {
            double u = van_der_corput(i) + shift;
            if (u >= 1.0) u -= 1.0;
            u = std::clamp(u, 1e-16, 1.0 - 1e-16);
            const double st = o.spot * std::exp(drift + diffusion * inverse_norm_cdf(u));
            sum += disc * payoff(o.type, st, o.strike);
        }
        across_shifts.add(sum / static_cast<double>(points), 0.0);
    }
    return summarise(across_shifts, points * static_cast<std::size_t>(shifts));
}

}  // namespace pricer
