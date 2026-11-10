// Self-contained test suite: no framework needed. Exits non-zero on any failure.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "pricer/american.hpp"
#include "pricer/black_scholes.hpp"
#include "pricer/hull_white.hpp"
#include "pricer/monte_carlo.hpp"
#include "pricer/parallel.hpp"

using namespace pricer;

static int failures = 0, checks = 0;

#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        ++checks;                                                          \
        if (!(cond)) {                                                     \
            ++failures;                                                    \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);      \
        }                                                                  \
    } while (0)

static bool close(double a, double b, double tol) { return std::abs(a - b) <= tol; }

// Textbook reference: Hull, Options, Futures and Other Derivatives, Example 15.6
// (S=42, K=40, r=10%, vol=20%, T=0.5): call 4.76, put 0.81.
static void test_reference_values() {
    EuropeanOption o{42, 40, 0.10, 0.0, 0.20, 0.5, OptionType::Call};
    CHECK(close(bs_price(o), 4.7594, 5e-4), "Hull call price");
    o.type = OptionType::Put;
    CHECK(close(bs_price(o), 0.8086, 5e-4), "Hull put price");
}

// C - P = S e^{-qT} - K e^{-rT} must hold to machine precision.
static void test_put_call_parity() {
    for (double s : {60.0, 100.0, 140.0})
        for (double vol : {0.05, 0.3, 0.9})
            for (double t : {0.05, 1.0, 5.0}) {
                EuropeanOption c{s, 100, 0.04, 0.015, vol, t, OptionType::Call};
                EuropeanOption p = c;
                p.type = OptionType::Put;
                const double lhs = bs_price(c) - bs_price(p);
                const double rhs = s * std::exp(-0.015 * t) - 100 * std::exp(-0.04 * t);
                CHECK(close(lhs, rhs, 1e-10), "put-call parity");
            }
}

// Each analytic Greek must match a central finite difference of the analytic price.
static void test_greeks_vs_finite_differences() {
    for (OptionType type : {OptionType::Call, OptionType::Put})
        for (double s : {80.0, 100.0, 120.0}) {
            const EuropeanOption o{s, 100, 0.05, 0.02, 0.25, 0.75, type};
            const Greeks g = bs_greeks(o);
            auto bumped = [&](auto field, double h) {
                EuropeanOption u = o, d = o;
                u.*field += h;
                d.*field -= h;
                return std::pair{bs_price(u), bs_price(d)};
            };
            // Central differences carry O(h^2) truncation error: small bump for delta,
            // larger for gamma (whose second difference is more sensitive to rounding).
            const double hd = 1e-4 * s;
            auto [du, dd] = bumped(&EuropeanOption::spot, hd);
            CHECK(close(g.delta, (du - dd) / (2 * hd), 1e-7), "delta vs FD");
            const double hs = 1e-3 * s;
            auto [su, sd] = bumped(&EuropeanOption::spot, hs);
            CHECK(close(g.gamma, (su - 2 * g.price + sd) / (hs * hs), 1e-5), "gamma vs FD");
            auto [vu, vd] = bumped(&EuropeanOption::vol, 1e-5);
            CHECK(close(g.vega, (vu - vd) / 2e-5, 1e-5), "vega vs FD");
            auto [ru, rd] = bumped(&EuropeanOption::rate, 1e-5);
            CHECK(close(g.rho, (ru - rd) / 2e-5, 1e-5), "rho vs FD");
            // theta = dV/dt = -dV/dT
            auto [tu, td] = bumped(&EuropeanOption::expiry, 1e-5);
            CHECK(close(g.theta, -(tu - td) / 2e-5, 1e-4), "theta vs FD");
        }
}

static void test_implied_vol_round_trip() {
    for (OptionType type : {OptionType::Call, OptionType::Put})
        for (double k : {70.0, 100.0, 130.0})
            for (double vol : {0.05, 0.2, 0.6, 1.5}) {
                EuropeanOption o{100, k, 0.03, 0.0, vol, 1.0, type};
                const double price = bs_price(o);
                // Vol is only identifiable through time value (price above intrinsic bound).
                const double fwd = 100.0, pv_k = k * std::exp(-0.03);
                const double bound = type == OptionType::Call ? std::max(0.0, fwd - pv_k)
                                                              : std::max(0.0, pv_k - fwd);
                if (price - bound < 1e-6) continue;
                CHECK(close(implied_vol(price, o), vol, 1e-7), "implied vol round trip");
            }
    bool threw = false;
    try {
        implied_vol(150.0, EuropeanOption{100, 100, 0.03, 0.0, 0.2, 1.0, OptionType::Call});
    } catch (const std::domain_error&) {
        threw = true;
    }
    CHECK(threw, "implied vol rejects price above no-arbitrage bound");
}

static void test_input_validation() {
    bool threw = false;
    try {
        bs_price(EuropeanOption{100, 100, 0.05, 0.0, -0.2, 1.0, OptionType::Call});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw, "negative vol rejected");
}

// Every Monte Carlo variant must land within 4 standard errors of the exact price.
static void test_mc_matches_black_scholes() {
    for (OptionType type : {OptionType::Call, OptionType::Put})
        for (double k : {80.0, 100.0, 125.0}) {
            const EuropeanOption o{100, k, 0.05, 0.02, 0.2, 1.0, type};
            const double exact = bs_price(o);
            for (bool anti : {false, true})
                for (bool cv : {false, true}) {
                    const McResult r = mc_european(o, {400'000, 123, anti, cv});
                    CHECK(std::abs(r.price - exact) < 4 * r.std_error, "MC within 4 SE");
                }
        }
}

static void test_variance_reduction_helps() {
    const EuropeanOption o{100, 100, 0.05, 0.02, 0.2, 1.0, OptionType::Call};
    const McResult plain = mc_european(o, {200'000, 5, false, false});
    const McResult both = mc_european(o, {200'000, 5, true, true});
    CHECK(both.std_error < 0.5 * plain.std_error, "antithetic + CV at least halves the SE");
}

// Standard error should fall as 1/sqrt(N): 100x the paths => ~10x smaller SE.
static void test_convergence_rate() {
    const EuropeanOption o{100, 100, 0.05, 0.02, 0.2, 1.0, OptionType::Call};
    const double se_small = mc_european(o, {10'000, 9, false, false}).std_error;
    const double se_large = mc_european(o, {1'000'000, 9, false, false}).std_error;
    CHECK(close(se_small / se_large, 10.0, 0.5), "SE scales as 1/sqrt(N)");
}

static void test_mc_greeks() {
    for (OptionType type : {OptionType::Call, OptionType::Put}) {
        const EuropeanOption o{100, 105, 0.05, 0.02, 0.25, 0.5, type};
        const Greeks g = bs_greeks(o);
        const McGreeks m = mc_european_greeks(o, {1'000'000, 21, true, false});
        CHECK(std::abs(m.delta.price - g.delta) < 4 * m.delta.std_error, "pathwise delta");
        CHECK(std::abs(m.vega.price - g.vega) < 4 * m.vega.std_error, "pathwise vega");
        CHECK(std::abs(m.gamma - g.gamma) < 0.03 * g.gamma, "CRN gamma within 3%");
    }
}

static void test_asian() {
    for (OptionType type : {OptionType::Call, OptionType::Put}) {
        const AsianOption a{100, 100, 0.05, 0.02, 0.3, 1.0, 52, type};
        const AsianMcResult r = mc_asian(a, {100'000, 77, false, false});
        CHECK(std::abs(r.geometric_mc.price - r.geometric_exact) < 4 * r.geometric_mc.std_error,
              "geometric Asian closed form matches simulation");
        CHECK(std::abs(r.arithmetic_cv.price - r.arithmetic_plain.price) <
                  4 * r.arithmetic_plain.std_error,
              "CV and plain arithmetic estimates agree");
        CHECK(r.arithmetic_cv.std_error < 0.2 * r.arithmetic_plain.std_error,
              "geometric control variate cuts SE by 5x+");
        // AM >= GM pathwise, so an arithmetic call is worth more (a put less) than the geometric.
        if (type == OptionType::Call)
            CHECK(r.arithmetic_cv.price > r.geometric_exact, "arithmetic call >= geometric call");
        else
            CHECK(r.arithmetic_cv.price < r.geometric_exact, "arithmetic put <= geometric put");
    }
    // One fixing at T: the geometric Asian is just a European option.
    const AsianOption one{100, 95, 0.05, 0.01, 0.2, 2.0, 1, OptionType::Call};
    CHECK(close(geometric_asian_price(one),
                bs_price(EuropeanOption{100, 95, 0.05, 0.01, 0.2, 2.0, OptionType::Call}), 1e-12),
          "1-fixing geometric Asian equals European");
}

// ---------------------------------------------------------------- American options

static void test_binomial() {
    for (OptionType type : {OptionType::Call, OptionType::Put}) {
        const EuropeanOption o{100, 95, 0.05, 0.02, 0.25, 1.0, type};
        CHECK(close(binomial_price(o, 4000, false), bs_price(o), 3e-3), "European binomial -> Black-Scholes");
    }
    // Without dividends an American call is never exercised early (Merton 1973).
    const EuropeanOption call{100, 100, 0.05, 0.0, 0.2, 1.0, OptionType::Call};
    CHECK(close(binomial_price(call, 4000, true), binomial_price(call, 4000, false), 1e-12),
          "American call, no dividends == European");
    // Early exercise has value for a put, and the American price never falls below intrinsic.
    const EuropeanOption put{36, 40, 0.06, 0.0, 0.2, 1.0, OptionType::Put};
    const double am = binomial_price(put, 4000, true);
    CHECK(am > bs_price(put) + 0.5, "American put carries an early-exercise premium");
    CHECK(close(am, 4.478, 0.012), "Longstaff-Schwartz (2001) finite-difference reference 4.478");
    const EuropeanOption deep{20, 40, 0.06, 0.0, 0.2, 1.0, OptionType::Put};
    CHECK(close(binomial_price(deep, 2000, true), 20.0, 1e-9), "deep ITM American put == intrinsic");
}

static void test_lsm() {
    struct Row {
        double s, vol, t;
    };
    for (const Row r : {Row{36, 0.2, 1}, Row{40, 0.2, 1}, Row{44, 0.4, 2}}) {
        const EuropeanOption put{r.s, 40, 0.06, 0.0, r.vol, r.t, OptionType::Put};
        const double binom = binomial_price(put, 4000, true);
        const LsmResult lsm = lsm_american(put, {60'000, static_cast<int>(50 * r.t), 3, 99});
        // LSM prices a Bermudan (finite exercise dates) with an estimated policy, so it sits slightly
        // below the continuously exercisable tree price: allow 4 SE plus a small bias.
        CHECK(std::abs(lsm.price.price - binom) < 4 * lsm.price.std_error + 0.02, "LSM matches binomial");
        CHECK(lsm.price.price > bs_price(put), "LSM American > European");
    }
}

// ---------------------------------------------------------------- Hull–White

static void test_hull_white() {
    const NelsonSiegel ns{0.04, -0.005, 0.01, 2.0};
    const auto curve = [ns](double t) { return ns.discount(t); };
    const HullWhite hw(0.05, 0.01, curve);
    CHECK(close(hw.V(3.0, 3.0), 0.0, 1e-15) && close(hw.B(2.0, 2.0), 0.0, 1e-15), "B and V vanish at t = T");
    CHECK(close(hw.bond_price(0.0, 7.0, 0.0), curve(7.0), 1e-14), "model bond at t=0 == market curve");
    for (double T : {1.0, 5.0, 10.0}) {
        const McResult r = mc_hw_zero_bond(hw, T, {200'000, 3, true, false});
        CHECK(std::abs(r.price - curve(T)) < 4 * r.std_error + 1e-12, "MC zero bond reproduces the curve");
    }
    // Bond option put-call parity: ZBC - ZBP = P(0,T2) - X P(0,T1).
    const double c = hw.bond_option(2, 7, 0.85, OptionType::Call), p = hw.bond_option(2, 7, 0.85, OptionType::Put);
    CHECK(close(c - p, curve(7) - 0.85 * curve(2), 1e-14), "zero-coupon bond option parity");
    for (double k : {0.03, 0.045}) {
        const double exact = hw.caplet(4.75, 5.0, k);
        const McResult r = mc_hw_caplet(hw, 4.75, 5.0, k, {400'000, 8, true, true});
        CHECK(std::abs(r.price - exact) < 4 * r.std_error, "MC caplet matches closed form");
    }
    // Caplet value must increase with vol and fall with strike.
    const HullWhite hw2(0.05, 0.02, curve);
    CHECK(hw2.caplet(4.75, 5.0, 0.04) > hw.caplet(4.75, 5.0, 0.04), "caplet increasing in sigma");
    CHECK(hw.caplet(4.75, 5.0, 0.03) > hw.caplet(4.75, 5.0, 0.04), "caplet decreasing in strike");
    const double cap = hw.cap(0, 10, 0.25, 0.04);
    CHECK(close(calibrate_hw_sigma(cap, 0.05, curve, 0, 10, 0.25, 0.04), 0.01, 1e-9), "sigma calibration round trip");
}

// ---------------------------------------------------------------- speed

static void test_parallel_and_qmc() {
    // Merging two halves must give exactly the statistics of the whole.
    PairStats a, b, all;
    std::mt19937_64 g(1);
    std::normal_distribution<double> n;
    for (int i = 0; i < 5000; ++i) {
        const double y = n(g), x = y + n(g);
        (i < 1700 ? a : b).add(y, x);
        all.add(y, x);
    }
    a.merge(b);
    CHECK(close(a.mean_y(), all.mean_y(), 1e-14) && close(a.var_y(), all.var_y(), 1e-12) &&
              close(a.cov(), all.cov(), 1e-12),
          "parallel merge == sequential");

    const EuropeanOption o{100, 100, 0.05, 0.02, 0.2, 1.0, OptionType::Call};
    const double exact = bs_price(o);
    for (unsigned t : {1u, 3u, 4u}) {
        const McResult r = mc_european_parallel(o, {300'000, 17, true, true}, t);
        CHECK(std::abs(r.price - exact) < 4 * r.std_error, "threaded MC within 4 SE");
        CHECK(r.paths == 300'000, "threaded MC uses every path");
    }
    const McResult r1 = mc_european_parallel(o, {100'000, 5, false, false}, 3);
    const McResult r2 = mc_european_parallel(o, {100'000, 5, false, false}, 3);
    CHECK(r1.price == r2.price, "threaded MC reproducible for fixed seed and thread count");

    double worst = 0.0;
    for (int k = 1; k < 2000; ++k) {
        const double p = k / 2000.0;
        worst = std::max(worst, std::abs(norm_cdf(inverse_norm_cdf(p)) - p));
    }
    CHECK(worst < 1e-14, "inverse normal CDF round trip");
    CHECK(close(inverse_norm_cdf(0.975), 1.959963984540054, 1e-13), "inverse normal at 97.5%");
    CHECK(van_der_corput(1) == 0.5 && van_der_corput(2) == 0.25 && van_der_corput(3) == 0.75 &&
              van_der_corput(5) == 0.625,
          "van der Corput sequence");

    const McResult q = rqmc_european(o, 16384, 16, 3);
    const McResult m = mc_european(o, {16384 * 16, 3, false, false});
    CHECK(std::abs(q.price - exact) < 4 * q.std_error, "RQMC within 4 SE");
    CHECK(q.std_error < m.std_error / 10.0, "RQMC beats plain MC by 100x+ in variance");
}

int main() {
    test_reference_values();
    test_put_call_parity();
    test_greeks_vs_finite_differences();
    test_implied_vol_round_trip();
    test_input_validation();
    test_mc_matches_black_scholes();
    test_variance_reduction_helps();
    test_convergence_rate();
    test_mc_greeks();
    test_asian();
    test_binomial();
    test_lsm();
    test_hull_white();
    test_parallel_and_qmc();
    std::printf("%d/%d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
