// Command-line driver.
//   pricer                          full demo report
//   pricer price S K r q vol T call|put [paths]
//   pricer american S K r q vol T call|put
//   pricer convergence out.csv      Monte Carlo / QMC convergence data for plotting
//   pricer benchmark                throughput vs thread count
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "pricer/american.hpp"
#include "pricer/black_scholes.hpp"
#include "pricer/hull_white.hpp"
#include "pricer/monte_carlo.hpp"
#include "pricer/parallel.hpp"

using namespace pricer;

namespace {

template <class F>
double time_ms(F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void print_mc(const char* label, const McResult& r, double exact, double ms) {
    std::printf("  %-28s %9.5f  SE %8.5f  95%% CI [%8.5f, %8.5f]  err %+8.5f  %7.1f ms\n", label,
                r.price, r.std_error, r.ci95_low(), r.ci95_high(), r.price - exact, ms);
}

NelsonSiegel demo_curve() { return {0.040, -0.005, 0.010, 2.0}; }  // illustrative, not market data

void section_european() {
    const EuropeanOption call{100.0, 100.0, 0.05, 0.02, 0.20, 1.0, OptionType::Call};
    const std::size_t paths = 1'000'000;
    std::printf("== 1. European call  S=100 K=100 r=5%% q=2%% vol=20%% T=1y\n\n");
    const Greeks g = bs_greeks(call);
    std::printf("Black-Scholes (exact)\n");
    std::printf("  price %.5f  delta %.5f  gamma %.5f  vega %.5f  theta %.5f  rho %.5f\n\n",
                g.price, g.delta, g.gamma, g.vega, g.theta, g.rho);

    std::printf("Monte Carlo, %zu paths\n", paths);
    struct Variant {
        const char* name;
        bool anti, cv;
    };
    const Variant variants[] = {{"plain", false, false},
                                {"antithetic", true, false},
                                {"control variate", false, true},
                                {"antithetic + control var.", true, true}};
    double plain_se = 0.0;
    for (const auto& v : variants) {
        McResult r;
        const double ms = time_ms([&] { r = mc_european(call, {paths, 42, v.anti, v.cv}); });
        print_mc(v.name, r, g.price, ms);
        if (!v.anti && !v.cv) plain_se = r.std_error;
        else
            std::printf("  %-28s variance cut %.1fx vs plain\n", "",
                        (plain_se * plain_se) / (r.std_error * r.std_error));
    }

    std::printf("\nMonte Carlo Greeks (pathwise delta/vega, gamma by CRN bump), %zu paths\n", paths);
    const McGreeks mg = mc_european_greeks(call, {paths, 7, true, false});
    std::printf("  delta %.5f +/- %.5f   (exact %.5f)\n", mg.delta.price, mg.delta.std_error, g.delta);
    std::printf("  vega  %.5f +/- %.5f   (exact %.5f)\n", mg.vega.price, mg.vega.std_error, g.vega);
    std::printf("  gamma %.5f             (exact %.5f)\n", mg.gamma, g.gamma);
    std::printf("\nImplied vol recovered from the exact price: %.10f (input 0.20)\n", implied_vol(g.price, call));
}

void section_asian() {
    const AsianOption asian{100.0, 100.0, 0.05, 0.02, 0.20, 1.0, 252, OptionType::Call};
    const std::size_t paths = 100'000;
    AsianMcResult ar;
    const double ms = time_ms([&] { ar = mc_asian(asian, {paths, 11, false, false}); });
    std::printf("\n== 2. Arithmetic Asian call, 252 daily fixings, %zu paths (%.0f ms)\n\n", paths, ms);
    std::printf("  geometric Asian: exact %.5f  vs MC %.5f +/- %.5f\n", ar.geometric_exact,
                ar.geometric_mc.price, ar.geometric_mc.std_error);
    std::printf("  arithmetic, plain MC        %.5f  SE %.5f\n", ar.arithmetic_plain.price, ar.arithmetic_plain.std_error);
    std::printf("  arithmetic, geometric CV    %.5f  SE %.5f\n", ar.arithmetic_cv.price, ar.arithmetic_cv.std_error);
    std::printf("  payoff correlation %.4f, variance cut %.0fx\n", ar.correlation,
                std::pow(ar.arithmetic_plain.std_error / ar.arithmetic_cv.std_error, 2));
}

void section_american() {
    std::printf("\n== 3. American put: binomial tree vs Longstaff-Schwartz (K=40 r=6%%)\n");
    std::printf("   Reference values from Longstaff & Schwartz (2001), Table 1\n\n");
    std::printf("   S   vol  T | European | binomial 5000 | LSM 100k paths, 50/yr | paper FD | paper LSM\n");
    struct Row {
        double s, vol, t, fd, lsm;
    };
    const Row rows[] = {{36, 0.2, 1, 4.478, 4.472}, {36, 0.4, 2, 8.508, 8.488},
                        {40, 0.2, 1, 2.314, 2.313}, {44, 0.2, 1, 1.110, 1.118}};
    for (const auto& r : rows) {
        const EuropeanOption put{r.s, 40.0, 0.06, 0.0, r.vol, r.t, OptionType::Put};
        const double binom = binomial_price(put, 5000, true);
        const LsmResult lsm = lsm_american(put, {100'000, static_cast<int>(50 * r.t), 3, 42});
        std::printf("  %3.0f  %.1f  %.0f |  %7.4f |       %7.4f | %7.4f +/- %.4f       |   %6.3f |    %6.3f\n",
                    r.s, r.vol, r.t, bs_price(put), binom, lsm.price.price, lsm.price.std_error, r.fd, r.lsm);
    }
    const EuropeanOption call{100, 100, 0.05, 0.0, 0.2, 1.0, OptionType::Call};
    std::printf("\n  No dividends => never exercise a call early: American %.5f vs Black-Scholes %.5f\n",
                binomial_price(call, 5000, true), bs_price(call));
}

void section_rates() {
    const NelsonSiegel ns = demo_curve();
    const auto curve = [ns](double t) { return ns.discount(t); };
    const HullWhite hw(0.05, 0.01, curve);
    std::printf("\n== 4. Hull-White short rate (a=0.05, sigma=1%%), fitted to an illustrative curve\n\n");
    std::printf("  Zero-coupon bonds, 400k paths (antithetic): model must reproduce the curve\n");
    for (double T : {1.0, 5.0, 10.0, 30.0}) {
        const McResult r = mc_hw_zero_bond(hw, T, {400'000, 3, true, false});
        std::printf("   T=%4.0fy  zero %.3f%%  curve %.6f  MC %.6f +/- %.6f\n", T, 100 * ns.zero_rate(T),
                    curve(T), r.price, r.std_error);
    }
    std::printf("\n  Caplets on 3M rate fixing in 4.75y (forward %.3f%%), per 1 notional, 1M paths\n",
                100 * hw.forward_rate(4.75, 5.0));
    for (double k : {0.03, 0.04, 0.05}) {
        const double exact = hw.caplet(4.75, 5.0, k);
        const McResult plain = mc_hw_caplet(hw, 4.75, 5.0, k, {1'000'000, 5, false, false});
        const McResult vr = mc_hw_caplet(hw, 4.75, 5.0, k, {1'000'000, 5, true, true});
        std::printf("   K=%.0f%%  exact %.7f | MC %.7f +/- %.7f | anti+CV %.7f +/- %.7f (variance cut %.1fx)\n",
                    100 * k, exact, plain.price, plain.std_error, vr.price, vr.std_error,
                    std::pow(plain.std_error / vr.std_error, 2));
    }
    const double cap = hw.cap(0.0, 10.0, 0.25, 0.04);
    std::printf("\n  10y quarterly cap, K=4%%: %.6f per 1 notional (%.0f on 10m)\n", cap, cap * 1e7);
    std::printf("  Calibrating sigma back from that cap price: %.8f (true 0.01)\n",
                calibrate_hw_sigma(cap, 0.05, curve, 0.0, 10.0, 0.25, 0.04));
}

void section_speed() {
    const EuropeanOption call{100.0, 100.0, 0.05, 0.02, 0.20, 1.0, OptionType::Call};
    const double exact = bs_price(call);
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    std::printf("\n== 5. Speed\n\n  Multi-threaded Monte Carlo, 20M paths (%u hardware threads)\n", hw);
    std::vector<unsigned> counts;  // 1, 2, 4, ... and always the full machine
    for (unsigned t = 1; t < hw; t *= 2) counts.push_back(t);
    counts.push_back(hw);
    double base = 0.0;
    for (unsigned t : counts) {
        McResult r;
        const double ms = time_ms([&] { r = mc_european_parallel(call, {20'000'000, 42, false, false}, t); });
        if (t == 1) base = ms;
        std::printf("   %2u thread(s): %6.0f ms  %5.1f M paths/s  speed-up %.2fx  price %.5f +/- %.5f\n", t, ms,
                    20.0 / ms * 1000.0, base / ms, r.price, r.std_error);
    }
    std::printf("\n  Randomised QMC (van der Corput / 1-D Sobol, 32 random shifts) vs plain MC, same paths\n");
    for (std::size_t pts : {1024ul, 16384ul, 262144ul}) {
        const McResult q = rqmc_european(call, pts, 32, 7);
        const McResult m = mc_european(call, {pts * 32, 7, false, false});
        std::printf("   N=%8zu  RQMC SE %.6f  MC SE %.6f  variance cut %7.0fx  RQMC err %+.6f\n", pts * 32,
                    q.std_error, m.std_error, std::pow(m.std_error / q.std_error, 2), q.price - exact);
    }
}

int demo() {
    section_european();
    section_asian();
    section_american();
    section_rates();
    section_speed();
    return 0;
}

EuropeanOption parse_option(int argc, char** argv) {
    if (argc < 9) throw std::invalid_argument("usage: pricer price|american S K r q vol T call|put [paths]");
    const std::string kind = argv[8];
    if (kind != "call" && kind != "put") throw std::invalid_argument("option type must be 'call' or 'put'");
    return {std::atof(argv[2]), std::atof(argv[3]), std::atof(argv[4]), std::atof(argv[5]),
            std::atof(argv[6]), std::atof(argv[7]), kind == "call" ? OptionType::Call : OptionType::Put};
}

int price_cmd(int argc, char** argv) {
    const EuropeanOption o = parse_option(argc, argv);
    const std::size_t paths = argc > 9 ? std::strtoull(argv[9], nullptr, 10) : 1'000'000;
    const Greeks g = bs_greeks(o);
    const McResult r = mc_european(o, {paths, 42, true, true});
    std::printf("exact %.6f | MC %.6f +/- %.6f (%zu paths)\n", g.price, r.price, r.std_error, r.paths);
    std::printf("delta %.6f gamma %.6f vega %.6f theta %.6f rho %.6f\n", g.delta, g.gamma, g.vega, g.theta, g.rho);
    return 0;
}

int american_cmd(int argc, char** argv) {
    const EuropeanOption o = parse_option(argc, argv);
    const double binom = binomial_price(o, 5000, true);
    const LsmResult lsm = lsm_american(o, {100'000, std::max(2, static_cast<int>(50 * o.expiry)), 3, 42});
    std::printf("European %.6f | American: binomial %.6f, LSM %.6f +/- %.6f | early-exercise premium %.6f\n",
                bs_price(o), binom, lsm.price.price, lsm.price.std_error, binom - bs_price(o));
    return 0;
}

// Standard error and RMS error across independent seeds, for paths = 10^2 .. 10^6.
int convergence_cmd(const char* path) {
    const EuropeanOption call{100.0, 100.0, 0.05, 0.02, 0.20, 1.0, OptionType::Call};
    const double exact = bs_price(call);
    const int seeds = 30, shifts = 16;
    std::ofstream out(path);
    if (!out) throw std::runtime_error(std::string("cannot open ") + path + " for writing");
    out << "method,paths,mean_std_error,rms_error\n";
    for (const char* method : {"plain", "antithetic+cv", "rqmc"}) {
        const std::string m = method;
        for (int k = 4; k <= 12; ++k) {  // 10^(k/2): 100, 316, 1000, ..., 1e6
            const auto n = static_cast<std::size_t>(std::llround(std::pow(10.0, k / 2.0)));
            double se_sum = 0.0, sq_err = 0.0;
            for (int s = 0; s < seeds; ++s) {
                McResult r;
                if (m == "rqmc") r = rqmc_european(call, std::max<std::size_t>(1, n / shifts), shifts, 1000u + s);
                else r = mc_european(call, {n, 1000u + s, m != "plain", m != "plain"});
                se_sum += r.std_error;
                sq_err += (r.price - exact) * (r.price - exact);
            }
            out << m << ',' << n << ',' << se_sum / seeds << ',' << std::sqrt(sq_err / seeds) << '\n';
        }
    }
    std::printf("wrote %s\n", path);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) return demo();
        const std::string cmd = argv[1];
        if (cmd == "price") return price_cmd(argc, argv);
        if (cmd == "american") return american_cmd(argc, argv);
        if (cmd == "convergence") return convergence_cmd(argc > 2 ? argv[2] : "convergence.csv");
        if (cmd == "benchmark") {
            section_speed();
            return 0;
        }
        std::fprintf(stderr, "unknown command '%s'\n", argv[1]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
