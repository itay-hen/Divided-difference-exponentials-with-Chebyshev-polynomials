// =============================================================================
// e5_interval_sensitivity.cpp -- Experiment E5b: sensitivity of the batch
// Chebyshev evaluator to the constructor interval [a, b].
//
// Nodes: uniform random in [-1, 1], q in {20, 100}, several node sets.
// Constructor interval, slack s in {0, 0.5, 1, 2, 5, 10, 20}:
//   symmetric : [a, b] = [-1 - s, 1 + s]
//   left      : [a, b] = [-1 - 2s, 1]      (slack below min x only; "right-anchored")
//   right     : [a, b] = [-1, 1 + 2s]      (slack above max x only; "left-anchored")
// All three have the same half-width c = 1 + s.
//
// For each case we record
//   * the RAW fixed-interval result (evaluate_normalized_scaled on [a,b], i.e.
//     the Chebyshev sum without the safety fallback): N, kappa*u*N, value;
//   * the production evaluate() result: N, kappa*u*N, whether it internally
//     fell back to [min x, max x] (it does so when kappa*u*N > 1e-10), whether
//     it refused (threw), and the value;
//   * amortized wall-clock time of evaluate() and of the raw core.
// Values are written so that make_reference.py can compute true errors
// against an mpmath reference for the same node sets.
//
// Expected behaviour (to be tested): slack ABOVE max x_i inflates kappa by
// ~e^{b - max x_i} (the Chebyshev terms are O(e^{b}) while the answer is
// O(e^{max x})); slack BELOW min x_i only increases N (c grows) but leaves
// kappa ~ O(1).
//
// Output: e5_interval.csv, e5_interval_nodes.csv
// =============================================================================
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "chebyshev_exp_divdiff.hpp"
#include "bench_util.hpp"

using Cheb = chebyshev::ExpDivDiff<double>;

int main(int argc, char** argv) {
    const std::string outdir = (argc > 1) ? argv[1] : ".";
    const std::vector<int>    qs = {20, 100};
    const std::vector<double> slacks = {0, 0.5, 1, 2, 5, 10, 20};
    const char* modes[3] = {"symmetric", "left", "right"};
    const int n_sets = 4;

    FILE* f  = bench::open_csv(outdir + "/e5_interval.csv");
    FILE* fn = bench::open_csv(outdir + "/e5_interval_nodes.csv");
    std::fprintf(f,
        "q,set,mode,s,a,b,c,min_x,max_x,"
        "N_raw,kuN_raw,raw_value,"
        "N_eval,kuN_eval,eval_fallback,eval_refused,eval_value,"
        "t_eval_us,t_raw_us\n");

    std::mt19937_64 rng(777ULL);
    std::uniform_real_distribution<double> U(-1.0, 1.0);

    for (int q : qs) {
        for (int set = 0; set < n_sets; ++set) {
            std::vector<double> x(q + 1);
            for (auto& v : x) v = U(rng);
            double mn = x[0], mx = x[0];
            for (double v : x) { mn = std::min(mn, v); mx = std::max(mx, v); }
            std::fprintf(fn, "%d,%d", q, set);
            for (double v : x) std::fprintf(fn, ",%.17g", v);
            std::fprintf(fn, "\n");

            for (int m = 0; m < 3; ++m) {
                for (double s : slacks) {
                    double a, b;
                    if (m == 0)      { a = -1 - s;     b = 1 + s; }
                    else if (m == 1) { a = -1 - 2 * s; b = 1; }
                    else             { a = -1;         b = 1 + 2 * s; }
                    Cheb E(a, b);

                    // Raw fixed-interval sum (no fallback).
                    int N_raw = -1; long double kuN_raw = NAN; double raw_val = NAN;
                    try {
                        auto S = E.evaluate_normalized_scaled(x, &N_raw);
                        kuN_raw = E.last_error_estimate();
                        raw_val = S.to_real() * E.get_scaling_factor(q);
                    } catch (...) {
                        // Refused by the kappa gate: N and kappa*u*N are
                        // still recorded by the core before it throws.
                        kuN_raw = E.last_error_estimate();
                    }
                    // Production evaluate().
                    int N_ev = -1; long double kuN_ev = NAN; double ev_val = NAN;
                    int refused = 0;
                    try {
                        ev_val = E.evaluate(x, &N_ev);
                        kuN_ev = E.last_error_estimate();
                    } catch (...) {
                        refused = 1; kuN_ev = E.last_error_estimate();
                    }
                    // evaluate() falls back exactly when the fixed-interval kappa*u*N exceeds
                    // its threshold (default 1e-10) and the nodes do not fill [a, b].
                    const int fallback = (!refused && kuN_raw > 1e-10L) ? 1 : 0;

                    double t_ev = NAN, t_raw = NAN;
                    if (!refused) {
                        t_ev = bench::time_kernel([&](long) {
                            return E.evaluate(x);
                        }, 0.05, 5).median_s * 1e6;
                    }
                    t_raw = bench::time_kernel([&](long) {
                        try { return E.evaluate_normalized_scaled(x).mantissa; }
                        catch (...) { return 0.0; }
                    }, 0.05, 5).median_s * 1e6;

                    std::fprintf(f,
                        "%d,%d,%s,%g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                        "%d,%.4Le,%.17g,%d,%.4Le,%d,%d,%.17g,%.5g,%.5g\n",
                        q, set, modes[m], s, a, b, E.get_c(), mn, mx,
                        N_raw, kuN_raw, raw_val, N_ev, kuN_ev, fallback, refused,
                        ev_val, t_ev, t_raw);
                    std::fflush(f);
                    if (set == 0)
                        std::printf("q=%-3d %-9s s=%-4g  raw: N=%-4d kuN=%.2Le | eval: N=%-4d "
                                    "kuN=%.2Le fb=%d ref=%d  t=%.3g us\n",
                                    q, modes[m], s, N_raw, kuN_raw, N_ev, kuN_ev,
                                    fallback, refused, t_ev);
                }
            }
        }
    }
    std::fclose(f); std::fclose(fn);
    return 0;
}
