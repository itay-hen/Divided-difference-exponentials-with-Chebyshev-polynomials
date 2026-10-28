// =============================================================================
// e5_refusal_stats.cpp -- companion to e5_runtime.cpp: how often does the
// batch Chebyshev evaluator fall back / refuse on uniform random nodes, and how
// does its rounding indicator kappa*u*N compare with the observed difference
// to MNP?  No timing; many node sets per grid point.
//
// Grid: c in {0.5, 5, 50}, q in {2, 5, 10, 20, 50, 100, 200, 400, 800},
// n_sets = 200 uniform random node sets in [-c, c] (fewer for q >= 400).
// For each set: evaluate() on the object constructed on [-c, c];
//   fallback = the fixed-interval core exceeded kappa*u*N > 1e-10 and
//              evaluate() rebuilt on [min x, max x];
//   refusal  = evaluate() threw (also too ill-conditioned on [min x, max x]);
//   rel diff vs MNP where MNP's double result is a normal number (q <~ 170).
//
// Output: e5_refusals.csv
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "chebyshev_exp_divdiff.hpp"
#include "mccurdy.hpp"
#include "bench_util.hpp"

using Cheb = chebyshev::ExpDivDiff<double>;
using MNP  = mccurdy_templated::ExpDivDiff<double>;

static double quant(std::vector<double> v, double p) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(p * (v.size() - 1) + 0.5))];
}

int main(int argc, char** argv) {
    const std::string outdir = (argc > 1) ? argv[1] : ".";
    const std::vector<int>    qs = {2, 5, 10, 20, 50, 100, 200, 400, 800};
    const std::vector<double> cs = {0.5, 5.0, 50.0};
    FILE* f = bench::open_csv(outdir + "/e5_refusals.csv");
    std::fprintf(f, "c,q,n_sets,fallbacks,refusals,kuN_fixed_median,kuN_fixed_max,"
                    "kuN_eval_median,kuN_eval_max,n_mnp_cmp,reldiff_median,reldiff_max,"
                    "max_reldiff_over_kuN\n");
    std::mt19937_64 rng(99ULL);
    for (double c : cs) for (int q : qs) {
        const int n_sets = (q >= 400) ? 50 : 200;
        std::uniform_real_distribution<double> U(-c, c);
        Cheb E(-c, c);
        int fb = 0, ref = 0;
        std::vector<double> kf, ke, rd; double worst_ratio = 0;
        for (int s = 0; s < n_sets; ++s) {
            std::vector<double> x(q + 1);
            for (auto& v : x) v = U(rng);
            long double kuN_fixed = NAN;
            try { (void)E.evaluate_normalized_scaled(x); kuN_fixed = E.last_error_estimate(); }
            catch (...) { kuN_fixed = E.last_error_estimate(); }
            kf.push_back(static_cast<double>(kuN_fixed));
            double v = NAN; bool ok = true;
            try { v = E.evaluate(x); } catch (...) { ok = false; ++ref; }
            if (ok && kuN_fixed > 1e-10L) ++fb;
            if (ok) ke.push_back(static_cast<double>(E.last_error_estimate()));
            if (ok && q <= 150) {
                const double m = MNP::compute(x);
                if (std::isnormal(m) && std::isnormal(v)) {
                    const double r = std::fabs(v - m) / std::fabs(m);
                    rd.push_back(r);
                    worst_ratio = std::max(worst_ratio, r / static_cast<double>(E.last_error_estimate()));
                }
            }
        }
        std::fprintf(f, "%g,%d,%d,%d,%d,%.3e,%.3e,%.3e,%.3e,%zu,%.3e,%.3e,%.3e\n",
                     c, q, n_sets, fb, ref, quant(kf, 0.5), quant(kf, 1.0),
                     quant(ke, 0.5), quant(ke, 1.0), rd.size(), quant(rd, 0.5), quant(rd, 1.0),
                     worst_ratio);
        std::printf("c=%-4g q=%-4d fb=%3d ref=%3d kuN(eval) med %.1e max %.1e | reldiff med %.1e max %.1e\n",
                    c, q, fb, ref, quant(ke, 0.5), quant(ke, 1.0), quant(rd, 0.5), quant(rd, 1.0));
        std::fflush(stdout); std::fflush(f);
    }
    std::fclose(f);
    return 0;
}
