// e7_ratio.cpp -- ratios exp[x]/exp[x'] of two node sets of the same
// order on the same interval, as in a PMR-QMC Metropolis step:
// Chebyshev--Bessel (ratio of normalized sums, evaluate_ratio) against
// MNP in double precision (two separate evaluations, then divided).
//
// Grid: q in {1,2,5,10,20,50,100,200,500}, c in {0.1,...,1000}; TRIALS
// pairs per cell.  x: uniform in [-c, c] with x_0 = -c, x_q = c; x' = x
// with one node (chosen uniformly) replaced by a fresh uniform value.
// The mapping interval is [-c, c] for both sets.
//
// Output (argv[1] = output prefix):
//   PREFIX_nodes.csv    instance,nodes      (instance 2k: x, 2k+1: x')
//                       for mp_reference.py
//   PREFIX_results.csv  pair,q,c,trial,cheb,cheb_est,cheb_refused,cheb_sq,mnp
//     cheb       ratio from evaluate_ratio_scaled with the default settings
//                (threshold 1e-10; if the common-interval ratio is
//                refused, both values are evaluated in full, with the
//                retries and, if needed, scaling and squaring)
//     cheb_log   natural log of the ratio (finite even if cheb overflows)
//     cheb_est   the indicator of the ratio (sum of both)
//     cheb_refused 1 if evaluate_ratio threw (no value)
//     cheb_sq    squarings used (0: common-interval series)
//     mnp        MNP ratio; nan if either evaluation failed or was not
//                finite and positive
//
// Build: g++ -std=c++17 -O3 -I../.. -I/opt/homebrew/include e7_ratio.cpp
#include "chebyshev_exp_divdiff.hpp"
#include "mccurdy.hpp"
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string pre = argc > 1 ? argv[1] : "e7_ratio";
    const int TRIALS = argc > 2 ? std::atoi(argv[2]) : 20;
    FILE* fn = std::fopen((pre + "_nodes.csv").c_str(), "w");
    FILE* fr = std::fopen((pre + "_results.csv").c_str(), "w");
    std::fprintf(fn, "instance,nodes\n");
    std::fprintf(fr, "pair,q,c,trial,cheb,cheb_log,cheb_est,cheb_refused,cheb_sq,mnp\n");
    const std::vector<int> qs = {1, 2, 5, 10, 20, 50, 100, 200, 500};
    const std::vector<double> cs = {0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000};
    std::mt19937_64 rng(20261119);
    int pair = 0;
    for (int q : qs) for (double c : cs) {
        chebyshev::ExpDivDiff<double> e(-c, c, 1e-14, 100000);
        std::uniform_real_distribution<double> U(-c, c);
        std::uniform_int_distribution<int> J(0, q);
        for (int t = 0; t < TRIALS; ++t, ++pair) {
            std::vector<double> x(q + 1);
            for (auto& v : x) v = U(rng);
            x[0] = -c; x[q] = c;
            std::vector<double> xp = x;
            xp[J(rng)] = U(rng);
            for (int s = 0; s < 2; ++s) {
                const auto& y = s ? xp : x;
                std::fprintf(fn, "%d,", 2 * pair + s);
                for (int i = 0; i <= q; ++i)
                    std::fprintf(fn, "%s%.17g", i ? " " : "", y[i]);
                std::fprintf(fn, "\n");
            }
            double cheb = NAN; long double est = NAN, lg = NAN;
            int refused = 1, sq = 0;
            try {
                const auto r = e.evaluate_ratio_scaled(x, xp);
                cheb = r.to_real();
                lg = r.log_abs();
                est = e.last_error_estimate();
                sq = e.last_squarings();
                refused = 0;
            } catch (...) {}
            double mnp = NAN;
            try {
                const double a = mccurdy_templated::ExpDivDiff<double>::compute(x);
                const double b = mccurdy_templated::ExpDivDiff<double>::compute(xp);
                if (std::isfinite(a) && std::isfinite(b) && a > 0 && b > 0) mnp = a / b;
            } catch (...) {}
            std::fprintf(fr, "%d,%d,%g,%d,%.17g,%.21Lg,%.3Le,%d,%d,%.17g\n", pair, q, c, t,
                         cheb, lg, est, refused, sq, mnp);
        }
    }
    std::fclose(fn); std::fclose(fr);
    return 0;
}
