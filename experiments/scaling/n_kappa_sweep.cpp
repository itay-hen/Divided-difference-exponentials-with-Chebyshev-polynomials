// n_kappa_sweep.cpp -- truncation length N and rounding-error indicator
// kappa*u*N of the Chebyshev--Bessel sum over a (q, c) grid.
//
// Nodes: uniform random in [-c, c].  Mapping interval: argv[2] = "fixed"
// uses [a, b] = [-c, c] (as for a fixed incremental interval); "tight"
// uses [a, b] = [min x_i, max x_i] (the best choice for one evaluation);
// "pinned" sets x_0 = -c, x_q = c (q >= 1) so that [-c, c] is both the
// mapping interval and the tightest interval (used for the figures).  The conditioning
// gate is disabled (set_max_error_estimate(huge)) so that N and the
// indicator are recorded for every case, including those evaluate()
// would refuse.  Output CSV: q,c,trial,N,kappa_u,est (= kappa*u*N).
//
// Build: g++ -std=c++17 -O2 -I../.. -I/opt/homebrew/include n_kappa_sweep.cpp
#include "chebyshev_exp_divdiff.hpp"
#include <cstdio>
#include <algorithm>
#include <random>
#include <string>

int main(int argc, char** argv) {
    const char* out = argc > 1 ? argv[1] : "n_kappa_sweep.csv";
    const std::string mode = argc > 2 ? argv[2] : "fixed";
    const bool tight = (mode == "tight");
    const bool pinned = (mode == "pinned");
    FILE* fp = std::fopen(out, "w");
    std::fprintf(fp, "q,c,trial,N,kappa_u,est\n");
    const std::vector<int> qs = {1, 2, 5, 10, 20, 50, 100, 200, 500};
    const std::vector<double> cs = {0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000};
    const int trials = 30;
    std::mt19937 rng(20261027);
    for (int q : qs) for (double c : cs) {
        chebyshev::ExpDivDiff<double> fixed_e(-c, c, 1e-14, 100000);
        fixed_e.set_max_error_estimate(1e300L);
        std::uniform_real_distribution<double> U(-c, c);
        for (int t = 0; t < trials; ++t) {
            std::vector<double> x(q + 1);
            for (auto& v : x) v = U(rng);
            if (pinned) { x[0] = -c; x[q] = c; }
            const double lo = *std::min_element(x.begin(), x.end());
            const double hi = *std::max_element(x.begin(), x.end());
            chebyshev::ExpDivDiff<double> tight_e(lo, hi, 1e-14, 100000);
            tight_e.set_max_error_estimate(1e300L);
            chebyshev::ExpDivDiff<double>& e = tight ? tight_e : fixed_e;
            int N = -1;
            try {
                (void) e.evaluate_normalized_scaled(x, &N);
                std::fprintf(fp, "%d,%g,%d,%d,%.3Le,%.3Le\n", q, c, t, N,
                             e.last_condition_estimate(), e.last_error_estimate());
            } catch (const std::exception& ex) {
                std::fprintf(fp, "%d,%g,%d,-1,nan,nan\n", q, c, t);
            }
        }
    }
    std::fclose(fp);
    return 0;
}
