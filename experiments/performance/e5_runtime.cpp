// =============================================================================
// e5_runtime.cpp -- Experiment E5: single-evaluation runtime and crossover.
//
// For uniform random nodes in [-c, c] and a grid of orders q, compares
//   (1) Chebyshev batch, amortized    : one chebyshev::ExpDivDiff object per
//                                       (c, q) point, Bessel-ratio cache warm;
//   (2) Chebyshev batch, incl. precomp: a FRESH object per evaluation, so the
//                                       Miller Bessel-ratio recurrence and
//                                       I_0(c) are paid on every call;
//   (3) MNP (McCurdy-Ng-Parlett)      : mccurdy_templated::ExpDivDiff::compute.
//
// Before timing, accuracy is checked on every node set of the pool:
//   * relative difference |cheb - mnp| / |mnp| (only where MNP's double result
//     is a normal number; MNP underflows once exp[x_0..x_q] < DBL_MIN, which
//     happens for q >~ 170 at small c, because the value is ~ e^xi / q!);
//   * Chebyshev N, kappa*u*N (last_error_estimate), whether evaluate()
//     internally fell back to the tight interval [min x, max x] (detected by
//     re-running the fixed-interval core via evaluate_normalized_scaled), and
//     refusals (exceptions).
// The first node set of each point and both methods' values are written to a
// side file so that an mpmath reference (make_reference.py) can adjudicate.
//
// Output (CSV, one row per (c, q)):  ../../../Data and figures/v18/
//   e5_runtime.csv        timings + accuracy summary
//   e5_runtime_nodes.csv  c,q,set,cheb_mantissa,cheb_exp2,mnp_value,x_0..x_q
//
// Build: see build_and_run.sh in this folder.
// =============================================================================
#include <cfloat>
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

int main(int argc, char** argv) {
    const std::string outdir = (argc > 1) ? argv[1] : ".";
    const std::vector<int>    qs = {2, 5, 10, 20, 50, 100, 200, 400, 800};
    const std::vector<double> cs = {0.5, 5.0, 50.0};
    const int pool_size = 8;   // node sets per grid point (timings cycle them)

    FILE* f  = bench::open_csv(outdir + "/e5_runtime.csv");
    FILE* fn = bench::open_csv(outdir + "/e5_runtime_nodes.csv");
    std::fprintf(f,
        "c,q,pool,N_max,N_mean,kuN_max,cheb_fallbacks,cheb_refusals,"
        "mnp_failures,mnp_underflow,max_rel_diff,"
        "t_amort_us,t_amort_min_us,t_fresh_us,t_fresh_min_us,t_mnp_us,t_mnp_min_us,"
        "reps_amort,reps_fresh,reps_mnp,"
        "t_per_qN_ns,t_per_cell_ns,precomp_share\n");

    std::mt19937_64 rng(20261027ULL);

    for (double c : cs) {
        for (int q : qs) {
            std::uniform_real_distribution<double> U(-c, c);
            std::vector<std::vector<double>> pool(pool_size, std::vector<double>(q + 1));
            for (auto& s : pool) for (auto& x : s) x = U(rng);

            // ---------------- accuracy / diagnostics ----------------------
            Cheb E(-c, c);
            int N_max = 0; double N_sum = 0; long double kuN_max = 0;
            int fallbacks = 0, refusals = 0, mnp_fail = 0, mnp_uf = 0;
            double max_rel = 0.0;
            std::vector<int> ok_idx;   // node sets the Chebyshev evaluator accepted
            for (int p = 0; p < pool_size; ++p) {
                const auto& x = pool[p];
                Cheb::ScaledReal v{}; bool ok = true; int N = 0;
                try {
                    v = E.evaluate_scaled(x, &N);
                } catch (const std::exception& e) {
                    ok = false; ++refusals;
                }
                const long double kuN = E.last_error_estimate();
                if (ok) {
                    ok_idx.push_back(p);
                    N_max = std::max(N_max, N); N_sum += N;
                    kuN_max = std::max(kuN_max, kuN);
                    // Did evaluate() run on the constructor interval, or did
                    // it rebuild on [min x, max x]?  Re-run the fixed-interval
                    // core (no fallback) and compare N.
                    int N_fixed = -1; bool fixed_ok = true;
                    try { (void)E.evaluate_normalized_scaled(x, &N_fixed); }
                    catch (...) { fixed_ok = false; }
                    if (!fixed_ok || N_fixed != N) ++fallbacks;
                }
                double m = 0.0; bool mok = true;
                try { m = MNP::compute(x); } catch (...) { mok = false; ++mnp_fail; }
                if (mok && !(std::isnormal(m) && std::fabs(m) > 1e-290)) { ++mnp_uf; mok = false; }
                if (ok && mok) {
                    const double r = std::fabs(v.to_real() - m) / std::fabs(m);
                    max_rel = std::max(max_rel, r);
                }
                if (p == 0) {
                    std::fprintf(fn, "%.17g,%d,%d,%.17g,%ld,%.17g", c, q, p,
                                 ok ? v.mantissa : NAN, ok ? v.exponent : 0L,
                                 mok ? m : NAN);
                    for (double xi : x) std::fprintf(fn, ",%.17g", xi);
                    std::fprintf(fn, "\n");
                }
            }

            // ---------------- timings ---------------------------------------
            // Chebyshev timings use only the node sets it accepted (refused
            // sets would throw); MNP is timed on the whole pool.
            bench::TimingResult t_am, t_fr;
            t_am.median_s = t_am.min_s = t_fr.median_s = t_fr.min_s = NAN;
            const size_t nok = ok_idx.size();
            if (nok > 0) {
                // (1) amortized: E's Bessel cache is warm after the loop above.
                t_am = bench::time_kernel([&](long i) {
                    return E.evaluate_scaled(pool[ok_idx[i % nok]]).mantissa;
                });
                // (2) fresh object per evaluation (includes R_n precomputation).
                t_fr = bench::time_kernel([&](long i) {
                    Cheb F(-c, c);
                    return F.evaluate_scaled(pool[ok_idx[i % nok]]).mantissa;
                });
            }
            // (3) MNP.
            auto t_mn = bench::time_kernel([&](long i) {
                return MNP::compute(pool[i % pool_size]);
            });

            const double N_mean = N_sum / std::max(1, pool_size - refusals);
            // Cells actually swept by the Chebyshev recurrence:
            //   sum_{n=0}^{N} (min(n, q) + 1).
            double cells = 0;
            for (int n = 0; n <= static_cast<int>(N_mean + 0.5); ++n) cells += std::min(n, q) + 1;
            const double t_qN   = t_am.median_s / (double(q) * N_mean) * 1e9;
            const double t_cell = t_am.median_s / cells * 1e9;
            const double share  = (t_fr.median_s - t_am.median_s) / t_fr.median_s;

            std::fprintf(f,
                "%.17g,%d,%d,%d,%.2f,%.3Le,%d,%d,%d,%d,%.3e,"
                "%.5g,%.5g,%.5g,%.5g,%.5g,%.5g,%ld,%ld,%ld,%.4g,%.4g,%.4f\n",
                c, q, pool_size, N_max, N_mean, kuN_max, fallbacks, refusals,
                mnp_fail, mnp_uf, max_rel,
                t_am.median_s * 1e6, t_am.min_s * 1e6,
                t_fr.median_s * 1e6, t_fr.min_s * 1e6,
                t_mn.median_s * 1e6, t_mn.min_s * 1e6,
                t_am.reps, t_fr.reps, t_mn.reps, t_qN, t_cell, share);
            std::fflush(f); std::fflush(fn);
            std::printf("c=%-5g q=%-4d N=%-5d kuN=%.1Le fb=%d ref=%d mnpUF=%d rel=%.1e | "
                        "amort %.3g us  fresh %.3g us  MNP %.3g us\n",
                        c, q, N_max, kuN_max, fallbacks, refusals, mnp_uf, max_rel,
                        t_am.median_s * 1e6, t_fr.median_s * 1e6, t_mn.median_s * 1e6);
            std::fflush(stdout);
        }
    }
    std::fclose(f); std::fclose(fn);
    return 0;
}
