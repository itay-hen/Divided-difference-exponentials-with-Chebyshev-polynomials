// =============================================================================
// e6_stack_micro.cpp -- Experiment E6a (revised): cost of one incremental
// update with chebyshev::IncrementalStack (chebyshev_stack.hpp).
//
// For c in {5, 50} and base order q in {10, 50, 100, 200, 400, 800}:
//   * IncrementalStack<double> st(-c, c, q_max = q + 10) with the automatic
//     cache depth Nmax = q_max + 1.3 sqrt(2 c ln(1/tol)) + 30;
//   * setup: constructor (Bessel ratios to Nmax) and initialize() with q+1
//     uniform random nodes (inserted in nonincreasing order, every prefix
//     gated); the base set is redrawn if initialize() refuses (counted);
//   * push (append, Theta(Nmax) column + O(N') sum) and pop (O(1)), timed per
//     call with steady_clock over many push/pop pairs (median of 7 samples of
//     >= 50 ms); new nodes are drawn from a pool of uniform random nodes from
//     which nodes refused by push() (kappa u N > 1e-10) are removed (counted);
//   * full recomputation of the new (q+2)-node set: batch Chebyshev
//     (amortized object, and fresh object) and MNP compute();
//   * accuracy: value_scaled() after one push vs batch evaluate_scaled() of
//     the same set (relative difference).
// Output: e6_stack.csv
// =============================================================================
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "chebyshev_exp_divdiff.hpp"
#include "chebyshev_stack.hpp"
#include "mccurdy.hpp"
#include "bench_util.hpp"

using Cheb  = chebyshev::ExpDivDiff<double>;
using Stack = chebyshev::IncrementalStack<double>;
using MNP   = mccurdy_templated::ExpDivDiff<double>;
using bench::clock_type;

int main(int argc, char** argv) {
    const std::string outdir = (argc > 1) ? argv[1] : ".";
    const std::vector<double> cs = {5.0, 50.0};
    const std::vector<int>    qs = {10, 50, 100, 200, 400, 800};
    FILE* f = bench::open_csv(outdir + "/e6_stack.csv");
    std::fprintf(f, "c,q,q_max,nmax,N_top,init_redraws,push_refused_frac,"
                    "t_ctor_us,t_init_us,t_push_us,t_pop_us,t_pair_us,"
                    "t_batch_amort_us,t_batch_fresh_us,t_mnp_us,rel_diff_stack_batch\n");
    std::mt19937_64 rng(4242ULL);
    for (double c : cs) for (int q : qs) {
        std::uniform_real_distribution<double> U(-c, c);
        const int q_max = q + 10;
        // Base set: redraw until initialize() accepts it.
        std::vector<double> base(q + 1);
        int redraws = -1;
        Stack st(-c, c, q_max);
        for (;;) {
            ++redraws;
            for (auto& x : base) x = U(rng);
            try { st.initialize(base); break; } catch (...) {}
            if (redraws > 1000) { std::fprintf(stderr, "no acceptable base set\n"); return 1; }
        }
        // Pool of appendable nodes.
        std::vector<double> pool; int tried = 0, refused = 0;
        while (pool.size() < 64 && tried < 5000) {
            const double x = U(rng); ++tried;
            try { st.push(x); st.pop(); pool.push_back(x); } catch (...) { ++refused; }
        }
        // Accuracy and N after one push.
        st.push(pool[0]);
        const int N_top = st.last_N();
        std::vector<double> set0 = base; set0.push_back(pool[0]);
        Cheb E(-c, c);
        double rel = NAN;
        try {
            const auto vb = E.evaluate_scaled(set0);
            rel = std::fabs((st.value_scaled() / vb).to_real() - 1.0);
        } catch (...) {}
        st.pop();

        const double t_ctor = bench::time_kernel([&](long) {
            Stack t(-c, c, q_max); return double(t.nmax()); }).median_s * 1e6;
        Stack si(-c, c, q_max);
        const double t_init = bench::time_kernel([&](long) {
            si.initialize(base); return double(si.order()); }).median_s * 1e6;

        // push / pop timing.
        const size_t P = pool.size();
        double acc = 0;
        long reps = 1; size_t idx = 0;
        auto one = [&](double& tp, double& tq) {
            const double x = pool[idx++ % P];
            auto t0 = clock_type::now(); st.push(x);
            auto t1 = clock_type::now(); acc += st.last_N(); st.pop();
            auto t2 = clock_type::now();
            tp += std::chrono::duration<double>(t1 - t0).count();
            tq += std::chrono::duration<double>(t2 - t1).count();
        };
        for (;;) {
            double a = 0, b = 0; auto t0 = clock_type::now();
            for (long r = 0; r < reps; ++r) one(a, b);
            const double dt = bench::seconds_since(t0);
            if (dt >= 0.05) break;
            reps = std::max(reps + 1, static_cast<long>(reps * std::min(10.0, 1.3 * 0.05 / std::max(dt, 1e-9))));
        }
        std::vector<double> vp, vq, vpair;
        for (int sm = 0; sm < 7; ++sm) {
            double a = 0, b = 0; auto t0 = clock_type::now();
            for (long r = 0; r < reps; ++r) one(a, b);
            vpair.push_back(bench::seconds_since(t0) / reps);
            vp.push_back(a / reps); vq.push_back(b / reps);
        }
        bench::g_sink = bench::g_sink + acc;

        std::vector<std::vector<double>> sets(P, base);
        for (size_t i = 0; i < P; ++i) sets[i].push_back(pool[i]);
        const double t_ba = bench::time_kernel([&](long i) {
            try { return E.evaluate_scaled(sets[i % P]).mantissa; } catch (...) { return 0.0; } }).median_s * 1e6;
        const double t_bf = bench::time_kernel([&](long i) {
            Cheb F(-c, c);
            try { return F.evaluate_scaled(sets[i % P]).mantissa; } catch (...) { return 0.0; } }).median_s * 1e6;
        const double t_mn = bench::time_kernel([&](long i) {
            return MNP::compute(sets[i % P]); }).median_s * 1e6;

        const double tp = bench::median(vp) * 1e6, tq = bench::median(vq) * 1e6;
        std::fprintf(f, "%g,%d,%d,%d,%d,%d,%.4f,%.5g,%.5g,%.5g,%.5g,%.5g,%.5g,%.5g,%.5g,%.3e\n",
                     c, q, q_max, st.nmax(), N_top, redraws, double(refused) / tried,
                     t_ctor, t_init, tp, tq, bench::median(vpair) * 1e6, t_ba, t_bf, t_mn, rel);
        std::fflush(f);
        std::printf("c=%-3g q=%-4d Nmax=%-5d N=%-5d redraw=%d refused=%.3f | ctor %.3g init %.3g push %.3g pop %.3g"
                    " | batch %.3g fresh %.3g MNP %.3g us | batch/push %.1f  rel %.1e\n",
                    c, q, st.nmax(), N_top, redraws, double(refused) / tried, t_ctor, t_init, tp, tq,
                    t_ba, t_bf, t_mn, t_ba / tp, rel);
        std::fflush(stdout);
    }
    std::fclose(f);
    return 0;
}
