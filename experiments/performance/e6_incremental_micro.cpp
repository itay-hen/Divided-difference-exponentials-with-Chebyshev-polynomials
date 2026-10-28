// =============================================================================
// e6_incremental_micro.cpp -- LEGACY E6a (IncrementalExpDivDiff): cost of one incremental update.
//
// For c in {5, 50} and base order q in {10, 50, 100, 200, 400, 800}:
//   * IncrementalExpDivDiff<double> inc(-c, c, max_q) with the automatic
//     max_order (cache depth N_max), for two choices of the capacity max_q:
//     "2q+10" (comfortable headroom, the primary setting) and "q+10" (tight).
//     The auto rule is N_max = max(c + sqrt(2c ln(1/tol)),
//                                   max_q + sqrt(2 max_q ln(1/tol))) + 50,
//     so N_max grows with the CAPACITY max_q, not with the current q.
//   * setup cost: constructor (Bessel-ratio cache to N_max) and initialize()
//     with q+1 nodes (fills the (q+1) x (N_max+1) Chebyshev table);
//   * append cost  : add_node_normalized(x)  (fills one column to depth N_max,
//                    then sums the series until the stopping rule fires);
//   * removal cost : remove_last_node_normalized() (LIFO pop; re-sums the
//                    parent column);
//   * the same with the full-value API add_node()/remove_last_node(), which
//     additionally forms e^d I_0(c) c^{-q} (one Bessel I_0 call per update);
//   * full recomputation of the new (q+2)-node set by the batch Chebyshev
//     evaluator (amortized object, and fresh object), and by MNP compute().
//     Note: MNP compute_all() returns ALL leading divided differences
//     exp[x_0..x_k], k = 0..q, at the same cost as compute(), so for a pure
//     append workload MNP can serve every prefix from one call.
//
// Append and removal are timed per call (steady_clock around each call,
// accumulated over many append/remove pairs); a pair is always applied so
// the order stays at q.  Median of 7 samples of >= 50 ms each.
//
// Output: e6_incremental.csv
// =============================================================================
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "chebyshev_exp_divdiff.hpp"
#include "incremental_chebyshev_exp_divdiff.hpp"
#include "mccurdy.hpp"
#include "bench_util.hpp"

using Cheb = chebyshev::ExpDivDiff<double>;
using Inc  = chebyshev_incremental::IncrementalExpDivDiff<double>;
using MNP  = mccurdy_templated::ExpDivDiff<double>;
using bench::clock_type;

struct PairTiming { double add_us, rem_us, pair_us; };

// Time append + LIFO removal on an initialized object.  `normalized` selects
// the *_normalized API.  Returns medians over 7 samples.
static PairTiming time_add_remove(Inc& inc, const std::vector<double>& new_nodes,
                                  bool normalized) {
    const size_t P = new_nodes.size();
    double acc = 0;
    auto one_pair = [&](size_t i, double& ta, double& tr) {
        const double x = new_nodes[i % P];
        auto t0 = clock_type::now();
        acc += normalized ? inc.add_node_normalized(x) : inc.add_node(x);
        auto t1 = clock_type::now();
        acc += normalized ? inc.remove_last_node_normalized() : inc.remove_last_node();
        auto t2 = clock_type::now();
        ta += std::chrono::duration<double>(t1 - t0).count();
        tr += std::chrono::duration<double>(t2 - t1).count();
    };
    // Calibrate.
    long reps = 1; size_t idx = 0;
    for (;;) {
        double ta = 0, tr = 0;
        auto t0 = clock_type::now();
        for (long r = 0; r < reps; ++r) one_pair(idx++, ta, tr);
        const double dt = bench::seconds_since(t0);
        if (dt >= 0.05) break;
        reps = std::max(reps + 1, static_cast<long>(reps * std::min(10.0, 1.3 * 0.05 / std::max(dt, 1e-9))));
    }
    std::vector<double> va, vr, vp;
    for (int s = 0; s < 7; ++s) {
        double ta = 0, tr = 0;
        auto t0 = clock_type::now();
        for (long r = 0; r < reps; ++r) one_pair(idx++, ta, tr);
        vp.push_back(bench::seconds_since(t0) / reps);
        va.push_back(ta / reps); vr.push_back(tr / reps);
    }
    bench::g_sink = bench::g_sink + acc;
    return {bench::median(va) * 1e6, bench::median(vr) * 1e6, bench::median(vp) * 1e6};
}

int main(int argc, char** argv) {
    const std::string outdir = (argc > 1) ? argv[1] : ".";
    const std::vector<double> cs = {5.0, 50.0};
    const std::vector<int>    qs = {10, 50, 100, 200, 400, 800};

    FILE* f = bench::open_csv(outdir + "/e6_incremental_legacy.csv");
    std::fprintf(f,
        "c,q,maxq_rule,max_q,max_order,last_used_order,N_batch,status,"
        "t_ctor_us,t_init_us,t_add_us,t_remove_us,t_pair_us,"
        "t_add_full_us,t_remove_full_us,"
        "t_batch_amort_us,t_batch_fresh_us,t_mnp_us,rel_diff_inc_batch\n");

    std::mt19937_64 rng(4242ULL);
    for (double c : cs) {
        for (int q : qs) {
            std::uniform_real_distribution<double> U(-c, c);
            std::vector<double> base(q + 1);
            for (auto& x : base) x = U(rng);
            std::vector<double> newx(64);
            for (auto& x : newx) x = U(rng);

            // ---- reference costs (independent of max_q) ----
            std::vector<std::vector<double>> sets(newx.size(), base);
            for (size_t i = 0; i < newx.size(); ++i) sets[i].push_back(newx[i]);
            Cheb E(-c, c);
            int N_batch = 0;
            (void)E.evaluate_scaled(sets[0], &N_batch);
            const double t_ba = bench::time_kernel([&](long i) {
                return E.evaluate_scaled(sets[i % sets.size()]).mantissa; }).median_s * 1e6;
            const double t_bf = bench::time_kernel([&](long i) {
                Cheb F(-c, c); return F.evaluate_scaled(sets[i % sets.size()]).mantissa; }).median_s * 1e6;
            const double t_mn = bench::time_kernel([&](long i) {
                return MNP::compute(sets[i % sets.size()]); }).median_s * 1e6;

            for (int rule = 0; rule < 2; ++rule) {
                const int max_q = (rule == 0) ? 2 * q + 10 : q + 10;
                const char* rule_name = (rule == 0) ? "2q+10" : "q+10";
                std::string status = "ok";
                double t_ctor = NAN, t_init = NAN, rel = NAN;
                PairTiming pn{NAN, NAN, NAN}, pf{NAN, NAN, NAN};
                int max_order = -1, last_used = -1;
                try {
                    t_ctor = bench::time_kernel([&](long) {
                        Inc tmp(-c, c, max_q); return double(tmp.max_order()); },
                        0.05, 7).median_s * 1e6;
                    Inc inc(-c, c, max_q);
                    max_order = inc.max_order();
                    t_init = bench::time_kernel([&](long) {
                        return inc.initialize_normalized(base); }, 0.05, 7).median_s * 1e6;
                    inc.initialize_normalized(base);
                    // Accuracy spot check of one append against the batch
                    // evaluator's normalized sum S on the SAME interval
                    // (exponent-scaled, so it cannot under/overflow).  The
                    // incremental table is unscaled double: S itself, or the
                    // factors R_n and D_q^(n), leave the double range for
                    // large q, which shows up here as NaN/inf/0.
                    const double S_inc = inc.add_node_normalized(newx[0]);
                    last_used = inc.last_used_order();
                    try {
                        const auto S_bat = E.evaluate_normalized_scaled(sets[0]);
                        const auto ratio = Cheb::ScaledReal{S_inc, 0} / S_bat;
                        rel = std::fabs(ratio.to_real() - 1.0);
                    } catch (...) {
                        // The batch fixed-interval core refused (kappa*u*N >
                        // 1e-10); compare full values instead, where the batch
                        // evaluate() auto-tightens the interval.
                        const double v_bat = E.evaluate(sets[0]);
                        const double v_inc = S_inc * inc.get_scaling_factor();
                        rel = std::fabs(v_inc - v_bat) / std::fabs(v_bat);
                    }
                    if (!std::isfinite(S_inc) || S_inc == 0.0) {
                        rel = NAN;
                        status = "invalid:S_inc=" + std::to_string(S_inc);
                    }
                    inc.remove_last_node_normalized();
                    pn = time_add_remove(inc, newx, true);
                    // The full-value API overflows/underflows long before the
                    // normalized one; only time it where the value is finite.
                    if (status == "ok" && std::isfinite(inc.add_node(newx[0]))) {
                        inc.remove_last_node();
                        pf = time_add_remove(inc, newx, false);
                    } else if (status == "ok") {
                        inc.remove_last_node_normalized();
                    }
                } catch (const std::exception& e) {
                    status = std::string("throw:") + e.what();
                    for (auto& ch : status) if (ch == ',' || ch == '\n') ch = ';';
                    status = status.substr(0, 120);
                }
                std::fprintf(f,
                    "%g,%d,%s,%d,%d,%d,%d,%s,%.5g,%.5g,%.5g,%.5g,%.5g,%.5g,%.5g,"
                    "%.5g,%.5g,%.5g,%.3e\n",
                    c, q, rule_name, max_q, max_order, last_used, N_batch, status.c_str(),
                    t_ctor, t_init, pn.add_us, pn.rem_us, pn.pair_us, pf.add_us, pf.rem_us,
                    t_ba, t_bf, t_mn, rel);
                std::fflush(f);
                std::printf("c=%-3g q=%-4d %-6s Nmax=%-5d used=%-5d Nb=%-5d %s | ctor %.3g init %.3g "
                            "add %.3g rem %.3g (full %.3g/%.3g) | batch %.3g fresh %.3g MNP %.3g us  rel=%.1e\n",
                            c, q, rule_name, max_order, last_used, N_batch, status.c_str(),
                            t_ctor, t_init, pn.add_us, pn.rem_us, pf.add_us, pf.rem_us,
                            t_ba, t_bf, t_mn, rel);
                std::fflush(stdout);
            }
        }
    }
    std::fclose(f);
    return 0;
}
