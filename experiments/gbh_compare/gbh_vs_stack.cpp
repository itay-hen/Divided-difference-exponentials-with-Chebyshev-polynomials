// gbh_vs_stack.cpp -- incremental updates: Gupta--Barash--Hen (CPC 2020)
// addition/removal routines vs chebyshev::IncrementalStack.
//
// Workload: a base set of q nodes uniform in [-c, c] (with the two
// endpoints included, so the spread, and hence GBH's scaling parameter
// s, stays fixed); then repeated "append one node, remove it" (LIFO),
// which is the update pattern both methods support at low cost.
//
// GBH is run unmodified (divdiff_gbh.hpp), with maxlen tuned to the order
// (maxlen = q + 16) and with its released default (maxlen = 10001); its
// per-append cost is O((maxlen - n) + s n), s = ceil(spread / 3.5).
// The stack uses q_max = q + 10 and its default Nmax.
//
// Accuracy: both values are compared with the batch evaluator through
// log values (GBH returns n! exp[z_0..z_n] in its own extended format).
//
// Output CSV on stdout: c,q,s_gbh,nmax,t_stack_ns,t_gbh_tuned_ns,
//                       t_gbh_default_ns,t_batch_ns,dlog_stack,dlog_gbh
//
// Build: g++ -std=c++17 -O3 -march=native -I../.. -I/opt/homebrew/include gbh_vs_stack.cpp
#include "chebyshev_stack.hpp"
#define private public          // read GBH's ExExFloat mantissa/exponent
#include "divdiff_gbh.hpp"
#undef private

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

double now_ns() {
    return std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

long double gbh_log_value(int n) {           // log exp[z_0..z_n]
    const gbh::ExExFloat& v = gbh::divdiff[n];
    return std::log(static_cast<long double>(v.mantissa)) +
           v.exponent * 0.693147180559945309417232121458176568L -
           std::lgamma(static_cast<long double>(n + 1));
}

// Set up GBH with the given base nodes; returns per-(append+remove) ns.
double time_gbh(const std::vector<double>& base, const std::vector<double>& extra,
                int maxlen, long double* dlog, const long double ref_log) {
    gbh::maxlen = maxlen;
    gbh::smax = 500;
    gbh::init();
    for (std::size_t i = 0; i < base.size(); ++i) gbh::z[i] = base[i];
    gbh::AddAll(static_cast<int>(base.size()));
    // Accuracy of the first append.
    gbh::AddElement(extra[0]);
    *dlog = gbh_log_value(gbh::CurrentLength - 1) - ref_log;
    gbh::RemoveElement();
    const int reps = static_cast<int>(extra.size());
    const double t0 = now_ns();
    for (int r = 0; r < reps; ++r) {
        gbh::AddElement(extra[r]);
        gbh::RemoveElement();
    }
    const double t = (now_ns() - t0) / reps;
    gbh::clear_up();
    return t;
}

} // namespace

int main() {
    std::printf("c,q,s_gbh,nmax,t_stack_ns,t_gbh_tuned_ns,t_gbh_default_ns,"
                "t_batch_ns,dlog_stack,dlog_gbh\n");
    std::mt19937 rng(2026);
    for (double c : {5.0, 50.0}) {
        for (int q : {10, 50, 100, 200, 400, 800}) {
            std::uniform_real_distribution<double> U(-c, c);
            std::vector<double> base(q);
            for (auto& v : base) v = U(rng);
            base[0] = c; base[1] = -c;
            const int reps = (q <= 100) ? 20000 : 4000;
            std::vector<double> extra(reps);
            for (auto& v : extra) v = U(rng);

            // Reference: batch evaluation of base + extra[0].
            std::vector<double> full = base; full.push_back(extra[0]);
            chebyshev::ExpDivDiff<double> E(-c, c);
            const long double ref_log = E.evaluate_scaled(full).log_abs();

            // Stack.
            chebyshev::IncrementalStack<double> st(-c, c, q + 10);
            st.initialize(base);
            st.push(extra[0]);
            const long double dlog_stack = st.value_scaled().log_abs() - ref_log;
            st.pop();
            volatile double sink = 0;
            double t0 = now_ns();
            for (int r = 0; r < reps; ++r) {
                st.push(extra[r]);
                sink += st.normalized_scaled().mantissa;
                st.pop();
            }
            const double t_stack = (now_ns() - t0) / reps;

            // Batch recomputation (amortized Bessel ratios).
            const int breps = std::max(20, 200000 / (q * q / 10 + 1));
            t0 = now_ns();
            for (int r = 0; r < breps; ++r) {
                full.back() = extra[r % reps];
                sink += E.evaluate_normalized_scaled(full).mantissa;
            }
            const double t_batch = (now_ns() - t0) / breps;

            long double dlog_gbh = 0, dlog_tmp = 0;
            const double t_gbh_tuned = time_gbh(base, extra, q + 16, &dlog_gbh, ref_log);
            const int dreps = std::min(reps, 2000);
            std::vector<double> extra_d(extra.begin(), extra.begin() + dreps);
            const double t_gbh_default = time_gbh(base, extra_d, 10001, &dlog_tmp, ref_log);
            const int s_gbh = static_cast<int>(std::ceil(2.0 * c / 3.5));

            std::printf("%g,%d,%d,%d,%.1f,%.1f,%.1f,%.1f,%.2Le,%.2Le\n", c, q, s_gbh,
                        st.nmax(), t_stack, t_gbh_tuned, t_gbh_default, t_batch,
                        dlog_stack, dlog_gbh);
            std::fflush(stdout);
        }
    }
    return 0;
}
