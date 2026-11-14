// =============================================================================
// e2_accuracy.cpp -- Experiment E2: accuracy of the Chebyshev--Bessel
// evaluator (and of the MNP baseline) over node configurations, orders q,
// half-widths c and node ORDERINGS.
//
// This program only produces double-precision results.  The independent
// high-precision reference values are computed afterwards by
// mp_reference.py from the node file written here (%.17g, so both sides see
// bit-identical doubles), and the errors are evaluated by e2_analyze.py.
//
// Build (from the "DDEF c++" folder):
//   g++ -std=c++17 -O2 -I. -I/opt/homebrew/include \
//       experiments/accuracy/e2_accuracy.cpp -o e2_accuracy
// Run:
//   ./e2_accuracy <out_dir>
// Writes:
//   <out_dir>/accuracy_e2_nodes.csv    one line per node multiset (instance)
//   <out_dir>/accuracy_e2_results.csv  one line per (instance, ordering)
//
// For every (instance, ordering) four Chebyshev runs are recorded.  The
// first three use the library default, which processes the nodes internally
// in nonincreasing order (set_sort_nodes(true)); the fourth keeps the
// caller's order:
//   prod  : E.evaluate_scaled() on [a,b] = [-c,c] with the default
//           threshold 1e-10 on kappa*u*N -- what a user gets.  Status is
//           "ok", "refused" (conditioning throw) or "noconv" (max_iter).
//   ab    : the same call with set_max_error_estimate(1e300), i.e. the
//           un-retried value on [a,b] (or on [min x, max x] when the
//           100x pre-screen of evaluate_scaled() applies).
//   tight : a fresh ExpDivDiff(min x, max x) with threshold 1e300, i.e.
//           the value the auto-tightening retry computes.  For a refused
//           case this is the value that was rejected.
//   unsorted: as `ab` but with set_sort_nodes(false), i.e. the recurrence is
//           run in the caller's node order (the situation of the incremental
//           scheme, whose order is fixed by the update sequence).
// Values are written as (mantissa %.17g, binary exponent) so that q = 300
// results (which underflow double) are kept exactly.
// MNP (McCurdy--Ng--Parlett) is evaluated in plain double precision.
// =============================================================================

#include "chebyshev_exp_divdiff.hpp"
#include "mccurdy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Cheb = chebyshev::ExpDivDiff<double>;
using Scaled = Cheb::ScaledReal;

// ---------------------------------------------------------------------------
// Node configurations.  All nodes lie in [-c, c].
// ---------------------------------------------------------------------------
static double clampc(double x, double c) { return std::min(c, std::max(-c, x)); }

static std::vector<double> make_nodes(const std::string& cfg, int q, double c,
                                      std::mt19937_64& rng) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<double> x(q + 1);
    if (cfg == "uniform") {
        for (auto& v : x) v = clampc(-c + 2.0 * c * U(rng), c);
    } else if (cfg == "cluster_left") {                 // dense near a = -c
        for (auto& v : x) { double u = U(rng); v = clampc(-c + 2.0 * c * u * u * u * u, c); }
    } else if (cfg == "cluster_right") {                // mirror: dense near b = c
        for (auto& v : x) { double u = U(rng); v = clampc(c - 2.0 * c * u * u * u * u, c); }
    } else if (cfg == "chebyshev") {                    // jittered Chebyshev points
        const double pi = std::acos(-1.0);
        for (int i = 0; i <= q; ++i) x[i] = clampc(c * std::cos(pi * (i + U(rng)) / (q + 1)), c);
    } else if (cfg == "near_coincident" || cfg == "near_coincident_far") {
        // all nodes within 1e-8 c of a random centre in [-0.9c, 0.9c]
        const double ctr = -0.9 * c + 1.8 * c * U(rng);
        for (auto& v : x) v = clampc(ctr + 1e-8 * c * (2.0 * U(rng) - 1.0), c);
        if (cfg == "near_coincident_far") x[q] = (ctr > 0 ? -c : c);  // one far node
    } else if (cfg == "repeated") {                     // m distinct values, multiplicity ~5
        const int m = std::max(2, q / 5);
        std::vector<double> vals(m);
        for (auto& v : vals) v = clampc(-c + 2.0 * c * U(rng), c);
        std::uniform_int_distribution<int> pick(0, m - 1);
        for (int i = 0; i <= q; ++i) x[i] = (i < m ? vals[i] : vals[pick(rng)]);  // every value used
    } else if (cfg == "two_clusters") {                 // [-c,-0.99c] and [0.99c, c]
        for (int i = 0; i <= q; ++i) {
            const double w = 0.01 * c * U(rng);
            x[i] = (i % 2 == 0) ? clampc(-c + w, c) : clampc(c - w, c);
        }
    } else if (cfg == "equispaced") {
        for (int i = 0; i <= q; ++i) x[i] = (i == q) ? c : -c + 2.0 * c * i / q;
    } else {
        throw std::invalid_argument("unknown config " + cfg);
    }
    return x;
}

// ---------------------------------------------------------------------------
// Orderings of one node multiset.
// ---------------------------------------------------------------------------
static std::vector<std::pair<std::string, std::vector<double>>>
make_orderings(const std::vector<double>& gen, unsigned long long seed) {
    std::vector<std::pair<std::string, std::vector<double>>> out;
    out.push_back({"generated", gen});
    std::vector<double> asc = gen;
    std::sort(asc.begin(), asc.end());
    out.push_back({"ascending", asc});
    out.push_back({"descending", std::vector<double>(asc.rbegin(), asc.rend())});
    for (int s = 0; s < 3; ++s) {
        std::vector<double> p = gen;
        std::mt19937_64 r(seed * 7919ULL + 101ULL * (s + 1));
        std::shuffle(p.begin(), p.end(), r);
        out.push_back({"random" + std::to_string(s + 1), p});
    }
    std::vector<double> alt;                             // min, max, 2nd min, 2nd max, ...
    for (size_t lo = 0, hi = asc.size(); lo < hi;) {
        alt.push_back(asc[lo++]);
        if (lo < hi) alt.push_back(asc[--hi]);
    }
    out.push_back({"alternating", alt});
    return out;
}

struct Run {
    std::string status = "ok";
    Scaled v;
    int N = -1;
    double ku = NAN, kuN = NAN;
};

// One Chebyshev run; catches the documented refusal / non-convergence throws.
static Run run_cheb(double a, double b, const std::vector<double>& x, double thresh,
                    bool sort_nodes = true) {
    Run r;
    Cheb E(a, b, 1e-14, 10000);
    E.set_max_error_estimate(thresh);
    E.set_sort_nodes(sort_nodes);
    try {
        r.v = E.evaluate_scaled(x, &r.N);
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        r.status = (msg.find("max_iterations") != std::string::npos) ? "noconv" : "refused";
    }
    r.ku = static_cast<double>(E.last_condition_estimate());
    r.kuN = static_cast<double>(E.last_error_estimate());
    return r;
}

static void write_run(FILE* f, const Run& r) {
    std::fprintf(f, ",%s,%.17g,%ld,%d,%.6e,%.6e", r.status.c_str(), r.v.mantissa,
                 r.v.exponent, r.N, r.ku, r.kuN);
}

int main(int argc, char** argv) {
    const std::string out_dir = (argc > 1) ? argv[1] : ".";
    FILE* fn = std::fopen((out_dir + "/accuracy_e2_nodes.csv").c_str(), "w");
    FILE* fr = std::fopen((out_dir + "/accuracy_e2_results.csv").c_str(), "w");
    if (!fn || !fr) { std::perror("open"); return 1; }

    std::fprintf(fn, "instance,q,c,config,rep,nodes\n");
    std::fprintf(fr, "instance,q,c,config,rep,ordering");
    for (const char* p : {"prod", "ab", "tight", "unsorted"})
        std::fprintf(fr, ",%s_status,%s_mant,%s_exp,%s_N,%s_ku,%s_kuN", p, p, p, p, p, p);
    std::fprintf(fr, ",mnp\n");

    const std::vector<int> qs = {5, 20, 50, 100, 150, 300};
    const std::vector<double> cs = {1.0, 10.0, 100.0};
    const std::vector<std::string> cfgs = {
        "uniform", "cluster_left", "cluster_right", "chebyshev", "near_coincident",
        "near_coincident_far", "repeated", "two_clusters", "equispaced"};
    const int reps = 10;

    int inst = 0;
    for (int q : qs) {
        for (double c : cs) {
            for (size_t ic = 0; ic < cfgs.size(); ++ic) {
                const std::string& cfg = cfgs[ic];
                const int nrep = (cfg == "equispaced") ? 1 : reps;   // deterministic
                for (int rep = 0; rep < nrep; ++rep, ++inst) {
                    // fixed, reproducible seed per (q, c, config, rep)
                    const unsigned long long seed =
                        1000003ULL * q + 10007ULL * static_cast<unsigned long long>(c) +
                        101ULL * ic + rep;
                    std::mt19937_64 rng(seed);
                    const std::vector<double> gen = make_nodes(cfg, q, c, rng);

                    std::fprintf(fn, "%d,%d,%g,%s,%d,", inst, q, c, cfg.c_str(), rep);
                    for (int i = 0; i <= q; ++i) std::fprintf(fn, "%s%.17g", i ? " " : "", gen[i]);
                    std::fprintf(fn, "\n");

                    const double xmin = *std::min_element(gen.begin(), gen.end());
                    const double xmax = *std::max_element(gen.begin(), gen.end());

                    for (const auto& [oname, x] : make_orderings(gen, seed)) {
                        const Run prod = run_cheb(-c, c, x, 1e-10);
                        const Run ab = run_cheb(-c, c, x, 1e300);
                        Run tight;
                        if (xmin < xmax) tight = run_cheb(xmin, xmax, x, 1e300);
                        else tight.status = "na";
                        const Run unsorted = run_cheb(-c, c, x, 1e300, false);
                        double mnp;
                        try { mnp = mccurdy_templated::ExpDivDiff<double>::compute(x); }
                        catch (...) { mnp = NAN; }

                        std::fprintf(fr, "%d,%d,%g,%s,%d,%s", inst, q, c, cfg.c_str(), rep,
                                     oname.c_str());
                        write_run(fr, prod);
                        write_run(fr, ab);
                        write_run(fr, tight);
                        write_run(fr, unsorted);
                        std::fprintf(fr, ",%.17g\n", mnp);
                    }
                }
                std::fflush(fr);
                std::fprintf(stderr, "q=%d c=%g %s done\n", q, c, cfg.c_str());
            }
        }
    }
    std::fclose(fn);
    std::fclose(fr);
    return 0;
}
