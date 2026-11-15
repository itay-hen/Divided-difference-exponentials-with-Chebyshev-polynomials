// replay.cpp -- replays recorded PMR-QMC divided-difference traces.
//
// A trace is the exact stream of operations that the PMR-QMC code of
// Barash, Babakhani and Hen (Phys. Rev. Research 6, 013281 (2024)) issues
// on its weight object during a run (recorded with divdiff_trace.hpp):
//
//   A x   append node x            R     remove the last node
//   V x   remove the (last) node equal to x, wherever it is
//   C     clear                    Q i   read exp[z_0..z_i], i = q
//
// Nodes are z = -beta E for the diagonal energies E along the operator
// string.  V is an interior removal; updates that change the energy string
// from a position j on (cycle completions, single-operator moves, spin
// flips) appear as runs of R followed by runs of A.  Only Q produces a
// value; all evaluators see the same operations.
//
// Evaluators
//   gbh     the divdiff class shipped with PMR-QMC (Gupta, Barash, Hen,
//           CPC 254, 107385 (2020)), unmodified (divdiff_pmrqmc.hpp),
//           driven exactly as PMR-QMC drives it.
//   batch   chebyshev::ExpDivDiff at every Q with the retry sequence of
//           evaluate(): the tight interval [min z, max z], then
//           [min z - s w, max z] for s = 1, 2 (w = max z - min z); the
//           objects are cached per interval, so the Bessel ratios are
//           computed once per distinct interval.
//   stack   chebyshev::IncrementalStack, using the symmetry of the divided
//           difference: the stack keeps its own order of the multiset.
//           When it is (re)built, the nodes are sorted in nonincreasing
//           order on [min z - w, max z] (midpoint at the smallest node,
//           Sec. 4.2 of the manuscript); appends go on top; removing a
//           value pops down to its topmost occurrence and re-pushes the
//           nodes above it.  A node outside the interval, or a refused
//           push, invalidates the stack; it is rebuilt at the next Q (stack
//           objects, with their Bessel ratios, are cached per interval).
//           (With REPLAY_PMR_ORDER set, the stack keeps PMR's own order
//           instead; then short prefixes of nodes far below max z are
//           ill-conditioned and almost every push is refused.)
//   stack0  the naive variant: one fixed interval for the whole run (the
//           range of all nodes in the trace) and PMR's own node order.
// A Q that a Chebyshev evaluator refuses (indicator above 1e-10) is
// answered by a fresh GBH evaluation of the set (counted as a fallback and
// timed with that evaluator).  Accuracy: largest |log difference| from the
// batch column, which holds either a value accepted by batch (indicator
// <= 1e-10) or a fresh GBH evaluation (both accurate to ~1e-13 against
// exact values on sampled queries), over the queries where the evaluator
// returned its own value.  gbh is driven incrementally, as in PMR-QMC,
// and its state accumulates rounding errors over long add/remove streams.
//
// Build: g++ -std=c++17 -O3 -march=native -I../.. -I/opt/homebrew/include replay.cpp -o replay
#include "chebyshev_exp_divdiff.hpp"
#include "chebyshev_stack.hpp"
#include "divdiff_pmrqmc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Stack = chebyshev::IncrementalStack<double>;
using Batch = chebyshev::ExpDivDiff<double>;

struct Op { char t; double x; int i; };

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Trace {
    std::string name;
    double beta = 0;
    std::vector<Op> ops;
    double zmin = 1e300, zmax = -1e300, csum = 0;
    int qmax = 0; long nQ = 0, nA = 0, nR = 0, nV = 0, nC = 0;
    double qsum = 0;
};

void erase_last(std::vector<double>& v, double x) {
    for (int k = static_cast<int>(v.size()) - 1; k >= 0; --k)
        if (v[k] == x) { v.erase(v.begin() + k); return; }
    throw std::runtime_error("value not found");
}

Trace load(const char* fn) {
    Trace T; T.name = fn;
    FILE* f = std::fopen(fn, "r");
    if (!f) throw std::runtime_error(std::string("cannot open ") + fn);
    char line[256];
    std::vector<double> sh;
    while (std::fgets(line, sizeof line, f)) {
        Op o{line[0], 0.0, 0};
        switch (line[0]) {
        case '#': { const char* p = std::strstr(line, "beta"); if (p) T.beta = std::atof(p + 4); continue; }
        case 'A': o.x = std::strtod(line + 2, nullptr); sh.push_back(o.x); ++T.nA;
                  T.zmin = std::min(T.zmin, o.x); T.zmax = std::max(T.zmax, o.x);
                  T.qmax = std::max<int>(T.qmax, static_cast<int>(sh.size()) - 1); break;
        case 'R': sh.pop_back(); ++T.nR; break;
        case 'V': o.x = std::strtod(line + 2, nullptr); ++T.nV; erase_last(sh, o.x); break;
        case 'C': sh.clear(); ++T.nC; break;
        case 'Q': { o.i = std::atoi(line + 2); ++T.nQ; T.qsum += static_cast<double>(sh.size()) - 1;
                    const auto mm = std::minmax_element(sh.begin(), sh.end());
                    T.csum += (*mm.second - *mm.first) / 2; break; }
        default: continue;
        }
        T.ops.push_back(o);
    }
    std::fclose(f);
    return T;
}

long double gbh_log(divdiff& d, int i) {
    ExExFloat v = d.divdiffs[i];
    return std::log(std::fabs(static_cast<long double>(v.get_mantissa()))) +
           v.get_exponent() * 0.693147180559945309417232121458176568L -
           std::lgamma(static_cast<long double>(i + 1));
}

// Fresh GBH evaluation of a set (the fallback for refused queries).
// (One object, reused: PMR-QMC's own GetWeight() does the same.)
long double gbh_fresh(const std::vector<double>& x) {
    static divdiff d(4, 500);
    d.CurrentLength = 0;
    for (double t : x) d.AddElement(t);
    return gbh_log(d, static_cast<int>(x.size()) - 1);
}

struct Stats { long fallbacks = 0, rebuilds = 0, pushes = 0; };

// ---------------------------------------------------------------- GBH
double replay_gbh(const Trace& T, std::vector<long double>& out) {
    out.clear(); out.reserve(T.nQ);
    divdiff d(4, 500);
    const double t0 = now_ms();
    for (const Op& o : T.ops) {
        switch (o.t) {
        case 'A': d.AddElement(o.x); break;
        case 'R': d.RemoveElement(); break;
        case 'V': if (!d.RemoveValue(o.x)) throw std::runtime_error("gbh: value not found"); break;
        case 'C': d.CurrentLength = 0; break;
        case 'Q': out.push_back(gbh_log(d, o.i)); break;
        }
    }
    return now_ms() - t0;
}

// ---------------------------------------------------------------- batch
double replay_batch(const Trace& T, std::vector<long double>& out,
                    std::vector<char>& own, Stats& S) {
    out.clear(); own.clear(); S = Stats();
    std::map<std::pair<double, double>, std::unique_ptr<Batch>> cache;
    std::vector<double> sh; sh.reserve(T.qmax + 2);
    const double t0 = now_ms();
    for (const Op& o : T.ops) {
        switch (o.t) {
        case 'A': sh.push_back(o.x); break;
        case 'R': sh.pop_back(); break;
        case 'V': erase_last(sh, o.x); break;
        case 'C': sh.clear(); break;
        case 'Q': {
            const auto mm = std::minmax_element(sh.begin(), sh.end());
            const double lo = *mm.first, hi = *mm.second;
            if (lo == hi) {                       // closed form, no interval needed
                Batch E(lo - 1.0, lo + 1.0);
                out.push_back(E.evaluate_scaled(sh).log_abs()); own.push_back(1); break;
            }
            bool done = false;
            for (int s = 0; s <= 2 && !done; ++s) {
                const double a = lo - s * (hi - lo);
                auto& E = cache[{a, hi}];
                if (!E) E.reset(new Batch(a, hi));
                try { out.push_back(E->evaluate_scaled_fixed(sh).log_abs()); own.push_back(1); done = true; }
                catch (const std::runtime_error&) {}
            }
            if (!done) { out.push_back(gbh_fresh(sh)); own.push_back(0); ++S.fallbacks; }
            break; }
        }
    }
    return now_ms() - t0;
}

// ------------------------------------------------- stack (multiset order)
double replay_stack(const Trace& T, std::vector<long double>& out,
                    std::vector<char>& own, Stats& S, bool sort_on_rebuild) {
    out.clear(); own.clear(); S = Stats();
    std::vector<double> sh;                 // PMR's sequence (for R)
    std::vector<double> ms;                 // the stack's order of the multiset
    std::map<std::pair<double, double>, std::unique_ptr<Stack>> cache;  // per interval
    Stack* st = nullptr;
    double a = 0, b = 0;
    int valid = 0;                          // ms[0..valid) are in the stack
    bool failed = false;                    // last rebuild of this set refused
    auto push_rest = [&]() {
        while (st && valid == st->order() + 1 && valid < static_cast<int>(ms.size())) {
            const double x = ms[valid];
            if (x < a || x > b) return;
            try { st->push(x); ++valid; ++S.pushes; } catch (const std::runtime_error&) { return; }
        }
    };
    auto remove_value = [&](double x) {
        int k = static_cast<int>(ms.size()) - 1;
        for (; k >= 0; --k) if (ms[k] == x) break;
        if (k < 0) throw std::runtime_error("stack: value not found");
        ms.erase(ms.begin() + k);
        if (k < valid) {
            while (valid > k) { st->pop(); --valid; }
            push_rest();
        }
        failed = false;
    };
    auto rebuild = [&]() {
        ++S.rebuilds;
        const auto mm = std::minmax_element(ms.begin(), ms.end());
        const double w = *mm.second - *mm.first;
        a = *mm.first - (w > 0 ? w : 1.0);
        b = *mm.second + 1e-9 * std::max(1.0, std::fabs(*mm.second));
        if (sort_on_rebuild) std::sort(ms.begin(), ms.end(), std::greater<double>());
        auto& slot = cache[{a, b}];
        if (!slot) slot.reset(new Stack(a, b, T.qmax + 8));
        st = slot.get(); st->clear();
        valid = 0;
        push_rest();
        failed = valid < static_cast<int>(ms.size());
    };
    const double t0 = now_ms();
    for (const Op& o : T.ops) {
        switch (o.t) {
        case 'A':
            sh.push_back(o.x); ms.push_back(o.x); failed = false;
            if (st && valid == static_cast<int>(ms.size()) - 1) push_rest();
            break;
        case 'R': { const double x = sh.back(); sh.pop_back(); remove_value(x); break; }
        case 'V': erase_last(sh, o.x); remove_value(o.x); break;
        case 'C': sh.clear(); ms.clear(); if (st) st->clear(); valid = 0; failed = false; break;
        case 'Q':
            if (valid < static_cast<int>(ms.size()) && !failed) rebuild();
            if (valid == static_cast<int>(ms.size())) {
                out.push_back(st->value_scaled().log_abs()); own.push_back(1);
            } else {
                out.push_back(gbh_fresh(ms)); own.push_back(0); ++S.fallbacks;
            }
            break;
        }
    }
    const double t = now_ms() - t0;
    if (std::getenv("REPLAY_DEBUG")) std::fprintf(stderr, "stack pushes %ld rebuilds %ld appends %ld\n", S.pushes, S.rebuilds, T.nA);
    return t;
}

// ------------------------------------------- stack0 (naive: fixed, PMR order)
double replay_stack0(const Trace& T, std::vector<long double>& out,
                     std::vector<char>& own, Stats& S) {
    out.clear(); own.clear(); S = Stats();
    const double pad = 1e-9 * std::max(1.0, std::fabs(T.zmax));
    Stack st(T.zmin - pad, T.zmax + pad, T.qmax + 1);
    std::vector<double> sh;
    int valid = 0;
    auto push_rest = [&]() {
        while (valid == st.order() + 1 && valid < static_cast<int>(sh.size())) {
            try { st.push(sh[valid]); ++valid; ++S.pushes; } catch (const std::runtime_error&) { return; }
        }
    };
    const double t0 = now_ms();
    for (const Op& o : T.ops) {
        switch (o.t) {
        case 'A': sh.push_back(o.x); push_rest(); break;
        case 'R': sh.pop_back(); if (valid > static_cast<int>(sh.size())) { st.pop(); --valid; } break;
        case 'V': {
            int k = static_cast<int>(sh.size()) - 1;
            for (; k >= 0; --k) if (sh[k] == o.x) break;
            sh.erase(sh.begin() + k);
            if (k < valid) { while (valid > k) { st.pop(); --valid; } push_rest(); }
            break; }
        case 'C': st.clear(); sh.clear(); valid = 0; break;
        case 'Q':
            if (valid == static_cast<int>(sh.size())) { out.push_back(st.value_scaled().log_abs()); own.push_back(1); }
            else { out.push_back(gbh_fresh(sh)); own.push_back(0); ++S.fallbacks; }
            break;
        }
    }
    return now_ms() - t0;
}

long double max_dlog(const std::vector<long double>& u, const std::vector<char>& own,
                     const std::vector<long double>& ref, const std::vector<char>& rown) {
    long double m = 0;
    (void)rown;
    for (std::size_t i = 0; i < u.size(); ++i)
        if (own[i]) m = std::max(m, std::fabs(u[i] - ref[i]));
    return m;
}

} // namespace

int main(int argc, char** argv) {
    int arg = 1; const int reps = 3;
    if (arg < argc && std::strcmp(argv[arg], "--header") == 0) {
        std::printf("trace,beta,ops,queries,frac_V,mean_q,max_q,mean_c,trace_c,"
                    "t_gbh_ms,t_batch_ms,t_stack_ms,t_stack0_ms,"
                    "batch_fallback,stack_fallback,stack_rebuilds,stack0_fallback,"
                    "dlog_gbh,dlog_stack,dlog_stack0\n");
        ++arg;
    }
    for (; arg < argc; ++arg) {
        const Trace T = load(argv[arg]);
        std::vector<long double> vg, vb, vs, v0;
        std::vector<char> ob, os, o0;
        double tg = 1e300, tb = 1e300, ts = 1e300, t0 = 1e300;
        Stats Sb, Ss, S0;
        for (int r = 0; r < reps; ++r) {
            tg = std::min(tg, replay_gbh(T, vg));
            tb = std::min(tb, replay_batch(T, vb, ob, Sb));
            ts = std::min(ts, replay_stack(T, vs, os, Ss, std::getenv("REPLAY_PMR_ORDER") == nullptr));
            t0 = std::min(t0, replay_stack0(T, v0, o0, S0));
        }
        const char* base = std::strrchr(T.name.c_str(), '/');
        const double nq = static_cast<double>(std::max(1L, T.nQ));
        std::printf("%s,%g,%zu,%ld,%.4f,%.1f,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,"
                    "%.4f,%.4f,%ld,%.4f,%.1Le,%.1Le,%.1Le\n",
                    base ? base + 1 : T.name.c_str(), T.beta, T.ops.size(), T.nQ,
                    double(T.nV) / T.ops.size(), T.qsum / nq, T.qmax, T.csum / nq,
                    (T.zmax - T.zmin) / 2, tg, tb, ts, t0,
                    Sb.fallbacks / nq, Ss.fallbacks / nq, Ss.rebuilds, S0.fallbacks / nq,
                    max_dlog(vg, std::vector<char>(vg.size(), 1), vb, ob),
                    max_dlog(vs, os, vb, ob), max_dlog(v0, o0, vb, ob));
        std::fflush(stdout);
    }
    return 0;
}
