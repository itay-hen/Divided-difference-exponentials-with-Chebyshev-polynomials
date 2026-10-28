// =============================================================================
// e6_qmc_chain_legacy.cpp -- LEGACY version of E6b (IncrementalExpDivDiff): end-to-end, PMR-QMC-LIKE Markov chain.
//
// THIS IS A SIMPLIFIED MODEL CHAIN, NOT A PRODUCTION PMR-QMC CODE.  It
// reproduces the data-access pattern of PMR-QMC (divided differences of
// exp over the diagonal energies of a sequence of basis states, modified
// by LIFO insertion/removal of operator pairs at the end of the sequence,
// plus occasional "classical" moves that change every node), but not its
// full move set or its combinatorics.
//
// Model.  Periodic transverse-field Ising chain, L spins, J = Gamma = 1:
//     H = -J sum_i z_i z_{i+1} - Gamma sum_i X_i .
// Configuration: classical state z_0 and a sequence of q single-spin flips
// (i_1, ..., i_q) whose product is the identity.  z_k = z_{k-1} with spin
// i_k flipped; nodes x_k = -beta E(z_k), E(z) = -J sum z_i z_{i+1}.
// Weight w = (beta Gamma)^q exp[x_0, ..., x_q] > 0.
// Moves (per step, random numbers r_type, r_site, r_acc ALWAYS drawn):
//   * p = 0.05 : classical move, flip spin j of z_0 (and hence of every z_k);
//                all nodes change; A = min(1, exp[new]/exp[old]).
//   * else 1/2 : append the pair (i, i), i uniform in {1..L} (two new nodes
//                -beta E(z_q with i flipped), -beta E(z_q));
//                A = min(1, (beta Gamma)^2 L exp[new]/exp[old]).
//   * else 1/2 : remove the last pair (rejected outright if q < 2);
//                A = min(1, exp[new]/exp[old] / ((beta Gamma)^2 L)).
// Because only pairs (i, i) are ever inserted, z_{2m} = z_0 for all m, so
// the node multiset contains x_0 (q/2 + 1 times) and q/2 nodes
// x_0 - beta dE_i with dE_i in {-4J, 0, +4J}: at most three distinct node
// values, spread <= 4 beta J.  The typical order q is ~ beta Gamma sqrt(L)
// here, smaller than in production PMR-QMC (~ beta Gamma L), because this
// chain lacks the operator-ordering entropy of the real expansion.
//
// Implementations (all driven by the SAME random numbers):
//   inc        : IncrementalExpDivDiff on the fixed interval
//                [a, b] = [-beta E_max, -beta E_min] = [-beta J L, beta J L];
//                appends = add_node_normalized twice; a REJECTED append is
//                undone by remove_last_node_normalized twice (cost counted);
//                classical moves use a persistent batch evaluator on the same
//                interval (evaluate_ratio) and, if accepted, re-initialize.
//   inc_local  : as inc, but on the interval [x_0 - 4 beta J, x_0 + 4 beta J]
//                (clipped), which contains every node reachable by pair
//                insertions until the next classical move; re-created after
//                each accepted classical move (cost included).
//   batch      : chebyshev::ExpDivDiff, FRESH object on [min x, max x] of the
//                proposed set every evaluation (pays the R_n precomputation).
//   batch_c    : as batch, but objects cached per interval (min x, max x);
//                legitimate here because energies are discrete.
//   mnp        : mccurdy_templated::ExpDivDiff::compute on the proposed set.
// Two bookkeeping modes, applied identically to all implementations:
//   naive : every proposal recomputes the proposed set (for inc, a removal
//           proposal pops twice and, if rejected, re-appends twice);
//   stack : each implementation keeps the values of all even-order prefixes
//           of the current sequence, so removal proposals cost nothing.  After
//           an accepted classical move: inc rebuilds by appending node by node
//           (which yields every prefix), mnp uses compute_all (all prefixes
//           at no extra cost), batch marks the prefixes stale and evaluates
//           them lazily on demand.
//
// Protocol per (L, beta):
//   1. lockstep pass: all implementations evaluate every proposal; the MNP
//      (naive) ratio drives the decisions; we record disagreements in the
//      accept/reject decision and the relative differences of the ratios,
//      plus chain statistics (q, spread, nodes near b, batch N and kappa u N).
//      n_therm thermalization moves, then n_meas measured moves.
//   2. timing pass: each implementation runs the n_meas measured moves alone,
//      starting from the post-thermalization snapshot with the same RNG
//      state, 3 repetitions, median wall time.  A checksum (number of
//      accepted moves, sum of q) verifies the trajectories are identical.
//
// Output: e6_qmc_timing.csv, e6_qmc_stats.csv, e6_qmc_agreement.csv,
//         e6_qmc_hist.csv
// Usage : e6_qmc_chain <outdir> [n_meas=100000] [n_therm=20000]
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "chebyshev_exp_divdiff.hpp"
#include "incremental_chebyshev_exp_divdiff.hpp"
#include "mccurdy.hpp"
#include "bench_util.hpp"

using Cheb   = chebyshev::ExpDivDiff<double>;
using Scaled = Cheb::ScaledReal;
using Inc    = chebyshev_incremental::IncrementalExpDivDiff<double>;
using MNP    = mccurdy_templated::ExpDivDiff<double>;

// ---------------------------------------------------------------------------
// Chain state shared by all implementations (each keeps its own copy).
// ---------------------------------------------------------------------------
struct Model { int L; double beta, J, G; };

struct ChainState {
    std::vector<int> z0;      // +-1
    std::vector<int> ops;     // flipped sites, length q (pairs i,i)
};

static double energy(const Model& m, const std::vector<int>& z) {
    double e = 0;
    for (int i = 0; i < m.L; ++i) e += z[i] * z[(i + 1) % m.L];
    return -m.J * e;
}
// Energy change for flipping site i of z.
static double dE(const Model& m, const std::vector<int>& z, int i) {
    const int L = m.L;
    return 2.0 * m.J * z[i] * (z[(i + L - 1) % L] + z[(i + 1) % L]);
}
// Full node list of a chain state.
static std::vector<double> build_nodes(const Model& m, const ChainState& s) {
    const double x0 = -m.beta * energy(m, s.z0);
    std::vector<double> x;
    x.reserve(s.ops.size() + 1);
    x.push_back(x0);
    for (size_t k = 0; k < s.ops.size(); k += 2) {
        x.push_back(x0 - m.beta * dE(m, s.z0, s.ops[k]));
        x.push_back(x0);
    }
    return x;
}

enum MoveType { ADD = 0, REMOVE = 1, CLASSICAL = 2 };
struct Move { MoveType type; int site; };

// Diagnostics: the first few node sets on which an evaluator threw are
// printed to stderr (as value:multiplicity pairs) so failures can be reproduced.
static std::map<std::string, int> g_dumped;   // per source, at most 3
static void dump_failure(const char* who, const std::vector<double>& x, const char* what) {
    if (g_dumped[who]++ >= 3) return;
    std::map<double, int> mult;
    for (double v : x) mult[v]++;
    std::fprintf(stderr, "[%s threw] q=%zu nodes(value:mult)=", who, x.size() - 1);
    for (auto& kv : mult) std::fprintf(stderr, " %.17g:%d", kv.first, kv.second);
    std::fprintf(stderr, "\n   order:");
    for (double v : x) std::fprintf(stderr, " %g", v);
    std::fprintf(stderr, "\n   what: %.200s\n", what);
}

static Scaled scaled_from_double(double v) {
    Scaled r; r.mantissa = v; r.exponent = 0; r.normalize(); return r;
}

// ---------------------------------------------------------------------------
// Implementation interface.  propose() returns exp[new]/exp[old] (no
// (beta Gamma)^2 L factor), or a negative value for "move impossible".
// commit() applies or discards the proposal.
// ---------------------------------------------------------------------------
struct Counters {
    long evals = 0;         // full evaluations (batch / MNP / inc rebuilds)
    long appends = 0;       // inc: add_node calls
    long removes = 0;       // inc: remove_last_node calls
    long throws_ = 0;       // exceptions caught (refusals / gate failures)
    long fallback_moves = 0;// moves decided by a fallback evaluator
    long ratio_gate = 0;    // inc: evaluate_ratio on the fixed interval refused
                            // (kappa u N > 1e-10) -> evaluate_scaled used instead
};

struct Impl {
    Model m; ChainState s; std::vector<double> nodes; bool stack_mode = false;
    Counters cnt;
    virtual ~Impl() = default;
    virtual std::string name() const = 0;
    virtual void reset(const Model& mm, const ChainState& st) = 0;
    virtual double propose(const Move& mv) = 0;
    virtual void commit(bool accept) = 0;
};

// Helper: proposed node set for a move (does not modify the impl).
static std::vector<double> proposed_nodes(const Impl& I, const Move& mv,
                                          ChainState* new_state = nullptr) {
    std::vector<double> y = I.nodes;
    if (mv.type == ADD) {
        const double x0 = I.nodes[0];
        y.push_back(x0 - I.m.beta * dE(I.m, I.s.z0, mv.site));
        y.push_back(x0);
    } else if (mv.type == REMOVE) {
        y.resize(y.size() - 2);
    } else {
        ChainState t = I.s;
        t.z0[mv.site] = -t.z0[mv.site];
        y = build_nodes(I.m, t);
        if (new_state) *new_state = t;
    }
    return y;
}

static void apply_move(Impl& I, const Move& mv, const std::vector<double>& y) {
    if (mv.type == ADD) { I.s.ops.push_back(mv.site); I.s.ops.push_back(mv.site); }
    else if (mv.type == REMOVE) { I.s.ops.resize(I.s.ops.size() - 2); }
    else { I.s.z0[mv.site] = -I.s.z0[mv.site]; }
    I.nodes = y;
}

// ------------------------------- batch / MNP --------------------------------
// Evaluators returning a Scaled value for a node set.
struct BatchEval {
    bool cached;
    std::map<std::pair<double, double>, std::unique_ptr<Cheb>> cache;
    // statistics of the last call
    int last_N = 0; long double last_kuN = 0; bool last_ok = true;
    Counters* cnt = nullptr;
    Scaled operator()(const std::vector<double>& x) {
        double mn = x[0], mx = x[0];
        for (double v : x) { mn = std::min(mn, v); mx = std::max(mx, v); }
        if (mn == mx) mx = mn + 1.0;   // all-equal: closed form inside evaluate_scaled
        last_ok = true;
        try {
            if (cached) {
                auto& p = cache[{mn, mx}];
                if (!p) p.reset(new Cheb(mn, mx));
                Scaled v = p->evaluate_scaled(x, &last_N);
                last_kuN = p->last_error_estimate();
                return v;
            }
            Cheb E(mn, mx);
            Scaled v = E.evaluate_scaled(x, &last_N);
            last_kuN = E.last_error_estimate();
            return v;
        } catch (const std::exception& e) {
            // Refusal: fall back to MNP (counted).
            dump_failure("batch", x, e.what());
            last_ok = false;
            if (cnt) { cnt->throws_++; cnt->fallback_moves++; }
            return scaled_from_double(MNP::compute(x));
        }
    }
};

struct BatchImpl : Impl {
    BatchEval ev;
    Scaled V;                       // value of current state
    std::vector<Scaled> stackV;     // value at order 2m (stack mode)
    std::vector<char> valid;
    Scaled Vnew; Move pending{ADD, 0}; std::vector<double> ynew;
    explicit BatchImpl(bool cached, bool stack) { ev.cached = cached; stack_mode = stack; ev.cnt = &cnt; }
    std::string name() const override { return ev.cached ? "batch_c" : "batch"; }
    void reset(const Model& mm, const ChainState& st) override {
        m = mm; s = st; nodes = build_nodes(m, s);
        V = ev(nodes); cnt.evals++;
        stackV.assign(nodes.size() / 2 + 1, Scaled{}); valid.assign(stackV.size(), 0);
        stackV.back() = V; valid.back() = 1;
    }
    double propose(const Move& mv) override {
        pending = mv;
        if (mv.type == REMOVE && nodes.size() < 3) return -1.0;
        ynew = proposed_nodes(*this, mv);
        const size_t mnew = ynew.size() / 2;
        if (stack_mode && mv.type == REMOVE && valid[mnew]) {
            Vnew = stackV[mnew];
        } else {
            Vnew = ev(ynew); cnt.evals++;
        }
        return (Vnew / V).to_real();
    }
    void commit(bool accept) override {
        if (!accept || (pending.type == REMOVE && nodes.size() < 3)) return;
        apply_move(*this, pending, ynew);
        V = Vnew;
        const size_t mm_ = nodes.size() / 2;
        if (pending.type == ADD) { stackV.push_back(V); valid.push_back(1); }
        else if (pending.type == REMOVE) { stackV.resize(mm_ + 1); valid.resize(mm_ + 1); stackV[mm_] = V; valid[mm_] = 1; }
        else { std::fill(valid.begin(), valid.end(), 0); stackV.back() = V; valid.back() = 1; }
    }
};

struct MnpImpl : Impl {
    double V = 0; std::vector<double> stackV; std::vector<char> valid;
    double Vnew = 0; std::vector<double> allnew; Move pending{ADD, 0}; std::vector<double> ynew;
    explicit MnpImpl(bool stack) { stack_mode = stack; }
    std::string name() const override { return "mnp"; }
    void reset(const Model& mm, const ChainState& st) override {
        m = mm; s = st; nodes = build_nodes(m, s);
        auto all = MNP::compute_all(nodes); cnt.evals++;
        V = all.back();
        stackV.assign(nodes.size() / 2 + 1, 0.0); valid.assign(stackV.size(), 1);
        for (size_t k = 0; k < stackV.size(); ++k) stackV[k] = all[2 * k];
    }
    double propose(const Move& mv) override {
        pending = mv;
        if (mv.type == REMOVE && nodes.size() < 3) return -1.0;
        ynew = proposed_nodes(*this, mv);
        const size_t mnew = ynew.size() / 2;
        if (stack_mode && mv.type == REMOVE && valid[mnew]) {
            Vnew = stackV[mnew];
        } else if (stack_mode && mv.type == CLASSICAL) {
            allnew = MNP::compute_all(ynew); cnt.evals++;
            Vnew = allnew.back();
        } else {
            Vnew = MNP::compute(ynew); cnt.evals++;
        }
        if (!(std::isnormal(Vnew) && std::isnormal(V))) cnt.throws_++;  // underflow flag
        return Vnew / V;
    }
    void commit(bool accept) override {
        if (!accept || (pending.type == REMOVE && nodes.size() < 3)) return;
        apply_move(*this, pending, ynew);
        V = Vnew;
        const size_t mm_ = nodes.size() / 2;
        if (pending.type == ADD) { stackV.push_back(V); valid.push_back(1); }
        else if (pending.type == REMOVE) { stackV.resize(mm_ + 1); valid.resize(mm_ + 1); }
        else if (stack_mode) {
            stackV.assign(mm_ + 1, 0.0); valid.assign(mm_ + 1, 1);
            for (size_t k = 0; k <= mm_; ++k) stackV[k] = allnew[2 * k];
        } else { stackV.back() = V; }
    }
};

// ------------------------------- incremental --------------------------------
struct IncImpl : Impl {
    std::unique_ptr<Inc> inc;
    std::unique_ptr<Cheb> Eg;        // batch evaluator on the same fixed interval
    int max_q = 0;
    double c = 0, c2 = 0;
    double S = 0;                    // normalized sum of current state
    std::vector<double> stackS;      // S at order 2m (stack mode)
    bool valid = false;              // inc object consistent with `nodes`?
    double Snew = 0; Move pending{ADD, 0}; std::vector<double> ynew;
    bool pending_fallback = false;   // proposal was evaluated by the fallback
    BatchEval fb;                    // fallback evaluator (fresh, tight)
    // local == false: fixed global interval [-beta J L, beta J L] for the run.
    // local == true : "physics-aware" interval [x_0 - 4 beta J, x_0 + 4 beta J]
    //                 (clipped to the global one), which contains every node
    //                 reachable by pair insertions until the next classical
    //                 move; the object is re-created (constructor + Bessel
    //                 cache, counted in the timing) after each accepted
    //                 classical move.
    bool local = false;
    IncImpl(bool stack, int maxq, bool loc = false) {
        stack_mode = stack; max_q = maxq; fb.cached = false; local = loc;
    }
    std::string name() const override { return local ? "inc_local" : "inc"; }

    void make_object() {
        const double B = m.beta * m.J * m.L;
        double a = -B, b = B;
        if (local) {
            a = std::max(-B, nodes[0] - 4.0 * m.beta * m.J);
            b = std::min( B, nodes[0] + 4.0 * m.beta * m.J);
        }
        c = 0.5 * (b - a); c2 = c * c;
        inc.reset(new Inc(a, b, max_q));
        if (!local || !Eg) Eg.reset(new Cheb(-B, B));
    }

    // (Re)build the incremental object for the current nodes.
    void rebuild() {
        try {
            if (stack_mode) {
                stackS.clear();
                S = inc->initialize_normalized({nodes[0]});
                stackS.push_back(S);
                for (size_t k = 1; k < nodes.size(); ++k) {
                    S = inc->add_node_normalized(nodes[k]); cnt.appends++;
                    if (k % 2 == 0) stackS.push_back(S);
                }
            } else {
                S = inc->initialize_normalized(nodes);
            }
            cnt.evals++;
            valid = true;
        } catch (const std::exception& e) {
            dump_failure("inc-rebuild", nodes, e.what());
            cnt.throws_++; valid = false;
        }
    }
    void reset(const Model& mm, const ChainState& st) override {
        m = mm; s = st; nodes = build_nodes(m, s);
        Eg.reset();
        make_object();
        fb.cnt = nullptr;
        rebuild();
    }
    // Fallback ratio exp[y]/exp[nodes] via fresh tight batch evaluations.
    double fallback_ratio(const std::vector<double>& y) {
        cnt.fallback_moves++;
        pending_fallback = true;
        return (fb(y) / fb(nodes)).to_real();
    }
    double propose(const Move& mv) override {
        pending = mv; pending_fallback = false;
        if (mv.type == REMOVE && nodes.size() < 3) return -1.0;
        ynew = proposed_nodes(*this, mv);
        if (!valid) return fallback_ratio(ynew);
        try {
            if (mv.type == ADD) {
                inc->add_node_normalized(ynew[ynew.size() - 2]); cnt.appends++;
                Snew = inc->add_node_normalized(ynew.back());    cnt.appends++;
                return Snew / S / c2;
            } else if (mv.type == REMOVE) {
                if (stack_mode) {
                    Snew = stackS[ynew.size() / 2];
                } else {
                    inc->remove_last_node_normalized();          cnt.removes++;
                    Snew = inc->remove_last_node_normalized();   cnt.removes++;
                }
                return Snew / S * c2;
            } else {
                // Classical move: batch evaluator on the fixed interval.
                cnt.evals++;
                try { return Eg->evaluate_ratio(ynew, nodes); }
                catch (...) {
                    cnt.ratio_gate++;
                    return (Eg->evaluate_scaled(ynew) / Eg->evaluate_scaled(nodes)).to_real();
                }
            }
        } catch (const std::exception& e) {
            dump_failure("inc", ynew, e.what());
            cnt.throws_++;
            valid = false;               // object may be half-updated
            return fallback_ratio(ynew);
        }
    }
    void commit(bool accept) override {
        if (pending.type == REMOVE && nodes.size() < 3) return;
        if (pending_fallback || !valid) {
            if (accept) apply_move(*this, pending, ynew);
            valid = false;
            rebuild();
            return;
        }
        try {
            if (pending.type == ADD) {
                if (accept) { S = Snew; if (stack_mode) stackS.push_back(S); }
                else {
                    inc->remove_last_node_normalized(); cnt.removes++;
                    inc->remove_last_node_normalized(); cnt.removes++;
                }
            } else if (pending.type == REMOVE) {
                if (accept) {
                    if (stack_mode) {
                        inc->remove_last_node_normalized(); cnt.removes++;
                        inc->remove_last_node_normalized(); cnt.removes++;
                        stackS.pop_back();
                    }
                    S = Snew;
                } else if (!stack_mode) {
                    // naive mode: undo the two pops.
                    inc->add_node_normalized(nodes[nodes.size() - 2]); cnt.appends++;
                    inc->add_node_normalized(nodes.back());            cnt.appends++;
                }
            }
        } catch (...) { cnt.throws_++; valid = false; }
        if (accept) apply_move(*this, pending, ynew);
        if (pending.type == CLASSICAL && accept) {
            valid = false;
            if (local) make_object();
            rebuild();
        }
        else if (!valid) rebuild();
    }
};

// ---------------------------------------------------------------------------
struct Rng {
    std::mt19937_64 g;
    std::uniform_real_distribution<double> U{0.0, 1.0};
    double operator()() { return U(g); }
};

static Move draw_move(const Model& m, Rng& r, double& u_acc) {
    const double rt = r(), rs = r(); u_acc = r();
    Move mv;
    const int site = std::min(m.L - 1, static_cast<int>(rs * m.L));
    if (rt < 0.05) mv = {CLASSICAL, site};
    else if (rt < 0.525) mv = {ADD, site};
    else mv = {REMOVE, site};
    return mv;
}
static double acceptance(const Model& m, const Move& mv, double ratio) {
    if (ratio < 0) return 0.0;
    const double f = m.beta * m.G * m.beta * m.G * m.L;
    if (mv.type == ADD) return std::min(1.0, f * ratio);
    if (mv.type == REMOVE) return std::min(1.0, ratio / f);
    return std::min(1.0, ratio);
}

static double quantile(std::vector<double> v, double p) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    const double idx = p * (v.size() - 1);
    const size_t i = static_cast<size_t>(idx);
    const double fr = idx - i;
    return (i + 1 < v.size()) ? v[i] * (1 - fr) + v[i + 1] * fr : v[i];
}

int main(int argc, char** argv) {
    const std::string outdir = (argc > 1) ? argv[1] : ".";
    const long n_meas  = (argc > 2) ? std::atol(argv[2]) : 100000;
    const long n_therm = (argc > 3) ? std::atol(argv[3]) : 20000;
    // Standard grid L in {8,16,32} x beta in {1,2,4}, plus two extended
    // points (L = 32, beta = 8, 16) that reach larger orders q.
    std::vector<std::pair<int, double>> params;
    for (int L : {8, 16, 32}) for (double be : {1.0, 2.0, 4.0}) params.push_back({L, be});
    params.push_back({32, 8.0});
    params.push_back({32, 16.0});

    FILE* ft = bench::open_csv(outdir + "/e6_qmc_timing_legacy.csv");
    FILE* fs = bench::open_csv(outdir + "/e6_qmc_stats_legacy.csv");
    FILE* fa = bench::open_csv(outdir + "/e6_qmc_agreement_legacy.csv");
    FILE* fh = bench::open_csv(outdir + "/e6_qmc_hist_legacy.csv");
    std::fprintf(ft, "L,beta,impl,mode,n_moves,t_move_us,t_move_min_us,t_move_max_us,"
                     "checksum_ok,evals,appends,removes,throws,fallback_moves,ratio_gate,max_q,max_order\n");
    std::fprintf(fs, "L,beta,c,n_therm,n_meas,acc_add,acc_remove,acc_classical,"
                     "q_mean,q_p05,q_p50,q_p95,q_max,spread_mean,spread_p05,spread_p95,"
                     "frac_within1_of_b,frac_at_b,"
                     "batch_N_mean,batch_N_max,batch_kuN_median,batch_kuN_p95,batch_kuN_max,"
                     "batch_refusals,mnp_nonnormal\n");
    std::fprintf(fa, "L,beta,impl,mode,n_proposals,decision_disagreements,max_rel_diff_ratio,"
                     "p99_rel_diff_ratio,throws\n");
    std::fprintf(fh, "L,beta,quantity,value,count\n");

    for (const auto& LB : params) {
        const int L = LB.first; const double beta = LB.second;
        const Model m{L, beta, 1.0, 1.0};
        // Initial state: all spins up, empty sequence.
        ChainState s0; s0.z0.assign(L, 1);

        // --- pilot thermalization with MNP only, to choose inc capacity ----
        Rng rng; rng.g.seed(1000003ULL * L + static_cast<unsigned long long>(beta * 7));
        MnpImpl pilot(false); pilot.reset(m, s0);
        int q_max_seen = 0;
        for (long t = 0; t < n_therm; ++t) {
            double u; Move mv = draw_move(m, rng, u);
            const double A = acceptance(m, mv, pilot.propose(mv));
            pilot.commit(u < A);
            q_max_seen = std::max(q_max_seen, static_cast<int>(pilot.s.ops.size()));
        }
        // Capacity of the incremental object: comfortably above the pilot's
        // maximum (exceeding it would throw; such throws are counted).
        const int max_q = std::max(32, 2 * q_max_seen + 16);
        const ChainState snap = pilot.s;
        const Rng rng_snap = rng;

        // --- implementations --------------------------------------------------
        std::vector<std::unique_ptr<Impl>> impls;
        for (int stack = 0; stack < 2; ++stack) {
            impls.emplace_back(new MnpImpl(stack));       // impls[0] (naive) drives
            impls.emplace_back(new IncImpl(stack, max_q));
            impls.emplace_back(new IncImpl(stack, max_q, true));
            impls.emplace_back(new BatchImpl(false, stack));
            impls.emplace_back(new BatchImpl(true, stack));
        }
        for (auto& I : impls) I->reset(m, snap);
        const int max_order = static_cast<IncImpl*>(impls[1].get())->inc->max_order();

        // --- lockstep measured pass ---------------------------------------------
        Rng r = rng_snap;
        std::vector<long> disagree(impls.size(), 0);
        std::vector<double> maxrel(impls.size(), 0.0);
        std::vector<std::vector<double>> rels(impls.size());
        long n_add = 0, n_rem = 0, n_cl = 0, a_add = 0, a_rem = 0, a_cl = 0;
        long acc_total = 0; double sum_q = 0;
        std::vector<double> qv, spv, Nv, kuv;
        double near_b = 0, at_b = 0, n_nodes_tot = 0;
        long batch_ref = 0;
        const double b = beta * L;
        BatchImpl* batch_naive = static_cast<BatchImpl*>(impls[3].get());
        for (long t = 0; t < n_meas; ++t) {
            double u; Move mv = draw_move(m, r, u);
            std::vector<double> ratios(impls.size());
            for (size_t k = 0; k < impls.size(); ++k) {
                ratios[k] = impls[k]->propose(mv);
                if (k == 3 && ratios[k] >= 0) {
                    Nv.push_back(batch_naive->ev.last_N);
                    kuv.push_back(static_cast<double>(batch_naive->ev.last_kuN));
                    if (!batch_naive->ev.last_ok) batch_ref++;
                }
            }
            const double A0 = acceptance(m, mv, ratios[0]);
            const bool acc = u < A0;
            for (size_t k = 1; k < impls.size(); ++k) {
                const double Ak = acceptance(m, mv, ratios[k]);
                if ((u < Ak) != acc) disagree[k]++;
                if (ratios[0] > 0) {
                    const double rd = std::fabs(ratios[k] - ratios[0]) / ratios[0];
                    maxrel[k] = std::max(maxrel[k], rd);
                    rels[k].push_back(rd);
                }
            }
            for (auto& I : impls) I->commit(acc);
            if (mv.type == ADD) { n_add++; a_add += acc; }
            else if (mv.type == REMOVE) { n_rem++; a_rem += acc; }
            else { n_cl++; a_cl += acc; }
            acc_total += acc;
            const auto& x = impls[0]->nodes;
            sum_q += x.size() - 1;
            qv.push_back(x.size() - 1);
            double mn = x[0], mx = x[0];
            for (double v : x) {
                mn = std::min(mn, v); mx = std::max(mx, v);
                near_b += (b - v <= 1.0); at_b += (v == b);
            }
            n_nodes_tot += x.size();
            spv.push_back(0.5 * (mx - mn));
        }
        const long ref_acc = acc_total; const double ref_sumq = sum_q;

        std::fprintf(fs, "%d,%g,%g,%ld,%ld,%.4f,%.4f,%.4f,%.3f,%g,%g,%g,%g,%.4f,%g,%g,%.4f,%.4f,"
                         "%.2f,%g,%.3e,%.3e,%.3e,%ld,%ld\n",
                     L, beta, b, n_therm, n_meas,
                     double(a_add) / std::max(1L, n_add), double(a_rem) / std::max(1L, n_rem),
                     double(a_cl) / std::max(1L, n_cl),
                     sum_q / n_meas, quantile(qv, 0.05), quantile(qv, 0.5), quantile(qv, 0.95),
                     *std::max_element(qv.begin(), qv.end()),
                     [&] { double a = 0; for (double v : spv) a += v; return a / spv.size(); }(),
                     quantile(spv, 0.05), quantile(spv, 0.95),
                     near_b / n_nodes_tot, at_b / n_nodes_tot,
                     [&] { double a = 0; for (double v : Nv) a += v; return a / Nv.size(); }(),
                     *std::max_element(Nv.begin(), Nv.end()),
                     quantile(kuv, 0.5), quantile(kuv, 0.95), *std::max_element(kuv.begin(), kuv.end()),
                     batch_ref, impls[0]->cnt.throws_);
        for (size_t k = 1; k < impls.size(); ++k)
            std::fprintf(fa, "%d,%g,%s,%s,%ld,%ld,%.3e,%.3e,%ld\n", L, beta,
                         impls[k]->name().c_str(), impls[k]->stack_mode ? "stack" : "naive",
                         n_meas, disagree[k], maxrel[k], quantile(rels[k], 0.99),
                         impls[k]->cnt.throws_);
        {   // histograms of q and spread
            std::map<int, long> hq; std::map<double, long> hs;
            for (double v : qv) hq[static_cast<int>(v)]++;
            for (double v : spv) hs[v]++;
            for (auto& kv : hq) std::fprintf(fh, "%d,%g,q,%d,%ld\n", L, beta, kv.first, kv.second);
            for (auto& kv : hs) std::fprintf(fh, "%d,%g,spread,%.6g,%ld\n", L, beta, kv.first, kv.second);
        }
        std::fflush(fs); std::fflush(fa); std::fflush(fh);
        std::printf("L=%d beta=%g: <q>=%.2f q95=%g spread=%.3f nearb=%.3f max_q=%d Nmax=%d "
                    "acc(add/rem/cl)=%.3f/%.3f/%.3f\n", L, beta, sum_q / n_meas,
                    quantile(qv, 0.95), quantile(spv, 0.5), near_b / n_nodes_tot, max_q, max_order,
                    double(a_add) / n_add, double(a_rem) / n_rem, double(a_cl) / std::max(1L, n_cl));
        for (size_t k = 1; k < impls.size(); ++k)
            std::printf("   %-8s %-5s disagreements=%ld maxrel=%.2e throws=%ld\n",
                        impls[k]->name().c_str(), impls[k]->stack_mode ? "stack" : "naive",
                        disagree[k], maxrel[k], impls[k]->cnt.throws_);
        std::fflush(stdout);

        // --- timing pass: each implementation alone ------------------------------
        for (size_t k = 0; k < impls.size(); ++k) {
            std::vector<double> tv;
            bool ok = true;
            Counters cnt_last;
            int mo = -1;
            for (int rep = 0; rep < 3; ++rep) {
                std::unique_ptr<Impl> I;
                const std::string nm = impls[k]->name();
                const bool st = impls[k]->stack_mode;
                if (nm == "mnp") I.reset(new MnpImpl(st));
                else if (nm == "inc") I.reset(new IncImpl(st, max_q));
                else if (nm == "inc_local") I.reset(new IncImpl(st, max_q, true));
                else if (nm == "batch") I.reset(new BatchImpl(false, st));
                else I.reset(new BatchImpl(true, st));
                I->reset(m, snap);
                I->cnt = Counters{};
                Rng rr = rng_snap;
                long acc_c = 0; double sq = 0;
                auto t0 = bench::clock_type::now();
                for (long t = 0; t < n_meas; ++t) {
                    double u; Move mv = draw_move(m, rr, u);
                    const bool acc = u < acceptance(m, mv, I->propose(mv));
                    I->commit(acc);
                    acc_c += acc; sq += I->nodes.size() - 1;
                }
                tv.push_back(bench::seconds_since(t0) / n_meas * 1e6);
                if (acc_c != ref_acc || sq != ref_sumq) ok = false;
                cnt_last = I->cnt;
                if (auto* II = dynamic_cast<IncImpl*>(I.get())) mo = II->inc->max_order();
                bench::g_sink = bench::g_sink + sq;
            }
            std::fprintf(ft, "%d,%g,%s,%s,%ld,%.5g,%.5g,%.5g,%d,%ld,%ld,%ld,%ld,%ld,%ld,%d,%d\n",
                         L, beta, impls[k]->name().c_str(), impls[k]->stack_mode ? "stack" : "naive",
                         n_meas, bench::median(tv), *std::min_element(tv.begin(), tv.end()),
                         *std::max_element(tv.begin(), tv.end()), ok ? 1 : 0,
                         cnt_last.evals, cnt_last.appends, cnt_last.removes, cnt_last.throws_,
                         cnt_last.fallback_moves, cnt_last.ratio_gate, max_q, mo);
            std::fflush(ft);
            std::printf("   timing %-8s %-5s %.3f us/move  checksum %s  throws=%ld\n",
                        impls[k]->name().c_str(), impls[k]->stack_mode ? "stack" : "naive",
                        bench::median(tv), ok ? "ok" : "MISMATCH", cnt_last.throws_);
            std::fflush(stdout);
        }
    }
    std::fclose(ft); std::fclose(fs); std::fclose(fa); std::fclose(fh);
    return 0;
}
