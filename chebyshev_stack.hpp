#ifndef CHEBYSHEV_STACK_HPP
#define CHEBYSHEV_STACK_HPP

// =============================================================================
// chebyshev_stack.hpp
//
// Incremental (append / LIFO-remove) evaluation of exp[x_0, ..., x_q] on a
// fixed interval [a, b] -- Section 3 of the manuscript.
//
// Storage.  The stack holds one column per node: column k contains the
// weighted Chebyshev divided differences
//
//     W_k^(n) = R_n(c) D_k^(n) / sigma_k,   n = k..Nmax  (n = 0..Nmax for k = 0),
//     D_k^(n) = T_n[y_0, ..., y_k],  sigma_0 = 1/2,  sigma_k = R_k(c) 2^{k-1},
//
// which is a diagonal similarity of the recurrence (Lemma 2.2):
//
//     W_k^(n+1) = 2 rho_{n+1} ( y_k W_k^(n) + W_{k-1}^(n) / (2 rho_k) )
//                 - rho_{n+1} rho_n W_k^(n-1),        rho_n = I_n(c)/I_{n-1}(c).
//
// The similarity does not change relative rounding errors, but it removes
// every range problem of the unweighted table: each row starts at
// W_k^(k) = 1 (D_k^(k) = 2^{k-1}), the entries are the series terms
// themselves divided by 2 sigma_k, so they neither overflow (unless the
// sum is catastrophically ill-conditioned) nor underflow before they are
// negligible.  sigma_k is kept in mantissa/exponent form; it cancels in
// ratios of divided differences of the same order.
//
// Costs.  The structural zeros W_k^(n) = 0, n < k, are not stored, so
// push() computes and stores Nmax - k + 1 entries: Theta(Nmax - q) work
// and memory.  The sum for the new top is accumulated until the stopping
// rule is met (O(N')), and is cached per level, so pop() is O(1).
//
// Cache depth.  If the stopping rule is not met within Nmax, the depth is
// increased and all stored columns are extended by the recurrence (a
// one-time Theta(q * Delta N) cost), and the push is retried.
//
// Stopping rules.  Window (default): Algorithm 1's heuristic -- stop at
// the first n >= k + 5 at which the last 5 terms are below tol*|S|.
// Certified: stop at the first N at which the rigorous relative tail
// bound (5.3) of the manuscript is below tol, evaluated in log space with
// an upper estimate of log I_0(c) and a geometric bound for the tail
// beyond Nmax (a conservative floating-point evaluation of a rigorous
// bound, not an interval-arithmetic certificate).
//
// Accuracy safeguard.  Every prefix x_0..x_k is evaluated and gated when
// it is pushed: if its error indicator kappa*u*N exceeds the threshold
// (default 1e-10), push() throws and leaves the stack unchanged.  The
// divided difference is symmetric, but the rounding errors of the
// recurrence depend on the processing order (manuscript, Sec. 4.3);
// gating every prefix detects unfavorable insertion orders, and
// initialize() inserts the initial set in nonincreasing order.
// =============================================================================

#include "chebyshev_exp_divdiff.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace chebyshev {

enum class StackStopping { Window, Certified };

template<typename Real = double>
class IncrementalStack {
public:
    using ScaledReal = typename ExpDivDiff<Real>::ScaledReal;

    /**
     * @param a, b      Fixed interval (b > a) containing every node ever pushed.
     * @param q_max     Largest order that will be reached (sets Nmax when
     *                  nmax < 0).
     * @param tol       Relative tolerance of the stopping rule.
     * @param nmax      Initial cache depth Nmax; default (-1) chooses
     *                  q_max + 1.3 sqrt(2 c ln(1/tol)) + 30.  The depth is
     *                  increased automatically when a push needs more.
     * @param stopping  StackStopping::Window (default) or ::Certified.
     */
    IncrementalStack(Real a, Real b, int q_max, Real tol = Real(1e-14), int nmax = -1,
                     StackStopping stopping = StackStopping::Window)
        : a_(a), b_(b), tol_(tol), q_max_(q_max), stopping_(stopping) {
        if (!(b > a) || !std::isfinite(static_cast<double>(a)) ||
            !std::isfinite(static_cast<double>(b))) {
            throw std::invalid_argument("IncrementalStack: need finite a < b");
        }
        if (q_max < 0) throw std::invalid_argument("IncrementalStack: q_max < 0");
        if (!(tol > Real(0))) throw std::invalid_argument("IncrementalStack: tol <= 0");
        c_ = b / 2 - a / 2;
        d_ = b / 2 + a / 2;
        if (nmax < 0) {
            const double L = std::log(1.0 / static_cast<double>(tol));
            nmax = q_max + static_cast<int>(std::ceil(
                       1.3 * std::sqrt(2.0 * static_cast<double>(c_) * L))) + 30;
        }
        if (nmax < q_max + W_ + 1) {
            throw std::invalid_argument("IncrementalStack: nmax must exceed q_max + 5");
        }
        nmax_ = nmax;
        // Practical ceiling for automatic depth increases.
        nmax_ceiling_ = std::max(20 * nmax_, 4 * q_max_ + 2000);
        compute_rho_(nmax_);
    }

    /// Clear the stack and push nodes in nonincreasing order.
    void initialize(std::vector<Real> nodes) {
        clear();
        std::sort(nodes.begin(), nodes.end(), std::greater<Real>());
        for (const Real& x : nodes) push(x);
    }

    void clear() {
        cols_.clear(); levels_.clear();
    }

    /**
     * @brief Append a node: Theta(Nmax - q) for the new column plus O(N')
     *        for its sum.  Throws (stack unchanged) if the node is
     *        outside [a, b], if the new top is too ill-conditioned
     *        (kappa*u*N above the threshold), or if the stopping rule is
     *        not met even after increasing the depth to its ceiling.
     */
    void push(Real x) {
        if (!std::isfinite(static_cast<double>(x)) || x < a_ || x > b_) {
            throw std::out_of_range("IncrementalStack::push: node outside [a, b]");
        }
        const int k = order() + 1;
        if (k > q_max_) {
            throw std::length_error("IncrementalStack::push: q_max exceeded");
        }
        const Real y = (x - d_) / c_;
        const Real xmin = (k == 0) ? x : std::min(levels_.back().xmin, x);

        for (;;) {
            Column col = build_column_(k, y);
            Level lv;
            if (sum_column_(col, k, xmin, lv)) {
                if (k == 0) {
                    lv.sigma.mantissa = Real(0.5); lv.sigma.exponent = 0; lv.sigma.normalize();
                } else {
                    ScaledReal two_rho;
                    two_rho.mantissa = Real(2) * rho_[k]; two_rho.exponent = 0;
                    two_rho.normalize();
                    lv.sigma = levels_.back().sigma * two_rho;
                }
                lv.xmin = xmin;
                if (lv.error_estimate > max_error_estimate_) {
                    last_rejected_estimate_ = lv.error_estimate;
                    throw std::runtime_error(
                        "IncrementalStack::push: error indicator kappa*u*N = " +
                        fmt_(lv.error_estimate) + " exceeds the threshold " +
                        fmt_(max_error_estimate_) + " for the new prefix; the "
                        "interval may lie too far above the nodes, the insertion "
                        "order may be unfavorable (insert large nodes first), or "
                        "(q, c) is in the ill-conditioned regime.");
                }
                cols_.push_back(std::move(col));
                levels_.push_back(lv);
                return;
            }
            // Stopping rule not met within Nmax: deepen the cache and retry.
            const int new_nmax = std::max(nmax_ + 64, nmax_ + nmax_ / 2);
            if (new_nmax > nmax_ceiling_) {
                throw std::runtime_error(
                    "IncrementalStack::push: stopping rule not met within the "
                    "maximal cache depth " + std::to_string(nmax_) + ".");
            }
            extend_(new_nmax);
        }
    }

    /// Remove the most recently pushed node: O(1).
    void pop() {
        if (cols_.empty()) throw std::logic_error("IncrementalStack::pop: empty");
        cols_.pop_back(); levels_.pop_back();
    }

    int order() const { return static_cast<int>(cols_.size()) - 1; }
    int nmax() const { return nmax_; }
    int depth_extensions() const { return extensions_; }
    Real c() const { return c_; }
    Real d() const { return d_; }

    /// Normalized sum S of the current set (exp[x] = e^d I_0(c) c^{-q} S).
    ScaledReal normalized_scaled() const {
        const Level& lv = top_();
        ScaledReal t; t.mantissa = lv.T; t.exponent = 0; t.normalize();
        return lv.sigma * t;
    }

    /// exp[x_0, ..., x_q] of the current set, in mantissa/exponent form.
    ScaledReal value_scaled() const {
        return normalized_scaled() * prefactor_(order());
    }

    Real value() const { return value_scaled().to_real(); }

    /// Error indicator of the current top: kappa*u*N of its sum (what
    /// push() gates on; sufficient for ratios, where the prefactor
    /// cancels) plus the prefactor's rounding error u(|d| + c + q + 8)
    /// for the full value.
    long double error_estimate() const {
        return top_().error_estimate + prefactor_error_estimate_(order());
    }
    long double sum_error_estimate() const { return top_().error_estimate; }
    int last_N() const { return top_().N; }
    long double last_rejected_estimate() const { return last_rejected_estimate_; }

    void set_max_error_estimate(long double v) {
        if (!(v > 0.0L)) throw std::invalid_argument("set_max_error_estimate: must be > 0");
        max_error_estimate_ = v;
    }

private:
    /// Column k: entries W_k^(n) for n = start..nmax_, stored at v[n - start].
    struct Column {
        int start = 0;
        Real y = Real(0);
        std::vector<Real> v;
        Real at(int n) const { return (n < start) ? Real(0) : v[n - start]; }
    };
    struct Level {
        Real T = Real(0);          // sum of weighted terms
        ScaledReal sigma;          // sigma_k
        Real xmin = Real(0);       // smallest node of the prefix
        int N = 0;
        long double error_estimate = 0.0L;
    };
    static constexpr int W_ = 5;

    Real a_, b_, c_, d_, tol_;
    int q_max_, nmax_, nmax_ceiling_ = 0, extensions_ = 0;
    StackStopping stopping_;
    std::vector<Real> rho_;                 // rho_[n], n = 1..nmax_+1
    std::vector<long double> logR_;         // log R_n, n = 0..nmax_+1
    std::vector<Column> cols_;
    std::vector<Level> levels_;
    long double max_error_estimate_ = 1e-10L;
    long double last_rejected_estimate_ = 0.0L;

    void compute_rho_(int nmax) {
        const int Ns = chebyshev_bessel::BesselRatiosMiller<Real>::recommended_Nstart(c_, nmax + 1);
        rho_ = chebyshev_bessel::BesselRatiosMiller<Real>::compute_rho(c_, nmax + 1, Ns);
        logR_.assign(nmax + 2, 0.0L);
        for (int n = 1; n <= nmax + 1; ++n) {
            if (!(rho_[n] > Real(0))) {
                throw std::runtime_error("IncrementalStack: Bessel ratio underflow");
            }
            logR_[n] = logR_[n - 1] + std::log(static_cast<long double>(rho_[n]));
        }
    }

    /// One recurrence step for column k at degree n -> n+1.
    Real step_(const Column& col, const Column* parent, int k, int n,
               Real wn, Real wnm1) const {
        Real v = col.y * wn;
        if (parent) v += parent->at(n) / (Real(2) * rho_[k]);
        return Real(2) * rho_[n + 1] * v - rho_[n + 1] * rho_[n] * wnm1;
    }

    Column build_column_(int k, Real y) const {
        Column col;
        col.y = y;
        const Real* rho = rho_.data();
        if (k == 0) {
            col.start = 0;
            col.v.resize(nmax_ + 1);
            Real* w = col.v.data();
            w[0] = Real(2);                                 // W_0^(0)
            w[1] = Real(2) * rho[1] * y;                    // W_0^(1)
            for (int n = 1; n < nmax_; ++n) {
                w[n + 1] = Real(2) * rho[n + 1] * (y * w[n])
                         - rho[n + 1] * rho[n] * w[n - 1];
            }
        } else {
            // Column k stores n = k..nmax_; its parent stores n >= k-1,
            // so parent entry n sits at p[n - (k-1)].
            col.start = k;
            col.v.resize(nmax_ - k + 1);
            Real* w = col.v.data() - k;                     // w[n], n >= k
            const Real* p = cols_.back().v.data() - (k - 1);
            const Real s = Real(1) / (Real(2) * rho[k]);   // sigma_{k-1}/sigma_k
            w[k] = Real(1);                                 // W_k^(k)
            Real wm1 = Real(0);                             // W_k^(k-1)
            for (int n = k; n < nmax_; ++n) {
                const Real wn = w[n];
                w[n + 1] = Real(2) * rho[n + 1] * (y * wn + s * p[n])
                         - rho[n + 1] * rho[n] * wm1;
                wm1 = wn;
            }
        }
        return col;
    }

    /// Increase the cache depth and extend every stored column.
    void extend_(int new_nmax) {
        const int old = nmax_;
        compute_rho_(new_nmax);
        nmax_ = new_nmax;
        for (std::size_t k = 0; k < cols_.size(); ++k) {
            Column& col = cols_[k];
            const Column* parent = (k == 0) ? nullptr : &cols_[k - 1];
            col.v.reserve(nmax_ - col.start + 1);
            for (int n = old; n < nmax_; ++n) {
                const Real wn   = col.at(n);
                const Real wnm1 = col.at(n - 1);
                col.v.push_back(step_(col, parent, static_cast<int>(k), n, wn, wnm1));
            }
        }
        ++extensions_;
    }

    long double prefactor_error_estimate_(int q) const {
        return static_cast<long double>(std::numeric_limits<Real>::epsilon()) *
               (std::abs(static_cast<long double>(d_)) +
                static_cast<long double>(c_) + q + 8.0L);
    }

    static std::string fmt_(long double v) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.3Le", v);
        return buf;
    }

    const Level& top_() const {
        if (levels_.empty()) throw std::logic_error("IncrementalStack: empty");
        return levels_.back();
    }

    /// log B(n,k): bound on |T_n[y_0..y_k]| (manuscript Eq. 5.1); B = 1 for k = 0.
    static long double log_B_(int n, int k) {
        if (k == 0) return 0.0L;
        if (n < k) return -std::numeric_limits<long double>::infinity();
        return (k - 1) * 0.693147180559945309417232121458176568L
             + std::log(static_cast<long double>(n) / k)
             + std::lgamma(static_cast<long double>(n + k))
             - std::lgamma(static_cast<long double>(2 * k))
             - std::lgamma(static_cast<long double>(n - k + 1));
    }

    /**
     * Certified truncation index: smallest N >= max(k, 1) with
     *     2 sum_{n > N} R_n B(n,k) / S_low <= tol,
     *     S_low = c^k e^{xmin - d} / (k! I_0(c))   (mean-value lower bound),
     * i.e. the bound (5.3).  Returns -1 if no N <= nmax_ qualifies (the
     * tail beyond nmax_ is bounded geometrically).
     */
    int certified_N_(int k, Real xmin) const {
        const long double lc = std::log(static_cast<long double>(c_));
        const long double log_S_low =
            k * lc + static_cast<long double>(xmin - d_)
            - std::lgamma(static_cast<long double>(k + 1))
            - static_cast<long double>(chebyshev_bessel::log_I0_upper(c_));
        const long double log_tol = std::log(static_cast<long double>(tol_));
        // Tail beyond nmax_: terms t_n = R_n B(n,k) with ratio
        // t_{n+1}/t_n <= c (n+k) / (2 n (n-k+1)) =: r_n, decreasing in n.
        const int M = nmax_;
        long double log_tail;
        {
            const long double r = static_cast<long double>(c_) * (M + k)
                                / (2.0L * M * (M - k + 1));
            if (!(r < 1.0L)) return -1;
            log_tail = logR_[M] + log_B_(M, k) + std::log(r / (1.0L - r));
        }
        auto lse = [](long double p, long double q2) {
            if (p == -std::numeric_limits<long double>::infinity()) return q2;
            if (q2 == -std::numeric_limits<long double>::infinity()) return p;
            const long double m = std::max(p, q2);
            return m + std::log(std::exp(p - m) + std::exp(q2 - m));
        };
        // Walk N downward from M, accumulating the tail; remember the
        // smallest N for which the bound holds.
        int best = -1;
        for (int N = M; N >= std::max(k, 1); --N) {
            // log_tail currently = log sum_{n > N} t_n
            if (std::log(2.0L) + log_tail - log_S_low <= log_tol) best = N;
            else break;                 // bound only grows as N decreases
            log_tail = lse(log_tail, logR_[N] + log_B_(N, k));
        }
        return best;
    }

    /// Sum the column with the active stopping rule.  Returns false if the
    /// rule is not met within nmax_ (caller extends the depth).
    bool sum_column_(const Column& col, int k, Real xmin, Level& lv) const {
        int N_cert = -1;
        if (stopping_ == StackStopping::Certified) {
            N_cert = certified_N_(k, xmin);
            if (N_cert < 0) return false;
        }
        // S / sigma_k = delta_{k0} W^(0) + 2 sum_{n >= 1} W^(n).
        Real T = (k == 0) ? col.at(0) : Real(0);
        Real A = std::abs(T);
        int run = 0, n = std::max(1, k);
        bool converged = false;
        for (; n <= nmax_; ++n) {
            const Real t = Real(2) * col.at(n);
            T += t;
            A += std::abs(t);
            if (!std::isfinite(T) || !std::isfinite(A)) {
                throw std::overflow_error(
                    "IncrementalStack: non-finite term (catastrophic conditioning)");
            }
            if (stopping_ == StackStopping::Certified) {
                if (n >= N_cert) { converged = true; break; }
            } else {
                run = (std::abs(t) < tol_ * std::abs(T)) ? run + 1 : 0;
                if (n >= k + W_ && run >= W_) { converged = true; break; }
            }
        }
        if (!converged) return false;
        lv.T = T;
        lv.N = n;
        const long double kappa = (T != Real(0))
            ? static_cast<long double>(A) / std::abs(static_cast<long double>(T))
            : std::numeric_limits<long double>::infinity();
        lv.error_estimate = kappa *
            static_cast<long double>(std::numeric_limits<Real>::epsilon()) * n;
        return true;
    }

    /// e^d I_0(c) / c^q in scaled form (same construction as ExpDivDiff).
    ScaledReal prefactor_(int q) const {
        ScaledReal ed, i0, cq, one;
        const long double ln2 = 0.693147180559945309417232121458176568L;
        auto from_log = [&](long double lv) {
            ScaledReal r; const long double l2 = lv / ln2; const long double ip = std::floor(l2);
            r.mantissa = static_cast<Real>(std::exp2(l2 - ip)); r.exponent = static_cast<long>(ip);
            r.normalize(); return r; };
        const Real e_direct = std::exp(d_);
        if (std::isfinite(e_direct) && e_direct > std::numeric_limits<Real>::min()) {
            ed.mantissa = e_direct; ed.exponent = 0; ed.normalize();
        } else {
            ed = from_log(static_cast<long double>(d_));
        }
        const Real safe_c = std::log(std::numeric_limits<Real>::max()) - Real(10);
        if (c_ < safe_c) {
            i0.mantissa = chebyshev_bessel::bessel_I0(c_); i0.exponent = 0; i0.normalize();
        } else {
            i0 = from_log(static_cast<long double>(chebyshev_bessel::log_I0_value(c_)));
        }
        one.mantissa = Real(1); one.exponent = 0; one.normalize();
        cq = one;
        ScaledReal base; base.mantissa = c_; base.exponent = 0; base.normalize();
        for (int p = q; p > 0; p >>= 1) {
            if (p & 1) cq = cq * base;
            base = base * base;
        }
        return (ed * i0) / cq;
    }
};

} // namespace chebyshev

#endif // CHEBYSHEV_STACK_HPP
