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
// Each row starts at W_k^(k) = 1, and the entries are the series terms
// divided by 2 sigma_k, so they cannot underflow before they are
// negligible.  They can, however, exceed the range of Real even when the
// sum is perfectly conditioned (e.g. q = 319 equal nodes at b on
// [-4000, 0]: every term is positive, but the terms reach ~e^{900}).
// Each column therefore carries a binary exponent e_k (stored value
// = W 2^{-e_k}): a column inherits its parent's exponent, its built prefix
// is shifted down by 2^400 whenever an entry exceeds 2^400 (exact up to
// underflow of negligible entries), and it is renormalized upward after
// construction if its largest entry is tiny.  sigma_k and the column sums
// are kept in mantissa/exponent form.
//
// Costs.  The structural zeros W_k^(n) = 0, n < k, are not stored, so
// push() computes, stores and sums Nmax - k + 1 entries: Theta(Nmax - q)
// work and memory.  The sum is cached per level, so pop() is O(1).
//
// Stopping rules.  Certified (default): every push sums its column up to
// the cache depth Nmax and accepts it only if the rigorous bound (5.3) on
// the tail beyond Nmax, 2 sum_{n>Nmax} R_n B(n,k), is at most tol/4 times
// the computed |S|; otherwise the depth is increased (all stored columns
// are extended by the recurrence, a one-time Theta(q * Delta N) cost) and
// the push is retried.  The bound is evaluated in floating point (a
// conservative evaluation of a rigorous bound, not an interval-arithmetic
// certificate).  Window: Algorithm 1's heuristic -- stop at the first
// n >= k + 5 at which the last 5 terms are below tol |S|; NOT safe, since
// D_k^(n) can vanish for long runs of n (Chebyshev--Lobatto nodes).
//
// Accuracy safeguard.  Every prefix x_0..x_k is evaluated and gated when
// it is pushed: if its error indicator kappa*u*N exceeds the threshold
// (default 1e-10), push() throws and leaves the stack unchanged.  N is
// the effective number of terms (the last index at which a term exceeds
// u times the running sum).  The divided difference is symmetric, but
// the rounding errors of the recurrence depend on the processing order
// (manuscript, Sec. 4.3); gating every prefix detects unfavorable
// insertion orders, and initialize() inserts the initial set in
// nonincreasing order.
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

using StackStopping = StoppingRule;   // Certified (default) or Window

template<typename Real = double>
class IncrementalStack {
public:
    using ScaledReal = typename ExpDivDiff<Real>::ScaledReal;

    /**
     * @param a, b      Fixed interval (b > a) containing every node ever pushed.
     * @param q_max     Largest order that will be reached (sets Nmax when
     *                  nmax < 0).
     * @param tol       Relative tolerance of the stopping rule.
     * @param nmax      Initial cache depth Nmax; default (-1) chooses the
     *                  smallest depth at which the certified tail bound
     *                  holds for order q_max and a sum of typical size
     *                  (nodes near the midpoint).  The depth is increased
     *                  automatically when a push needs more.
     * @param stopping  StoppingRule::Certified (default) or ::Window.
     */
    IncrementalStack(Real a, Real b, int q_max, Real tol = Real(1e-14), int nmax = -1,
                     StackStopping stopping = StackStopping::Certified)
        : a_(a), b_(b), tol_(tol), q_max_(q_max), stopping_(stopping) {
        if (!(b > a) || !std::isfinite(static_cast<double>(a)) ||
            !std::isfinite(static_cast<double>(b))) {
            throw std::invalid_argument("IncrementalStack: need finite a < b");
        }
        if (q_max < 0) throw std::invalid_argument("IncrementalStack: q_max < 0");
        if (!(tol > Real(0))) throw std::invalid_argument("IncrementalStack: tol <= 0");
        c_ = b / 2 - a / 2;
        d_ = b / 2 + a / 2;
        c_ = detail::fit_halfwidth(a, b, d_, c_);   // mapped nodes in [-1, 1]
        const double L = std::log(1.0 / static_cast<double>(tol));
        const int guess = q_max + static_cast<int>(std::ceil(
                              1.3 * std::sqrt(2.0 * static_cast<double>(c_) * L))) + 30;
        if (nmax < 0) {
            if (stopping_ == StackStopping::Window) {
                nmax = guess;
            } else {
                // Smallest depth M with 2 sum_{n>M} R_n B(n,q_max) <=
                // (tol/4) S_typ, S_typ = c^q / (q! I_0(c)) (the value of S
                // for nodes at the midpoint).
                int M = std::max(q_max + W_ + 1, 8);
                compute_rho_(std::max(guess, M) + 64);
                const long double lc = std::log(static_cast<long double>(c_));
                const long double log_S_typ =
                    q_max * lc - std::lgamma(static_cast<long double>(q_max + 1))
                    - static_cast<long double>(chebyshev_bessel::log_I0_upper(c_));
                const long double log_goal =
                    std::log(static_cast<long double>(tol) / 4.0L) + log_S_typ;
                for (;; ++M) {
                    if (M + 1 >= static_cast<int>(rho_.size())) compute_rho_(2 * M + 64);
                    if (log_tail_beyond_(q_max, M) <= log_goal) break;
                }
                nmax = M;
            }
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

        ScaledReal sigma;
        if (k == 0) {
            sigma.mantissa = Real(0.5); sigma.exponent = 0; sigma.normalize();
        } else {
            ScaledReal two_rho;
            two_rho.mantissa = Real(2) * rho_[k]; two_rho.exponent = 0;
            two_rho.normalize();
            sigma = levels_.back().sigma * two_rho;
        }

        for (;;) {
            Column col;
            Level lv;
            lv.sigma = sigma;
            if (build_column_(k, y, col, lv)) {
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
        return lv.sigma * lv.T;
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
    /// Column k: entries W_k^(n) 2^{-e} for n = start..nmax_, at v[n - start].
    struct Column {
        int start = 0;
        long e = 0;
        Real y = Real(0);
        std::vector<Real> v;
        Real at(int n) const { return (n < start) ? Real(0) : v[n - start]; }
    };
    struct Level {
        ScaledReal T;              // S / sigma_k
        ScaledReal sigma;          // sigma_k
        Real xmin = Real(0);       // smallest node of the prefix
        int N = 0;                 // effective number of terms
        long double error_estimate = 0.0L;
    };
    static constexpr int W_ = 5;
    static constexpr int kRescaleBits = 400;
    static constexpr long double kLn2 = 0.693147180559945309417232121458176568L;

    Real a_, b_, c_, d_, tol_;
    int q_max_, nmax_ = 0, nmax_ceiling_ = 0, extensions_ = 0;
    StackStopping stopping_;
    std::vector<Real> rho_;                 // rho_[n], n = 1..size-1
    std::vector<long double> logR_;         // log R_n
    std::vector<long double> tail_cache_;   // log tail beyond nmax_, per k (NaN = unset)
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
        tail_cache_.assign(q_max_ + 1, std::numeric_limits<long double>::quiet_NaN());
    }

    /// log of 2 sum_{n>M} R_n B(n,k) (needs rho_ up to index M+1).  The
    /// terms decrease at least geometrically from n = M+1 on once their
    /// ratio bound is < 1; +inf if it is not.
    long double log_tail_beyond_(int k, int M) const {
        const long double r = detail::tail_ratio_bound(
            M + 1, k, static_cast<long double>(rho_[M + 1]));   // rho_{M+2} <= rho_{M+1}
        if (!(r < 1.0L)) return std::numeric_limits<long double>::infinity();
        return kLn2 + logR_[M + 1] + detail::log_B(M + 1, k) - std::log1p(-r);
    }

    long double tail_beyond_nmax_(int k) {
        long double& t = tail_cache_[k];
        if (std::isnan(t)) t = log_tail_beyond_(k, nmax_);
        return t;
    }

    static void scale_down_(Real* w, int len, int bits) {
        for (int i = 0; i < len; ++i) w[i] = std::ldexp(w[i], -bits);
    }

    /**
     * Build column k (entries n = k..nmax_; n = 0..nmax_ for k = 0) and
     * sum it with the active stopping rule.  Returns false if the rule is
     * not met within nmax_ (caller extends the depth).
     */
    bool build_column_(int k, Real y, Column& col, Level& lv) {
        col.y = y;
        const Real* rho = rho_.data();
        const Real big = std::ldexp(Real(1), kRescaleBits);
        const Real eps = std::numeric_limits<Real>::epsilon();
        // Running sums (in units of 2^{col.e}): T = S/sigma_k, A = sum |terms|.
        Real T = 0, A = 0;
        int n_eff = std::max(1, k);
        bool window_done = false;
        int run = 0, n_window = -1;
        auto add_term = [&](Real t, int n) {
            if (window_done) return;
            T += t; A += std::abs(t);
            if (std::abs(t) > eps * std::abs(T)) n_eff = n;
            if (stopping_ == StackStopping::Window && n >= 1) {
                run = (std::abs(t) < tol_ * std::abs(T)) ? run + 1 : 0;
                if (n >= k + W_ && run >= W_) { window_done = true; n_window = n; }
            }
        };

        if (k == 0) {
            col.start = 0; col.e = 0;
            col.v.resize(nmax_ + 1);
            Real* w = col.v.data();
            w[0] = Real(2);                                 // W_0^(0)
            w[1] = Real(2) * rho[1] * y;                    // W_0^(1)
            add_term(w[0], 0);
            add_term(Real(2) * w[1], 1);
            for (int n = 1; n < nmax_; ++n) {
                Real v = Real(2) * rho[n + 1] * (y * w[n]) - rho[n + 1] * rho[n] * w[n - 1];
                if (std::abs(v) > big) {
                    scale_down_(w, n + 1, kRescaleBits);
                    v = std::ldexp(v, -kRescaleBits);
                    T = std::ldexp(T, -kRescaleBits); A = std::ldexp(A, -kRescaleBits);
                    col.e += kRescaleBits;
                }
                w[n + 1] = v;
                add_term(Real(2) * v, n + 1);
            }
        } else {
            // Column k stores n = k..nmax_; its parent stores n >= k-1,
            // so parent entry n sits at p[n - (k-1)].
            const Column& par = cols_.back();
            col.start = k; col.e = par.e;
            col.v.resize(nmax_ - k + 1);
            Real* w0 = col.v.data();
            Real* w = w0 - k;                               // w[n], n >= k
            const Real* p = par.v.data() - (k - 1);
            const Real s0 = Real(1) / (Real(2) * rho[k]);  // sigma_{k-1}/sigma_k
            Real s = s0;                                    // times 2^{e_par - e_col}
            w[k] = std::ldexp(Real(1), static_cast<int>(std::max(-100000L, -col.e)));
            add_term(Real(2) * w[k], k);
            Real wm1 = Real(0);                             // W_k^(k-1)
            for (int n = k; n < nmax_; ++n) {
                const Real wn = w[n];
                Real v = Real(2) * rho[n + 1] * (y * wn + s * p[n])
                       - rho[n + 1] * rho[n] * wm1;
                if (std::abs(v) > big) {
                    scale_down_(w0, n - k + 1, kRescaleBits);
                    v = std::ldexp(v, -kRescaleBits);
                    wm1 = std::ldexp(wm1, -kRescaleBits);
                    T = std::ldexp(T, -kRescaleBits); A = std::ldexp(A, -kRescaleBits);
                    col.e += kRescaleBits;
                    s = std::ldexp(s0, static_cast<int>(std::max(-100000L, par.e - col.e)));
                }
                w[n + 1] = v;
                wm1 = w[n];
                add_term(Real(2) * v, n + 1);
            }
        }
        if (!std::isfinite(T) || !std::isfinite(A)) {
            throw std::overflow_error(
                "IncrementalStack: non-finite term (catastrophic conditioning)");
        }
        // Renormalize a column whose entries are all tiny (exact shift).
        if (A > Real(0) && A < Real(1) / big) {
            const int sh = -std::ilogb(A);
            for (Real& v : col.v) v = std::ldexp(v, sh);
            T = std::ldexp(T, sh); A = std::ldexp(A, sh);
            col.e -= sh;
        }

        int N;
        if (stopping_ == StackStopping::Window) {
            if (!window_done) return false;
            N = n_window;
        } else {
            N = n_eff;
        }
        lv.T.mantissa = T; lv.T.exponent = col.e; lv.T.normalize();
        lv.N = N;
        if (T == Real(0)) {
            lv.error_estimate = std::numeric_limits<long double>::infinity();
            return true;                    // gated by the caller
        }
        if (stopping_ == StackStopping::Certified) {
            const long double log_S = lv.T.log_abs() + lv.sigma.log_abs();
            if (!(tail_beyond_nmax_(k) <=
                  std::log(static_cast<long double>(tol_) / 4.0L) + log_S)) {
                return false;
            }
        }
        const long double kappa = static_cast<long double>(A) /
                                  std::abs(static_cast<long double>(T));
        lv.error_estimate = kappa * static_cast<long double>(eps) * N;
        return true;
    }

    /// Increase the cache depth and extend every stored column.
    void extend_(int new_nmax) {
        const int old = nmax_;
        compute_rho_(new_nmax);
        nmax_ = new_nmax;
        const Real big = std::ldexp(Real(1), kRescaleBits);
        for (std::size_t k = 0; k < cols_.size(); ++k) {
            Column& col = cols_[k];
            const Column* par = (k == 0) ? nullptr : &cols_[k - 1];
            col.v.reserve(nmax_ - col.start + 1);
            for (int n = old; n < nmax_; ++n) {
                Real v = col.y * col.at(n);
                if (par) {
                    v += std::ldexp(par->at(n), static_cast<int>(std::max(-100000L,
                             std::min(100000L, par->e - col.e)))) / (Real(2) * rho_[k]);
                }
                v = Real(2) * rho_[n + 1] * v - rho_[n + 1] * rho_[n] * col.at(n - 1);
                if (std::abs(v) > big) {
                    scale_down_(col.v.data(), static_cast<int>(col.v.size()), kRescaleBits);
                    v = std::ldexp(v, -kRescaleBits);
                    col.e += kRescaleBits;
                }
                col.v.push_back(v);
            }
        }
        ++extensions_;
    }

    long double prefactor_error_estimate_(int q) const {
        const long double ul = std::numeric_limits<long double>::epsilon();
        const long double ad = std::abs(static_cast<long double>(d_));
        return static_cast<long double>(std::numeric_limits<Real>::epsilon()) *
               (static_cast<long double>(c_) + q + 8.0L) +
               4.0L * ul * (1.0L + ad * ul);
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
        } else {                                // exact reduction, any |d|
            long double m, err; long k;
            detail::exp_split(static_cast<long double>(d_), &m, &k, &err);
            ed.mantissa = static_cast<Real>(m); ed.exponent = k; ed.normalize();
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
