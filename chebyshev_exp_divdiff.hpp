#ifndef CHEBYSHEV_EXP_DIVDIFF_TEMPLATED_HPP
#define CHEBYSHEV_EXP_DIVDIFF_TEMPLATED_HPP

// =============================================================================
// chebyshev_exp_divdiff.hpp
//
// Reference implementation of the Chebyshev-Bessel evaluator for
// exp[x_0, ..., x_q] from the manuscript.
//
// Portability requirement
// -----------------------
// This header uses std::cyl_bessel_i (C++17 mathematical special
// functions, ISO 29124).  Verified to compile with g++ 9+ on libstdc++,
// and with recent libc++ on FreeBSD/Linux distributions that ship the
// special-functions extension.  As of this writing, Apple's bundled
// libc++ on macOS does NOT yet ship std::cyl_bessel_i; users on that
// platform need either a recent libc++ from MacPorts/Homebrew or a
// drop-in Boost.Math replacement (`boost::math::cyl_bessel_i`).
//
// The same dependency exists in incremental_chebyshev_exp_divdiff.hpp
// and mccurdy.hpp; replacing the Bessel calls in those three files is
// sufficient to swap implementations.
// =============================================================================


// All Bessel infrastructure (numeric_traits, BesselRatiosMiller,
// bessel_I0, bessel_In, bessel_ratio) now lives in chebyshev_bessel.hpp
// so it can be shared with incremental_chebyshev_exp_divdiff.hpp without
// duplication.
#include "chebyshev_bessel.hpp"
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <functional>

namespace chebyshev {

/// Truncation rule of the Chebyshev--Bessel series.
///   Certified (default): stop at the first N for which a rigorous bound
///     on the omitted tail, 2 sum_{n>N} R_n B(n,q) (manuscript (5.3)),
///     is at most tol/4 times the computed |S_N|.  Since |S - S_N| <=
///     tail, this guarantees a relative truncation error below tol
///     (up to the rounding of the bound's own floating-point evaluation).
///   Window: Algorithm 1's heuristic (stop once W = 5 consecutive terms
///     are below tol |S|).  Faster, but NOT safe: D_q^(n) can vanish for
///     long runs of consecutive n (e.g. Chebyshev--Lobatto nodes), and the
///     rule then stops early with an arbitrarily large silent error.
enum class StoppingRule { Certified, Window };

namespace detail {
/// log B(n,k), B(n,k) = 2^{k-1} (n/k) (2k)_{n-k} / (n-k)! >= max |T_n[y_0..y_k]|
/// over y_i in [-1,1] (manuscript Eq. 5.1); B(n,0) = 1.
inline long double log_B(int n, int k) {
    if (k == 0) return 0.0L;
    if (n < k) return -std::numeric_limits<long double>::infinity();
    return (k - 1) * 0.693147180559945309417232121458176568L
         + std::log(static_cast<long double>(n) / k)
         + std::lgamma(static_cast<long double>(n + k))
         - std::lgamma(static_cast<long double>(2 * k))
         - std::lgamma(static_cast<long double>(n - k + 1));
}
/// Upper bound on the ratio t_{m+1}/t_m, t_m = R_m B(m,k), for all
/// m >= m0 >= max(k,1), given rho_up >= rho_{m0+1} (rho_n is decreasing):
/// rho_{m+1} (m+1)/m (m+k)/(m-k+1), decreasing in m.
inline long double tail_ratio_bound(int m0, int k, long double rho_up) {
    long double r = rho_up;
    if (k > 0) r *= (static_cast<long double>(m0 + 1) / m0) *
                   (static_cast<long double>(m0 + k) / (m0 - k + 1));
    return r;
}

/// e^x as m * 2^k with m in [0.5, 1), for an exactly given x (a node
/// or the midpoint d, which are exact Real values).  Reduction
/// x = k ln2 + r with k ln2 formed exactly: ln2 = Lh + Ll (Lh = ln2
/// rounded to long double), k Lh = p + e exactly via fma, x - p exact
/// (Sterbenz), so r = ((x - p) - e) - k Ll carries an absolute error
/// O(u (1 + |x| u)), u = long double epsilon; this works whether long
/// double has 64 (x87) or 53 (arm64) significant bits.  The naive route,
/// exp2(frac(x / ln2)), loses ~|x| u in the exponent's fractional part
/// (relative error ~2e-10 at x = 1e10 with x87, ~1e-6 with 53 bits).
/// *err_out receives a bound on the relative error of m * 2^k from the
/// reduction and exp (not counting the final rounding of m to Real).
/// Half-width c' >= c such that the mapped endpoints satisfy
/// fl(fl(a - d) / c') >= -1 and fl(fl(b - d) / c') <= 1 in Real
/// arithmetic, where d is the rounded midpoint.  Rounding of d (and of c)
/// can otherwise put mapped nodes outside [-1, 1] -- e.g. [2^51 + 0.5,
/// 2^51 + 2] gives d = 2^51 + 1, c = 0.75, y(b) = 4/3 -- which voids the
/// hypotheses of the tail bound (5.3)-(5.4) and of the expansion's
/// normalization.  Since rounding is monotone, every node in [a, b] then
/// maps into [-1, 1].  Using c' in place of (b - a)/2 is exact: the series
/// represents e^x on [d - c', d + c'], which contains [a, b].
template <typename Real>
inline Real fit_halfwidth(Real a, Real b, Real d, Real c) {
    c = std::max(c, std::max(b - d, d - a));
    for (int i = 0; i < 64; ++i) {
        if ((b - d) / c <= Real(1) && (a - d) / c >= Real(-1)) return c;
        c = std::nextafter(c, std::numeric_limits<Real>::infinity());
    }
    throw std::runtime_error("fit_halfwidth: cannot cover [a, b]");
}

inline void exp_split(long double x, long double* m_out, long* k_out,
                      long double* err_out) {
    // ln2 = A + B + C to ~1e-49 (A, B, C doubles, exact in long double).
    constexpr long double A = 0x1.62e42fefa39efp-1L;
    constexpr long double B = 0x1.abc9e3b39803fp-56L;
    constexpr long double C = 0x1.7b57a079a1934p-111L;
    constexpr long double Lh = A + B;            // ln2 rounded to long double
    constexpr long double Ll = ((A - Lh) + B) + C;
    constexpr long double u = std::numeric_limits<long double>::epsilon();
    const long double kf = std::nearbyint(x / Lh);
    if (!(std::abs(kf) < 0x1p52L)) {
        throw std::overflow_error("exp_split: argument too large");
    }
    const long double p = kf * Lh;
    const long double e = std::fma(kf, Lh, -p);  // k Lh - p, exact
    const long double r = ((x - p) - e) - kf * Ll;
    *err_out = 4.0L * u * (1.0L + std::abs(x) * u);
    int ex;
    const long double m = std::frexp(std::exp(r), &ex);
    *m_out = m;
    *k_out = static_cast<long>(kf) + ex;
}
} // namespace detail

// Re-export the shared Bessel infrastructure as chebyshev::<name>
// for backward compatibility with existing call sites.
using chebyshev_bessel::numeric_traits;
using chebyshev_bessel::BesselRatiosMiller;
using chebyshev_bessel::bessel_I0;
using chebyshev_bessel::bessel_In;
using chebyshev_bessel::bessel_ratio;
using chebyshev_bessel::log_I0_value;
using chebyshev_bessel::log_I0_upper;

/**
 * @brief Main class for exponential divided difference computation
 *
 * Implements Algorithm 1 of the manuscript (batch evaluation).
 */
template<typename Real = double>
class ExpDivDiff {
private:
    using traits = numeric_traits<Real>;
    
    Real a_, b_;           // Interval [a, b]
    Real c_, d_;           // Affine mapping parameters
    Real I0_c_;            // I_0(c), populated on first call to get_I0_c()
    bool bessel_I0_initialized_;
    Real tolerance_;       // Convergence tolerance
    int max_iterations_;   // Maximum Chebyshev orders

    // Cached R_n(c) shared across calls.  Empty until first
    // get_bessel_ratio() call, then lazily expanded.
    std::vector<Real> bessel_ratios_;

public:
    /**
     * @brief Constructor with fixed interval.
     *
     * @param a Lower bound of interval.
     * @param b Upper bound of interval (must be > a).
     * @param tolerance Convergence tolerance (default 1e-14).
     * @param max_iterations Maximum Chebyshev orders (default 10000).
     */
    ExpDivDiff(Real a, Real b, Real tolerance = Real(1e-14), int max_iterations = 10000)
        : a_(a), b_(b), tolerance_(tolerance), max_iterations_(max_iterations) {

        if (!std::isfinite(static_cast<long double>(a)) ||
            !std::isfinite(static_cast<long double>(b))) {
            throw std::invalid_argument(
                "ExpDivDiff: interval endpoints a and b must be finite");
        }
        if (!(b > a)) {
            throw std::invalid_argument(
                "ExpDivDiff: interval must satisfy b > a (also rejects "
                "NaN-tainted comparisons)");
        }
        if (!(tolerance > Real(0))) {
            throw std::invalid_argument(
                "ExpDivDiff: tolerance must be strictly positive");
        }
        if (max_iterations < 2) {
            throw std::invalid_argument(
                "ExpDivDiff: max_iterations must be at least 2 (need T_0 "
                "and at least one Chebyshev correction)");
        }

        // A3 R13: compute c_ and d_ in a way that does not overflow
        // when b - a does, even though both endpoints are finite.
        // For Real = double with b = 1e308, a = -1e308, the previous
        //   c_ = (b - a) / 2
        // overflowed to +inf while  b/2 - a/2  stays finite.  After
        // this rearrangement, verify both are finite and refuse the
        // constructor if not -- a non-finite c_ would silently
        // poison every subsequent call that touches the affine
        // mapping or the I_0(c) cache.
        c_ = b / traits::two() - a / traits::two();
        d_ = b / traits::two() + a / traits::two();
        if (std::isfinite(static_cast<long double>(c_)) &&
            std::isfinite(static_cast<long double>(d_))) {
            c_ = detail::fit_halfwidth(a, b, d_, c_);   // mapped nodes in [-1, 1]
        }
        if (!std::isfinite(static_cast<long double>(c_)) ||
            !std::isfinite(static_cast<long double>(d_))) {
            throw std::invalid_argument(
                "ExpDivDiff: interval endpoints are too extreme -- "
                "the derived half-width c = (b-a)/2 or midpoint "
                "d = (b+a)/2 is not representable as Real.  Use a "
                "narrower interval or a higher-precision Real.");
        }

        // A4 (R9 review): defer the linear I_0(c) and the Bessel-ratio
        // cache.  For huge constructor intervals (e.g. [-1e8, 1e8]),
        // eager seeding of bessel_ratios_ to size ~3c+50 would allocate
        // multi-GB up front, and modified_bessel_I0(c_) would overflow
        // even when the caller only ever calls the auto-tightening
        // evaluate() on tight node sets that never need them.
        // bessel_I0_initialized_ stays false; I0_c_ and bessel_ratios_
        // are filled on first access.  Not thread-safe -- the same as
        // std::vector growth.  Use distinct instances per thread, or
        // call get_I0_c() / get_bessel_ratio(0) under a lock once
        // before sharing.
        I0_c_                   = traits::zero();
        bessel_I0_initialized_  = false;
    }
    
    /**
     * @brief Return I_0(c_), computing it on first call.  Used by the
     *        few code paths that need the linear-scale value rather
     *        than its log (which can be computed via log_I0_value
     *        without overflow).
     */
    Real get_I0_c() {
        if (!bessel_I0_initialized_) {
            I0_c_ = bessel_I0(c_);
            bessel_I0_initialized_ = true;
        }
        return I0_c_;
    }
    
    /**
     * @brief A real number held as mantissa * 2^exponent.
     *
     * Used for quantities whose magnitude can leave the range of Real
     * even though the final divided difference (or a ratio of two of
     * them) is representable: the normalized sum S at high order
     * (S ~ c^q / q!), the full value exp[x_0..x_q] for q > ~170, and
     * the prefactor e^d I_0(c) / c^q.
     */
    struct ScaledReal {
        Real mantissa = Real(0);
        long exponent = 0;

        /// Value as Real; overflows to +-inf / underflows to 0 when out of range.
        Real to_real() const {
            if (mantissa == Real(0)) return mantissa;
            const long e = std::max(-100000L, std::min(100000L, exponent));
            return std::ldexp(mantissa, static_cast<int>(e));
        }
        /// Natural log of |value| (finite whenever mantissa != 0).
        long double log_abs() const {
            return std::log(std::abs(static_cast<long double>(mantissa))) +
                   static_cast<long double>(exponent) * kLn2;
        }
        void normalize() {
            if (mantissa != Real(0) && std::isfinite(mantissa)) {
                int ex;
                mantissa = std::frexp(mantissa, &ex);
                exponent += ex;
            }
        }
        friend ScaledReal operator*(const ScaledReal& u, const ScaledReal& v) {
            ScaledReal w;
            w.mantissa = u.mantissa * v.mantissa;
            w.exponent = u.exponent + v.exponent;
            w.normalize();
            return w;
        }
        friend ScaledReal operator/(const ScaledReal& u, const ScaledReal& v) {
            ScaledReal w;
            w.mantissa = u.mantissa / v.mantissa;
            w.exponent = u.exponent - v.exponent;
            w.normalize();
            return w;
        }
    };

    /**
     * @brief Evaluate exponential divided difference exp[x_0, ..., x_q]
     *
     * @param nodes Vector of nodes (must be within [a, b])
     * @param n_out Optional output: number of Chebyshev orders used
     * @return exp[x_0, ..., x_q]; +inf / 0 if the value itself lies
     *         outside the range of Real (use evaluate_scaled() then).
     */
    Real evaluate(const std::vector<Real>& nodes, int* n_out = nullptr) {
        return evaluate_scaled(nodes, n_out).to_real();
    }

    /**
     * @brief Evaluate exp[x_0, ..., x_q] in mantissa/exponent form.
     *
     * Strategy:
     *   1. All-equal nodes: closed form e^{x_0}/q!, in log space.
     *   2. Run the Chebyshev--Bessel sum on the constructor interval
     *      [a, b] and record kappa * u, where
     *          kappa = (sum_n |2 R_n D_q^(n)|) / |S|
     *      is the condition number of the truncated series and u the
     *      unit roundoff.  kappa * u * N (N = orders used) is an
     *      a-posteriori indicator of the relative rounding error.
     *   3. If kappa * u * N > max_error_estimate (default 1e-10),
     *      re-run on the tightest
     *      interval [min x_i, max x_i].  Anchoring the right endpoint
     *      at max x_i removes the e^{b - max x_i} cancellation factor
     *      that a loose fixed interval introduces.  (Revision note:
     *      an earlier version rebuilt on [mid - 1.5 s, mid + 1.5 s],
     *      which is not anchored at max x_i and frequently not
     *      narrower than [a, b], and then silently returned the
     *      ill-conditioned sum.)
     *   4. If still too large, retry on [min x - s w, max x], s = 1, 2
     *      (w = max x - min x; see evaluate_tight_), which removes the
     *      cancellation of order-dominated sets concentrated below their
     *      largest node.
     *   5. If the indicator is still too large on these intervals,
     *      the problem is intrinsically ill-conditioned for this
     *      representation in working precision; throw rather than
     *      return an inaccurate number.
     */
    ScaledReal evaluate_scaled(const std::vector<Real>& nodes, int* n_out = nullptr) {
        if (nodes.empty()) {
            throw std::invalid_argument("Node vector cannot be empty");
        }
        const int q = static_cast<int>(nodes.size()) - 1;
        validate_nodes_(nodes);

        Real node_min = nodes[0], node_max = nodes[0];
        for (int i = 1; i <= q; ++i) {
            node_min = std::min(node_min, nodes[i]);
            node_max = std::max(node_max, nodes[i]);
        }

        // All-equal closed form: exp[x, ..., x] = e^x / q!.  The formula
        // is exact; its evaluation is not, so the rounding of e^x (in
        // mantissa/exponent form) and of q! is accounted for and the
        // threshold enforced like every other branch.
        if (node_min == node_max) {
            if (n_out) *n_out = 0;
            long double err_exp = 0.0L, err_fact = 0.0L;
            const ScaledReal ex = exp_scaled_(nodes[0], &err_exp);
            const ScaledReal fq = factorial_scaled_(q, &err_fact);
            last_kappa_eps_ = 0.0L;
            last_error_estimate_ = err_exp + err_fact +
                2.0L * static_cast<long double>(std::numeric_limits<Real>::epsilon());
            if (last_error_estimate_ > max_error_estimate_) {
                throw_conditioning_(last_error_estimate_);
            }
            return ex / fq;
        }

        // Pre-screen: if [a, b] is more than 100x wider than the node
        // spread, go straight to the tight interval (the wide one would
        // need O(c) terms and be hopelessly ill-conditioned).
        const Real spread = node_max - node_min;
        if (c_ > Real(100) * spread) {
            return evaluate_tight_(nodes, node_min, node_max, n_out);
        }

        long double kappa_eps;
        int n_used = 0;
        const ScaledReal S = evaluate_core_(nodes, &kappa_eps, &n_used);
        if (n_out) *n_out = n_used;
        last_kappa_eps_ = kappa_eps;
        // Full value: add the rounding error of the prefactor.
        last_error_estimate_ = kappa_eps * core_n_eff_ + prefactor_error_estimate(q);
        if (last_error_estimate_ > max_error_estimate_) {
            const bool tight = !(node_min > a_ || node_max < b_);
            if (!tight || lowered_retry_) {
                return evaluate_tight_(nodes, node_min, node_max, n_out, tight ? 1 : 0);
            }
            throw_conditioning_(last_error_estimate_);
        }
        return S * scaling_factor_scaled_(q);
    }

    /**
     * @brief Estimated relative rounding error kappa * u of the most
     *        recent evaluate()/evaluate_scaled()/evaluate_normalized()
     *        call on this object (0 for the closed-form branches).
     */
    long double last_condition_estimate() const { return last_kappa_eps_; }

    /**
     * @brief Relative rounding error of the prefactor e^d I_0(c) / c^q,
     *        estimated as u (c + q + 8) + (error of e^d): I_0(c) ~ e^c and
     *        c^q amplify the rounding of c by ~c and q; d is an exact Real
     *        and e^d is formed by an exact argument reduction (exp_split),
     *        so a large common shift |d| costs only O(u^2 |d|).  Included
     *        in last_error_estimate() for full values (evaluate,
     *        evaluate_scaled); not for normalized sums or ratios, where
     *        the prefactor cancels.
     */
    long double prefactor_error_estimate(int q) const {
        const long double ul = std::numeric_limits<long double>::epsilon();
        const long double ad = std::abs(static_cast<long double>(d_));
        return static_cast<long double>(std::numeric_limits<Real>::epsilon()) *
               (static_cast<long double>(c_) + q + 8.0L) +
               4.0L * ul * (1.0L + ad * ul);
    }

    /**
     * @brief Rounding-error indicator of the most recent evaluation:
     *        kappa * u * N (N = number of Chebyshev orders used), plus
     *        prefactor_error_estimate(q) for full values.  The
     *        factor N accounts for round-off accumulated in the
     *        Chebyshev divided-difference recurrence; in the
     *        manuscript's experiments the observed relative error did
     *        not exceed ~2 kappa u N.
     */
    long double last_error_estimate() const { return last_error_estimate_; }

    /**
     * @brief Process nodes in nonincreasing order (default true).  The
     *        result is mathematically order-independent; the ordering
     *        only affects rounding (see evaluate_core_).
     */
    void set_sort_nodes(bool v) { sort_nodes_ = v; }

    /// Truncation rule (default StoppingRule::Certified).
    void set_stopping_rule(StoppingRule r) { stopping_ = r; }

    /**
     * @brief exp[x_0..x_q] on the constructor interval only: no closed
     *        form, no retry on another interval; throws if the indicator
     *        exceeds the threshold.  For callers that manage their own
     *        intervals (e.g. with cached objects).
     */
    ScaledReal evaluate_scaled_fixed(const std::vector<Real>& nodes, int* n_out = nullptr) {
        if (nodes.empty()) throw std::invalid_argument("Node vector cannot be empty");
        const int q = static_cast<int>(nodes.size()) - 1;
        long double kappa_eps;
        int n_used = 0;
        const ScaledReal S = evaluate_core_(nodes, &kappa_eps, &n_used);
        if (n_out) *n_out = n_used;
        last_kappa_eps_ = kappa_eps;
        last_error_estimate_ = kappa_eps * core_n_eff_ + prefactor_error_estimate(q);
        if (last_error_estimate_ > max_error_estimate_) throw_conditioning_(last_error_estimate_);
        return S * scaling_factor_scaled_(q);
    }

    /**
     * @brief After a refusal on [a, b] and on [min x, max x], also try
     *        [min x - s w, max x], s = 1, 2 (default true).  See
     *        evaluate_tight_.  Affects only evaluate()/evaluate_scaled().
     */
    void set_lowered_retry(bool v) { lowered_retry_ = v; }
    StoppingRule stopping_rule() const { return stopping_; }

    /**
     * @brief Threshold on kappa * u * N above which evaluate() rebuilds
     *        on [min x_i, max x_i] and, failing that, throws instead of
     *        returning an inaccurate value.  Default 1e-10.
     */
    void set_max_error_estimate(long double v) {
        if (!(v > 0.0L)) {
            throw std::invalid_argument("set_max_error_estimate: must be > 0");
        }
        max_error_estimate_ = v;
    }

    /**
     * @brief Evaluate normalized form: S = D_q^(0) + 2 sum_n R_n(c) D_q^(n)
     *
     * Full result = exp(d) * I_0(c) / c^q * S.  For q above ~150 the
     * normalized sum itself underflows in double (S ~ c^q / q!); use
     * evaluate_normalized_scaled() in that case.  Ratios of two
     * normalized sums on the same interval and of the same order
     * equal the ratio of the divided differences.
     *
     * @param nodes Vector of nodes (must be within [a, b])
     * @param n_out Optional output: number of Chebyshev orders used
     * @return Normalized sum S
     */
    Real evaluate_normalized(const std::vector<Real>& nodes, int* n_out = nullptr) {
        return evaluate_normalized_scaled(nodes, n_out).to_real();
    }

    /**
     * @brief Normalized sum S in mantissa/exponent form.  Throws when
     *        the error indicator kappa * u * N exceeds the threshold
     *        (set_max_error_estimate); unlike evaluate(), this entry point
     *        cannot change the interval, because S is defined relative
     *        to [a, b].
     */
    ScaledReal evaluate_normalized_scaled(const std::vector<Real>& nodes,
                                          int* n_out = nullptr) {
        long double kappa_eps;
        int n_used = 0;
        const ScaledReal S = evaluate_core_(nodes, &kappa_eps, &n_used);
        if (n_out) *n_out = n_used;
        last_kappa_eps_ = kappa_eps;
        last_error_estimate_ = kappa_eps * core_n_eff_;
        if (last_error_estimate_ > max_error_estimate_) {
            throw_conditioning_(last_error_estimate_);
        }
        return S;
    }

    /**
     * @brief Ratio exp[x_0..x_q] / exp[x'_0..x'_q] for two node sets of
     *        the same order on this interval.  The common factor
     *        e^d I_0(c) / c^q cancels, so only the normalized sums are
     *        needed and the ratio is representable even when both
     *        divided differences under- or overflow.
     */
    Real evaluate_ratio(const std::vector<Real>& nodes_num,
                        const std::vector<Real>& nodes_den) {
        if (nodes_num.size() != nodes_den.size()) {
            throw std::invalid_argument(
                "evaluate_ratio: node sets must have the same order");
        }
        // Diagnostics: the relative errors of the two sums add, so the
        // ratio's indicator is the sum of both (the prefactor cancels).
        const ScaledReal Sn = evaluate_normalized_scaled(nodes_num);
        const long double k1 = last_kappa_eps_, e1 = last_error_estimate_;
        const ScaledReal Sd = evaluate_normalized_scaled(nodes_den);
        last_kappa_eps_ = k1 + last_kappa_eps_;
        last_error_estimate_ = e1 + last_error_estimate_;
        if (last_error_estimate_ > max_error_estimate_) {
            throw_conditioning_(last_error_estimate_);
        }
        return (Sn / Sd).to_real();
    }

private:
    static constexpr long double kLn2 =
        0.693147180559945309417232121458176568L;

    // Exponent step used when rescaling rows of the Chebyshev table and
    // the running Bessel ratio.  2^400 leaves ample head-room below the
    // double overflow threshold (2^1024) for the O(1) growth per step.
    static constexpr int kRescaleBits = 400;

    long double last_kappa_eps_ = 0.0L;
    bool sort_nodes_ = true;
    int core_n_eff_ = 0;
    StoppingRule stopping_ = StoppingRule::Certified;
    long double last_error_estimate_ = 0.0L;
    long double max_error_estimate_ = 1e-10L;
    bool lowered_retry_ = true;

    void validate_nodes_(const std::vector<Real>& nodes) const {
        for (const Real& x : nodes) {
            if (!std::isfinite(static_cast<long double>(x)) ||
                x < a_ || x > b_) {
                throw std::out_of_range("Node must be finite and in [a, b]");
            }
        }
    }

    [[noreturn]] void throw_conditioning_(long double est) const {
        char buf[96];
        std::snprintf(buf, sizeof buf, "%.3Le exceeds the threshold %.3Le",
                      est, max_error_estimate_);
        throw std::runtime_error(
            std::string("ExpDivDiff: series conditioning failure -- error "
                        "estimate kappa*u*N = ") + buf +
            " on the tightest available interval.  The Chebyshev--Bessel "
            "series cancels too strongly for these nodes in the current "
            "precision (typically large q and c with q < c, or nodes "
            "concentrated far below their maximum); use higher precision "
            "or the MNP method.");
    }

    /// One evaluation on [lo, hi], no retries: throws if the indicator
    /// exceeds the threshold.
    ScaledReal evaluate_once_on_(const std::vector<Real>& nodes, Real lo, Real hi,
                                 int* n_out) {
        ExpDivDiff<Real> E(lo, hi, tolerance_, max_iterations_);
        E.max_error_estimate_ = max_error_estimate_;
        E.sort_nodes_ = sort_nodes_;
        E.stopping_ = stopping_;
        const int q = static_cast<int>(nodes.size()) - 1;
        long double kappa_eps;
        int n_used = 0;
        const ScaledReal S = E.evaluate_core_(nodes, &kappa_eps, &n_used);
        if (n_out) *n_out = n_used;
        last_kappa_eps_ = kappa_eps;
        last_error_estimate_ = kappa_eps * E.core_n_eff_ + E.prefactor_error_estimate(q);
        if (last_error_estimate_ > max_error_estimate_) {
            throw_conditioning_(last_error_estimate_);
        }
        return S * E.scaling_factor_scaled_(q);
    }

    /**
     * Retries after the indicator was too large on the constructor
     * interval: first the tight interval [min x, max x]; then, if
     * lowered_retry is on, [min x - s w, max x] for s = 1, 2
     * (w = max x - min x).  Lowering a moves the midpoint d to or below
     * the smallest node.  When the order dominates (q >~ c) the leading
     * terms of the series have the size of the sum for nodes at d, so a
     * set concentrated below d cancels by roughly e^{d - xi}
     * (exp[x] = e^xi / q!); with d <= min x this cancellation is absent.
     * (Manuscript, Sec. 4.2; it does not help when c >> q.)  Each value
     * is still accepted only if its own indicator is below the threshold.
     */
    ScaledReal evaluate_tight_(const std::vector<Real>& nodes,
                               Real node_min, Real node_max, int* n_out,
                               int first = 0) {
        const Real w = node_max - node_min;
        const int tries = lowered_retry_ ? 3 : 1;
        for (int s = first; s < tries; ++s) {
            try {
                return evaluate_once_on_(nodes, node_min - Real(s) * w, node_max, n_out);
            } catch (const std::runtime_error&) {
                if (s + 1 == tries) throw;
            }
        }
        throw std::logic_error("unreachable");
    }

    /// sign * exp(log_v) as a ScaledReal.
    static ScaledReal from_log_(long double log_v, Real sign) {
        const long double l2 = log_v / kLn2;
        const long double ip = std::floor(l2);
        ScaledReal r;
        r.mantissa = sign * static_cast<Real>(std::exp2(l2 - ip));
        r.exponent = static_cast<long>(ip);
        r.normalize();
        return r;
    }

    /// e^x for an exact Real x as a ScaledReal; *err = reduction error.
    static ScaledReal exp_scaled_(Real x, long double* err) {
        long double m; long k;
        detail::exp_split(static_cast<long double>(x), &m, &k, err);
        ScaledReal r;
        r.mantissa = static_cast<Real>(m);
        r.exponent = k;
        r.normalize();
        return r;
    }

    /// q! as a ScaledReal; *err = bound on its relative error.
    /// Compensated (double-word) product: hi + lo carries the exact
    /// product of hi and i via fma, so the relative error is
    /// O(u + q u^2) instead of O(q u) for a plain product (or
    /// O(u_ld log q!) for exp(lgamma(q + 1))).  O(q) multiplications.
    static ScaledReal factorial_scaled_(int q, long double* err) {
        const long double uR =
            static_cast<long double>(std::numeric_limits<Real>::epsilon());
        Real hi = Real(1), lo = Real(0);
        long ex = 0;
        for (int i = 2; i <= q; ++i) {
            const Real fi = static_cast<Real>(i);
            const Real p = hi * fi;
            const Real e = std::fma(hi, fi, -p);     // hi * i - p, exact
            lo = lo * fi + e;
            const Real s = p + lo;                   // renormalize hi + lo
            lo = lo - (s - p);
            hi = s;
            int k;
            hi = std::frexp(hi, &k);
            lo = std::ldexp(lo, -k);
            ex += k;
        }
        ScaledReal f;
        f.mantissa = hi + lo;
        f.exponent = ex;
        f.normalize();
        *err = 2.0L * uR + 4.0L * uR * uR * (q + 1);
        return f;
    }

    static ScaledReal from_real_(Real x) {
        ScaledReal r;
        r.mantissa = x;
        r.normalize();
        return r;
    }

    /// x^p for p >= 0 by binary powering in mantissa/exponent form.
    static ScaledReal pow_scaled_(Real x, int p) {
        ScaledReal result = from_real_(Real(1));
        ScaledReal base = from_real_(x);
        while (p > 0) {
            if (p & 1) result = result * base;
            base = base * base;
            p >>= 1;
        }
        return result;
    }

    /// e^d I_0(c) / c^q without overflow or underflow.
    ScaledReal scaling_factor_scaled_(int q) {
        // e^d: direct when representable (accurate to ~1 ulp), else in
        // log space.
        ScaledReal ed;
        const Real e_direct = traits::exp(d_);
        if (std::isfinite(e_direct) && e_direct > std::numeric_limits<Real>::min()) {
            ed = from_real_(e_direct);
        } else {
            long double err;
            ed = exp_scaled_(d_, &err);   // accurate for any |d| < 2^52 ln2
        }
        // I_0(c): direct below the overflow threshold, else asymptotic log.
        ScaledReal i0;
        const Real safe_c = std::log(std::numeric_limits<Real>::max()) - Real(10);
        if (c_ < safe_c) {
            i0 = from_real_(get_I0_c());
        } else {
            i0 = from_log_(static_cast<long double>(log_I0_value(c_)), Real(1));
        }
        return (ed * i0) / pow_scaled_(c_, q);
    }

    /// Running sum in mantissa/exponent form.
    struct ScaledSum {
        ScaledReal v;
        void add(Real t, long te) {
            if (t == Real(0)) return;
            if (v.mantissa == Real(0)) {
                v.mantissa = t; v.exponent = te; v.normalize(); return;
            }
            if (te > v.exponent) {
                const long sh = v.exponent - te;
                v.mantissa = (sh < -2000 ? Real(0)
                              : std::ldexp(v.mantissa, static_cast<int>(sh))) + t;
                v.exponent = te;
            } else {
                const long sh = te - v.exponent;
                if (sh >= -2000) v.mantissa += std::ldexp(t, static_cast<int>(sh));
            }
            v.normalize();
        }
    };

    /// |t| 2^te < tol |S| (tol = tolerance_ unless given), robust to
    /// exponent ranges.
    bool small_relative_(Real t, long te, const ScaledReal& S, Real tol = Real(-1)) const {
        if (tol < Real(0)) tol = tolerance_;
        if (t == Real(0)) return true;
        if (S.mantissa == Real(0)) return false;
        const long sh = te - S.exponent;
        if (sh < -2000) return true;
        if (sh > 2000) return false;
        return std::ldexp(traits::abs(t), static_cast<int>(sh)) <
               tol * traits::abs(S.mantissa);
    }

    static Real pow2_(long k) {
        if (k < -1070) return Real(0);
        if (k > 1000) k = 1000;
        return std::ldexp(Real(1), static_cast<int>(k));
    }

    /**
     * @brief Core of Algorithm 1: the normalized sum S on [a, b] plus
     *        kappa * u, without throwing on bad conditioning.
     *
     * Scaling (overflow/underflow control):
     *   - Each row k of the Chebyshev table (D_k^(n) for all n) carries
     *     its own binary exponent e_k: the stored mantissa M_k^(n)
     *     satisfies D_k^(n) = M_k^(n) 2^{e_k}.  The recurrence becomes
     *       M_k^(n) = 2 (y_k M_k^(n-1) + 2^{e_{k-1} - e_k} M_{k-1}^(n-1))
     *                 - M_k^(n-2),
     *     and a row is shifted down by 2^kRescaleBits (its three live
     *     layers together) whenever a new entry exceeds 2^kRescaleBits.
     *     The recurrence is linear and homogeneous in each row, so this
     *     is exact up to rounding.  Without it D_q^(q) = 2^{q-1} alone
     *     overflows double for q > 1024.
     *   - R_n(c) is accumulated as R_n = rho_1 ... rho_n with
     *     rho_k = I_k/I_{k-1} in (0,1) from the stable backward ratio
     *     recurrence, in mantissa/exponent form.  This avoids the
     *     underflow of R_n(c) ~ (c/2)^n/n! at small c and large n.
     *   - The running sum S and the absolute sum sum|terms| are kept
     *     in mantissa/exponent form.
     *
     * Stopping rule (StoppingRule::Certified, default): whenever the
     * current term is below tol |S_n| (a cheap trigger), evaluate the
     * rigorous tail bound
     *     |S - S_n| <= 2 sum_{m>n} R_m B(m,q) <= 2 t_{n+1} / (1 - r),
     *     t_m = R_m B(m,q),  r = bound on t_{m+1}/t_m for m > n,
     * in log space, and stop once it is <= (tol/4) |S_n|.  The terms
     * D_q^(n) themselves are never used to decide truncation, so runs of
     * vanishing coefficients (parity, Chebyshev--Lobatto nodes) cannot
     * cause early termination.  StoppingRule::Window keeps the earlier
     * heuristic (W = 5 consecutive small terms), which is unsafe for
     * such node sets.
     */
    ScaledReal evaluate_core_(const std::vector<Real>& nodes,
                              long double* kappa_eps_out, int* n_out) {
        if (nodes.empty()) {
            throw std::invalid_argument("Node vector cannot be empty");
        }
        const int q = static_cast<int>(nodes.size()) - 1;
        validate_nodes_(nodes);

        bool all_equal = true;
        for (int i = 1; i <= q; ++i) {
            if (nodes[i] != nodes[0]) { all_equal = false; break; }
        }
        if (all_equal) {
            // Normalized closed form, in log space:
            //   log S = c y_0 + q log c - log q! - log I_0(c).
            // Error: the four log terms are each rounded (relative
            // u), and exp turns absolute log error into relative error.
            if (n_out) *n_out = 0;
            const long double y0 = static_cast<long double>((nodes[0] - d_) / c_);
            const long double lc = std::log(static_cast<long double>(c_));
            const long double t1 = static_cast<long double>(c_) * y0;
            const long double t2 = q * lc;
            const long double t3 = std::lgamma(static_cast<long double>(q + 1));
            const long double t4 = static_cast<long double>(log_I0_value(c_));
            const long double uR =
                static_cast<long double>(std::numeric_limits<Real>::epsilon());
            if (kappa_eps_out) {
                *kappa_eps_out = 4.0L * uR *
                    (std::abs(t1) + std::abs(t2) + t3 + std::abs(t4) + 1.0L);
            }
            core_n_eff_ = 1;
            return from_log_(t1 + t2 - t3 - t4, Real(1));
        }

        Real ps_min = nodes[0], ps_max = nodes[0];
        for (int i = 1; i <= q; ++i) {
            ps_min = std::min(ps_min, nodes[i]);
            ps_max = std::max(ps_max, nodes[i]);
        }
        if (c_ > Real(100) * (ps_max - ps_min)) {
            throw std::runtime_error(
                "ExpDivDiff::evaluate_normalized: fixed interval "
                "[a, b] is more than 100x wider than the node spread; "
                "running the Chebyshev sum on this interval would "
                "require O(c) Bessel-ratio work and the result would "
                "be dominated by round-off.  Reconstruct on a tighter "
                "[a, b] or call evaluate() (which auto-tightens for "
                "full-value computation).");
        }

        // Processing order.  The divided difference is symmetric in the
        // nodes, but the rounding errors of the row recurrence are not:
        // a long leading run of nodes far below b (e.g. ascending order
        // on a wide interval) can produce large errors that the
        // final-row condition number kappa does not see.  In all our
        // experiments, processing the nodes in nonincreasing order kept
        // the observed error below ~kappa*u*N, so this is the default
        // (set_sort_nodes(false) keeps the caller's order).
        std::vector<Real> y(q + 1);
        for (int i = 0; i <= q; ++i) y[i] = (nodes[i] - d_) / c_;
        if (sort_nodes_) std::sort(y.begin(), y.end(), std::greater<Real>());

        // Mantissas of layers n-2, n-1, n and per-row exponents.
        std::vector<Real> M2(q + 1, Real(0)), M1(q + 1, Real(0)), Mc(q + 1, Real(0));
        std::vector<long> e(q + 1, 0);
        std::vector<Real> f(q + 1, Real(1));   // f[k] = 2^{e[k-1] - e[k]}
        M2[0] = traits::one();                  // T_0 = 1
        M1[0] = y[0];                           // T_1[y_0] = y_0
        if (q >= 1) M1[1] = traits::one();      // T_1[y_0, y_1] = 1

        const Real big = std::ldexp(Real(1), kRescaleBits);

        // Running Bessel ratio R_n = mR * 2^eR.
        Real mR = traits::one();
        long eR = 0;

        ScaledSum S, A;
        S.add(M2[q], e[q]);                    // n = 0 (R_0 = 1, weight 1)
        A.add(traits::abs(M2[q]), e[q]);
        mR *= get_bessel_rho(1);
        {
            const Real t1 = traits::two() * mR * M1[q];
            S.add(t1, eR + e[q]);
            A.add(traits::abs(t1), eR + e[q]);
        }

        // Certified rule: B(n,q) = mB 2^eB, tracked from n_cert0 = max(q,2)
        // (B(q,q) = 2^{q-1}; B(n,0) = 1; B(1,1) = 1, B(2,1) = 4).
        const int n_cert0 = std::max(q, 2);
        Real mB = Real(1);
        long eB = 0;
        if (q >= 2) { eB = q - 1; }
        else if (q == 1) { mB = Real(4); }
        // Effective number of terms: last n with |term| >= u |S_n|.
        const Real eps_rel = std::numeric_limits<Real>::epsilon();
        int n_eff = 1;

        constexpr int W = 5;
        Real last_t[W] = {};
        long last_e[W] = {};
        int n;
        bool converged = false;

        for (n = 2; n < max_iterations_; ++n) {
            mR *= get_bessel_rho(n);
            if (mR < Real(1) / big) {
                mR = std::ldexp(mR, kRescaleBits);
                eR -= kRescaleBits;
            }

            // Rows k > n vanish identically (T_n[y_0..y_k] = 0 for
            // n < k), so the sweep stops at min(n, q).  Row k starts at
            // n = k with D_k^(k) = 2 D_{k-1}^(k-1); give it the exponent
            // of row k-1 at that moment so the coupling factor f[k]
            // starts at 1 instead of 2^{e_{k-1}}.
            const int kmax = std::min(n, q);
            if (n <= q && n >= 1) {
                e[n] = e[n - 1];
                f[n] = traits::one();
                if (n < q) f[n + 1] = pow2_(e[n] - e[n + 1]);
            }
            for (int k = 0; k <= kmax; ++k) {
                Real v = y[k] * M1[k];
                if (k > 0) v += f[k] * M1[k - 1];
                v = traits::two() * v - M2[k];
                if (traits::abs(v) > big) {
                    v     = std::ldexp(v,     -kRescaleBits);
                    M1[k] = std::ldexp(M1[k], -kRescaleBits);
                    M2[k] = std::ldexp(M2[k], -kRescaleBits);
                    e[k] += kRescaleBits;
                    if (k > 0) f[k] = pow2_(e[k - 1] - e[k]);
                    if (k < q) f[k + 1] = pow2_(e[k] - e[k + 1]);
                }
                Mc[k] = v;
            }

            const Real t = traits::two() * mR * Mc[q];
            const long te = eR + e[q];
            S.add(t, te);
            A.add(traits::abs(t), te);

            for (int i = W - 1; i > 0; --i) {
                last_t[i] = last_t[i - 1];
                last_e[i] = last_e[i - 1];
            }
            last_t[0] = t;
            last_e[0] = te;

            if (!small_relative_(t, te, S.v, eps_rel)) n_eff = n;
            if (stopping_ == StoppingRule::Certified) {
                if (n >= n_cert0) {
                    if (n > n_cert0) {          // B(n,q) from B(n-1,q)
                        const int m = n - 1;
                        Real g = Real(1);
                        if (q > 0) g = (Real(m + 1) / Real(m)) * (Real(m + q) / Real(m - q + 1));
                        mB *= g;
                        if (mB > big) { mB = std::ldexp(mB, -kRescaleBits); eB += kRescaleBits; }
                    }
                    if (small_relative_(t, te, S.v) &&
                        certified_tail_ok_(q, n, mR, eR, mB, eB, S.v)) {
                        converged = true; break;
                    }
                }
            } else if (n >= q + W) {
                bool all_small = true;
                for (int i = 0; i < W; ++i) {
                    if (!small_relative_(last_t[i], last_e[i], S.v)) {
                        all_small = false;
                        break;
                    }
                }
                if (all_small) { converged = true; break; }
            }

            std::swap(M2, M1);
            std::swap(M1, Mc);
        }

        if (n_out) *n_out = n;
        if (!converged) {
            throw std::runtime_error(
                "ExpDivDiff::evaluate_normalized: stopping rule not satisfied "
                "within max_iterations_; increase max_iterations_ or loosen the "
                "tolerance; raising the interval half-width makes the Chebyshev "
                "expansion harder, not easier.");
        }

        // kappa = sum |terms| / |S|; kappa * u bounds the relative
        // round-off of the final summation and, empirically, tracks the
        // observed error of the whole evaluation (see the manuscript).
        long double kappa_eps = std::numeric_limits<long double>::infinity();
        if (S.v.mantissa != Real(0)) {
            const long double kappa =
                std::ldexp(static_cast<long double>(A.v.mantissa) /
                           std::abs(static_cast<long double>(S.v.mantissa)),
                           static_cast<int>(std::max(-100000L, std::min(100000L,
                               A.v.exponent - S.v.exponent))));
            kappa_eps = kappa *
                static_cast<long double>(std::numeric_limits<Real>::epsilon());
        }
        if (kappa_eps_out) *kappa_eps_out = kappa_eps;
        // N used in the indicator kappa*u*N: the effective number of terms
        // (later terms are below u |S| and change neither the sum nor its
        // rounding).  The certified rule may compute a few more orders.
        core_n_eff_ = (stopping_ == StoppingRule::Certified) ? std::max(n_eff, 1) : n;
        return S.v;
    }

    /// Certified truncation test after term n (R_n = mR 2^eR,
    /// B(n,q) = mB 2^eB):  2 sum_{m>n} R_m B(m,q) <= (tol/4) |S_n|, using
    /// sum_{m>n} t_m <= t_{n+1} / (1 - r), r >= t_{m+1}/t_m for m > n.
    bool certified_tail_ok_(int q, int n, Real mR, long eR, Real mB, long eB,
                            const ScaledReal& S) {
        if (S.mantissa == Real(0)) return false;
        const Real rho1 = get_bessel_rho(n + 1);
        const Real rho2 = get_bessel_rho(n + 2);
        const long double r = detail::tail_ratio_bound(n + 1, q, rho2);
        if (!(r < 1.0L)) return false;
        Real g = Real(1);                       // B(n+1,q) / B(n,q)
        if (q > 0) g = (Real(n + 1) / Real(n)) * (Real(n + q) / Real(n - q + 1));
        // tail = 2 R_{n+1} B(n+1,q) / (1 - r), as mantissa * 2^exp.
        const Real tail_m = Real(2) * mR * rho1 * mB * g / static_cast<Real>(1.0L - r);
        if (!std::isfinite(tail_m)) return false;
        const long sh = eR + eB - S.exponent;
        if (sh > 2000) return false;
        if (sh < -2000) return true;
        return std::ldexp(tail_m, static_cast<int>(sh)) <=
               (tolerance_ / Real(4)) * traits::abs(S.mantissa);
    }

public:
    /**
     * @brief rho_n(c) = I_n(c) / I_{n-1}(c), expanding the cache if necessary.
     */
    Real get_bessel_rho(int n) {
        if (n < 1) {
            throw std::out_of_range("get_bessel_rho: order n must be >= 1");
        }
        if (n >= static_cast<int>(bessel_rhos_.size())) {
            const int new_size = std::max(n + 100,
                static_cast<int>(2 * bessel_rhos_.size()));
            const int Nstart =
                BesselRatiosMiller<Real>::recommended_Nstart(c_, new_size - 1);
            bessel_rhos_ = BesselRatiosMiller<Real>::compute_rho(c_, new_size - 1, Nstart);
        }
        return bessel_rhos_[n];
    }

private:
    std::vector<Real> bessel_rhos_;

public:
    
    /**
     * @brief Get scaling factor: exp(d) * I_0(c) / c^q.
     *
     * Throws std::out_of_range for q < 0.  This is the linear-scale
     * factor that converts a normalized sum S to the full divided
     * difference; for large c and/or large d it can overflow even when
     * the final divided difference is finite.  Prefer
     * log_scaling_factor() if you intend to combine it with a
     * normalized sum in log space.
     */
    Real get_scaling_factor(int q) {
        if (q < 0) {
            throw std::out_of_range(
                "get_scaling_factor: q must be non-negative");
        }
        return traits::exp(d_) * get_I0_c() / traits::pow(c_, q);
    }

    /**
     * @brief log of the scaling factor: d + log I_0(c) - q log c.
     *
     * Useful when combining with a normalized sum in log space to
     * avoid overflow in the multiplicative step at large c, d, or q.
     */
    Real log_scaling_factor(int q) const {
        if (q < 0) {
            throw std::out_of_range(
                "log_scaling_factor: q must be non-negative");
        }
        const Real log_c = (q == 0) ? traits::zero() : traits::log(c_);
        return d_ + log_I0_value(c_) - Real(q) * log_c;
    }
    
    /**
     * @brief Get interval parameters
     */
    Real get_a() const { return a_; }
    Real get_b() const { return b_; }
    Real get_c() const { return c_; }
    Real get_d() const { return d_; }
    
public:
    /**
     * @brief Return R_n(c), expanding the cache if necessary.
     *
     * Exposed publicly so external tests can cross-check the
     * Miller-based ratios against an independent reference such as
     * std::cyl_bessel_i(n, c) / std::cyl_bessel_i(0, c).
     *
     * Throws std::out_of_range for n < 0.
     */
    Real get_bessel_ratio(int n) {
        if (n < 0) {
            throw std::out_of_range(
                "get_bessel_ratio: order n must be non-negative");
        }
        if (n >= static_cast<int>(bessel_ratios_.size())) {
            // Expand cache using efficient batch computation
            int new_size = n + 100;
            int Nstart = BesselRatiosMiller<Real>::recommended_Nstart(c_, new_size - 1);
            bessel_ratios_ = BesselRatiosMiller<Real>::compute_R(c_, new_size - 1, Nstart);
        }
        return bessel_ratios_[n];
    }

private:
};

} // namespace chebyshev

#endif // CHEBYSHEV_EXP_DIVDIFF_TEMPLATED_HPP
