#ifndef CHEBYSHEV_BESSEL_HPP
#define CHEBYSHEV_BESSEL_HPP

// =============================================================================
// chebyshev_bessel.hpp
//
// Shared Bessel infrastructure for the Chebyshev exp-divdiff
// implementations.  Factored out of chebyshev_exp_divdiff.hpp so that
// the batch and incremental headers use one authoritative copy of
// the numeric_traits, the I_0 wrapper, the Miller-backward-recurrence
// I_n routine, and the BesselRatiosMiller class.
//
// Portability requirement
// -----------------------
// This header uses std::cyl_bessel_i (C++17 mathematical special
// functions, ISO 29124).  Verified with g++ 9+ on libstdc++, and with
// recent libc++ on Linux.  As of this writing, Apple's bundled libc++
// on macOS does NOT ship std::cyl_bessel_i; users on that platform
// need either a recent libc++ from MacPorts/Homebrew or a drop-in
// boost::math::cyl_bessel_i replacement.
// =============================================================================

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <type_traits>

// Portable cyl_bessel_i: use the C++17 special-math function where the
// standard library ships it (libstdc++), and Boost.Math otherwise
// (Apple libc++).  All call sites in this package go through
// chebyshev_bessel::cyl_bessel_i so that the dispatch lives in one place.
#if defined(__STDCPP_MATH_SPEC_FUNCS__) || defined(__GLIBCXX__)
#define CHEBYSHEV_BESSEL_HAVE_STD_CYL_BESSEL_I 1
#else
#include <boost/math/special_functions/bessel.hpp>
#endif

namespace chebyshev_bessel {

template<typename Real>
inline Real cyl_bessel_i(Real nu, Real x) {
#ifdef CHEBYSHEV_BESSEL_HAVE_STD_CYL_BESSEL_I
    return std::cyl_bessel_i(nu, x);
#else
    return boost::math::cyl_bessel_i(nu, x);
#endif
}

// Numeric traits — the union of what either evaluator needs.
template<typename Real>
struct numeric_traits {
    static Real abs(const Real& x)  { return std::abs(x); }
    static Real sqrt(const Real& x) { return std::sqrt(x); }
    static Real exp(const Real& x)  { return std::exp(x); }
    static Real log(const Real& x)  { return std::log(x); }
    static Real pow(const Real& x, int n) { return std::pow(x, n); }
    static Real max(const Real& a, const Real& b) { return std::max(a, b); }
    static Real epsilon() { return std::numeric_limits<Real>::epsilon(); }
    static constexpr Real zero() { return Real(0); }
    static constexpr Real one()  { return Real(1); }
    static constexpr Real two()  { return Real(2); }
    static Real pi() { return Real(3.14159265358979323846264338327950288L); }
};

/**
 * @brief Efficient batch computation of Bessel ratios R_n(c) = I_n(c)/I_0(c)
 *
 * Uses Miller's backward recurrence to compute all ratios R_0, R_1, ..., R_nmax
 * in a single pass. This is much more efficient than computing each ratio individually.
 *
 * Performance: O(Nstart) for batch vs O(nmax × Nstart) for individual computation
 * Speedup: ~50-100× for typical use cases
 */
template<typename Real>
class BesselRatiosMiller {
public:
    using traits = numeric_traits<Real>;

private:
    // Continued-fraction evaluation of r_n = I_{n+1}(c) / I_n(c) using modified Lentz's method.
    // This is particularly robust for small |c| where Miller's backward recurrence can be ill-conditioned.
    static Real ratio_I_np1_over_I_n_cf(const Real& c, int n) {
        // r_n = c / ( 2(n+1) + c^2 / ( 2(n+2) + c^2 / ( 2(n+3) + ... )))
        const Real x = c;
        const Real x2 = x * x;

        // Lentz parameters
        const Real eps  = traits::epsilon() * Real(16);
        const Real tiny = std::numeric_limits<Real>::min() * Real(16);

        Real f = traits::two() * Real(n + 1); // b0
        if (traits::abs(f) < tiny) f = (f < traits::zero() ? -tiny : tiny);
        Real C = f;
        Real D = traits::zero();

        // Iterate CF
        for (int k = 1; k <= 10000; ++k) {
            const Real b = traits::two() * Real(n + 1 + k); // b_k
            const Real a = x2;                               // a_k

            // D = 1 / (b + a*D)
            Real denomD = b + a * D;
            if (traits::abs(denomD) < tiny) denomD = (denomD < traits::zero() ? -tiny : tiny);
            D = traits::one() / denomD;

            // C = b + a/C
            Real denomC = C;
            if (traits::abs(denomC) < tiny) denomC = (denomC < traits::zero() ? -tiny : tiny);
            C = b + a / denomC;
            if (traits::abs(C) < tiny) C = (C < traits::zero() ? -tiny : tiny);

            const Real delta = C * D;
            f *= delta;

            if (traits::abs(delta - traits::one()) <= eps) break;
        }

        return x / f;
    }

public:
    
    /**
     * @brief Compute R_n(c) = I_n(c)/I_0(c) for n=0..nmax in one pass
     *
     * @param c Argument (must be positive)
     * @param nmax Maximum order to compute
     * @param Nstart Starting point for backward recurrence (must satisfy Nstart >> max(nmax, c))
     * @return Vector of ratios R[0], R[1], ..., R[nmax]
     */
    static std::vector<Real> compute_R(const Real& c, int nmax, int Nstart) {
        // A2 R12: reject NaN / +-inf c.  `c < 0` is silently false for
        // NaN, so without an explicit isfinite() check NaN would slip
        // through to the recurrence.
        if (!std::isfinite(static_cast<long double>(c))) {
            throw std::invalid_argument(
                "compute_R: c must be finite");
        }
        if (c < traits::zero())
            throw std::invalid_argument("c must be non-negative");
        if (nmax < 0)
            throw std::invalid_argument("compute_R: nmax must be non-negative");
        if (Nstart <= nmax + 2)
            throw std::invalid_argument("Nstart too small");

        const Real c_abs = traits::abs(c);

        // c == 0 closed form: I_0(0) = 1, I_n(0) = 0 for n >= 1, so
        // the ratios are exactly {1, 0, 0, ...}.
        if (c_abs == traits::zero()) {
            std::vector<Real> R(nmax + 1, traits::zero());
            R[0] = traits::one();
            return R;
        }

        // Small-|c| safeguard:
        // For very small arguments, the Miller backward recurrence contains factors (2k/c) that
        // can lead to severe loss of accuracy (conditioning), even with overflow rescaling.
        // In this regime we instead build ratios via a continued-fraction evaluation of
        // r_n = I_{n+1}(c)/I_n(c), then accumulate R_{n+1} = R_n * r_n.
        if (c_abs < Real(1)) {
            std::vector<Real> R(nmax + 1);
            R[0] = traits::one();
            if (nmax == 0) return R;

            for (int n = 0; n < nmax; ++n) {
                const Real r = ratio_I_np1_over_I_n_cf(c, n);
                R[n + 1] = R[n] * r;
            }
            return R;
        }


        // Temporary unnormalized solution
        std::vector<Real> I(Nstart + 2);

        // Arbitrary normalization at the top
        I[Nstart + 1] = traits::zero();
        I[Nstart]     = traits::one();

        // Miller backward recurrence can overflow even though the final ratios are well-scaled.
        // When the unnormalized values get close to overflow, rescale
        // the entire array by the same factor.  This is safe because
        // the ratios R_n = I_n / I_0 are invariant under common
        // rescaling.  An earlier version of this code accumulated the
        // rescaling factor into `total_scale`, but it was never read
        // -- ratios needed no compensation -- so the variable has been
        // removed.
        const Real BIG = traits::sqrt(std::numeric_limits<Real>::max()) * Real(0.25);

        // Backward recurrence: I_{k-1} = I_{k+1} + (2k/c) * I_k
        for (int k = Nstart; k >= 1; --k) {
            I[k - 1] = I[k + 1] + (traits::two() * Real(k) / c) * I[k];

            // Dynamic rescaling to prevent overflow.  When we rescale,
            // we must rescale ALL values in the array so the ratios
            // remain consistent.
            const Real a0 = traits::abs(I[k - 1]);
            const Real a1 = traits::abs(I[k]);
            const Real a2 = traits::abs(I[k + 1]);
            Real amax = a0;
            if (a1 > amax) amax = a1;
            if (a2 > amax) amax = a2;

            if (amax > BIG && amax > traits::zero()) {
                const Real scale = BIG / amax;
                for (int j = k - 1; j <= Nstart + 1; ++j) {
                    I[j] *= scale;
                }
            }
        }

        // Normalize by I[0] to get ratios R_n = I_n / I_0
        if (I[0] == traits::zero())
            throw std::runtime_error("Miller recurrence underflow: I_0 computed as zero");

        const Real invI0 = traits::one() / I[0];
        std::vector<Real> R(nmax + 1);
        for (int n = 0; n <= nmax; ++n)
            R[n] = I[n] * invI0;

        return R;
    }

    /**
     * @brief Successive Bessel ratios rho_n(c) = I_n(c) / I_{n-1}(c),
     *        n = 1..nmax (rho[0] is set to 1 and unused).
     *
     * Computed by the backward continued-fraction recurrence
     *     rho_k = 1 / (2k/c + rho_{k+1}),   rho_{Nstart+1} = 0,
     * which is the three-term Bessel recurrence written for ratios
     * (Miller's algorithm without the unnormalized values).  Every
     * rho_k lies in (0, 1) and the map rho_{k+1} -> rho_k contracts
     * (derivative -rho_k^2), so the recurrence is stable and can
     * neither overflow nor underflow.  The products
     *     R_n(c) = rho_1 rho_2 ... rho_n
     * reproduce compute_R() but can be accumulated by the caller in
     * mantissa/exponent form, which avoids the underflow of R_n(c)
     * itself (R_n(c) ~ (c/2)^n / n! for n >> c) that limits the
     * unscaled ratios at small c and high order.
     */
    static std::vector<Real> compute_rho(const Real& c, int nmax, int Nstart) {
        if (!std::isfinite(static_cast<long double>(c))) {
            throw std::invalid_argument("compute_rho: c must be finite");
        }
        if (c < traits::zero())
            throw std::invalid_argument("compute_rho: c must be non-negative");
        if (nmax < 0)
            throw std::invalid_argument("compute_rho: nmax must be non-negative");
        if (Nstart <= nmax + 2)
            throw std::invalid_argument("compute_rho: Nstart too small");

        std::vector<Real> rho(nmax + 1, traits::zero());
        rho[0] = traits::one();
        if (c == traits::zero()) return rho;   // I_n(0) = 0 for n >= 1

        Real r = traits::zero();               // rho_{Nstart+1}
        for (int k = Nstart; k >= 1; --k) {
            r = traits::one() / (traits::two() * Real(k) / c + r);
            if (k <= nmax) rho[k] = r;
        }
        return rho;
    }

    /**
     * @brief Heuristic for choosing Nstart
     *
     * Safe for both double and long double.
     * The starting point must be large enough that I_Nstart ≈ 0 for numerical purposes,
     * but not so large that backward recurrence causes overflow.
     *
     * @param c Argument
     * @param nmax Maximum order needed
     * @return Recommended starting point for Miller's recurrence
     */
    static int recommended_Nstart(const Real& c, int nmax) {
        // A2/B1 R12: reject non-finite c (otherwise the comparisons
        // below silently succeed for NaN), and compute the 2c+50 in
        // long double so we can detect int overflow before casting.
        // B1 R13: also reject negative c, consistent with compute_R
        // and bessel_ratio.
        if (!std::isfinite(static_cast<long double>(c))) {
            throw std::invalid_argument(
                "recommended_Nstart: argument c must be finite");
        }
        if (c < traits::zero()) {
            throw std::invalid_argument(
                "recommended_Nstart: argument c must be non-negative");
        }
        if (nmax < 0) {
            throw std::invalid_argument(
                "recommended_Nstart: nmax must be non-negative");
        }
        const long double margin = 50.0L;
        const long double n_from_c = 2.0L *
            static_cast<long double>(c) + margin;
        const long double n_from_nmax = static_cast<long double>(nmax) + margin;
        const long double n_required = std::max(n_from_c, n_from_nmax);
        if (n_required >
            static_cast<long double>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "recommended_Nstart: required Miller starting index "
                "exceeds the int range; the interval half-width c is "
                "too large for this routine.");
        }
        return static_cast<int>(n_required);
    }
};

/**
 * @brief Computes the modified Bessel function I_0(x).
 *
 * Wraps std::cyl_bessel_i(0, x) (C++17), which delivers full
 * working-precision accuracy. The earlier Abramowitz--Stegun polynomial
 * approximation used here was limited to ~10^-7 relative accuracy and
 * was the dominant error source for unnormalized evaluations at large c.
 *
 * For non-floating-point Real types, the user should specialize this
 * template to a precision-matched Bessel routine.
 *
 * @param x Argument (real, finite)
 * @return I_0(x)
 */
template<typename Real>
inline Real bessel_I0(Real x) {
    using traits = numeric_traits<Real>;
    if (!std::isfinite(static_cast<long double>(x))) {
        throw std::invalid_argument(
            "bessel_I0: argument must be finite");
    }
    return static_cast<Real>(cyl_bessel_i(0.0L,
        static_cast<long double>(traits::abs(x))));
}

/**
 * @brief Modified Bessel function I_n(x) by Miller's backward recurrence.
 *
 * For x small relative to n, upward recurrence is unstable.  Miller's
 * algorithm starts from a high-order seed and recurses downward, then
 * normalizes against I_0.
 *
 * Range note (moderate values only): this routine returns I_n(x) in
 * its linear, unscaled form.  For x larger than ~700 (double) or
 * ~11000 (long double), I_n(x) itself overflows even when its ratio
 * to I_0(x) -- the quantity actually used by the Chebyshev
 * evaluator -- is fine.  The evaluator therefore does not call this
 * function in its main path; the BesselRatiosMiller class computes
 * R_n(c) = I_n(c)/I_0(c) directly and stays bounded by 1.  Use
 * bessel_In only in moderate-x ranges and check for overflow at
 * the call site, or use BesselRatiosMiller::compute_R if you need
 * many ratios at large c.
 *
 * @param n Order (non-negative integer)
 * @param x Argument
 * @return I_n(x) in linear scale; may overflow for x >~ 700 (double).
 */
template<typename Real>
inline Real bessel_In(int n, Real x) {
    using traits = numeric_traits<Real>;
    
    if (n < 0) throw std::invalid_argument("n must be non-negative");
    if (!std::isfinite(static_cast<long double>(x))) {
        throw std::invalid_argument(
            "bessel_In: argument x must be finite");
    }
    if (n == 0) return bessel_I0(x);
    
    x = traits::abs(x);
    
    if (x < Real(1e-10)) {
        // For very small x, use series: I_n(x) ~ (x/2)^n / n!
        if (n > 100) return Real(0);  // Effectively zero
        Real result = traits::one();
        Real x_half = x / traits::two();
        for (int k = 1; k <= n; ++k) {
            result *= x_half / Real(k);
        }
        return result;
    }
    
    // Miller's backward recurrence.  We use Bessel's three-term identity
    //     I_{k-1}(x) = (2k/x) * I_k(x) + I_{k+1}(x)
    // starting from an arbitrary seed at high order and stepping down
    // toward k = 0.  In code we maintain a sliding window:
    //     I_curr -> currently the I_k value just computed (or seed),
    //     I_next -> currently the I_{k+1} value,
    // and on each iteration compute I_prev = I_{k-1} from those.
    // We seed at order m where m >> max(n, 2x); at that scale
    // I_{m+1}(x) is small compared to I_m(x), so taking I_next = 0,
    // I_curr = 1 (an arbitrary nonzero seed) is a good approximation.
    // The full unnormalized sequence is recovered by exact downward
    // recursion; one final normalization step replaces I_0_unnorm with
    // the true I_0(x).
    // A1 R13: compute the Miller seed index in long double, with both
    // an int-range check and a practical-work cap.  The previous
    // `static_cast<int>(Real(2) * x) + 50` had two problems for huge
    // finite x: (1) the cast was implementation-defined / UB if 2x
    // exceeds INT_MAX (UBSAN flags this for x = 1e20), and (2) even
    // when it didn't overflow, a value like 2x = 1e9 would request
    // a billion-iteration loop with no upfront diagnostic.  Refuse
    // either case at the top of the moderate-range routine.
    constexpr long double kMaxStandaloneMillerOrders = 200000.0L;
    const long double m_required_ld =
        std::max(static_cast<long double>(n) + 50.0L,
                 2.0L * static_cast<long double>(x) + 50.0L);
    if (m_required_ld > kMaxStandaloneMillerOrders) {
        throw std::invalid_argument(
            "bessel_In: argument x is too large for the standalone "
            "Miller helper (required Miller start exceeds the 2e5 "
            "practical work cap).  Use the Bessel-ratio routines "
            "(BesselRatiosMiller::compute_R or bessel_ratio) for "
            "values at large x where only the ratio I_n/I_0 is "
            "meaningful, or move to a higher-precision library.");
    }
    int m = static_cast<int>(m_required_ld);

    Real I_next = Real(0);          // approximates I_{m+1}(x)
    Real I_curr = traits::one();    // arbitrary seed for I_m(x)
    Real I_n_unnorm = Real(0);

    // Loop index k tracks the order we are *about to produce* via the
    // recurrence (i.e., we compute I_{k-1} in this iteration).  At the
    // bottom of the loop, I_curr becomes I_{k-1}.  When the loop
    // finishes (k == 0 after the last decrement), I_curr is the
    // unnormalized I_{-1+1} = I_0(x).
    for (int k = m; k >= 0; --k) {
        Real I_prev = (traits::two() * (k + 1) / x) * I_curr + I_next;

        if (k == n) {
            // I_curr was I_{n+1} on the previous iteration; we want
            // I_n, which is the value of I_prev at *this* iteration
            // after stepping down.  Equivalently, save I_curr after
            // the assignments below.
            I_n_unnorm = I_prev;
        }

        I_next = I_curr;
        I_curr = I_prev;

        // Prevent overflow during the descent (the unnormalized values
        // grow as we step down).  We can rescale freely because the
        // final value depends only on the ratio I_n_unnorm / I_curr.
        if (traits::abs(I_curr) > Real(1e100)) {
            I_curr /= Real(1e100);
            I_next /= Real(1e100);
            if (k <= n) I_n_unnorm /= Real(1e100);
        }
    }

    // After the loop, I_curr holds the unnormalized I_0(x).  Normalize
    // by dividing by it and multiplying by the true I_0(x).
    Real I0_true = bessel_I0(x);
    return I_n_unnorm * (I0_true / I_curr);
}

/**
 * @brief Modified Bessel-ratio R_n(c) = I_n(c) / I_0(c).
 *
 * Forming I_n(c) and I_0(c) separately overflows in linear scale
 * when c is much past log(max(Real)) (about 700 for double), even
 * though the ratio itself is bounded by 1 for n >= 0.  Route the
 * computation through BesselRatiosMiller, which produces all ratios
 * R_0, R_1, ..., R_n in one pass without ever forming the
 * unnormalized I_n's at full magnitude.
 *
 * For n = 0 the answer is exactly 1.
 *
 * @param n  Order (must be non-negative).
 * @param c  Argument (must be non-negative; c == 0 gives R_0 = 1,
 *           R_n = 0 for n >= 1).
 * @return   R_n(c), accurate to a few ulps for all c.
 */
template<typename Real>
inline Real bessel_ratio(int n, Real c, Real /*I0_c_ignored*/ = Real(-1)) {
    if (n < 0) {
        throw std::out_of_range("bessel_ratio: n must be non-negative");
    }
    if (!std::isfinite(static_cast<long double>(c))) {
        throw std::invalid_argument(
            "bessel_ratio: argument c must be finite");
    }
    if (n == 0) return Real(1);
    // A2 R13: practical work cap on the public helper.  The Miller
    // recurrence underlying this routine costs O(2c) iterations and
    // O(2c) memory.  At c = 1e7 (Nstart ~ 2e7) that is ~160 MB and a
    // multi-second loop -- not what a single-shot helper call
    // implies.  Reject above 100 000 standalone Miller orders.
    // BesselRatiosMiller::compute_R itself remains uncapped so that
    // the in-class lazy caches in ExpDivDiff/IncrementalExpDivDiff
    // (which have their own higher-level guards) can use it freely.
    constexpr long double kMaxStandaloneMillerOrders = 100000.0L;
    const long double c_ld = static_cast<long double>(c);
    if (2.0L * c_ld + 50.0L > kMaxStandaloneMillerOrders) {
        throw std::invalid_argument(
            "bessel_ratio: argument c is too large for the public "
            "helper (the Miller recurrence would require more than "
            "1e5 starting orders).  For wide intervals, use "
            "chebyshev::ExpDivDiff::evaluate() (which tight-falls-"
            "back automatically and never materializes a wide-c "
            "Miller cache for the full-value path).");
    }
    const int Nstart = BesselRatiosMiller<Real>::recommended_Nstart(c, n);
    const std::vector<Real> R =
        BesselRatiosMiller<Real>::compute_R(c, n, Nstart);
    return R[n];
}

namespace detail {
/**
 * @brief Large-argument asymptotic series of I_0, summed to convergence:
 *     I_0(c) ~ e^c / sqrt(2 pi c) * sum_k a_k / c^k,
 *     a_0 = 1,  a_k = a_{k-1} (2k-1)^2 / (8k)      (DLMF 10.40.1, nu = 0),
 * i.e. 1 + 1/(8c) + 9/(128 c^2) + 225/(3072 c^3) + 11025/(98304 c^4) + ...
 * All terms are positive for real c > 0.  Terms are added (in long
 * double) until they fall below 1e-30 relative, which for the c >~ 700
 * where this is used takes a handful of terms; *rem_out receives the
 * first omitted term as a relative remainder estimate.
 *
 * Revision note: the earlier four-term version used the coefficient
 * 75/3072 instead of 225/3072 for the c^-3 term (relative error
 * ~1.4e-10 at c = 700), which propagated silently into every
 * divided difference evaluated through the large-c prefactor.
 */
inline long double log_I0_asymptotic(long double c, long double* rem_out) {
    long double sum = 1.0L, term = 1.0L;
    for (int k = 1; k < 200; ++k) {
        const long double next = term * (2.0L * k - 1.0L) * (2.0L * k - 1.0L)
                                 / (8.0L * k * c);
        if (next >= term) break;            // series starts to diverge
        term = next;
        if (term < 1e-30L * sum) { break; }
        sum += term;
    }
    if (rem_out) *rem_out = term;
    const long double two_pi = 6.283185307179586476925286766559005768L;
    return c - 0.5L * std::log(two_pi * c) + std::log(sum);
}
} // namespace detail

/**
 * @brief Accurate log(I_0(c)).
 *
 * For c small enough that I_0(c) fits in Real, std::log of the direct
 * value; otherwise the full large-argument asymptotic series
 * (detail::log_I0_asymptotic), whose truncation error is far below
 * working precision for c >~ 700.
 *
 * Use log_I0_upper() instead when the value will feed into a
 * one-sided certificate bound.
 */
template<typename Real>
inline Real log_I0_value(const Real& c) {
    using std::log;
    // I_0(c) <= e^c, so below log(max(Real)) - 10 the direct value fits.
    const Real safe_c = log(std::numeric_limits<Real>::max()) - Real(10);
    if (c < safe_c) {
        return log(bessel_I0(c));
    }
    return static_cast<Real>(detail::log_I0_asymptotic(
        static_cast<long double>(c), nullptr));
}

/**
 * @brief Conservative floating-point upper estimate of log(I_0(c)).
 *
 * log_I0_value(c) plus a pad covering the rounding of the evaluation
 * (16 ulps of the result) and, in the asymptotic regime, twice the
 * first omitted term of the series.  A conservative floating-point
 * estimate, not an interval-arithmetic certificate.
 */
template<typename Real>
inline Real log_I0_upper(const Real& c) {
    const Real safe_c = std::log(std::numeric_limits<Real>::max()) - Real(10);
    const Real rounding_pad = [&](Real v) {
        return Real(16) * std::numeric_limits<Real>::epsilon() * std::abs(v);
    }(log_I0_value(c));
    if (c < safe_c) {
        return log_I0_value(c) + rounding_pad;
    }
    long double rem = 0.0L;
    const long double v = detail::log_I0_asymptotic(static_cast<long double>(c), &rem);
    return static_cast<Real>(v + 2.0L * rem) + rounding_pad;
}

/// Deprecated alias kept for source compatibility; new code should
/// pick log_I0_value (accurate) or log_I0_upper (bound) deliberately.
template<typename Real>
[[deprecated("Use log_I0_value for accurate values or log_I0_upper for bounds")]]
inline Real log_I0_safe(const Real& c) {
    return log_I0_upper(c);
}

} // namespace chebyshev_bessel

#endif // CHEBYSHEV_BESSEL_HPP
