#ifndef CHEBYSHEV_EXP_DIVDIFF_TEMPLATED_HPP
#define CHEBYSHEV_EXP_DIVDIFF_TEMPLATED_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <type_traits>

namespace chebyshev {

// Numeric traits for different types
template<typename Real>
struct numeric_traits {
    static Real abs(Real x) { return std::abs(x); }
    static Real sqrt(Real x) { return std::sqrt(x); }
    static Real exp(Real x) { return std::exp(x); }
    static Real pow(Real x, int n) { return std::pow(x, n); }
    static Real epsilon() { return std::numeric_limits<Real>::epsilon(); }
    static Real one() { return Real(1); }
    static Real two() { return Real(2); }
    static Real pi() { return Real(3.14159265358979323846264338327950288L); }
    static Real zero() { return Real(0); }
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
        if (c <= traits::zero())
            throw std::invalid_argument("c must be positive");
        if (Nstart <= nmax + 2)
            throw std::invalid_argument("Nstart too small");
        // Small-|c| safeguard:
        // For very small arguments, the Miller backward recurrence contains factors (2k/c) that
        // can lead to severe loss of accuracy (conditioning), even with overflow rescaling.
        // In this regime we instead build ratios via a continued-fraction evaluation of
        // r_n = I_{n+1}(c)/I_n(c), then accumulate R_{n+1} = R_n * r_n.
        const Real c_abs = traits::abs(c);
        if (c_abs < Real(1)) {
            std::vector<Real> R(nmax + 1);
            R[0] = traits::one();
            if (nmax == 0) return R;

            // Handle c=0 exactly
            if (c_abs == traits::zero()) {
                for (int n = 1; n <= nmax; ++n) R[n] = traits::zero();
                return R;
            }

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
        // We need to track ALL rescaling to preserve ratios correctly.
        //
        // The key insight: we need to apply the SAME total rescaling to ALL values,
        // not just rescale locally. We'll accumulate the total scale factor.
        const Real BIG = traits::sqrt(std::numeric_limits<Real>::max()) * Real(0.25);
        Real total_scale = traits::one();

        // Backward recurrence: I_{k-1} = I_{k+1} + (2k/c) * I_k
        for (int k = Nstart; k >= 1; --k) {
            I[k - 1] = I[k + 1] + (traits::two() * Real(k) / c) * I[k];

            // Dynamic rescaling to prevent overflow
            // When we rescale, we must rescale ALL values in the array
            const Real a0 = traits::abs(I[k - 1]);
            const Real a1 = traits::abs(I[k]);
            const Real a2 = traits::abs(I[k + 1]);
            Real amax = a0;
            if (a1 > amax) amax = a1;
            if (a2 > amax) amax = a2;

            if (amax > BIG && amax > traits::zero()) {
                const Real scale = BIG / amax;
                // Rescale ALL computed values so far
                for (int j = k - 1; j <= Nstart + 1; ++j) {
                    I[j] *= scale;
                }
                total_scale *= scale;
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
        // Use the same heuristic as the original bessel_In implementation:
        // Nstart ≈ max(nmax, 2c) + margin
        return std::max(nmax + 50, static_cast<int>(traits::two() * c) + 50);
    }
};

/**
 * @brief Computes modified Bessel function I_0(x) using Abramowitz & Stegun approximation
 *
 * Uses polynomial approximation for |x| < 3.75 and asymptotic expansion for |x| >= 3.75.
 * This matches the implementation in Numerical Recipes and provides high accuracy.
 *
 * @param x Argument (should be real and positive for physical applications)
 * @return I_0(x)
 */
template<typename Real>
inline Real bessel_I0(Real x) {
    using traits = numeric_traits<Real>;
    Real ax = traits::abs(x);
    
    if (ax < Real(3.75)) {
        // Polynomial approximation for |x| < 3.75
        Real y = (x / Real(3.75)) * (x / Real(3.75));
        return Real(1.0) + y * (Real(3.5156229) + y * (Real(3.0899424) +
               y * (Real(1.2067492) + y * (Real(0.2659732) +
               y * (Real(0.0360768) + y * Real(0.0045813))))));
    }
    
    // Asymptotic expansion for |x| >= 3.75
    Real y = Real(3.75) / ax;
    Real result = (traits::exp(ax) / traits::sqrt(ax)) *
                   (Real(0.39894228) + y * (Real(0.01328592) +
                   y * (Real(0.00225319) + y * (Real(-0.00157565) +
                   y * (Real(0.00916281) + y * (Real(-0.02057706) +
                   y * (Real(0.02635537) + y * (Real(-0.01647633) +
                   y * Real(0.00392377)))))))));
    return result;
}

/**
 * @brief Computes modified Bessel function I_n(x) using Miller's backward recurrence
 *
 * For x small relative to n, upward recurrence is unstable.
 * Use Miller's algorithm: start from high order and recurse downward,
 * then normalize using I_0.
 *
 * @param n Order (non-negative integer)
 * @param x Argument
 * @return I_n(x)
 */
template<typename Real>
inline Real bessel_In(int n, Real x) {
    using traits = numeric_traits<Real>;
    
    if (n < 0) throw std::invalid_argument("n must be non-negative");
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
    
    // Miller's backward recurrence for moderate x
    // Start from large m >> n and recurse downward
    int m = std::max(n + 50, static_cast<int>(Real(2) * x) + 50);
    
    // Backward recurrence: I_k = (2*(k+1)/x) * I_{k+1} + I_{k+2}
    Real I_next = Real(0);      // I_{m+1} ~ 0
    Real I_curr = traits::one();      // I_m ~ arbitrary
    Real I_n_unnorm = Real(0);
    
    for (int k = m; k >= 0; --k) {
        Real I_prev = (traits::two() * (k + 1) / x) * I_curr + I_next;
        
        if (k == n) {
            I_n_unnorm = I_prev;
        }
        
        I_next = I_curr;
        I_curr = I_prev;
        
        // Prevent overflow
        if (traits::abs(I_curr) > Real(1e100)) {
            I_curr /= Real(1e100);
            I_next /= Real(1e100);
            if (k <= n) I_n_unnorm /= Real(1e100);
        }
    }
    
    // I_curr is now unnormalized I_0
    // Normalize: I_n = I_n_unnorm * (I_0_true / I_0_unnorm)
    Real I0_true = bessel_I0(x);
    return I_n_unnorm * (I0_true / I_curr);
}

/**
 * @brief Computes the ratio R_n(c) = I_n(c) / I_0(c)
 *
 * This normalized form avoids exponential growth and is numerically stable.
 *
 * @param n Order
 * @param c Argument
 * @param I0_c Precomputed I_0(c) (optional, computed if negative)
 * @return R_n(c)
 */
template<typename Real>
inline Real bessel_ratio(int n, Real c, Real I0_c = Real(-1)) {
    if (I0_c < Real(0)) {
        I0_c = bessel_I0(c);
    }
    return bessel_In(n, c) / I0_c;
}

/**
 * @brief Main class for exponential divided difference computation
 *
 * Implements Algorithm 1 from the paper with incremental update capabilities.
 */
template<typename Real = double>
class ExpDivDiff {
private:
    using traits = numeric_traits<Real>;
    
    Real a_, b_;           // Interval [a, b]
    Real c_, d_;           // Affine mapping parameters
    Real I0_c_;            // I_0(c)
    Real tolerance_;       // Convergence tolerance
    int max_iterations_;     // Maximum Chebyshev orders
    
    // Cache for incremental updates
    std::vector<Real> nodes_;           // Current nodes in [a, b]
    std::vector<Real> y_;               // Mapped nodes in [-1, 1]
    std::vector<Real> bessel_ratios_;   // Cached R_n(c)
    std::vector<Real> D_curr_;          // Current Chebyshev divided differences
    std::vector<Real> D_prev_;          // Previous layer
    std::vector<Real> D_prev2_;         // Two layers back
    int last_n_;                          // Last Chebyshev order computed
    Real last_sum_;                     // Last accumulated sum
    
    bool cache_valid_;                    // Whether cache is valid
    
public:
    /**
     * @brief Constructor with fixed interval
     *
     * @param a Lower bound of interval
     * @param b Upper bound of interval (must be > a)
     * @param tolerance Convergence tolerance (default 1e-14)
     * @param max_iterations Maximum Chebyshev orders (default 10000)
     */
    ExpDivDiff(Real a, Real b, Real tolerance = Real(1e-14), int max_iterations = 10000)
        : a_(a), b_(b), tolerance_(tolerance), max_iterations_(max_iterations),
          cache_valid_(false), last_n_(1), last_sum_(Real(0)) {
        
        if (b <= a) {
            throw std::invalid_argument("Interval must satisfy b > a");
        }
        
        c_ = (b - a) / traits::two();
        d_ = (b + a) / traits::two();
        I0_c_ = bessel_I0(c_);
        
        // Precompute Bessel ratios using efficient batch computation
        // For large nmax but small c, we don't need all ratios
        int nmax = 99;  // Initial cache size
        int effective_max = std::min(nmax, std::max(100, static_cast<int>(Real(3) * c_) + 50));
        int Nstart = BesselRatiosMiller<Real>::recommended_Nstart(c_, effective_max);
        std::vector<Real> computed_ratios = BesselRatiosMiller<Real>::compute_R(c_, effective_max, Nstart);
        
        // Resize to 100 and fill
        bessel_ratios_.resize(100, traits::zero());
        for (int n = 0; n <= effective_max && n < 100; ++n) {
            bessel_ratios_[n] = computed_ratios[n];
        }
    }
    
    /**
     * @brief Evaluate exponential divided difference exp[x_0, ..., x_q]
     *
     * @param nodes Vector of nodes (must be within [a, b])
     * @param n_out Optional output: number of Chebyshev orders used
     * @return exp[x_0, ..., x_q]
     */
    Real evaluate(const std::vector<Real>& nodes, int* n_out = nullptr) {
        // Compute using normalized evaluation and scaling factor
        const int q = nodes.size() - 1;
        Real S = evaluate_normalized(nodes, n_out);
        return get_scaling_factor(q) * S;
    }
    
    /**
     * @brief Evaluate normalized form: S = sum R_n(c) D_n^q
     *
     * This avoids overflow by computing the normalized sum without the scaling factor.
     * Full result = exp(d) * I_0(c) / c^q * S
     *
     * @param nodes Vector of nodes (must be within [a, b])
     * @param n_out Optional output: number of Chebyshev orders used
     * @return Normalized sum S
     */
    Real evaluate_normalized(const std::vector<Real>& nodes, int* n_out = nullptr) {
        if (nodes.empty()) {
            throw std::invalid_argument("Node vector cannot be empty");
        }
        
        const int q = nodes.size() - 1;
        
        // Check if all nodes are the same
        Real min_node = *std::min_element(nodes.begin(), nodes.end());
        Real max_node = *std::max_element(nodes.begin(), nodes.end());
        
        if (traits::abs(max_node - min_node) < Real(1e-15)) {
            // Degenerate case: exp[x, x, ..., x] = exp(x) / q!
            Real factorial = traits::one();
            for (int i = 1; i <= q; ++i) factorial *= Real(i);
            Real full_result = traits::exp(nodes[0]) / factorial;
            
            // Return normalized: S = full_result * c^q / (exp(d) * I_0(c))
            return full_result * traits::pow(c_, q) / (traits::exp(d_) * I0_c_);
        }
        
        // Map nodes to [-1, 1]
        std::vector<Real> y(q + 1);
        for (int i = 0; i <= q; ++i) {
            if (nodes[i] < a_ || nodes[i] > b_) {
                throw std::out_of_range("Node outside interval [a, b]");
            }
            y[i] = (nodes[i] - d_) / c_;
        }
        
        // Initialize divided difference arrays
        std::vector<Real> D_prev2(q + 1, Real(0));
        std::vector<Real> D_prev(q + 1, Real(0));
        std::vector<Real> D_curr(q + 1, Real(0));
        
        // n=0: T_0 = 1
        D_prev2[0] = traits::one();
        
        // n=1: T_1 = y
        D_prev[0] = y[0];
        if (q >= 1) {
            D_prev[1] = traits::one();
        }
        
        // Initialize sum
        Real S = D_prev2[q] + traits::two() * get_bessel_ratio(1) * D_prev[q];
        
        // Array to track last 5 terms for convergence check
        Real last_5_terms[5] = {Real(0), Real(0), Real(0), Real(0), Real(0)};
        
        // Main iteration
        int n;
        bool converged = false;
        
        for (n = 2; n < max_iterations_; ++n) {
            Real Rn = get_bessel_ratio(n);
            
            // Chebyshev divided difference recurrence
            D_curr[0] = traits::two() * (y[0] * D_prev[0]) - D_prev2[0];
            
            for (int k = 1; k <= q; ++k) {
                D_curr[k] = traits::two() * (y[k] * D_prev[k] + D_prev[k-1]) - D_prev2[k];
            }
            
            // Update sum
            Real term = traits::two() * Rn * D_curr[q];
            S += term;
            
            // Track last 5 terms for convergence check
            if (n >= 2) {
                for (int i = 4; i > 0; --i) {
                    last_5_terms[i] = last_5_terms[i-1];
                }
                last_5_terms[0] = traits::abs(term);
                
                if (n >= q + 5) {
                    bool all_small = true;
                    for (int i = 0; i < 5; ++i) {
                        if (last_5_terms[i] >= tolerance_ * traits::abs(S)) {
                            all_small = false;
                            break;
                        }
                    }
                    if (all_small) {
                        converged = true;
                        break;
                    }
                }
            }
            
            std::swap(D_prev2, D_prev);
            std::swap(D_prev, D_curr);
        }
        
        if (n_out) *n_out = n;
        
        return S;
    }
    
    /**
     * @brief Get scaling factor: exp(d) * I_0(c) / c^q
     */
    Real get_scaling_factor(int q) const {
        return traits::exp(d_) * I0_c_ / traits::pow(c_, q);
    }
    
    /**
     * @brief Get interval parameters
     */
    Real get_a() const { return a_; }
    Real get_b() const { return b_; }
    Real get_c() const { return c_; }
    Real get_d() const { return d_; }
    
private:
    /**
     * @brief Get Bessel ratio R_n(c), expanding cache if needed
     */
    Real get_bessel_ratio(int n) {
        if (n >= static_cast<int>(bessel_ratios_.size())) {
            // Expand cache using efficient batch computation
            int new_size = n + 100;
            int Nstart = BesselRatiosMiller<Real>::recommended_Nstart(c_, new_size - 1);
            bessel_ratios_ = BesselRatiosMiller<Real>::compute_R(c_, new_size - 1, Nstart);
        }
        return bessel_ratios_[n];
    }
};

} // namespace chebyshev

#endif // CHEBYSHEV_EXP_DIVDIFF_TEMPLATED_HPP
