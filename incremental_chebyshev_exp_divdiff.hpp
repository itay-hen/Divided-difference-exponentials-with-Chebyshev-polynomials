#ifndef INCREMENTAL_EXP_DIVDIFF_HPP
#define INCREMENTAL_EXP_DIVDIFF_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace chebyshev_incremental {

/**
 * @brief Type traits for numeric operations
 */
template<typename T>
struct numeric_traits {
    static T abs(const T& x) { return std::abs(x); }
    static T sqrt(const T& x) { return std::sqrt(x); }
    static T exp(const T& x) { return std::exp(x); }
    static T pow(const T& x, int n) { return std::pow(x, n); }
    static T max(const T& a, const T& b) { return std::max(a, b); }
    
    static constexpr T zero() { return T(0); }
    static constexpr T one() { return T(1); }
    static constexpr T two() { return T(2); }
    
    static T epsilon() { return std::numeric_limits<T>::epsilon(); }
    static T pi() { return T(3.14159265358979323846264338327950288L); }
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
template<typename T>
class BesselRatiosMiller {
public:
    using traits = numeric_traits<T>;

private:
    // Continued-fraction evaluation of r_n = I_{n+1}(c) / I_n(c) using modified Lentz's method.
    // This is particularly robust for small |c| where Miller's backward recurrence can be ill-conditioned.
    static T ratio_I_np1_over_I_n_cf(const T& c, int n) {
        // r_n = c / ( 2(n+1) + c^2 / ( 2(n+2) + c^2 / ( 2(n+3) + ... )))
        const T x = c;
        const T x2 = x * x;

        // Lentz parameters
        const T eps  = traits::epsilon() * T(16);
        const T tiny = std::numeric_limits<T>::min() * T(16);

        T f = traits::two() * T(n + 1); // b0
        if (traits::abs(f) < tiny) f = (f < traits::zero() ? -tiny : tiny);
        T C = f;
        T D = traits::zero();

        // Iterate CF
        for (int k = 1; k <= 10000; ++k) {
            const T b = traits::two() * T(n + 1 + k); // b_k
            const T a = x2;                            // a_k

            // D = 1 / (b + a*D)
            T denomD = b + a * D;
            if (traits::abs(denomD) < tiny) denomD = (denomD < traits::zero() ? -tiny : tiny);
            D = traits::one() / denomD;

            // C = b + a/C
            T denomC = C;
            if (traits::abs(denomC) < tiny) denomC = (denomC < traits::zero() ? -tiny : tiny);
            C = b + a / denomC;
            if (traits::abs(C) < tiny) C = (C < traits::zero() ? -tiny : tiny);

            const T delta = C * D;
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
    static std::vector<T> compute_R(const T& c, int nmax, int Nstart) {
        if (c <= traits::zero())
            throw std::invalid_argument("c must be positive");
        if (Nstart <= nmax + 2)
            throw std::invalid_argument("Nstart too small");
        // Small-|c| safeguard:
        // For very small arguments, the Miller backward recurrence contains factors (2k/c) that
        // can lead to severe loss of accuracy (conditioning), even with overflow rescaling.
        // In this regime we instead build ratios via a continued-fraction evaluation of
        // r_n = I_{n+1}(c)/I_n(c), then accumulate R_{n+1} = R_n * r_n.
        const T c_abs = traits::abs(c);
        if (c_abs < T(1)) {
            std::vector<T> R(nmax + 1);
            R[0] = traits::one();
            if (nmax == 0) return R;

            // Handle c=0 exactly
            if (c_abs == traits::zero()) {
                for (int n = 1; n <= nmax; ++n) R[n] = traits::zero();
                return R;
            }

            for (int n = 0; n < nmax; ++n) {
                const T r = ratio_I_np1_over_I_n_cf(c, n);
                R[n + 1] = R[n] * r;
            }
            return R;
        }


        // Temporary unnormalized solution
        std::vector<T> I(Nstart + 2);

        // Arbitrary normalization at the top
        I[Nstart + 1] = traits::zero();
        I[Nstart]     = traits::one();

        // Miller backward recurrence can overflow even though the final ratios are well-scaled.
        // We need to track ALL rescaling to preserve ratios correctly.
        //
        // The key insight: we need to apply the SAME total rescaling to ALL values,
        // not just rescale locally. We'll accumulate the total scale factor.
        const T BIG = traits::sqrt(std::numeric_limits<T>::max()) * T(0.25);
        T total_scale = traits::one();

        // Backward recurrence: I_{k-1} = I_{k+1} + (2k/c) * I_k
        for (int k = Nstart; k >= 1; --k) {
            I[k - 1] = I[k + 1] + (traits::two() * T(k) / c) * I[k];

            // Dynamic rescaling to prevent overflow
            // When we rescale, we must rescale ALL values in the array
            const T a0 = traits::abs(I[k - 1]);
            const T a1 = traits::abs(I[k]);
            const T a2 = traits::abs(I[k + 1]);
            T amax = a0;
            if (a1 > amax) amax = a1;
            if (a2 > amax) amax = a2;

            if (amax > BIG && amax > traits::zero()) {
                const T scale = BIG / amax;
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

        const T invI0 = traits::one() / I[0];
        std::vector<T> R(nmax + 1);
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
    static int recommended_Nstart(const T& c, int nmax) {
        // Use the same heuristic as the original bessel_In implementation:
        // Nstart ≈ max(nmax, 2c) + margin
        return std::max(nmax + 50, static_cast<int>(traits::two() * c) + 50);
    }
};


/**
 * @brief Incremental exponential divided difference evaluator using Chebyshev polynomials.
 *
 * This class implements the incremental update algorithm from Section 4 of the paper
 * "Exponential Divided Differences via Chebyshev Polynomials". It enables efficient
 * insertion and removal of nodes in O(N) time (independent of q) when the affine
 * mapping interval [a,b] is fixed.
 *
 * Key capabilities:
 * - add_node(): Insert a node in O(N) time
 * - remove_last_node(): Remove most recent node in O(N) time
 * - Both operations much faster than full recomputation at O(qN)
 *
 * Typical use case: Quantum Monte Carlo with dynamically evolving energy sets.
 *
 * @tparam T Floating-point type (double, long double, or arbitrary precision)
 */
template<typename T = double>
class IncrementalExpDivDiff {
private:
    using traits = numeric_traits<T>;
    
    // Fixed interval parameters
    T a_, b_;           // Interval [a, b]
    T c_, d_;           // c = (b-a)/2, d = (b+a)/2
    T tolerance_;       // Convergence tolerance
    int max_terms_;     // Maximum Chebyshev terms
    
    // Precomputed Bessel ratios R_n(c) = I_n(c) / I_0(c)
    std::vector<T> bessel_ratios_;
    
    // Current state
    std::vector<T> current_nodes_;        // Original nodes x_i
    std::vector<T> normalized_nodes_;     // Normalized nodes y_i = (x_i - d)/c
    int current_q_;                       // Current divided difference order
    
    // Cached Chebyshev divided differences D^(n)_q for the CURRENT node set
    // cached_D_[n][k] = T_n[y_0, ..., y_k] for the current nodes
    std::vector<std::vector<T>> cached_D_;
    
    // Most recent exponential divided difference value
    T current_value_;
    bool is_initialized_;
    
    /**
     * @brief Compute modified Bessel function I_0(x).
     */
    T modified_bessel_I0(T x) const {
        T ax = traits::abs(x);
        
        if (ax < T(3.75)) {
            T y = (x / T(3.75)) * (x / T(3.75));
            return T(1.0) + y * (T(3.5156229) + y * (T(3.0899424) +
                   y * (T(1.2067492) + y * (T(0.2659732) +
                   y * (T(0.0360768) + y * T(0.0045813))))));
        }
        
        T y = T(3.75) / ax;
        T result = (traits::exp(ax) / traits::sqrt(ax)) *
                   (T(0.39894228) + y * (T(0.01328592) +
                   y * (T(0.00225319) + y * (T(-0.00157565) +
                   y * (T(0.00916281) + y * (T(-0.02057706) +
                   y * (T(0.02635537) + y * (T(-0.01647633) +
                   y * T(0.00392377)))))))));
        return result;
    }
    
    /**
     * @brief Compute modified Bessel function I_n(x) using Miller's backward recurrence.
     */
    T bessel_In(int n, T x) const {
        if (n < 0) throw std::invalid_argument("n must be non-negative");
        if (n == 0) return modified_bessel_I0(x);
        
        T x_abs = traits::abs(x);
        
        if (x_abs < T(1e-10)) {
            // For very small x, use series: I_n(x) ~ (x/2)^n / n!
            if (n > 100) return traits::zero();
            T result = traits::one();
            T x_half = x_abs / traits::two();
            for (int k = 1; k <= n; ++k) {
                result = result * x_half / T(k);
            }
            return result;
        }
        
        // Miller's backward recurrence for n > 0
        int m = std::max(n + 50, static_cast<int>(T(2) * x_abs) + 50);
        
        // Backward recurrence: I_k = (2*(k+1)/x) * I_{k+1} + I_{k+2}
        T I_next = traits::zero();      // I_{m+1} ~ 0
        T I_curr = traits::one();       // I_m ~ arbitrary
        T I_n_unnorm = traits::zero();
        
        for (int k = m; k >= 0; --k) {
            T I_prev = (traits::two() * T(k + 1) / x_abs) * I_curr + I_next;
            
            if (k == n) {
                I_n_unnorm = I_prev;
            }
            
            I_next = I_curr;
            I_curr = I_prev;
            
            // Prevent overflow
            if (traits::abs(I_curr) > T(1e100)) {
                I_curr = I_curr / T(1e100);
                I_next = I_next / T(1e100);
                if (k <= n) I_n_unnorm = I_n_unnorm / T(1e100);
            }
        }
        
        // I_curr is now unnormalized I_0
        // Normalize: I_n = I_n_unnorm * (I_0_true / I_0_unnorm)
        T I0_true = modified_bessel_I0(x_abs);
        return I_n_unnorm * (I0_true / I_curr);
    }
    
    /**
     * @brief Compute modified Bessel function ratio R_n(c) = I_n(c) / I_0(c).
     */
    T bessel_ratio(int n, T c) const {
        if (n == 0) return traits::one();
        T I0_c = modified_bessel_I0(c);
        return bessel_In(n, c) / I0_c;
    }
    
    /**
     * @brief Precompute Bessel ratios R_n(c) using efficient batch computation.
     *
     * For large max_terms but small c, we don't need to compute all ratios
     * because they become negligibly small. This function computes only what's needed.
     */
    void precompute_bessel_ratios() {
        // Estimate how many terms we actually need based on c
        // For large n >> c, R_n(c) ~ (c/2n)^n which decays exponentially
        // Rule of thumb: compute up to max(100, 3*c)
        int effective_max = std::min(max_terms_, std::max(100, static_cast<int>(T(3) * c_) + 50));
        
        int Nstart = BesselRatiosMiller<T>::recommended_Nstart(c_, effective_max);
        std::vector<T> computed_ratios = BesselRatiosMiller<T>::compute_R(c_, effective_max, Nstart);
        
        // Resize to full max_terms and fill with computed values
        // Any values beyond effective_max will be essentially zero
        bessel_ratios_.resize(max_terms_ + 1, traits::zero());
        for (int n = 0; n <= effective_max; ++n) {
            bessel_ratios_[n] = computed_ratios[n];
        }
    }
    
    /**
     * @brief Initialize Chebyshev divided differences for T_0 and T_1.
     */
    void initialize_base_layers(int q) {
        // D^(0) = [1, 0, 0, ..., 0]
        std::vector<T> D0(q + 1, traits::zero());
        D0[0] = traits::one();
        
        // D^(1) = [y_0, 1, 0, ..., 0]
        std::vector<T> D1(q + 1, traits::zero());
        if (q >= 0) D1[0] = normalized_nodes_[0];
        if (q >= 1) D1[1] = traits::one();
        
        cached_D_.clear();
        cached_D_.push_back(D0);
        cached_D_.push_back(D1);
    }
    
    /**
     * @brief Compute next Chebyshev layer using recurrence (7) from the paper.
     * T_{n+1}[y_0,...,y_k] = 2(y_k*T_n[y_0,...,y_k] + T_n[y_0,...,y_{k-1}]) - T_{n-1}[y_0,...,y_k]
     *
     * CRITICAL: Note the parentheses! The 2 multiplies the entire sum.
     */
    std::vector<T> compute_next_layer(const std::vector<T>& D_n,
                                       const std::vector<T>& D_n_minus_1,
                                       int q) const {
        std::vector<T> D_n_plus_1(q + 1);
        
        for (int k = 0; k <= q; ++k) {
            T term1 = normalized_nodes_[k] * D_n[k];
            T term2 = (k > 0) ? D_n[k-1] : traits::zero();
            T term3 = D_n_minus_1[k];
            
            // Apply equation (7) WITH PARENTHESES:
            D_n_plus_1[k] = traits::two() * (term1 + term2) - term3;
        }
        
        return D_n_plus_1;
    }
    
    /**
     * @brief Evaluate exponential divided difference from cached Chebyshev layers.
     */
    T evaluate_from_cache(int q, bool normalized = false) const {
        if (cached_D_.empty() || q < 0) {
            throw std::runtime_error("No cached data available");
        }
        
        // Compute S = D^(0)_q + 2 * sum_{n=1}^N R_n(c) * D^(n)_q
        T S = cached_D_[0][q];
        
        // Array to track last 5 terms for convergence check
        T last_5_terms[5] = {traits::zero(), traits::zero(), traits::zero(),
                             traits::zero(), traits::zero()};
        
        for (size_t n = 1; n < cached_D_.size(); ++n) {
            T term = traits::two() * bessel_ratios_[n] * cached_D_[n][q];
            S += term;
            
            // Track last 5 terms for convergence check (same as batch version)
            if (n >= 2) {
                // Shift history
                for (int i = 4; i > 0; --i) {
                    last_5_terms[i] = last_5_terms[i-1];
                }
                last_5_terms[0] = traits::abs(term);
                
                // Check convergence: wait until n >= q + 5, then check if all 5 are small
                if (n >= static_cast<size_t>(q + 5)) {
                    bool all_small = true;
                    for (int i = 0; i < 5; ++i) {
                        if (last_5_terms[i] >= tolerance_ * traits::max(traits::abs(S), traits::one())) {
                            all_small = false;
                            break;
                        }
                    }
                    if (all_small) {
                        break;
                    }
                }
            }
        }
        
        if (normalized) {
            return S;
        }
        
        // Apply scaling: exp(d) * I_0(c) / c^q * S
        T I0_c = modified_bessel_I0(c_);
        T scale = traits::exp(d_) * I0_c / traits::pow(c_, q);
        
        return scale * S;
    }
    
public:
    /**
     * @brief Constructor with fixed interval.
     *
     * @param a Lower bound of interval
     * @param b Upper bound of interval (must be > a)
     * @param tolerance Convergence tolerance (default 1e-14)
     * @param max_terms Maximum Chebyshev terms (default 10000)
     */
    IncrementalExpDivDiff(T a, T b, T tolerance = T(1e-14), int max_terms = 10000)
        : a_(a), b_(b), tolerance_(tolerance), max_terms_(max_terms),
          current_q_(-1), current_value_(traits::zero()), is_initialized_(false) {
        
        if (b <= a) {
            throw std::invalid_argument("Interval must satisfy b > a");
        }
        
        c_ = (b - a) / traits::two();
        d_ = (b + a) / traits::two();
        
        // Precompute Bessel ratios
        precompute_bessel_ratios();
    }
    
    /**
     * @brief Initialize with a starting set of nodes.
     *
     * This computes the exponential divided difference for the given nodes
     * and sets up the cache for incremental updates.
     *
     * Complexity: O(qN) for q nodes and N Chebyshev terms
     *
     * @param initial_nodes Vector of initial nodes (must be in [a, b])
     * @return exp[x_0, ..., x_q]
     */
    T initialize(const std::vector<T>& initial_nodes) {
        if (initial_nodes.empty()) {
            throw std::invalid_argument("Initial node vector cannot be empty");
        }
        
        // Validate and store nodes
        current_nodes_ = initial_nodes;
        current_q_ = static_cast<int>(initial_nodes.size()) - 1;
        
        // Check for degenerate case: all nodes identical
        T min_node = *std::min_element(current_nodes_.begin(), current_nodes_.end());
        T max_node = *std::max_element(current_nodes_.begin(), current_nodes_.end());
        
        if (traits::abs(max_node - min_node) < T(1e-15)) {
            // Degenerate case: exp[x, x, ..., x] = exp(x) / q!
            T factorial = traits::one();
            for (int i = 1; i <= current_q_; ++i) {
                factorial *= T(i);
            }
            current_value_ = traits::exp(current_nodes_[0]) / factorial;
            
            // Still need to set up normalized nodes for consistency
            normalized_nodes_.resize(current_q_ + 1);
            for (int i = 0; i <= current_q_; ++i) {
                normalized_nodes_[i] = (current_nodes_[i] - d_) / c_;
            }
            
            is_initialized_ = true;
            return current_value_;
        }
        
        // Normalize nodes to [-1, 1]
        normalized_nodes_.resize(current_q_ + 1);
        for (int i = 0; i <= current_q_; ++i) {
            if (current_nodes_[i] < a_ || current_nodes_[i] > b_) {
                throw std::out_of_range("Node outside interval [a, b]");
            }
            normalized_nodes_[i] = (current_nodes_[i] - d_) / c_;
        }
        
        // Initialize base layers (T_0 and T_1)
        initialize_base_layers(current_q_);
        
        // Build up all Chebyshev layers and accumulate sum
        T S = cached_D_[0][current_q_]; // D^(0)_q
        
        // Add n=1 term
        S += traits::two() * bessel_ratios_[1] * cached_D_[1][current_q_];
        
        // Array to track last 5 terms for convergence check (same as batch version)
        T last_5_terms[5] = {traits::zero(), traits::zero(), traits::zero(),
                             traits::zero(), traits::zero()};
        
        // Build remaining layers
        for (int n = 2; n < max_terms_; ++n) {
            // Compute next layer
            std::vector<T> D_next = compute_next_layer(
                cached_D_[cached_D_.size() - 1],
                cached_D_[cached_D_.size() - 2],
                current_q_
            );
            
            cached_D_.push_back(D_next);
            
            // Accumulate
            T term = traits::two() * bessel_ratios_[n] * D_next[current_q_];
            S += term;
            
            // Track last 5 terms for convergence check
            if (n >= 2) {
                // Shift history
                for (int i = 4; i > 0; --i) {
                    last_5_terms[i] = last_5_terms[i-1];
                }
                last_5_terms[0] = traits::abs(term);
                
                // Check convergence: wait until n >= q + 5, then check if all 5 are small
                if (n >= current_q_ + 5) {
                    bool all_small = true;
                    for (int i = 0; i < 5; ++i) {
                        if (last_5_terms[i] >= tolerance_ * traits::max(traits::abs(S), traits::one())) {
                            all_small = false;
                            break;
                        }
                    }
                    if (all_small) {
                        break;
                    }
                }
            }
        }
        
        // Apply scaling: exp(d) * I_0(c) / c^q * S
        T I0_c = modified_bessel_I0(c_);
        T scale = traits::exp(d_) * I0_c / traits::pow(c_, current_q_);
        current_value_ = scale * S;
        
        is_initialized_ = true;
        return current_value_;
    }
    
    /**
     * @brief Add a new node to the current set.
     *
     * Implements Section 4.3 of the paper using incremental recurrence.
     * This is much faster than recomputing from scratch.
     *
     * Complexity: O(N) (independent of q!)
     *
     * @param new_node Node to add (must be in [a, b])
     * @return Updated exp[x_0, ..., x_q, new_node]
     */
    T add_node(T new_node) {
        if (!is_initialized_) {
            throw std::runtime_error("Must call initialize() before add_node()");
        }
        
        if (new_node < a_ || new_node > b_) {
            throw std::out_of_range("Node outside interval [a, b]");
        }
        
        // Add to current node set
        current_nodes_.push_back(new_node);
        T y_new = (new_node - d_) / c_;
        normalized_nodes_.push_back(y_new);
        
        int old_q = current_q_;
        current_q_++;
        
        // Resize all cached layers to accommodate new column
        for (auto& layer : cached_D_) {
            layer.resize(current_q_ + 1);
        }
        
        // Initialize base values for the new column
        // D^(0)_{q+1} = T_0[y_0,...,y_{q+1}] = 0 (T_0 constant, divided diff vanishes for k>0)
        cached_D_[0][current_q_] = traits::zero();
        
        // D^(1)_{q+1} = T_1[y_0,...,y_{q+1}]
        // For T_1(y) = y (linear), the divided difference is:
        // - T_1[y_0, y_1] = (y_1 - y_0)/(y_1 - y_0) = 1
        // - T_1[y_0, ..., y_k] = 0 for k > 1 (degree 1 polynomial)
        if (current_q_ == 1) {
            cached_D_[1][current_q_] = traits::one();  // Special case: two nodes
        } else {
            cached_D_[1][current_q_] = traits::zero();  // General case: more than two nodes
        }
        
        // Now build up D^(n)_{q+1} for n = 2, 3, 4, ... using the recurrence
        // D^(n)_{q+1} = 2*y_{q+1}*D^(n-1)_{q+1} + 2*D^(n-1)_q - D^(n-2)_{q+1}
        //
        // Note: This is equation (7) specialized to column q+1:
        // T_{n+1}[y_0,...,y_{q+1}] = 2*y_{q+1}*T_n[y_0,...,y_{q+1}] + T_n[y_0,...,y_q] - T_{n-1}[y_0,...,y_{q+1}]
        // Shifting indices: T_n becomes D^(n)
        
        int n = 2;
        T S = cached_D_[0][current_q_]; // Start accumulation with D^(0)_{q+1}
        
        // Add contribution from n=1
        if (cached_D_.size() > 1) {
            S += traits::two() * bessel_ratios_[1] * cached_D_[1][current_q_];
        }
        
        // Build up remaining layers
        // Array to track last 5 terms for convergence check (same as batch version)
        T last_5_terms[5] = {traits::zero(), traits::zero(), traits::zero(),
                             traits::zero(), traits::zero()};
        
        while (n < max_terms_) {
            // Ensure we have this layer
            if (n >= static_cast<int>(cached_D_.size())) {
                // Need to create a new layer for all columns first
                std::vector<T> new_layer(current_q_ + 1, traits::zero());
                
                // Fill in this layer for columns 0 through old_q using standard recurrence
                for (int k = 0; k <= old_q; ++k) {
                    T term1 = normalized_nodes_[k] * cached_D_[n-1][k];
                    T term2 = (k > 0) ? cached_D_[n-1][k-1] : traits::zero();
                    T term3 = cached_D_[n-2][k];
                    // Equation (7) WITH PARENTHESES:
                    new_layer[k] = traits::two() * (term1 + term2) - term3;
                }
                
                cached_D_.push_back(new_layer);
            }
            
            // Now compute the new column entry using incremental recurrence
            // From Eq. (7): T_{n+1}[y_0,...,y_k] = 2(y_k*T_n[y_0,...,y_k] + T_n[y_0,...,y_{k-1}]) - T_{n-1}[y_0,...,y_k]
            // With k = q+1, n -> n-1:
            // T_n[y_0,...,y_{q+1}] = 2(y_{q+1}*T_{n-1}[y_0,...,y_{q+1}] + T_{n-1}[y_0,...,y_q]) - T_{n-2}[y_0,...,y_{q+1}]
            // Or: D^(n)_{q+1} = 2(y_{q+1}*D^(n-1)_{q+1} + D^(n-1)_q) - D^(n-2)_{q+1}
            T D_n_q_plus_1 = traits::two() * (y_new * cached_D_[n-1][current_q_] + cached_D_[n-1][old_q])
                           - cached_D_[n-2][current_q_];
            
            cached_D_[n][current_q_] = D_n_q_plus_1;
            
            // Accumulate to sum
            T term = traits::two() * bessel_ratios_[n] * D_n_q_plus_1;
            S += term;
            
            // Track last 5 terms for convergence check
            if (n >= 2) {
                // Shift history
                for (int i = 4; i > 0; --i) {
                    last_5_terms[i] = last_5_terms[i-1];
                }
                last_5_terms[0] = traits::abs(term);
                
                // Check convergence: wait until n >= current_q + 5, then check if all 5 are small
                if (n >= current_q_ + 5) {
                    bool all_small = true;
                    for (int i = 0; i < 5; ++i) {
                        if (last_5_terms[i] >= tolerance_ * traits::max(traits::abs(S), traits::one())) {
                            all_small = false;
                            break;
                        }
                    }
                    if (all_small) {
                        break;
                    }
                }
            }
            
            n++;
        }
        
        // Apply scaling: exp(d) * I_0(c) / c^{q+1} * S
        T I0_c = modified_bessel_I0(c_);
        T scale = traits::exp(d_) * I0_c / traits::pow(c_, current_q_);
        current_value_ = scale * S;
        
        return current_value_;
    }
    
    /**
     * @brief Remove the most recently added node.
     *
     * Implements Section 4.4 of the paper. Since we cache D^(n)_q values,
     * removal simply requires re-evaluating the sum with the previous cached values.
     *
     * Complexity: O(N) (independent of q)
     *
     * @return Updated exp[x_0, ..., x_{q-1}]
     */
    T remove_last_node() {
        if (!is_initialized_) {
            throw std::runtime_error("Must call initialize() before remove_last_node()");
        }
        
        if (current_q_ < 0) {
            throw std::runtime_error("No nodes to remove");
        }
        
        // Remove from current node set
        current_nodes_.pop_back();
        normalized_nodes_.pop_back();
        current_q_--;
        
        // Special case: after removal, if q=0, just evaluate exp(x0) directly
        if (current_q_ == 0) {
            current_value_ = traits::exp(current_nodes_[0]);
            return current_value_;
        }
        
        // Normal case: q >= 1
        // The cached values D^(n)_q for the reduced node set are still valid!
        // Just re-evaluate with the lower order
        current_value_ = evaluate_from_cache(current_q_, false);
        
        return current_value_;
    }
    
    /**
     * @brief Get current exponential divided difference value.
     */
    T current_value() const {
        if (!is_initialized_) {
            throw std::runtime_error("Not initialized");
        }
        return current_value_;
    }
    
    /**
     * @brief Get current divided difference order.
     */
    int current_order() const {
        return current_q_;
    }
    
    /**
     * @brief Get current node set.
     */
    const std::vector<T>& current_nodes() const {
        return current_nodes_;
    }
    
    /**
     * @brief Check if initialized.
     */
    bool is_initialized() const {
        return is_initialized_;
    }
    
    /**
     * @brief Get number of cached Chebyshev layers.
     */
    size_t num_cached_layers() const {
        return cached_D_.size();
    }
    
    /**
     * @brief Get current normalized value: S = sum R_n(c) D^(n)_q
     *
     * Returns the normalized sum without the scaling factor.
     * Full result = exp(d) * I_0(c) / c^q * S
     *
     * This is useful for avoiding overflow when computing ratios of divided differences.
     *
     * @return Normalized sum S
     */
    T get_normalized_value() const {
        if (!is_initialized_) {
            throw std::runtime_error("Not initialized");
        }
        return evaluate_from_cache(current_q_, true);
    }
    
    /**
     * @brief Add node and return normalized value.
     *
     * Same as add_node() but returns normalized sum S instead of full value.
     *
     * @param new_node Node to add (must be in [a, b])
     * @return Normalized sum S for updated node set
     */
    T add_node_normalized(T new_node) {
        add_node(new_node);  // This updates current_value_ with full result
        return evaluate_from_cache(current_q_, true);  // Return normalized
    }
    
    /**
     * @brief Get scaling factor for current state.
     *
     * Returns the scaling factor exp(d) * I_0(c) / c^q
     * Full result = scaling_factor * normalized_sum
     *
     * @return Scaling factor
     */
    T get_scaling_factor() const {
        if (!is_initialized_) {
            throw std::runtime_error("Not initialized");
        }
        T I0_c = modified_bessel_I0(c_);
        return traits::exp(d_) * I0_c / traits::pow(c_, current_q_);
    }
    
    /**
     * @brief Reset to uninitialized state.
     */
    void reset() {
        current_nodes_.clear();
        normalized_nodes_.clear();
        cached_D_.clear();
        current_q_ = -1;
        current_value_ = traits::zero();
        is_initialized_ = false;
    }
};

} // namespace chebyshev_incremental

#endif // INCREMENTAL_EXP_DIVDIFF_HPP
