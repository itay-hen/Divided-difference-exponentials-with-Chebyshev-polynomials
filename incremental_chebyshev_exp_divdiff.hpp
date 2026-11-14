#ifndef INCREMENTAL_EXP_DIVDIFF_HPP
#define INCREMENTAL_EXP_DIVDIFF_HPP

// LEGACY.  Superseded by chebyshev::IncrementalStack (chebyshev_stack.hpp),
// which is the implementation described in the revised manuscript.  This
// class is kept because the older test groups exercise it.  Known
// limitations found in the revision: the cached table is not exponent-
// scaled (fails for q >~ 250 at small c), the conditioning gate
// (kappa*u > 1e-6) is much looser than the batch gate and does not see
// unfavorable insertion orders, and the layer-major storage makes an
// append several times slower than necessary.

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>

#include "chebyshev_bessel.hpp"

namespace chebyshev_incremental {

// Shared with the batch evaluator via chebyshev_bessel.hpp; the
// re-exports below let existing call sites continue to use the
// unqualified names numeric_traits / BesselRatiosMiller inside the
// chebyshev_incremental:: namespace.
using chebyshev_bessel::numeric_traits;
using chebyshev_bessel::BesselRatiosMiller;
using chebyshev_bessel::log_I0_value;
using chebyshev_bessel::log_I0_upper;


/**
 * @brief Incremental exponential divided difference evaluator using
 *        Chebyshev polynomials.
 *
 * Implements the incremental update scheme from Section 5 of the paper
 * "Exponential Divided Differences via Chebyshev Polynomials". Under
 * the standing assumptions of that section, namely a fixed affine
 * mapping interval [a,b] (no rescaling between updates), appended
 * nodes contained in [a,b], and a global Chebyshev cutoff N_max with
 * full columns cached up to that cutoff, an append move costs
 * O(N(q,c;eps)) and last-in--first-out node removal costs the same.
 *
 * The notation here is N(q,c;eps): the truncation length depends on
 * both the divided-difference order q and the interval half-width c.
 * The advantage of the incremental update over full recomputation is
 * the removal of the explicit O(q) sweep over the divided-difference
 * table, not q-independence of the cost: in the width-dominated regime
 * c >> q, N is essentially independent of q and the update cost
 * approaches O(c) at fixed tolerance; in the order-dominated regime
 * q >~ c, the structural lower bound N >= q forces the update cost to
 * scale with q through N.
 *
 * Key capabilities:
 *   - add_node(): append a node, cost O(N(q+1,c;eps)).
 *   - remove_last_node(): LIFO removal of the last appended node,
 *     cost O(N(q,c;eps)).
 *   - Memory footprint of the cached column stack at order q is
 *     O(q * N_max).
 *
 * Typical use case: quantum Monte Carlo with dynamically evolving
 * energy sets, where appends and last-in--first-out removals dominate.
 *
 * @tparam T Floating-point type (double, long double, or a multi-
 *           precision type whose std:: overloads of exp/sqrt/pow are
 *           available).
 */
/**
 * @brief Choice of stopping rule for the Chebyshev sum.
 *
 * HeuristicFiveTerm
 *     The default.  After n >= q+5, declare convergence when the last
 *     five consecutive added terms have absolute value below
 *     tolerance * |S|.  Fast, but not certified: in principle a
 *     pathological coincidence of small terms could trigger early
 *     termination.
 *
 * ConservativeTailEstimate
 *     The safety mode.  At each step n, evaluate the analytic tail
 *     bound from Eq. (33) of the paper,
 *
 *         2 e^d I_0(c) q! / (c^q e^a) * sum_{n'>n} R_{n'}(c) * B(n',q),
 *
 *     and stop when it falls below the requested tolerance.  The
 *     evaluation is performed in log space against an upper estimate
 *     of log(I_0(c)) from log_I0_upper(), and the infinite tail beyond
 *     max_order_ is bounded analytically (see certified_tail_residual).
 *
 *     This is a *conservative floating-point* implementation of the
 *     analytic bound, not a proven interval-arithmetic certificate:
 *     log_I0_upper() and certified_tail_residual() rely on
 *     std::cyl_bessel_i rounding properties and an asymptotic
 *     remainder estimate, neither of which is formally proven inside
 *     this code.  In practice the implementation has been verified to
 *     meet its stated tolerance on every tested case (including the
 *     c=100, q=3 adversarial scenario from the review), but if you
 *     need proof-level certification, substitute an interval-arithmetic
 *     library (Arb / Boost.Interval / MPFI) for the Bessel quantities.
 */
enum class StoppingMode {
    HeuristicFiveTerm,
    ConservativeTailEstimate
};

template<typename T = double>
class IncrementalExpDivDiff {
private:
    using traits = numeric_traits<T>;

    // Fixed interval parameters.
    T a_, b_;           // Interval [a, b].
    T c_, d_;           // c = (b-a)/2, d = (b+a)/2.
    T tolerance_;       // Convergence tolerance.
    // max_order_ : the maximum Chebyshev order N retained in the
    //              column cache and the Bessel-ratio cache.  After
    //              initialize(), cached_D_ has max_order_+1 layers
    //              (indices 0..max_order_), bessel_ratios_ has size
    //              max_order_+1, and certified_B_ has size max_order_+1.
    //              The certified-mode residual covers the infinite
    //              tail beyond n = max_order_.
    int max_order_;
    int max_q_;         // Maximum divided-difference order to support.
    StoppingMode stopping_mode_;
    
    // Precomputed Bessel ratios R_n(c) = I_n(c) / I_0(c).
    // Mutable because the lazy expand-on-demand strategy in
    // get_bessel_ratio() can extend the cache from logically-const
    // contexts (e.g. evaluate_from_cache).
    mutable std::vector<T> bessel_ratios_;
    
    // Current state.
    std::vector<T> current_nodes_;        // Original nodes x_i.
    std::vector<T> normalized_nodes_;     // Normalized nodes y_i = (x_i - d)/c.
    int current_q_;                       // Current divided-difference order.

    // Cached Chebyshev divided differences D^(n)_q for the CURRENT node
    // set.  cached_D_[n][k] = T_n[y_0, ..., y_k] for the current nodes.
    // Under the A1 design, cached_D_ has N_max layers from initialize()
    // onward, and each layer is exactly current_q_+1 wide.  Appends
    // extend each layer's column width by one; removals shrink it.
    std::vector<std::vector<T>> cached_D_;

    // Most recent exponential divided difference (full, unscaled) value
    // and the matching normalized Chebyshev sum S.  We keep both so that
    // a normalized read does not have to form the full value (avoiding
    // overflow in the prefactor at large c), and vice versa.
    T current_value_;
    T current_S_;
    bool is_initialized_;
    // True when current_value_ holds the up-to-date full divided
    // difference; false when a *_normalized path was last taken and
    // the full value would have to be recomputed via scale_full.
    bool full_value_valid_ = false;

    // Precomputed structural bound coefficients for the certified
    // stopping mode (Eq. 33 of the paper):
    //   B(n,q) = 2^{q-1} * n/q * (2q)_{n-q} / (n-q)! for n >= q,
    //   B(n,q) = 0 otherwise.
    // certified_B_[n] holds B(n, current_q_) at indices 0..max_order_;
    // rebuilt by precompute_certified_bounds(q) whenever current_q_
    // changes.
    // log_certified_B_[n] holds log(B(n, current_q_)) at indices
    // 0..max_order_; rebuilt by precompute_certified_bounds(q) whenever
    // current_q_ changes.  Stored in log space so that B(n, q) itself
    // does not need to be representable -- for large q and large n
    // the linear-scale value can exceed double's range even though
    // the product R_n * B(n,q) (which is what feeds the bound) is
    // tiny because R_n is exponentially small.
    std::vector<T> log_certified_B_;
    int certified_B_q_ = -1;

    // C2: last Chebyshev order n actually consulted in evaluate_from_cache.
    // Reported via last_used_order() so external code can show "n used"
    // alongside "Nmax cached" rather than always reporting the (fixed)
    // cache depth.
    mutable int last_used_order_ = -1;
    
    /**
     * @brief Compute the modified Bessel function I_0(x).
     *
     * Wraps std::cyl_bessel_i(0, x) (C++17), which provides full
     * working-precision accuracy. The earlier Abramowitz--Stegun
     * polynomial approximation was limited to ~10^-7 relative error.
     */
    T modified_bessel_I0(T x) const {
        return static_cast<T>(chebyshev_bessel::cyl_bessel_i(0.0L,
            static_cast<long double>(traits::abs(x))));
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
     * @brief Seed the Bessel-ratio cache up to max_order_.
     *
     * Under the A1 global-N_max design, we pay the full Miller cost
     * once at construction.  get_bessel_ratio() then only reads from
     * the cache (no lazy expansion path is exercised in normal use).
     */
    void precompute_bessel_ratios() {
        const int Nstart = BesselRatiosMiller<T>::recommended_Nstart(c_, max_order_);
        bessel_ratios_ = BesselRatiosMiller<T>::compute_R(c_, max_order_, Nstart);
    }

    /**
     * @brief Return R_n(c), extending the cache if necessary.
     *
     * Mirrors the lazy-expansion behavior of the batch ExpDivDiff class.
     * Throws if the requested order exceeds max_order_, which signals
     * a configuration error rather than something to silently round.
     */
    T get_bessel_ratio(int n) const {
        if (n < 0) {
            throw std::out_of_range("get_bessel_ratio: n must be non-negative");
        }
        if (n > max_order_) {
            throw std::out_of_range(
                "get_bessel_ratio: requested order exceeds max_order_");
        }
        if (n < static_cast<int>(bessel_ratios_.size())) {
            return bessel_ratios_[n];
        }
        // Defensive lazy expand; in the A1 design this is normally
        // unreachable because precompute_bessel_ratios() already built
        // to max_order_.
        const int new_max = std::min(max_order_, n + 100);
        const int Nstart = BesselRatiosMiller<T>::recommended_Nstart(c_, new_max);
        bessel_ratios_ = BesselRatiosMiller<T>::compute_R(c_, new_max, Nstart);
        return bessel_ratios_[n];
    }

    /**
     * @brief Precompute the structural bound B(n, q) used by the
     *        certified stopping rule (Eq. 33 of the paper).
     *
     * For n >= q, the uniform bound on |D^{(n)}_q| reads
     *     B(n,q) = 2^{q-1} * (n/q) * (2q)_{n-q} / (n-q)!
     * where (2q)_m is the rising Pochhammer symbol.  We compute this
     * via the ratio recurrence
     *     B(n+1,q)/B(n,q) = ((n+1)/n) * (n+q)/(n-q+1)
     * which is overflow-safe in the regime we care about (the tail
     * sum R_n * B(n,q) decays rapidly past n ~ c + q).  For q = 0 we
     * use B(0,0)=1 and B(n,0)=0 for n>=1 (since D^{(n)}_0 = T_n(y_0)
     * with |T_n| <= 1; the structural bound's role at q=0 is moot
     * because the heuristic and full sum agree exactly).
     */
    void precompute_certified_bounds(int q) {
        if (certified_B_q_ == q
            && (int)log_certified_B_.size() == max_order_ + 1) {
            return;  // already cached for this q
        }
        log_certified_B_.assign(max_order_ + 1,
            -std::numeric_limits<T>::infinity());
        if (q == 0) {
            // For q=0, the safe envelope is B(n,0) = 1 for all n (since
            // |T_n(y_0)| <= 1).  The expression in the paper has 0/0
            // at q=0; the analytic continuation that respects the
            // pointwise bound is exactly this constant 1.  Earlier
            // versions of this comment claimed B(n,0)=0 for n>=1; that
            // is wrong.  In log space: log(1) = 0.
            for (int n = 0; n <= max_order_; ++n) {
                log_certified_B_[n] = traits::zero();
            }
            certified_B_q_ = 0;
            return;
        }
        // q >= 1: B(q,q) = 2^{q-1}, so log_B(q,q) = (q-1) * log 2.
        log_certified_B_[q] = T(q - 1) * traits::log(traits::two());
        // Recurrence: log_B(n+1,q) = log_B(n,q)
        //                          + log((n+1)/n) + log((n+q)/(n-q+1)).
        for (int n = q; n < max_order_; ++n) {
            const T log_ratio =
                traits::log(T(n + 1) / T(n)) +
                traits::log(T(n + q) / T(n - q + 1));
            log_certified_B_[n + 1] = log_certified_B_[n] + log_ratio;
        }
        certified_B_q_ = q;
    }

    T certified_log_prefactor(int q) const {
        T log_fact = traits::zero();
        for (int i = 2; i <= q; ++i) log_fact += traits::log(T(i));
        const T log_c = (q == 0) ? traits::zero() : traits::log(c_);
        return traits::log(traits::two()) + c_ + log_I0_upper(c_) +
               log_fact - T(q) * log_c;
    }

    /**
     * @brief log of the conservative upper bound on the omitted tail
     *        Σ_{n > max_order_} R_n(c) B(n, q).
     *
     * Strategy:
     *   - R_{N+1} = I_{N+1}(c) / I_0(c) via std::cyl_bessel_i (safer
     *     than the subtractive Bessel recurrence, which suffers
     *     cancellation in the tail).
     *   - log B(N+1, q) from the log-space B-recurrence and the
     *     already-cached log_certified_B_[N].
     *   - Geometric bound for the rest of the tail using the
     *     worst-case term ratio at n = N+1.
     *
     * Returns +infinity if the recurrence / ratio bound fails to
     * apply.  The caller turns that into a clear "increase max_order"
     * exception rather than silently pretending to certify.
     */
    T certified_log_tail_residual(int q) const {
        const T pos_inf = std::numeric_limits<T>::infinity();
        if (max_order_ < q + 2) return pos_inf;
        const int N = max_order_;

        // log R_{N+1} = log(I_{N+1}(c)) - log(I_0(c)).  Use long double
        // for the Bessel evaluation so that c values up to ~11000 stay
        // representable; for larger c we'd need a log-scaled Bessel
        // wrapper.  We bound log(I_0(c)) above by log_I0_upper.
        const long double I_Np1 = chebyshev_bessel::cyl_bessel_i(
            static_cast<long double>(N + 1),
            static_cast<long double>(c_));
        if (!std::isfinite(I_Np1) || I_Np1 < 0.0L) return pos_inf;
        if (I_Np1 == 0.0L) {
            // Tail contribution from this term is exactly zero; the
            // geometric extension is also zero.  Return -infinity so
            // log-sum-exp ignores the residual.
            return -pos_inf;
        }
        const T log_R_Np1 =
            static_cast<T>(std::log(I_Np1)) - log_I0_value(c_);

        if (N < q || N - q + 1 <= 0) return pos_inf;
        const T log_B_Np1 = log_certified_B_[N]
            + traits::log(T(N + 1) / T(N))
            + traits::log(T(N + q) / T(N - q + 1));

        const T log_term_Np1 = log_R_Np1 + log_B_Np1;

        if (N + 2 - q <= 0) return pos_inf;
        const T r = (c_ / T(2 * (N + 2))) *
                    (T(N + 2) / T(N + 1)) *
                    (T(N + 1 + q) / T(N + 2 - q));
        if (!(r < traits::one())) return pos_inf;
        return log_term_Np1 - traits::log(traits::one() - r);
    }

    /**
     * @brief Stable log(e^a + e^b).
     *
     * Handles -inf (correctly returns the other argument) and avoids
     * overflow/underflow in the exponentiation.
     */
    static T log_sum_exp(T a, T b) {
        const T neg_inf = -std::numeric_limits<T>::infinity();
        if (a == neg_inf) return b;
        if (b == neg_inf) return a;
        const T m  = (a > b) ? a : b;
        const T dx = (a > b) ? (b - a) : (a - b);   // <= 0
        return m + traits::log(traits::one() + traits::exp(dx));
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
     * @brief Local variant of compute_next_layer that uses an explicit
     *        normalized-nodes vector instead of the member field.
     *
     * Used by the transactional initialize_compute_S_, which builds
     * the new cache into a local vector before touching member state.
     */
    std::vector<T> compute_next_layer_local(const std::vector<T>& D_n,
                                             const std::vector<T>& D_n_minus_1,
                                             int q,
                                             const std::vector<T>& y_nodes) const {
        std::vector<T> D_n_plus_1(q + 1);
        for (int k = 0; k <= q; ++k) {
            T term1 = y_nodes[k] * D_n[k];
            T term2 = (k > 0) ? D_n[k - 1] : traits::zero();
            T term3 = D_n_minus_1[k];
            D_n_plus_1[k] = traits::two() * (term1 + term2) - term3;
        }
        return D_n_plus_1;
    }

    /**
     * @brief Read-only sibling of evaluate_from_cache that operates
     *        on an explicitly supplied cached_D vector.
     *
     * Used by the transactional initialize_compute_S_ to compute S
     * from the locally-built cache before committing to member state.
     * Convergence rule and certified-mode treatment match
     * evaluate_from_cache.
     */
    T evaluate_from_cache_local(const std::vector<std::vector<T>>& cache,
                                 int q,
                                 bool normalized) const {
        if (cache.empty() || q < 0) {
            throw std::runtime_error("No cached data available");
        }
        T S = cache[0][q];
        // A2 (R8 review): conditioning tracking; see evaluate_from_cache.
        T abs_sum = traits::zero();
        const T abs_S0 = traits::abs(S);
        T last_5[5] = { traits::zero(), traits::zero(), traits::zero(),
                        traits::zero(), traits::zero() };
        // Certified-mode log-space machinery.  We never form B(n,q) or
        // R_n * B(n,q) in linear scale: each can overflow at high q.
        // Instead we precompute log_term[n] = log R_n + log_B[n] and
        // the partial log-sum-exp arrays log_tail_partial[k] =
        //     log( Σ_{j>=k} R_j B_j  +  residual_tail ),
        // so that at step n the bound on the remaining tail is just
        // a lookup of log_tail_partial[n+1].
        T log_cert_prefactor = traits::zero();
        std::vector<T> log_tail_partial;
        T log_tol  = traits::zero();
        if (stopping_mode_ == StoppingMode::ConservativeTailEstimate) {
            const_cast<IncrementalExpDivDiff*>(this)->precompute_certified_bounds(q);
            log_cert_prefactor = certified_log_prefactor(q);
            log_tol = traits::log(tolerance_);

            std::vector<T> log_term(max_order_ + 1,
                -std::numeric_limits<T>::infinity());
            for (int n = 1; n <= max_order_; ++n) {
                const T Rn = get_bessel_ratio(n);
                if (Rn > traits::zero()) {
                    log_term[n] = traits::log(Rn) + log_certified_B_[n];
                }
            }
            const T log_residual = certified_log_tail_residual(q);
            // Up-front feasibility check: even if every cached term is
            // consumed, what remains is at least the residual.  If
            // log_pref + log_residual already exceeds log_tol, no
            // amount of iteration can satisfy the bound -- fail fast
            // with a clear message rather than running through the
            // whole cache and reporting "cached layers exhausted."
            if (log_residual == std::numeric_limits<T>::infinity() ||
                log_cert_prefactor + log_residual > log_tol) {
                throw std::runtime_error(
                    "Certified mode: max_order is too small for the "
                    "requested tolerance at this c, q.  The residual "
                    "tail bound beyond max_order alone already exceeds "
                    "tolerance, so no amount of iteration can satisfy "
                    "the certificate.  Reconstruct with a larger "
                    "max_order or relax the tolerance.");
            }
            log_tail_partial.assign(max_order_ + 2,
                -std::numeric_limits<T>::infinity());
            log_tail_partial[max_order_ + 1] = log_residual;
            for (int k = max_order_; k >= 1; --k) {
                log_tail_partial[k] =
                    log_sum_exp(log_term[k], log_tail_partial[k + 1]);
            }
        }
        bool converged = false;
        const int N_terms = static_cast<int>(cache.size());
        int n = 0;
        for (n = 1; n < N_terms; ++n) {
            const T R_n  = get_bessel_ratio(n);
            const T term = traits::two() * R_n * cache[n][q];
            S += term;
            abs_sum += traits::abs(term);
            if (stopping_mode_ == StoppingMode::ConservativeTailEstimate) {
                const int idx = (n + 1 <= max_order_ + 1)
                              ? (n + 1)
                              : (max_order_ + 1);
                const T log_bound = log_cert_prefactor + log_tail_partial[idx];
                if (log_bound <= log_tol) {
                    converged = true; break;
                }
            } else {
                if (n >= 2) {
                    for (int i = 4; i > 0; --i) last_5[i] = last_5[i - 1];
                    last_5[0] = traits::abs(term);
                    if (n >= q + 5) {
                        bool all_small = true;
                        for (int i = 0; i < 5; ++i) {
                            if (last_5[i] >= tolerance_ * traits::abs(S)) {
                                all_small = false; break;
                            }
                        }
                        if (all_small) { converged = true; break; }
                    }
                }
            }
        }
        last_used_order_ = std::min(n, N_terms - 1);
        // Revision: a NaN sum (unscaled table overflow at high q) made
        // every "term < tol*|S|" comparison false, i.e. the window test
        // succeeded and NaN was returned silently.
        if (!std::isfinite(S)) {
            throw std::overflow_error(
                "IncrementalExpDivDiff: non-finite Chebyshev sum (the unscaled "
                "table over/underflowed); use chebyshev::IncrementalStack "
                "(chebyshev_stack.hpp), which is scaled.");
        }
        if (!converged) {
            throw std::runtime_error(
                "evaluate_from_cache_local: cached Chebyshev layers "
                "exhausted without satisfying the stopping rule.");
        }

        // A2 conditioning gate (R9 review): fire regardless of the
        // `normalized` flag, because the round-off problem is in S
        // itself.  initialize_compute_S_ calls this with
        // normalized=true and then forms the full value separately;
        // if we only gated on normalized=false, the over-wide-interval
        // case at initialize() would silently return garbage to the
        // caller.  See evaluate_from_cache for the full discussion.
        const T abs_S = traits::abs(S);
        if (abs_S > traits::zero()) {
            const long double kappa =
                (static_cast<long double>(abs_S0) +
                 static_cast<long double>(abs_sum))
                / static_cast<long double>(abs_S);
            const long double eps_mach =
                static_cast<long double>(std::numeric_limits<T>::epsilon());
            if (kappa * eps_mach > 1e-6L) {
                throw std::runtime_error(
                    "Series-conditioning failure: the fixed interval "
                    "[a, b] is too wide for the active node spread in "
                    "the current precision (kappa * epsilon_mach ~ "
                    + std::to_string(static_cast<double>(kappa * eps_mach)) +
                    ").  Reconstruct on a tighter interval, use higher "
                    "precision, or evaluate in batch mode "
                    "(chebyshev::ExpDivDiff::evaluate) which auto-"
                    "tightens the interval for full-value computation.");
            }
        }

        if (normalized) return S;
        return scale_full(S, q);
    }
    
    /**
     * @brief Evaluate the exponential divided difference from cached
     *        Chebyshev layers.
     *
     * The Chebyshev sum is
     *     S = D^{(0)}_q + 2 * sum_{n>=1} R_n(c) * D^{(n)}_q,
     * and the full value (if requested) is
     *     exp[x_0,...,x_q] = (e^d * I_0(c) / c^q) * S.
     *
     * The stopping rule is controlled by stopping_mode_:
     *
     *   HeuristicFiveTerm
     *       After n >= q + 5, declare convergence when the last five
     *       added terms are below tolerance_ * |S| in absolute value.
     *
     *   ConservativeTailEstimate
     *       At each step, compute a conservative floating-point
     *       estimate of an upper bound on the still-to-add tail using
     *       the structural B(n,q) bounds and the residual Bessel-ratio
     *       mass.  Stop when that estimate, multiplied by the
     *       prefactor (also computed as a conservative upper estimate),
     *       falls below tolerance_.  Slower than the heuristic, and
     *       gives a relative-error bound that holds modulo the
     *       floating-point rounding assumptions in log_I0_upper and
     *       certified_log_tail_residual.  For proof-level certification
     *       substitute an interval-arithmetic backend (Arb / MPFI /
     *       Boost.Interval).
     *
     * Throws std::runtime_error if the cache is exhausted before
     * either stopping rule fires.
     *
     * @param q          The divided-difference order to evaluate.
     * @param normalized If true, return the normalized sum S; if false,
     *                   return the full value (e^d * I_0(c) / c^q) * S.
     */
    T evaluate_from_cache(int q, bool normalized = false) const {
        if (cached_D_.empty() || q < 0) {
            throw std::runtime_error("No cached data available");
        }

        // S starts at D^{(0)}_q.
        T S = cached_D_[0][q];
        // A2 (R8 review): conditioning diagnostic.
        // Track abs_sum = sum |2 R_n D_q^(n)| alongside the signed S.
        // The series-conditioning estimate is
        //     kappa = (|D_q^(0)| + abs_sum) / |S|
        // and kappa * epsilon_mach approximates the relative round-off
        // floor in S.  When the caller forms the full value with a
        // huge prefactor (over-wide-interval case), that round-off
        // becomes the final relative error.  We throw later (after
        // the loop) when kappa * epsilon is above a threshold and the
        // caller asked for a non-normalized return.
        T abs_sum = traits::zero();
        const T abs_S0 = traits::abs(S);

        // Heuristic-mode history.
        T last_5_terms[5] = { traits::zero(), traits::zero(),
                              traits::zero(), traits::zero(),
                              traits::zero() };

        // Certified-mode tail-bound machinery.  Maintain log_prefactor
        // and a running positive tail_sum, then check the convergence
        // Certified-mode log-space machinery; see evaluate_from_cache_local
        // for the full discussion.  Maintain log_tail_partial[k] = log of
        // ( Σ_{j>=k} R_j B(j,q)  +  analytic residual beyond max_order_ )
        // so that at step n the residual bound is a lookup.  The residual
        // makes the rule a conservative estimate over the full infinite
        // sum, not just the in-cache portion -- subject to the
        // floating-point assumptions in log_I0_upper and
        // certified_log_tail_residual.
        T log_cert_prefactor = traits::zero();
        std::vector<T> log_tail_partial;
        T log_tol            = traits::zero();
        if (stopping_mode_ == StoppingMode::ConservativeTailEstimate) {
            const_cast<IncrementalExpDivDiff*>(this)->precompute_certified_bounds(q);
            log_cert_prefactor = certified_log_prefactor(q);

            std::vector<T> log_term(max_order_ + 1,
                -std::numeric_limits<T>::infinity());
            for (int n = 1; n <= max_order_; ++n) {
                const T Rn = get_bessel_ratio(n);
                if (Rn > traits::zero()) {
                    log_term[n] = traits::log(Rn) + log_certified_B_[n];
                }
            }
            const T log_residual = certified_log_tail_residual(q);
            log_tol = traits::log(tolerance_);
            if (log_residual == std::numeric_limits<T>::infinity() ||
                log_cert_prefactor + log_residual > log_tol) {
                throw std::runtime_error(
                    "Certified mode: max_order is too small for the "
                    "requested tolerance at this c, q.  The residual "
                    "tail bound beyond max_order alone already exceeds "
                    "tolerance, so no amount of iteration can satisfy "
                    "the certificate.  Reconstruct with a larger "
                    "max_order or relax the tolerance.");
            }
            log_tail_partial.assign(max_order_ + 2,
                -std::numeric_limits<T>::infinity());
            log_tail_partial[max_order_ + 1] = log_residual;
            for (int k = max_order_; k >= 1; --k) {
                log_tail_partial[k] =
                    log_sum_exp(log_term[k], log_tail_partial[k + 1]);
            }
        }

        bool converged = false;
        const int N_terms = static_cast<int>(cached_D_.size());
        int n = 0;

        for (n = 1; n < N_terms; ++n) {
            const T R_n  = get_bessel_ratio(n);
            const T term = traits::two() * R_n * cached_D_[n][q];
            S += term;
            abs_sum += traits::abs(term);

            if (stopping_mode_ == StoppingMode::ConservativeTailEstimate) {
                const int idx = (n + 1 <= max_order_ + 1)
                              ? (n + 1)
                              : (max_order_ + 1);
                const T log_bound = log_cert_prefactor + log_tail_partial[idx];
                if (log_bound <= log_tol) {
                    converged = true;
                    break;
                }
            } else {
                if (n >= 2) {
                    for (int i = 4; i > 0; --i) {
                        last_5_terms[i] = last_5_terms[i - 1];
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
            }
        }
        // Record the actual last consulted order (regardless of mode);
        // when convergence fired, n is the firing index, else n == N_terms.
        last_used_order_ = std::min(n, N_terms - 1);

        // Revision: a NaN sum (unscaled table overflow at high q) made
        // every "term < tol*|S|" comparison false, i.e. the window test
        // succeeded and NaN was returned silently.
        if (!std::isfinite(S)) {
            throw std::overflow_error(
                "IncrementalExpDivDiff: non-finite Chebyshev sum (the unscaled "
                "table over/underflowed); use chebyshev::IncrementalStack "
                "(chebyshev_stack.hpp), which is scaled.");
        }
        if (!converged) {
            throw std::runtime_error(
                "evaluate_from_cache: cached Chebyshev layers exhausted "
                "without satisfying the stopping rule; raise max_order_ "
                "or loosen tolerance_.");
        }

        // A2 conditioning gate (R8 review).  Track
        //     kappa = (|D_q^(0)| + Σ|2 R_n D_q^(n)|) / |S|.
        // When the fixed interval is much wider than the active node
        // spread, S is the small residue of a near-perfect
        // cancellation among order-1 partial sums.  In that regime
        // kappa * epsilon_mach exceeds 1, so the computed S has no
        // resolved digits.  We throw rather than return either the
        // garbage full value (after the huge scaling factor) or a
        // misleading normalized value (which looks "small but
        // representable" while actually being noise).  Threshold of
        // 1e-6 means "less than 6 resolved digits of S"; tighten if
        // your application needs more headroom.  Fires regardless of
        // the `normalized` flag because the round-off issue is in S
        // itself, not in the scaling step.
        const T abs_S = traits::abs(S);
        if (abs_S > traits::zero()) {
            const long double kappa =
                (static_cast<long double>(abs_S0) +
                 static_cast<long double>(abs_sum))
                / static_cast<long double>(abs_S);
            const long double eps_mach =
                static_cast<long double>(std::numeric_limits<T>::epsilon());
            if (kappa * eps_mach > 1e-6L) {
                throw std::runtime_error(
                    "Series-conditioning failure: the fixed interval "
                    "[a, b] is too wide for the active node spread in "
                    "the current precision (kappa * epsilon_mach ~ "
                    + std::to_string(static_cast<double>(kappa * eps_mach)) +
                    ").  Reconstruct the evaluator on a tighter "
                    "interval, use a higher-precision type, or evaluate "
                    "in batch mode (chebyshev::ExpDivDiff::evaluate) "
                    "which auto-tightens the interval for full-value "
                    "computation.");
            }
        }

        if (normalized) {
            return S;
        }
        return scale_full(S, q);
    }
    
public:
    /**
     * @brief Constructor with fixed interval and global N_max caching.
     *
     * @param a              Lower bound of interval.
     * @param b              Upper bound of interval (must be > a).
     * @param max_q          Maximum divided-difference order to support.
     *                       The cache is pre-sized so that subsequent
     *                       add_node()/remove_last_node() calls each
     *                       cost O(N_max) without layer creation.
     *                       add_node() throws if q would exceed max_q.
     * @param tolerance      Convergence tolerance for the Chebyshev sum.
     * @param max_order      Hard cap on the Chebyshev truncation length.
     *                       If <= 0, an automatic choice
     *                           N_max = ceil(c + sqrt(2c log(1/eps))) +
     *                                   max_q + safety
     *                       is used, which covers both the width- and
     *                       order-dominated regimes of Sec. 5 of the
     *                       paper.
     * @param stopping_mode  StoppingMode::HeuristicFiveTerm (default,
     *                       fast) or StoppingMode::ConservativeTailEstimate
     *                       (slower per step; gives a conservative
     *                       floating-point bound on the truncation
     *                       error rather than a heuristic stop, but
     *                       is not a proof-level certificate -- see
     *                       the enum docstring for details).
     */
    IncrementalExpDivDiff(T a, T b,
                          int max_q = 100,
                          T tolerance = T(1e-14),
                          int max_order = -1,
                          StoppingMode stopping_mode = StoppingMode::HeuristicFiveTerm)
        : a_(a), b_(b), tolerance_(tolerance),
          max_q_(max_q), stopping_mode_(stopping_mode),
          current_q_(-1), current_value_(traits::zero()),
          current_S_(traits::zero()), is_initialized_(false) {

        if (!std::isfinite(static_cast<long double>(a)) ||
            !std::isfinite(static_cast<long double>(b))) {
            throw std::invalid_argument(
                "IncrementalExpDivDiff: a and b must be finite");
        }
        if (!(b > a)) {
            throw std::invalid_argument(
                "IncrementalExpDivDiff: interval must satisfy b > a "
                "(also rejects NaN-tainted comparisons)");
        }
        if (max_q < 0) {
            throw std::invalid_argument("max_q must be non-negative");
        }
        if (!(tolerance > T(0))) {
            throw std::invalid_argument(
                "IncrementalExpDivDiff: tolerance must be strictly positive");
        }
        if (max_order != -1 && max_order < max_q + 2) {
            throw std::invalid_argument(
                "IncrementalExpDivDiff: max_order, if specified, must be "
                "at least max_q + 2 to leave room for the Chebyshev sum "
                "and the residual recurrence");
        }

        // A3 R13: overflow-safe affine computation; reject endpoint
        // pairs that produce non-representable c or d (see the batch
        // constructor for the same fix).
        c_ = b / traits::two() - a / traits::two();
        d_ = b / traits::two() + a / traits::two();
        if (!std::isfinite(static_cast<long double>(c_)) ||
            !std::isfinite(static_cast<long double>(d_))) {
            throw std::invalid_argument(
                "IncrementalExpDivDiff: interval endpoints are too "
                "extreme -- the derived half-width c = (b-a)/2 or "
                "midpoint d = (b+a)/2 is not representable as T.  "
                "Use a narrower interval or a higher-precision T.");
        }

        // A1 R12: hard cap on c_.  precompute_bessel_ratios() invokes
        // BesselRatiosMiller::compute_R, whose `recommended_Nstart`
        // is ~2*c regardless of the requested max_order.  For c=1e7
        // that is a 2e7-iteration backward recurrence and ~160 MB of
        // memory, even if the user passed `max_order = 200`.  Since
        // there is no way to get useful results out of a fixed-
        // interval shared-cache evaluator at that c (the conditioning
        // gate would refuse anyway), refuse construction.  The cap
        // is well above the realistic working range and well below
        // the resource-hazard range; users with a genuine need for
        // wide intervals should call the batch `evaluate()` API,
        // which tight-falls-back without per-instance Miller state.
        constexpr long double kCMax = 1.0e6L;
        if (static_cast<long double>(c_) > kCMax) {
            throw std::invalid_argument(
                "IncrementalExpDivDiff: interval half-width c = " +
                std::to_string(static_cast<double>(c_)) + " exceeds "
                "the 1e6 cap.  At this c, the Miller-recurrence "
                "Bessel-ratio cache the incremental class needs "
                "would require O(c) memory and O(c) work regardless "
                "of the requested max_order, and the series-"
                "conditioning gate would refuse any non-trivial node "
                "set anyway.  Use chebyshev::ExpDivDiff::evaluate() "
                "(which tight-falls-back automatically) if you need "
                "full-value evaluation on a wide interval.");
        }

        // Choose N_max.  The two regime estimates of Sec. 5 are:
        //   width regime  : N >~ c + sqrt(2 c log(1/eps))
        //   order regime  : N >~ q + sqrt(2 q log(1/eps))
        // We take the max, then pad.
        //
        // A1 R12: when the auto-computed N_max would be huge (~c for
        // wide intervals), refuse construction rather than silently
        // entering an O(c) memory allocation + O(c) recurrence path.
        // For an incremental class whose whole point is shared-cache
        // fixed-interval evaluation, "I asked for [-1e7, 1e7] and the
        // constructor hung" is a worse outcome than a clear error.
        // The user can still pass max_order explicitly to bypass this
        // guard if they really mean it (in which case they are
        // responsible for the resource cost).
        if (max_order <= 0) {
            const double log_inv_eps =
                std::max(1.0, -std::log(static_cast<double>(tolerance_)));
            const double c_d = static_cast<double>(c_);

            // Conservative ceiling: 100k Chebyshev orders is several
            // MB of cache and seconds of work in the recurrence; past
            // that we want the user to opt in explicitly.
            constexpr long double kAutoCeil = 100000.0L;
            const long double n_width_ld =
                static_cast<long double>(c_d) +
                std::sqrt(2.0L * static_cast<long double>(c_d) *
                          static_cast<long double>(log_inv_eps));
            if (n_width_ld > kAutoCeil) {
                throw std::invalid_argument(
                    "IncrementalExpDivDiff: interval half-width c=" +
                    std::to_string(c_d) + " is too large for "
                    "automatic max_order selection (auto N_max ~ " +
                    std::to_string(static_cast<double>(n_width_ld)) +
                    " exceeds the 100000 ceiling).  Either reconstruct "
                    "on a tighter interval or pass max_order "
                    "explicitly to opt in to the resource cost.");
            }
            const int N_width = static_cast<int>(
                std::ceil(c_d + std::sqrt(2.0 * c_d * log_inv_eps)));
            const int N_order = static_cast<int>(
                max_q + std::sqrt(2.0 * max_q * log_inv_eps));
            int safety = 50;
            // Certified mode needs noticeably more headroom because the
            // honest residual bound (using std::cyl_bessel_i directly,
            // padded log_I0_upper, and no R_n recurrence cancellation)
            // is a one-sided over-estimate -- in contrast with the
            // heuristic five-term rule which is happy to terminate as
            // soon as the actual numerical tail is small.  Empirically
            // for the c=100, q=3, tol=1e-8 adversarial case the
            // certified bound needs about 1.5 extra c worth of terms
            // beyond what the heuristic uses.  We add 2*c so that the
            // typical case has comfortable headroom; the user can
            // always specify max_order explicitly for tight memory.
            if (stopping_mode == StoppingMode::ConservativeTailEstimate) {
                safety += static_cast<int>(std::ceil(2.0 * c_d));
            }
            max_order_ = std::max(N_width, N_order) + safety;
        } else {
            max_order_ = max_order;
        }

        // Precompute Bessel ratios up to max_order_.
        precompute_bessel_ratios();
    }
    
    /**
     * @brief Initialize with a starting set of nodes.
     *
     * Computes exp[x_0, ..., x_q] for the supplied nodes and pre-fills
     * the Chebyshev column stack to depth max_order_, so that every
     * subsequent add_node()/remove_last_node() costs O(max_order_)
     * without any layer creation.  Memory at this point is
     *     (max_order_ + 1) * (q + 1) * sizeof(T).
     *
     * For all-equal node sets, the routine short-circuits the
     * Chebyshev sum: S is computed in log space from the closed
     * form
     *     log S = (x_0 - d) + q log c - log(q!) - log I_0(c),
     * and the full value (when form_full is true) returns
     * exp(x_0) / q! directly.  This bypasses the Chebyshev path
     * entirely, both because the closed form is exact and because
     * the cancellation in the Chebyshev sum at all-equal nodes in a
     * wide interval would trip the conditioning gate.  This is also
     * the path that handles arbitrarily wide constructor intervals
     * at all-equal nodes without overflow.
     *
     * For distinct nodes, the Chebyshev sum is evaluated normally and
     * the series-conditioning gate fires whenever the constructor
     * interval is too wide for the active node spread in the current
     * precision.
     *
     * @param initial_nodes Vector of initial nodes (must be in [a, b]).
     * @return exp[x_0, ..., x_q].
     */
private:
    /**
     * @brief Private helper that builds the Chebyshev column stack
     *        for a starting node set and returns the normalized sum.
     *
     * Public wrappers initialize() and initialize_normalized() call
     * this with form_full = true / false respectively, so that the
     * normalized path never forms the e^d * I_0(c) / c^q prefactor
     * (which can overflow at extreme c).
     */
    T initialize_compute_S_(const std::vector<T>& initial_nodes, bool form_full) {
        // A1: transactional initialization.  All mutating writes go into
        // local variables; the member state is only updated at the very
        // end, after validation, cache build, and convergence succeed.
        // A throw at any earlier step leaves the object exactly as it
        // was before the call (whether previously initialized or not).
        //
        // A3: the diagnostic field last_used_order_ is set by the inner
        // evaluator if it runs, so guard it with a save/restore RAII
        // helper so that even a convergence-failure throw cannot leak
        // through.  The same applies to the (also diagnostic) flag
        // certified_B_q_, which would be left pointing at a stale
        // certified-bound cache after a failed initialize at a new q.
        struct DiagGuard {
            IncrementalExpDivDiff* self;
            int  saved_last_used_order;
            int  saved_certified_B_q;
            bool committed = false;
            ~DiagGuard() {
                if (committed) return;
                self->last_used_order_ = saved_last_used_order;
                self->certified_B_q_   = saved_certified_B_q;
            }
        };
        DiagGuard dg{ this, last_used_order_, certified_B_q_ };

        if (initial_nodes.empty()) {
            throw std::invalid_argument("Initial node vector cannot be empty");
        }
        const int q_init = static_cast<int>(initial_nodes.size()) - 1;
        if (q_init > max_q_) {
            throw std::invalid_argument(
                "initialize: number of initial nodes exceeds max_q_; "
                "construct with a larger max_q.");
        }

        // Validate node ranges and compute normalized coordinates into
        // local buffers; no member mutation yet.
        std::vector<T> new_nodes = initial_nodes;
        std::vector<T> new_normalized(q_init + 1);
        for (int i = 0; i <= q_init; ++i) {
            if (!std::isfinite(static_cast<long double>(new_nodes[i])) ||
                new_nodes[i] < a_ || new_nodes[i] > b_) {
                throw std::out_of_range(
                    "Node must be finite and in [a, b]");
            }
            new_normalized[i] = (new_nodes[i] - d_) / c_;
        }

        // Build the new Chebyshev column stack into a local cache.
        // initialize_base_layers writes member cached_D_, so we use a
        // local builder instead.
        std::vector<std::vector<T>> new_cached;
        new_cached.reserve(static_cast<size_t>(max_order_) + 1);
        {
            // Layer 0: D^(0)_k = 1 if k == 0 else 0.
            std::vector<T> D0(q_init + 1, traits::zero());
            D0[0] = traits::one();
            new_cached.push_back(std::move(D0));
            // Layer 1: D^(1)_0 = y_0, D^(1)_1 = 1 (if q >= 1), else 0.
            std::vector<T> D1(q_init + 1, traits::zero());
            if (q_init >= 0) D1[0] = new_normalized[0];
            if (q_init >= 1) D1[1] = traits::one();
            new_cached.push_back(std::move(D1));
        }
        while (static_cast<int>(new_cached.size()) <= max_order_) {
            const std::vector<T>& D_n   = new_cached[new_cached.size() - 1];
            const std::vector<T>& D_nm1 = new_cached[new_cached.size() - 2];
            new_cached.push_back(
                compute_next_layer_local(D_n, D_nm1, q_init, new_normalized));
        }

        // Evaluate the sum into a local S.  evaluate_from_cache_local
        // is a const-style read-only sibling of evaluate_from_cache
        // that operates on the supplied local cache, so it does not
        // require us to have mutated the member state yet.  It may
        // throw on convergence failure; if it does, no member fields
        // have been touched.
        T new_S;
        // All-equal nodes (including q == 0 as a special case): the
        // normalized sum has a closed form
        //     S = exp(c y_0) c^q / (q! I_0(c)),  y_0 = (x_0 - d)/c,
        // which collapses to exp(x_0 - d)/I_0(c) when q == 0.  We
        // form it in log space so neither factor overflows at large
        // c; the linear S may legitimately underflow to 0 when the
        // interval is much wider than the node spread.  This
        // bypasses evaluate_from_cache_local, whose Chebyshev sum
        // would be a near-zero cancellation residue dominated by
        // round-off, and would correctly trip the conditioning gate.
        bool init_all_equal = true;
        for (size_t i = 1; i < new_nodes.size(); ++i) {
            if (new_nodes[i] != new_nodes[0]) {
                init_all_equal = false; break;
            }
        }
        if (init_all_equal) {
            const T x0 = new_nodes[0];
            long double log_fact = 0.0L;
            for (int k = 2; k <= q_init; ++k) {
                log_fact += std::log(static_cast<long double>(k));
            }
            const long double log_S_ld =
                static_cast<long double>(x0 - d_)
                + static_cast<long double>(q_init) *
                  std::log(static_cast<long double>(c_))
                - log_fact
                - static_cast<long double>(log_I0_value(c_));
            const long double log_max_ld =
                std::log(static_cast<long double>(std::numeric_limits<T>::max()));
            if (log_S_ld > log_max_ld) {
                new_S = std::numeric_limits<T>::infinity();
            } else if (log_S_ld < std::log(static_cast<long double>(
                std::numeric_limits<T>::min())) - 10.0L) {
                new_S = traits::zero();
            } else {
                new_S = static_cast<T>(std::exp(log_S_ld));
            }
        } else {
            new_S = evaluate_from_cache_local(new_cached, q_init,
                                              /*normalized=*/true);
        }

        // -------- COMMIT --------
        // From here on no further operations can throw (move-assign
        // and POD assignment are noexcept).  We have a complete,
        // self-consistent new state.
        current_nodes_      = std::move(new_nodes);
        normalized_nodes_   = std::move(new_normalized);
        current_q_          = q_init;
        cached_D_           = std::move(new_cached);
        current_S_          = new_S;
        is_initialized_     = true;
        certified_B_q_      = -1;   // q changed; invalidate B(n,q) cache

        if (form_full) {
            // A2: all-equal closed form (includes q == 0 as a special
            // case).  exp[x, ..., x] = exp(x_0) / q! exactly, no
            // round-trip through I_0(c) needed -- so this branch is
            // immune to the overflow at large c that scale_full hits.
            bool all_equal = true;
            const T x0 = current_nodes_[0];
            for (int i = 1; i <= q_init; ++i) {
                if (current_nodes_[i] != x0) { all_equal = false; break; }
            }
            if (all_equal) {
                // A3 R10: lgamma-based form so q! does not overflow before
                // exp(x_0) / q! does.  See ExpDivDiff::evaluate.
                const long double log_v_le =
                    static_cast<long double>(x0) -
                    std::lgamma(static_cast<long double>(q_init + 1));
                current_value_ = static_cast<T>(std::exp(log_v_le));
            } else {
                current_value_ = scale_full(current_S_, current_q_);
            }
            full_value_valid_ = true;
        } else {
            current_value_    = std::numeric_limits<T>::quiet_NaN();
            full_value_valid_ = false;
        }
        dg.committed = true;
        return current_S_;
    }

public:
    /**
     * @brief Initialize with a starting set of nodes and return the
     *        full exponential divided difference exp[x_0,...,x_q].
     *
     * See initialize_compute_S_ for the column-build / cache-fill
     * details.  This wrapper additionally forms the full value
     *     e^d I_0(c) c^{-q} S
     * via scale_full.
     */
    T initialize(const std::vector<T>& initial_nodes) {
        initialize_compute_S_(initial_nodes, /*form_full=*/true);
        return current_value_;
    }

    /**
     * @brief Initialize and return the normalized sum S only.
     *
     * Does not form the full e^d I_0(c) c^{-q} prefactor, so it is
     * usable at extreme c where that prefactor would overflow.
     * current_value() will throw until compute_full_value() is called
     * or a non-normalized API path is taken.
     */
    T initialize_normalized(const std::vector<T>& initial_nodes) {
        return initialize_compute_S_(initial_nodes, /*form_full=*/false);
    }

private:
    /**
     * @brief Apply the full-value scaling e^d * I_0(c) / c^q to a
     *        normalized Chebyshev sum.
     */
    T scale_full(T S, int q) const {
        const T I0_c = modified_bessel_I0(c_);
        const T scale = traits::exp(d_) * I0_c / traits::pow(c_, q);
        return scale * S;
    }

    /**
     * @brief Append a node to the current set.
     *
     * Implements the column-update recurrence of Sec. 5.1 of the paper.
     * Under the A1 design the Chebyshev column stack already has
     * max_order_ layers from initialize(); this method only widens
     * each layer by one column (the new node), so the worst-case cost
     * is O(max_order_) with no layer-creation overhead.
     *
     * **Exception safety (A4).**  All mutations to the object's state
     * (current_nodes_, normalized_nodes_, current_q_, the column
     * extension of every cached layer, current_value_, current_S_)
     * occur through a transactional path: the data is built first,
     * and the object is only committed at the end via swap with the
     * pre-update snapshot.  If any step throws (convergence failure,
     * allocation, get_bessel_ratio out of range, ...) the destructor
     * of the local guard restores the object to its pre-call state.
     * Important for QMC samplers that catch convergence exceptions
     * around a rejected move.
     *
     * Throws std::invalid_argument if q would exceed max_q_.
     *
     * @param new_node Node to append (must lie in [a, b]).
     * @return Updated exp[x_0, ..., x_q, new_node].
     */
    /**
     * @brief Private helper that performs the column-update work and
     *        returns the normalized Chebyshev sum S.
     *
     * Public wrappers add_node() and add_node_normalized() call this
     * with form_full = true / false respectively.  Splitting the work
     * this way ensures add_node_normalized() never forms the full
     * e^d * I_0(c) * c^{-q} prefactor (which can overflow at extreme
     * c even when S itself is fine).
     *
     * Throws std::invalid_argument if q would exceed max_q_.
     *
     * @param new_node Node to append (must lie in [a, b]).
     * @param form_full Whether to additionally form the full value
     *                  via scale_full.  When false, current_value_ is
     *                  marked invalid until the next full-value call.
     * @return S, the normalized Chebyshev sum at the updated order.
     */
    T add_node_compute_S_(T new_node, bool form_full) {
        if (!is_initialized_) {
            throw std::runtime_error("Must call initialize() before add_node()");
        }
        if (!std::isfinite(static_cast<long double>(new_node)) ||
            new_node < a_ || new_node > b_) {
            throw std::out_of_range(
                "Node must be finite and in [a, b]");
        }
        if (current_q_ + 1 > max_q_) {
            throw std::invalid_argument(
                "add_node: current_q_+1 exceeds max_q_; construct with a "
                "larger max_q.");
        }

        // -- Snapshot for rollback ---------------------------------------
        // Save just enough state to restore on exception.  The cached
        // layer widths are uniform (= current_q_+1) before the call, so
        // a single integer captures their old width; the data at
        // position current_q_+1 is what we will write, so rolling back
        // is "resize each layer back to old_width".
        struct Rollback {
            IncrementalExpDivDiff* self;
            int saved_q;
            int saved_layer_width;
            T   saved_S;
            T   saved_value;
            T   saved_last_node;          // valid iff saved_pushed
            T   saved_last_normalized;    // valid iff saved_pushed
            int saved_last_used_order;
            bool saved_full_value_valid;
            bool saved_pushed = false;
            bool committed   = false;
            ~Rollback() {
                if (committed) return;
                if (saved_pushed) {
                    self->current_nodes_.pop_back();
                    self->normalized_nodes_.pop_back();
                }
                self->current_q_         = saved_q;
                self->current_S_         = saved_S;
                self->current_value_     = saved_value;
                self->last_used_order_   = saved_last_used_order;
                self->full_value_valid_  = saved_full_value_valid;
                for (auto& layer : self->cached_D_) {
                    if (static_cast<int>(layer.size()) > saved_layer_width) {
                        layer.resize(saved_layer_width);
                    }
                }
            }
        };
        Rollback rb{ this,
                     current_q_,
                     current_q_ + 1,
                     current_S_,
                     current_value_,
                     T(0), T(0),
                     last_used_order_,
                     full_value_valid_,
                     false, false };

        // -- Begin transactional update ----------------------------------
        const int old_q = current_q_;
        const T y_new = (new_node - d_) / c_;
        current_nodes_.push_back(new_node);
        normalized_nodes_.push_back(y_new);
        rb.saved_pushed = true;
        current_q_++;

        // Widen every cached layer by one entry; the new entry will be
        // filled below.  Resize is the only step that can throw an
        // allocation; rollback will shrink any successful resizes back.
        for (auto& layer : cached_D_) {
            layer.resize(current_q_ + 1, traits::zero());
        }

        // D^{(0)}_{q+1} = T_0 divided difference over q+2 nodes:
        // = 1 if q+1 == 0 (impossible here, since q+1 >= 1), else 0.
        cached_D_[0][current_q_] = traits::zero();

        // D^{(1)}_{q+1}: T_1(y) = y, so its divided difference over the
        // new column is 1 if q+1 == 1, else 0.
        cached_D_[1][current_q_] = (current_q_ == 1) ? traits::one()
                                                     : traits::zero();

        // For n >= 2, the column-update recurrence is
        //   D^{(n)}_{q+1} = 2 (y_{q+1} D^{(n-1)}_{q+1} + D^{(n-1)}_q)
        //                 - D^{(n-2)}_{q+1}
        // which is just Eq. (7) with k = q+1 specialized to the new
        // column (the parent column entries D^{(n-1)}_q already live
        // in cached_D_).
        const int N_terms = static_cast<int>(cached_D_.size());
        for (int n = 2; n < N_terms; ++n) {
            cached_D_[n][current_q_] =
                traits::two() *
                    (y_new * cached_D_[n - 1][current_q_]
                           + cached_D_[n - 1][old_q])
                - cached_D_[n - 2][current_q_];
        }

        // Evaluate the normalized Chebyshev sum via the active stopping
        // rule.  This is the only step that can throw a convergence
        // failure; rollback restores the pre-call state cleanly if it
        // does.  If the appended node makes the full set all-equal,
        // the Chebyshev sum is a near-zero residue and the
        // conditioning gate would correctly fire; short-circuit with
        // the closed-form S to bypass that.
        T S;
        {
            bool aeq = true;
            const T x0 = current_nodes_[0];
            for (size_t i = 1; i < current_nodes_.size(); ++i) {
                if (current_nodes_[i] != x0) { aeq = false; break; }
            }
            if (aeq) {
                long double log_fact = 0.0L;
                for (int k = 2; k <= current_q_; ++k) {
                    log_fact += std::log(static_cast<long double>(k));
                }
                const long double log_S_ld =
                    static_cast<long double>(x0 - d_)
                    + static_cast<long double>(current_q_) *
                      std::log(static_cast<long double>(c_))
                    - log_fact
                    - static_cast<long double>(log_I0_value(c_));
                const long double log_max_ld = std::log(
                    static_cast<long double>(std::numeric_limits<T>::max()));
                if (log_S_ld > log_max_ld) {
                    S = std::numeric_limits<T>::infinity();
                } else {
                    S = static_cast<T>(std::exp(log_S_ld));
                }
            } else {
                S = evaluate_from_cache(current_q_, /*normalized=*/true);
            }
        }

        current_S_ = S;
        if (form_full) {
            // A2: all-equal closed form bypasses scale_full (which can
            // overflow at large c).  exp[x, ..., x] = exp(x_0) / q!
            // exactly.
            bool all_equal = true;
            const T x0 = current_nodes_[0];
            for (int i = 1; i <= current_q_; ++i) {
                if (current_nodes_[i] != x0) { all_equal = false; break; }
            }
            if (all_equal) {
                // A3 R10: lgamma-based form so q! does not overflow before
                // exp(x_0) / q! does.  See ExpDivDiff::evaluate.
                const long double log_v_le =
                    static_cast<long double>(x0) -
                    std::lgamma(static_cast<long double>(current_q_ + 1));
                current_value_     = static_cast<T>(std::exp(log_v_le));
            } else {
                current_value_     = scale_full(S, current_q_);
            }
            full_value_valid_  = true;
        } else {
            current_value_     = std::numeric_limits<T>::quiet_NaN();
            full_value_valid_  = false;
        }
        rb.committed = true;
        return S;
    }

public:
    /**
     * @brief Append a node and return the full divided difference.
     *
     * See add_node_compute_S_ for the column-update / RAII rollback
     * details.  This wrapper additionally forms the full value
     *     e^d I_0(c) c^{-q} S
     * via scale_full; on extreme c that prefactor can overflow even
     * when S is well-behaved.  Use add_node_normalized() in that case.
     */
    T add_node(T new_node) {
        add_node_compute_S_(new_node, /*form_full=*/true);
        return current_value_;
    }

    /**
     * @brief Append a node and return the normalized sum S only.
     *
     * No full-value scaling is performed; the e^d * I_0(c) * c^{-q}
     * prefactor (which can overflow at extreme c) is never formed.
     * On return, current_value() is invalidated (will throw if called
     * before the next full-value computation); current_normalized_value()
     * and current_S_ are the supported queries until the user runs
     * the full-value path again.
     */
    T add_node_normalized(T new_node) {
        return add_node_compute_S_(new_node, /*form_full=*/false);
    }

    /**
     * @brief Remove the most recently appended node (LIFO).
     *
     * Implements the removal half of Sec. 5.2 of the paper.  The
     * parent column was retained on append, so the reduced sum can be
     * read straight off cached_D_[*][current_q_-1] at the active
     * stopping rule.  Cost is O(max_order_).
     *
     * **Exception safety (A4).**  Computes the post-removal value
     * first by evaluating the cache at order current_q_-1 (a read,
     * not a write).  Only after that evaluation succeeds does the
     * routine commit by popping the node, decrementing current_q_,
     * and shrinking the cached layers.  If the evaluation throws,
     * the object is unchanged.
     *
     * Refuses to remove from a single-node state.
     *
     * @return Updated exp[x_0, ..., x_{q-1}].
     */
private:
    T remove_last_node_compute_S_(bool form_full) {
        if (!is_initialized_) {
            throw std::runtime_error("Must call initialize() before remove_last_node()");
        }
        if (current_q_ <= 0) {
            throw std::runtime_error(
                "remove_last_node(): refusing to remove from a single-node "
                "state; call initialize() with a new node set instead.");
        }

        // Compute the new value first.  evaluate_from_cache reads
        // cached_D_[n][q-1], which is independent of the column we
        // intend to drop.  For new_q == 0, use the closed-form
        // exp(x_0) instead of the Chebyshev sum (same reason as in
        // initialize()).
        const int new_q = current_q_ - 1;
        T new_S;
        if (new_q == 0) {
            // q == 0 closed form in log space; see initialize_compute_S_.
            const T x0     = current_nodes_[0];
            const T log_S0 = (x0 - d_) - log_I0_value(c_);
            const T log_max = traits::log(std::numeric_limits<T>::max());
            new_S = (log_S0 > log_max)
                  ? std::numeric_limits<T>::infinity()
                  : traits::exp(log_S0);
        } else {
            new_S = evaluate_from_cache(new_q, /*normalized=*/true);
        }

        // Commit: shrink the data structures.
        current_nodes_.pop_back();
        normalized_nodes_.pop_back();
        current_q_ = new_q;
        for (auto& layer : cached_D_) {
            if (static_cast<int>(layer.size()) > current_q_ + 1) {
                layer.resize(current_q_ + 1);
            }
        }
        current_S_ = new_S;
        if (form_full) {
            bool all_equal = true;
            const T x0 = current_nodes_[0];
            for (int i = 1; i <= new_q; ++i) {
                if (current_nodes_[i] != x0) { all_equal = false; break; }
            }
            if (all_equal) {
                // A3 R10: lgamma-based form so q! does not overflow before
                // exp(x_0) / q! does.  See ExpDivDiff::evaluate.
                const long double log_v_le =
                    static_cast<long double>(x0) -
                    std::lgamma(static_cast<long double>(new_q + 1));
                current_value_ = static_cast<T>(std::exp(log_v_le));
            } else {
                current_value_ = scale_full(new_S, new_q);
            }
            full_value_valid_ = true;
        } else {
            current_value_    = std::numeric_limits<T>::quiet_NaN();
            full_value_valid_ = false;
        }
        return new_S;
    }

public:
    /**
     * @brief Remove the most recently appended node; return the full
     *        divided difference at the reduced order.
     */
    T remove_last_node() {
        remove_last_node_compute_S_(/*form_full=*/true);
        return current_value_;
    }

    /**
     * @brief Remove the most recently appended node; return the
     *        normalized sum S without forming the full prefactor.
     */
    T remove_last_node_normalized() {
        return remove_last_node_compute_S_(/*form_full=*/false);
    }

    /**
     * @brief Get the current full exponential divided difference value.
     *
     * Throws if the last state-changing call was a *_normalized
     * variant: in that case the full value was not formed (deliberately,
     * to avoid the overflow-prone scale_full step at extreme c).  Call
     * compute_full_value() to opt into forming it now, or use
     * get_normalized_value() to read the normalized sum S.
     */
    T current_value() const {
        if (!is_initialized_) {
            throw std::runtime_error("Not initialized");
        }
        if (!full_value_valid_) {
            throw std::runtime_error(
                "current_value(): full value is invalid because the last "
                "state-changing call used a *_normalized API.  Either "
                "call compute_full_value() to form it now, or read "
                "current_normalized_value() / get_normalized_value().");
        }
        return current_value_;
    }

    /**
     * @brief Form the full value e^d * I_0(c) * c^{-q} * S from the
     *        currently-cached normalized sum S.
     *
     * Idempotent: if the full value is already valid this is a no-op.
     * At extreme c this can overflow; the function does not catch
     * that — overflow is what the normalized API exists to avoid.
     */
    T compute_full_value() {
        if (!is_initialized_) throw std::runtime_error("Not initialized");
        if (!full_value_valid_) {
            // All-equal closed form, including the q == 0 special
            // case: exp[x, ..., x] = exp(x_0) / q!.  This bypasses the
            // scale_full step, which can overflow at large c even when
            // the divided difference itself is finite.  For the
            // genuinely-distinct case we round-trip through S as usual.
            bool all_equal = true;
            const T x0 = current_nodes_[0];
            for (int i = 1; i <= current_q_; ++i) {
                if (current_nodes_[i] != x0) { all_equal = false; break; }
            }
            if (all_equal) {
                // A3 R10: lgamma-based form so q! does not overflow before
                // exp(x_0) / q! does.  See ExpDivDiff::evaluate.
                const long double log_v_le =
                    static_cast<long double>(x0) -
                    std::lgamma(static_cast<long double>(current_q_ + 1));
                current_value_ = static_cast<T>(std::exp(log_v_le));
            } else {
                current_value_ = scale_full(current_S_, current_q_);
            }
            full_value_valid_ = true;
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
     * Returns the normalized sum S = D^{(0)}_q + 2 * sum R_n D^{(n)}_q
     * without the e^d I_0(c) / c^q prefactor.  Useful for ratios and
     * for any caller who would overflow when forming the full value
     * (which can happen at large c regardless of how small the ratio
     * is).
     *
     * The cached S is kept current by initialize(), add_node(), and
     * remove_last_node(), so this method is O(1).
     *
     * @return Normalized sum S.
     */
    T get_normalized_value() const {
        if (!is_initialized_) {
            throw std::runtime_error("Not initialized");
        }
        return current_S_;
    }

    /**
     * @brief Select between the heuristic and certified stopping rules.
     *
     * The default is StoppingMode::HeuristicFiveTerm.  Switching to
     * StoppingMode::ConservativeTailEstimate makes every subsequent
     * convergence check use the Eq. (33) tail bound: slower per step
     * but provides a conservative floating-point estimate of the
     * analytic truncation error.  See the enum docstring for the
     * caveats -- this is not an interval-arithmetic certificate.
     *
     * Safe to call between updates; takes effect on the next
     * evaluation that consults the stopping rule.
     */
    void set_stopping_mode(StoppingMode mode) {
        stopping_mode_ = mode;
        // The certified mode caches B(n,q); invalidate so the next call
        // recomputes for the current q.
        certified_B_q_ = -1;
    }

    /// Current stopping rule.
    StoppingMode stopping_mode() const { return stopping_mode_; }

    /// max_q supplied at construction.
    int max_q() const { return max_q_; }

    /// max_order (either user-supplied or auto-chosen at construction).
    int max_order() const { return max_order_; }

    /**
     * @brief Most recent Chebyshev order n at which the stopping rule
     *        fired.  Distinct from max_order(): max_order is the cached
     *        cache depth, while last_used_order is the n at which
     *        evaluate_from_cache actually stopped on the most recent
     *        state-changing call.  Useful for reporting "n used"
     *        alongside "Nmax cached" in benchmarks.  Returns -1 if no
     *        evaluation has been performed yet.
     */
    int last_used_order() const { return last_used_order_; }

    /**
     * @brief Get the scaling factor exp(d) * I_0(c) / c^q for the
     *        current state.  Full value = scaling_factor * S.
     */
    T get_scaling_factor() const {
        if (!is_initialized_) {
            throw std::runtime_error("Not initialized");
        }
        const T I0_c = modified_bessel_I0(c_);
        return traits::exp(d_) * I0_c / traits::pow(c_, current_q_);
    }

    /**
     * @brief Reset to uninitialized state.  Clears node and cache
     *        state plus all diagnostic fields, so e.g.
     *        last_used_order() correctly reports -1 after reset.
     */
    void reset() {
        current_nodes_.clear();
        normalized_nodes_.clear();
        cached_D_.clear();
        current_q_         = -1;
        current_value_     = traits::zero();
        current_S_         = traits::zero();
        is_initialized_    = false;
        certified_B_q_     = -1;
        full_value_valid_  = false;
        last_used_order_   = -1;
    }
};

} // namespace chebyshev_incremental

#endif // INCREMENTAL_EXP_DIVDIFF_HPP
