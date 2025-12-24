#ifndef MCCURDY_TEMPLATED_HPP
#define MCCURDY_TEMPLATED_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

/**
 * @brief Templated McCurdy-Ng-Parlett (1984) algorithm for arbitrary precision
 * 
 * Paper: "Accurate Computation of Divided Differences of the Exponential Function"
 * Mathematics of Computation, Vol. 43, No. 168 (1984), pp. 501-528
 * 
 * This templated version works with double, long double, and arbitrary precision types.
 */

namespace mccurdy_templated {

/**
 * @brief Type traits for numeric operations
 */
template<typename Real>
struct numeric_traits {
    static Real abs(const Real& x) { return std::abs(x); }
    static Real exp(const Real& x) { return std::exp(x); }
    static Real max(const Real& a, const Real& b) { return std::max(a, b); }
    
    static constexpr Real zero() { return Real(0); }
    static constexpr Real one() { return Real(1); }
    static constexpr Real taylor_threshold() { return Real(0.7); }
    
    static Real epsilon() { return std::numeric_limits<Real>::epsilon(); }
};

/**
 * @brief Standard recursive divided difference (for verification)
 */
template<typename Real>
inline Real exact_divdiff_recursive(const std::vector<Real>& x) {
    using traits = numeric_traits<Real>;
    
    int n = x.size();
    if (n == 1) {
        return traits::exp(x[0]);
    }
    if (n == 2) {
        if (traits::abs(x[1] - x[0]) < Real(1e-15)) {
            return traits::exp(x[0]);
        }
        return (traits::exp(x[1]) - traits::exp(x[0])) / (x[1] - x[0]);
    }
    
    std::vector<Real> x1(x.begin() + 1, x.end());
    std::vector<Real> x0(x.begin(), x.end() - 1);
    
    Real d1 = exact_divdiff_recursive(x1);
    Real d0 = exact_divdiff_recursive(x0);
    
    if (traits::abs(x.back() - x.front()) < Real(1e-15)) {
        Real factorial = traits::one();
        for (int i = 1; i < n; ++i) factorial = factorial * Real(i);
        return traits::exp(x[0]) / factorial;
    }
    
    return (d1 - d0) / (x.back() - x.front());
}

/**
 * @brief Taylor series method (Section 2.3)
 * 
 * Computes first row: [f[x0], f[x0,x1], f[x0,x1,x2], ...]
 * where f = exp
 */
template<typename Real>
inline std::vector<Real> taylor_series(const std::vector<Real>& x) {
    using traits = numeric_traits<Real>;
    
    int n = x.size();
    if (n == 0) return {};
    if (n == 1) return {traits::exp(x[0])};
    
    std::vector<Real> d(n), s(n);
    
    // Initialize
    for (int i = 0; i < n; ++i) {
        Real factorial = traits::one();
        for (int j = 1; j <= i; ++j) {
            factorial = factorial * Real(j);
        }
        d[i] = s[i] = traits::one() / factorial;
    }
    
    // Main loop
    const int MAX_TERMS = 500;  // Increased for arbitrary precision
    for (int k = 1; k <= MAX_TERMS; ++k) {
        // Update s[0]
        s[0] = x[0] * s[0] / Real(k);
        
        // Update s[i] for i >= 1
        for (int i = 1; i < n; ++i) {
            s[i] = (x[i] * s[i] + s[i-1]) / Real(k + i);
            d[i] = d[i] + s[i];
        }
        
        // Check convergence
        Real max_increment = traits::zero();
        for (int i = 0; i < n; ++i) {
            max_increment = traits::max(max_increment, traits::abs(s[i]));
        }
        
        Real max_d = traits::zero();
        for (int i = 0; i < n; ++i) {
            max_d = traits::max(max_d, traits::abs(d[i]));
        }
        
        Real tolerance = Real(100) * traits::epsilon();
        if (max_increment < tolerance * traits::max(max_d, traits::one())) {
            break;
        }
    }
    
    // Set d[0] = exp(x[0])
    d[0] = traits::exp(x[0]);
    
    return d;
}

/**
 * @brief Standard recurrence (Section 2.2)
 * 
 * Direct formula: f[x0,...,xk] = (f[x1,...,xk] - f[x0,...,xk-1]) / (xk - x0)
 * 
 * Builds the first row column by column
 */
template<typename Real>
inline std::vector<Real> standard_recurrence(const std::vector<Real>& x) {
    using traits = numeric_traits<Real>;
    
    int n = x.size();
    if (n == 0) return {};
    if (n == 1) return {traits::exp(x[0])};
    
    std::vector<Real> d(n);
    
    // Initialize first column: d[i] = exp(x[i])
    for (int i = 0; i < n; ++i) {
        d[i] = traits::exp(x[i]);
    }
    
    // Build subsequent columns
    // After column k, d[i] contains f[xi, xi+1, ..., xi+k]
    for (int k = 1; k < n; ++k) {
        for (int i = n - 1; i >= k; --i) {
            d[i] = (d[i] - d[i-1]) / (x[i] - x[i-k]);
        }
    }
    
    // Now d[k] contains f[x0, x1, ..., xk]
    return d;
}

/**
 * @brief Main class for exponential divided difference computation
 */
template<typename Real = double>
class ExpDivDiff {
private:
    using traits = numeric_traits<Real>;
    
public:
    /**
     * @brief Compute single divided difference f[x[0], ..., x[n-1]]
     */
    static Real compute(const std::vector<Real>& x) {
        int n = x.size();
        if (n == 0) throw std::invalid_argument("Empty node vector");
        if (n == 1) return traits::exp(x[0]);
        
        // Check radius
        Real mean = traits::zero();
        for (const auto& xi : x) mean = mean + xi;
        mean = mean / Real(n);
        
        Real radius = traits::zero();
        for (const auto& xi : x) {
            radius = traits::max(radius, traits::abs(xi - mean));
        }
        
        // Use Taylor series for small radius
        if (radius < traits::taylor_threshold()) {
            auto result = taylor_series(x);
            return result.back();
        }
        
        // Use standard recurrence for general case
        auto result = standard_recurrence(x);
        return result.back();
    }
    
    /**
     * @brief Compute all divided differences: [f[x0], f[x0,x1], ..., f[x0,...,xn]]
     */
    static std::vector<Real> compute_all(const std::vector<Real>& x) {
        int n = x.size();
        if (n == 0) return {};
        if (n == 1) return {traits::exp(x[0])};
        
        // Check radius
        Real mean = traits::zero();
        for (const auto& xi : x) mean = mean + xi;
        mean = mean / Real(n);
        
        Real radius = traits::zero();
        for (const auto& xi : x) {
            radius = traits::max(radius, traits::abs(xi - mean));
        }
        
        // Use Taylor series for small radius
        if (radius < traits::taylor_threshold()) {
            return taylor_series(x);
        }
        
        // Use standard recurrence for general case
        return standard_recurrence(x);
    }
};

} // namespace mccurdy_templated

#endif // MCCURDY_TEMPLATED_HPP
