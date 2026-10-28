#ifndef MCCURDY_TEMPLATED_HPP
#define MCCURDY_TEMPLATED_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

/**
 * @file mccurdy.hpp
 * @brief Full McCurdy-Ng-Parlett (1984) algorithm for divided differences
 *        of the exponential function.
 *
 * Implements the Opitz matrix-exponential formulation with spectral
 * shifting, Taylor approximation of the scaled matrix, and repeated
 * squaring, following
 *
 *   A. McCurdy, K. C. Ng, B. N. Parlett,
 *   "Accurate Computation of Divided Differences of the Exponential
 *    Function," Math. Comp. 43(168), 1984, pp. 501-528.
 *
 * Algorithm
 * ---------
 *   1. Build the (q+1) x (q+1) Opitz bidiagonal matrix J with
 *      diagonal entries x_0, ..., x_q and superdiagonal entries 1.
 *      Opitz's theorem says (f(J))_{i,j} = f[x_i, ..., x_j] for any
 *      analytic f and i <= j. In particular, the divided differences
 *      of exp lie in the first row of exp(J).
 *
 *   2. Spectral shift: with d = (min_i x_i + max_i x_i)/2, set
 *      M = J - d I. Eigenvalues of M lie in [-c, c] where c is the
 *      half-width of the node set, so the bulk of the magnitude of
 *      exp(J) is concentrated in the scalar factor e^d.
 *
 *   3. Choose an integer s >= 0 such that ||M||_inf / 2^s <= 1/2.
 *      Replace M by M/2^s.
 *
 *   4. Compute exp(M) by truncated Taylor series. With ||M||_inf <= 1/2
 *      the kth term has norm <= (1/2)^k / k!, so the series converges
 *      to working precision in O(log(1/eps)) terms.
 *
 *   5. Recover exp(J - dI) = (exp(M))^{2^s} by squaring s times.
 *
 *   6. Multiply by e^d to get exp(J), then read divided differences
 *      from the first row: exp[x_0,...,x_k] = (exp(J))_{0,k}.
 *
 * The matrix is upper triangular throughout (Opitz J is upper
 * bidiagonal; powers and exponentials of upper-triangular matrices
 * are upper triangular). The implementation exploits this in the
 * multiply routine, giving an effective cost of about q^3 / 6 per
 * multiply and an overall O(q^3 log(1+c) / 6) wall-clock cost. The
 * paper quotes O(q^2 log(1+c)) using a more aggressive structured
 * multiply that we do not implement here; for the regimes used in
 * the accompanying paper's experiments this is not the bottleneck.
 *
 * Numerical properties
 * --------------------
 *   - No node-difference subtractions appear, so the catastrophic
 *     cancellation that ruins the classical recurrence at clustered
 *     nodes does not happen here.
 *   - Coincident nodes are handled automatically: the Opitz matrix
 *     becomes a shifted nilpotent and the Taylor series terminates
 *     after q+1 terms exactly.
 *   - Stability is governed by the squaring step, which is well
 *     behaved when ||M_scaled||_inf is small.
 *
 * Public API
 * ----------
 *   mccurdy_templated::ExpDivDiff<Real>::compute(nodes)        -> Real
 *   mccurdy_templated::ExpDivDiff<Real>::compute_all(nodes)    -> vector<Real>
 *
 * Both are stateless static methods; the class is templated on the
 * floating-point type.
 */

namespace mccurdy_templated {

// ---------------------------------------------------------------------------
// Numeric traits.
// ---------------------------------------------------------------------------
template<typename Real>
struct numeric_traits {
    static Real abs(const Real& x) { return std::abs(x); }
    static Real exp(const Real& x) { return std::exp(x); }
    static Real max(const Real& a, const Real& b) { return std::max(a, b); }
    static constexpr Real zero() { return Real(0); }
    static constexpr Real one()  { return Real(1); }
    static constexpr Real half() { return Real(0.5); }
    static Real epsilon() { return std::numeric_limits<Real>::epsilon(); }
};

namespace detail {

// ---------------------------------------------------------------------------
// Minimal matrix utilities. We use std::vector<std::vector<Real>> for clarity
// since the matrices here are small and the cost is dominated by matrix
// multiplications, not by container overhead.
// ---------------------------------------------------------------------------
template<typename Real>
using Matrix = std::vector<std::vector<Real>>;

template<typename Real>
inline Matrix<Real> zeros(int n) {
    return Matrix<Real>(n, std::vector<Real>(n, Real(0)));
}

template<typename Real>
inline Matrix<Real> identity(int n) {
    Matrix<Real> I = zeros<Real>(n);
    for (int i = 0; i < n; ++i) I[i][i] = Real(1);
    return I;
}

// Infinity norm: max over rows of the sum of absolute row entries.
template<typename Real>
inline Real inf_norm(const Matrix<Real>& A) {
    using traits = numeric_traits<Real>;
    Real m = traits::zero();
    for (const auto& row : A) {
        Real s = traits::zero();
        for (const Real& x : row) s += traits::abs(x);
        if (s > m) m = s;
    }
    return m;
}

// Upper-triangular matrix product C = A * B. Both A and B are assumed
// upper triangular; this is true throughout the MNP algorithm because
// the Opitz matrix is upper bidiagonal and the entire flow (Taylor
// terms, exponential, repeated squarings) preserves upper-triangular
// structure.
template<typename Real>
inline Matrix<Real> mul_upper(const Matrix<Real>& A, const Matrix<Real>& B) {
    const int n = static_cast<int>(A.size());
    Matrix<Real> C = zeros<Real>(n);
    for (int i = 0; i < n; ++i) {
        for (int k = i; k < n; ++k) {
            const Real a = A[i][k];
            if (a == Real(0)) continue;
            for (int j = k; j < n; ++j) {
                C[i][j] += a * B[k][j];
            }
        }
    }
    return C;
}

// A += c * B, upper triangle only.
template<typename Real>
inline void axpy_upper(Matrix<Real>& A, const Real& c, const Matrix<Real>& B) {
    const int n = static_cast<int>(A.size());
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j)
            A[i][j] += c * B[i][j];
}

// A := c * A, upper triangle only.
template<typename Real>
inline void scale_upper(Matrix<Real>& A, const Real& c) {
    const int n = static_cast<int>(A.size());
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j)
            A[i][j] *= c;
}

} // namespace detail


// ---------------------------------------------------------------------------
// Full MNP exponential-divided-difference engine.
// ---------------------------------------------------------------------------
template<typename Real = double>
class ExpDivDiff {
public:
    using traits = numeric_traits<Real>;

    /// Return the single divided difference exp[x_0, ..., x_{n-1}].
    static Real compute(const std::vector<Real>& x) {
        auto all = compute_all(x);
        return all.back();
    }

    /// Return all divided differences:
    ///   result[k] = exp[x_0, ..., x_k], for k = 0, ..., n-1.
    static std::vector<Real> compute_all(const std::vector<Real>& x) {
        const int n = static_cast<int>(x.size());
        if (n == 0) throw std::invalid_argument("MNP: empty node vector");
        if (n == 1) return { traits::exp(x[0]) };

        // -----------------------------------------------------------
        // Step 1-2: Build M = J - d*I.
        //   J is upper bidiagonal with diagonal x_i and superdiag 1.
        //   d is chosen as the midpoint of the node set so that the
        //   shifted diagonal entries x_i - d lie in [-c, c] where c
        //   is the node-set half-width.
        // -----------------------------------------------------------
        const Real min_x = *std::min_element(x.begin(), x.end());
        const Real max_x = *std::max_element(x.begin(), x.end());
        const Real d = (min_x + max_x) / Real(2);

        // Shifted nodes z_i = x_i - d lie in [-c, c].
        std::vector<Real> z(n);
        Real c = traits::zero();
        for (int i = 0; i < n; ++i) {
            z[i] = x[i] - d;
            c = traits::max(c, traits::abs(z[i]));
        }

        // -----------------------------------------------------------
        // Step 3: Choose s with c / 2^s <= 1/2 and scale the NODES
        // (not the superdiagonal).  Halving is exact in binary
        // floating point.
        //
        // Revision note: an earlier version scaled the whole Opitz
        // matrix, superdiagonal included.  The (i,j) entry of the
        // scaled exponential is then 2^{-s(j-i)} exp[z_i/2^s, ...],
        // which underflows for large s*(j-i) and was the main source
        // of the spurious "MNP failures" at large c and q.  Keeping
        // the superdiagonal equal to 1 means every matrix entry below
        // is itself a divided difference, E_{ij} = exp[w_i, ..., w_j],
        // of magnitude ~ 1/(j-i)!.
        // -----------------------------------------------------------
        int s = 0;
        while (c > traits::half()) {
            ++s;
            c *= traits::half();
            for (Real& zi : z) zi *= traits::half();
        }

        // -----------------------------------------------------------
        // Step 4: Taylor series for exp(M), M = Opitz(z) with unit
        // superdiagonal, exploiting the bidiagonal structure of M:
        //   (term * M)_{ij} = term_{ij} z_j + term_{i,j-1},
        // so each Taylor step costs O(n^2) instead of O(n^3).
        //
        // Revision note: the earlier stopping test was norm-wise,
        // ||term|| <= tol ||E||.  Entries far from the diagonal are
        // ~1/(j-i)! and much smaller than ||E|| = O(1), so they were
        // not converged (for j-i > ~13 they were still exactly zero
        // when the loop stopped).  We now require entry-wise
        // convergence, |term_{ij}| <= tol |E_{ij}| for all i <= j,
        // which needs at least n-1 terms.
        // -----------------------------------------------------------
        detail::Matrix<Real> E    = detail::identity<Real>(n);
        detail::Matrix<Real> term = detail::identity<Real>(n);

        const int  MAX_TERMS = n + 200;
        const Real tol = traits::epsilon();
        bool converged = false;

        for (int k = 1; k <= MAX_TERMS; ++k) {
            const Real inv_k = traits::one() / Real(k);
            for (int i = 0; i < n; ++i) {
                // Sweep j downward so term_{i,j-1} is still the old value.
                for (int j = n - 1; j >= i; --j) {
                    Real v = term[i][j] * z[j];
                    if (j > i) v += term[i][j - 1];
                    term[i][j] = v * inv_k;
                }
            }
            bool small = (k >= n - 1);
            for (int i = 0; i < n; ++i) {
                for (int j = i; j < n; ++j) {
                    E[i][j] += term[i][j];
                    if (small && traits::abs(term[i][j]) >
                                 tol * traits::abs(E[i][j])) {
                        small = false;
                    }
                }
            }
            if (small) { converged = true; break; }
        }
        if (!converged) {
            throw std::runtime_error(
                "MNP: Taylor series for exp(M_scaled) did not converge in "
                "MAX_TERMS iterations; this should not happen for "
                "max_i |z_i| <= 1/2.");
        }

        // -----------------------------------------------------------
        // Step 5: Undo the scaling with the divided-difference form of
        // squaring.  If F_{ij} = exp[w_i, ..., w_j] with unit
        // superdiagonal, then for the doubled nodes 2w,
        //   exp[2w_i, ..., 2w_j] = 2^{-(j-i)} sum_{k=i}^{j} F_{ik} F_{kj},
        // which follows from exp(2 J_w) = exp(J_w)^2 and the diagonal
        // similarity diag(2^i) that maps superdiagonal 2 back to 1.
        // Only the first row is needed after the final squaring.
        // -----------------------------------------------------------
        std::vector<Real> pow2inv(n);
        pow2inv[0] = traits::one();
        for (int m = 1; m < n; ++m) pow2inv[m] = pow2inv[m - 1] * traits::half();

        for (int r = 0; r < s; ++r) {
            const bool last = (r == s - 1);
            const int rows = last ? 1 : n;
            detail::Matrix<Real> F = detail::zeros<Real>(n);
            for (int i = 0; i < rows; ++i) {
                for (int k = i; k < n; ++k) {
                    const Real a = E[i][k];
                    if (a == Real(0)) continue;
                    for (int j = k; j < n; ++j) F[i][j] += a * E[k][j];
                }
                for (int j = i; j < n; ++j) F[i][j] *= pow2inv[j - i];
            }
            E.swap(F);
        }

        // -----------------------------------------------------------
        // Step 6: Multiply by e^d. The result is exp(J).
        // -----------------------------------------------------------
        const Real e_d = traits::exp(d);
        detail::scale_upper(E, e_d);

        // -----------------------------------------------------------
        // Read divided differences from the first row of exp(J).
        // -----------------------------------------------------------
        std::vector<Real> result(n);
        for (int k = 0; k < n; ++k) result[k] = E[0][k];
        return result;
    }
};

} // namespace mccurdy_templated

#endif // MCCURDY_TEMPLATED_HPP
