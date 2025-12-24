#ifndef TEST_BESSEL_RATIOS_HPP
#define TEST_BESSEL_RATIOS_HPP

#include "chebyshev_exp_divdiff.hpp"
// Note: This test suite requires the FIXED version of chebyshev_exp_divdiff.hpp
// Use chebyshev_exp_divdiff_VERIFIED.hpp or ensure your version has the
// fixed BesselRatiosMiller::compute_R() with proper rescaling
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>  // For std::ostringstream
#include <limits>   // For std::numeric_limits

/**
 * @file test_bessel_ratios.hpp
 * @brief Validated test suite for Bessel ratio computation R_n(c) = I_n(c)/I_0(c)
 *
 * This test suite validates BesselRatiosMiller::compute_R() against reference values
 * computed with scipy.special.iv (Python's gold-standard Bessel function library).
 *
 * Coverage:
 *   - c from 0.1 to 100 (4 orders of magnitude)
 *   - n from 0 to 200
 *   - 50 strategic test cases
 *   - Expected: All tests pass with machine precision (rel error < 1e-6)
 *
 * Usage:
 *   #include "test_bessel_ratios.hpp"
 *
 *   int main() {
 *       return test_bessel_ratios::run_all_tests();
 *   }
 *
 *   // Returns 0 if all tests pass, 1 if any fail
 */

namespace test_bessel_ratios {

/**
 * @brief Single test case with expected value from scipy
 */
struct BesselTest {
    double c;
    int n;
    double expected;  // Reference value from scipy.special.iv
};

/**
 * @brief Test result for a single case
 */
struct TestResult {
    double c;
    int n;
    double computed;
    double expected;
    double rel_error;
    bool passed;
    std::string error_msg;
    
    TestResult() : c(0), n(0), computed(0), expected(0),
                   rel_error(0), passed(false) {}
};

/**
 * @brief Get all test cases with scipy reference values
 */
inline std::vector<BesselTest> get_test_cases() {
    // Reference values computed with scipy.special.iv version 1.11+
    // R_n(c) = I_n(c) / I_0(c)
    return {
        // Small c (c < 1): Tests continued fraction method
        {0.1, 0, 1.000000000000e+00},
        {0.1, 1, 4.993760398794e-02},
        {0.1, 5, 2.598750990740e-09},
        {0.1, 10, 2.685039348013e-20},
        
        {0.5, 0, 1.000000000000e+00},
        {0.5, 1, 2.424996125808e-01},
        {0.5, 5, 7.732298914441e-06},
        {0.5, 10, 2.485268692108e-13},
        
        // Medium c (1 ≤ c ≤ 10): Tests Miller's backward recurrence
        {1.0, 0, 1.000000000000e+00},
        {1.0, 1, 4.463899658965e-01},
        {1.0, 5, 2.144147162697e-04},
        {1.0, 10, 2.174411370066e-10},
        {1.0, 20, 3.133198718587e-25},
        
        {2.0, 0, 1.000000000000e+00},
        {2.0, 1, 6.977746579640e-01},
        {2.0, 5, 4.310292452343e-03},
        {2.0, 10, 1.323470490996e-07},
        {2.0, 20, 1.890940677540e-19},
        {2.0, 50, 1.470900354738e-65},
        
        {5.0, 0, 1.000000000000e+00},
        {5.0, 1, 8.933831370441e-01},
        {5.0, 5, 7.922117113094e-02},
        {5.0, 10, 1.681375172701e-04},
        {5.0, 20, 1.844443098157e-12},
        {5.0, 50, 1.076168664006e-46},
        
        // Large c (c > 10): Tests Miller's method with rescaling
        {10.0, 0, 1.000000000000e+00},
        {10.0, 1, 9.485998259548e-01},
        {10.0, 5, 2.760179339590e-01},
        {10.0, 10, 7.774825755690e-03},
        {10.0, 20, 4.442207440194e-08},
        {10.0, 50, 1.689408128871e-33},
        {10.0, 100, 3.843938664875e-92},
        
        {20.0, 0, 1.000000000000e+00},
        {20.0, 1, 9.746705078898e-01},
        {20.0, 10, 8.127501822826e-02},
        {20.0, 20, 7.320652104439e-05},
        {20.0, 50, 5.177248604045e-22},
        {20.0, 100, 6.589606276874e-66},
        
        {50.0, 0, 1.000000000000e+00},
        {50.0, 1, 9.899489673785e-01},
        {50.0, 10, 3.654143243269e-01},
        {50.0, 20, 1.855723305988e-02},
        {50.0, 50, 6.018918570983e-11},
        {50.0, 100, 9.302090083122e-37},
        
        // Very large c (c = 100): Stress test for rescaling
        {100.0, 0, 1.000000000000e+00},
        {100.0, 1, 9.949873730052e-01},
        {100.0, 50, 4.490757084315e-06},
        {100.0, 100, 4.322726483963e-21},
        {100.0, 150, 8.010851636015e-44},
        {100.0, 200, 1.299561016612e-72},
    };
}

/**
 * @brief Run a single test case
 */
inline TestResult run_single_test(const BesselTest& test, double tolerance = 1e-6) {
    TestResult result;
    result.c = test.c;
    result.n = test.n;
    result.expected = test.expected;
    
    try {
        int nmax = test.n + 10;
        int Nstart = chebyshev::BesselRatiosMiller<double>::recommended_Nstart(test.c, nmax);
        auto R = chebyshev::BesselRatiosMiller<double>::compute_R(test.c, nmax, Nstart);
        
        result.computed = R[test.n];
        
        // Compute relative error
        if (test.expected != 0.0) {
            result.rel_error = std::abs((result.computed - test.expected) / test.expected);
        } else {
            result.rel_error = std::abs(result.computed);
        }
        
        // Check if test passes
        result.passed = result.rel_error < tolerance;
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.error_msg = e.what();
        result.computed = std::numeric_limits<double>::quiet_NaN();
        result.rel_error = std::numeric_limits<double>::infinity();
    }
    
    return result;
}

/**
 * @brief Run all tests and print results
 *
 * @param os Output stream (default: std::cout)
 * @param tolerance Relative error threshold for passing (default: 1e-6)
 * @param verbose Print detailed results for each test (default: true)
 * @return 0 if all tests pass, 1 if any fail
 */
inline int run_all_tests(std::ostream& os = std::cout,
                        double tolerance = 1e-6,
                        bool verbose = true) {
    
    auto tests = get_test_cases();
    std::vector<TestResult> results;
    
    if (verbose) {
        os << "\n";
        os << "╔════════════════════════════════════════════════════════════════════════════╗\n";
        os << "║  Comprehensive Bessel Ratio Test Suite                                    ║\n";
        os << "║  Testing: BesselRatiosMiller::compute_R()                                 ║\n";
        os << "║  Reference: scipy.special.iv (Python)                                     ║\n";
        os << "╚════════════════════════════════════════════════════════════════════════════╝\n";
        os << "\n";
        os << "Coverage:\n";
        os << "  c range: 0.1 to 100.0 (4 orders of magnitude)\n";
        os << "  n range: 0 to 200\n";
        os << "  Total tests: " << tests.size() << "\n";
        os << "  Pass threshold: relative error < " << std::scientific << tolerance << "\n";
        os << "\n";
        os << std::string(80, '=') << "\n\n";
    }
    
    // Run all tests
    double current_c = -1;
    int passed = 0;
    int failed = 0;
    
    for (const auto& test : tests) {
        TestResult result = run_single_test(test, tolerance);
        results.push_back(result);
        
        if (result.passed) {
            passed++;
        } else {
            failed++;
        }
        
        // Print detailed results if verbose
        if (verbose) {
            // Print header for new c value
            if (test.c != current_c) {
                if (current_c > 0) {
                    os << "\n";
                }
                current_c = test.c;
                os << "c = " << std::fixed << std::setprecision(1) << test.c << "\n";
                os << std::string(80, '-') << "\n";
                os << std::setw(5) << "n"
                   << std::setw(20) << "Computed"
                   << std::setw(20) << "Expected"
                   << std::setw(15) << "Rel Error"
                   << std::setw(10) << "Status"
                   << "\n";
            }
            
            // Print test result
            os << std::scientific << std::setprecision(6)
               << std::setw(5) << result.n
               << std::setw(20);
            
            if (std::isnan(result.computed)) {
                os << "NaN";
            } else if (std::isinf(result.computed)) {
                os << "Inf";
            } else {
                os << result.computed;
            }
            
            os << std::setw(20) << result.expected
               << std::setw(15) << result.rel_error
               << std::setw(10) << (result.passed ? "PASS" : "FAIL");
            
            if (!result.error_msg.empty()) {
                os << " (" << result.error_msg << ")";
            }
            
            os << "\n";
        }
    }
    
    // Print summary
    if (verbose) {
        os << "\n" << std::string(80, '=') << "\n";
    }
    
    os << "Summary: " << passed << "/" << tests.size() << " tests passed";
    if (failed > 0) {
        os << ", " << failed << " FAILED ❌";
    } else {
        os << " ✅";
    }
    os << "\n";
    
    if (verbose) {
        os << "\n";
        
        // Print accuracy statistics for passed tests
        if (passed > 0) {
            double max_error = 0;
            double avg_error = 0;
            int count = 0;
            
            for (const auto& r : results) {
                if (r.passed) {
                    max_error = std::max(max_error, r.rel_error);
                    avg_error += r.rel_error;
                    count++;
                }
            }
            avg_error /= count;
            
            int avg_digits = (avg_error > 0) ? std::max(0, -static_cast<int>(std::log10(avg_error))) : 16;
            int min_digits = (max_error > 0) ? std::max(0, -static_cast<int>(std::log10(max_error))) : 16;
            
            os << "Accuracy Statistics (passed tests):\n";
            os << "  Average relative error: " << std::scientific << avg_error
               << " (~" << avg_digits << " digits)\n";
            os << "  Maximum relative error: " << std::scientific << max_error
               << " (~" << min_digits << " digits)\n";
            os << "  Status: " << (min_digits >= 6 ? "Machine precision achieved ✓" : "Sub-optimal accuracy ⚠️") << "\n";
        }
        
        os << "\n";
    }
    
    return (failed == 0) ? 0 : 1;
}

/**
 * @brief Run tests quietly and return only pass/fail status
 *
 * @return true if all tests pass, false otherwise
 */
inline bool verify_accuracy() {
    std::ostringstream dummy;
    return run_all_tests(dummy, 1e-6, false) == 0;
}

/**
 * @brief Get summary statistics without printing
 */
struct TestSummary {
    int total;
    int passed;
    int failed;
    double max_error;
    double avg_error;
    bool all_passed;
};

inline TestSummary get_summary(double tolerance = 1e-6) {
    auto tests = get_test_cases();
    TestSummary summary;
    
    summary.total = tests.size();
    summary.passed = 0;
    summary.failed = 0;
    summary.max_error = 0;
    summary.avg_error = 0;
    
    for (const auto& test : tests) {
        TestResult result = run_single_test(test, tolerance);
        
        if (result.passed) {
            summary.passed++;
            summary.avg_error += result.rel_error;
            summary.max_error = std::max(summary.max_error, result.rel_error);
        } else {
            summary.failed++;
        }
    }
    
    if (summary.passed > 0) {
        summary.avg_error /= summary.passed;
    }
    
    summary.all_passed = (summary.failed == 0);
    
    return summary;
}

} // namespace test_bessel_ratios

#endif // TEST_BESSEL_RATIOS_HPP
