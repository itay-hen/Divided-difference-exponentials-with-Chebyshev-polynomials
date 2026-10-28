// =============================================================================
// test_against_reference.hpp -- compare Chebyshev and MNP against an
//                               mpmath-generated reference (parsed into
//                               long double on the C++ side).
//
// Reads reference.csv (produced by mpmath_reference.py at ~100 decimal
// digits) and, for each case, computes exp[x_0, ..., x_q] via both
// Chebyshev and MNP at a user-chosen working precision (double or long
// double).  Reports the relative error of each method against the
// reference.  Crucially, *both methods are evaluated at the same
// precision in any given run* -- there is no more "method A at double
// vs method B at long double" precision-mismatch artifact.
//
// Note: the mpmath reference is generated at ~100 decimal digits, but
// the C++ side parses each value into long double via std::stold,
// truncating to ~19 decimal digits.  This is more than enough headroom
// for the double-precision test pass, and sets a ~10^-18 floor on the
// long-double pass.  For genuinely 100-digit C++ comparisons one would
// need to add a multi-precision library (e.g., Boost.Multiprecision) on
// the C++ side; the simple long-double parse here suffices for checking
// either method's working-precision accuracy.
//
// Output to stdout: one line per case (verbose mode), plus per-method
// summaries (success rate, median rel err, max rel err) at the chosen
// precision.
// =============================================================================

#ifndef TEST_AGAINST_REFERENCE_HPP
#define TEST_AGAINST_REFERENCE_HPP

#include "chebyshev_exp_divdiff.hpp"
#include "mccurdy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace test_against_reference {

struct ReferenceCase {
    int                 case_id;
    double              b;
    int                 q;
    std::vector<double> nodes;          // node values (full precision is in `ref_str`)
    std::string         ref_str;        // mpmath value, decimal string
    long double         ref_ld;         // parsed once, used for both runs
};

inline std::vector<ReferenceCase>
load_reference_csv(const std::string& cases_path,
                   const std::string& reference_path) {
    std::ifstream fcases(cases_path);
    std::ifstream fref(reference_path);
    if (!fcases) {
        std::fprintf(stderr,
                     "test_against_reference: cannot open %s\n",
                     cases_path.c_str());
        return {};
    }
    if (!fref) {
        std::fprintf(stderr,
                     "test_against_reference: cannot open %s\n",
                     reference_path.c_str());
        return {};
    }

    std::vector<ReferenceCase> out;
    std::string cases_line, ref_line;
    std::getline(fcases, cases_line);   // header
    std::getline(fref, ref_line);       // header

    // Manifests, populated when the trailing line is parsed.
    int manifest_cases = -1;
    int manifest_ref   = -1;

    while (std::getline(fcases, cases_line)) {
        // Strip CR.
        if (!cases_line.empty() && cases_line.back() == '\r') cases_line.pop_back();
        if (cases_line.empty()) continue;
        // Detect manifest line in cases.csv: starts with "#manifest".
        if (cases_line.rfind("#manifest", 0) == 0) {
            std::istringstream cs(cases_line);
            std::string field;
            std::getline(cs, field, ',');   // "#manifest"
            std::getline(cs, field, ',');   // "n_cases=N"
            const auto eq = field.find('=');
            if (eq != std::string::npos) manifest_cases = std::stoi(field.substr(eq + 1));
            continue;
        }

        // Read a matching reference line, skipping its manifest if present.
        bool got_ref = false;
        while (std::getline(fref, ref_line)) {
            if (!ref_line.empty() && ref_line.back() == '\r') ref_line.pop_back();
            if (ref_line.empty()) continue;
            if (ref_line.rfind("#manifest", 0) == 0) {
                std::istringstream rs(ref_line);
                std::string field;
                std::getline(rs, field, ',');
                std::getline(rs, field, ',');
                const auto eq = field.find('=');
                if (eq != std::string::npos) manifest_ref = std::stoi(field.substr(eq + 1));
                continue;
            }
            got_ref = true;
            break;
        }
        if (!got_ref) {
            std::fprintf(stderr,
                "test_against_reference: cases.csv has more entries than "
                "reference.csv; reference file is incomplete.\n");
            return {};
        }

        ReferenceCase c;
        std::istringstream cs(cases_line), rs(ref_line);
        std::string field;

        // cases.csv: case_id, b, q, nodes_semicolon
        std::getline(cs, field, ','); c.case_id = std::stoi(field);
        std::getline(cs, field, ','); c.b       = std::stod(field);
        std::getline(cs, field, ','); c.q       = std::stoi(field);
        std::getline(cs, field, ',');           // nodes_semicolon
        if (!field.empty() && field.back() == '\r') field.pop_back();
        c.nodes.clear();
        std::istringstream ns(field);
        std::string tok;
        while (std::getline(ns, tok, ';')) {
            if (!tok.empty() && tok.back() == '\r') tok.pop_back();
            c.nodes.push_back(std::stod(tok));
        }

        // reference.csv: case_id, b, q, exp_dd  (string)
        int ref_case_id;
        std::getline(rs, field, ','); ref_case_id = std::stoi(field);
        std::getline(rs, field, ',');   // b
        std::getline(rs, field, ',');   // q
        std::getline(rs, field, ',');   // exp_dd string
        if (!field.empty() && field.back() == '\r') field.pop_back();
        c.ref_str = field;

        if (ref_case_id != c.case_id) {
            std::fprintf(stderr,
                "test_against_reference: case_id mismatch at line "
                "(cases=%d, reference=%d); files are inconsistent.\n",
                c.case_id, ref_case_id);
            return {};
        }

        // Parse the high-precision string into long double.  Note this
        // truncates the mpmath value to ~19 decimal digits in C++; for
        // long-double test passes, that sets the reportable rel_err
        // floor.  For double passes the ~19-digit reference is well
        // above the double-precision floor.
        c.ref_ld = std::stold(c.ref_str);

        out.push_back(std::move(c));
    }

    // After the cases loop, drain the rest of reference.csv to pick up
    // its trailing manifest line.
    while (manifest_ref < 0 && std::getline(fref, ref_line)) {
        if (!ref_line.empty() && ref_line.back() == '\r') ref_line.pop_back();
        if (ref_line.rfind("#manifest", 0) == 0) {
            std::istringstream rs(ref_line);
            std::string field;
            std::getline(rs, field, ',');
            std::getline(rs, field, ',');
            const auto eq = field.find('=');
            if (eq != std::string::npos) manifest_ref = std::stoi(field.substr(eq + 1));
            break;
        }
    }

    // Sanity-check manifests if present.
    if (manifest_cases < 0) {
        std::fprintf(stderr,
            "test_against_reference: cases.csv has no #manifest line; "
            "file may be incomplete (regenerate with mpmath_reference.py).\n");
        return {};
    }
    if (manifest_ref < 0) {
        std::fprintf(stderr,
            "test_against_reference: reference.csv has no #manifest line; "
            "file may be incomplete (regenerate with mpmath_reference.py).\n");
        return {};
    }
    if (static_cast<int>(out.size()) != manifest_cases ||
        static_cast<int>(out.size()) != manifest_ref) {
        std::fprintf(stderr,
            "test_against_reference: line count mismatch "
            "(loaded %zu, cases manifest %d, ref manifest %d).\n",
            out.size(), manifest_cases, manifest_ref);
        return {};
    }
    return out;
}

template<typename Real>
struct MethodStats {
    int    n_finite       = 0;
    int    n_total        = 0;
    double median_rel_err = 0.0;
    double max_rel_err    = 0.0;
};

template<typename Real>
inline MethodStats<Real>
summarize(const std::vector<double>& rel_errs, int n_total) {
    MethodStats<Real> s;
    s.n_total = n_total;
    std::vector<double> finite;
    finite.reserve(rel_errs.size());
    for (double e : rel_errs) {
        if (std::isfinite(e)) {
            finite.push_back(e);
            s.max_rel_err = std::max(s.max_rel_err, e);
        }
    }
    s.n_finite = static_cast<int>(finite.size());
    if (!finite.empty()) {
        std::sort(finite.begin(), finite.end());
        s.median_rel_err = finite[finite.size() / 2];
    }
    return s;
}

// Run both methods at the working precision Real against the reference.
// Returns true if both methods stay within tol on every comparable case.
//
//   verbose     : if true, print one line per case.
//   tol         : pass/fail tolerance.
//   skip_below  : skip cases where |ref| falls below this threshold
//                 (set by the test type's epsilon -- ~1e-14 for double,
//                 1e-18 for long double).
template<typename Real>
bool run_at_precision(const std::vector<ReferenceCase>& cases,
                      const char* label,
                      double tol,
                      double skip_below,
                      bool verbose = false) {
    std::vector<double> rel_cheb_all, rel_mnp_all;
    int skipped = 0;
    int passes  = 0;
    int passes_cheb = 0;
    int passes_mnp  = 0;
    int total   = 0;

    std::printf("\n--- %s (n=%zu cases) ---\n", label, cases.size());
    if (verbose) {
        std::printf("%4s %6s %3s %16s %16s %16s %12s %12s\n",
                    "id", "b", "q", "reference", "cheb", "mnp",
                    "rel_cheb", "rel_mnp");
    }

    for (const auto& c : cases) {
        if (std::abs(static_cast<double>(c.ref_ld)) < skip_below) {
            ++skipped;
            continue;
        }
        ++total;

        // Build same-precision node vector for both methods.
        std::vector<Real> nodes_R(c.nodes.size());
        for (size_t i = 0; i < c.nodes.size(); ++i) {
            nodes_R[i] = static_cast<Real>(c.nodes[i]);
        }
        const Real b_R = static_cast<Real>(c.b);

        // Chebyshev at Real precision.
        Real cheb_v;
        try {
            chebyshev::ExpDivDiff<Real> solver(-b_R, b_R);
            cheb_v = solver.evaluate(nodes_R);
        } catch (...) {
            cheb_v = std::numeric_limits<Real>::quiet_NaN();
        }

        // MNP at Real precision.
        Real mnp_v;
        try {
            mnp_v = mccurdy_templated::ExpDivDiff<Real>::compute(nodes_R);
        } catch (...) {
            mnp_v = std::numeric_limits<Real>::quiet_NaN();
        }

        // Relative error vs the mpmath reference (stored as long double).
        const long double ref = c.ref_ld;
        const double rc = static_cast<double>(
            std::abs((static_cast<long double>(cheb_v) - ref) / ref));
        const double rm = static_cast<double>(
            std::abs((static_cast<long double>(mnp_v) - ref) / ref));
        rel_cheb_all.push_back(rc);
        rel_mnp_all .push_back(rm);
        if (rc < tol)              ++passes_cheb;
        if (rm < tol)              ++passes_mnp;
        if (rc < tol && rm < tol)  ++passes;

        if (verbose) {
            std::printf("%4d %6.2f %3d %16.6Le %16.6Le %16.6Le %12.2e %12.2e\n",
                        c.case_id, c.b, c.q,
                        c.ref_ld,
                        static_cast<long double>(cheb_v),
                        static_cast<long double>(mnp_v),
                        rc, rm);
        }
    }

    auto sc = summarize<Real>(rel_cheb_all, total);
    auto sm = summarize<Real>(rel_mnp_all,  total);

    std::printf("Chebyshev: %d/%d finite, median rel_err = %.2e, "
                "max = %.2e\n", sc.n_finite, sc.n_total,
                sc.median_rel_err, sc.max_rel_err);
    std::printf("MNP      : %d/%d finite, median rel_err = %.2e, "
                "max = %.2e\n", sm.n_finite, sm.n_total,
                sm.median_rel_err, sm.max_rel_err);
    std::printf("Within tol=%.0e (skipped %d below |ref| floor %.0e):\n",
                tol, skipped, skip_below);
    std::printf("  Chebyshev: %d/%d\n", passes_cheb, total);
    std::printf("  MNP      : %d/%d\n", passes_mnp, total);
    std::printf("  Both     : %d/%d\n", passes, total);

    return passes == total;
}

enum class InfoResult { Passed, FailedInformational, Skipped };

// Top-level entry.  Looks for cases.csv + reference.csv in cwd; if not
// found, returns InfoResult::Skipped so the CI driver can report
// "SKIPPED" rather than misleadingly saying "PASS" on a test that
// did not actually run.
inline InfoResult run_all_reference_tests(bool verbose = false) {
    auto cases = load_reference_csv("cases.csv", "reference.csv");
    if (cases.empty()) {
        std::printf("test_against_reference: no reference data found "
                    "(generate with mpmath_reference.py --generate); "
                    "skipping.\n");
        return InfoResult::Skipped;
    }
    std::printf("test_against_reference: loaded %zu cases\n",
                cases.size());

    // Double-precision run: both methods at double.  Floor is ~1e-14.
    const bool ok_d = run_at_precision<double>(
        cases, "double precision", 1e-9, 1e-14, verbose);
    // Long-double run: both methods at long double.  Floor is ~1e-18.
    const bool ok_ld = run_at_precision<long double>(
        cases, "long double precision", 1e-12, 1e-18, verbose);

    return (ok_d && ok_ld) ? InfoResult::Passed
                           : InfoResult::FailedInformational;
}

} // namespace test_against_reference

#endif // TEST_AGAINST_REFERENCE_HPP
