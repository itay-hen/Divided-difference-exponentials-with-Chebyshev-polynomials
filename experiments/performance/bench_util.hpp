// =============================================================================
// bench_util.hpp -- shared timing helpers for the performance experiments
//                   (E5, E6) of the Chebyshev--Bessel exp-divided-difference
//                   paper.
//
// Timing protocol (used by every program in this folder):
//   * std::chrono::steady_clock.
//   * A "sample" runs the kernel `reps` times back to back and records the
//     mean time per call.  `reps` is calibrated once so that one sample
//     lasts at least `min_sample_seconds` (default 50 ms).
//   * The reported figure is the MEDIAN of `n_samples` (default 7) samples,
//     which suppresses outliers caused by other processes on the machine.
//   * Every kernel returns a double that is accumulated into a volatile
//     sink, so the compiler cannot eliminate the work.
// =============================================================================
#ifndef DDEF_BENCH_UTIL_HPP
#define DDEF_BENCH_UTIL_HPP

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace bench {

/// Global sink: results are added here so the optimiser keeps the work.
inline volatile double g_sink = 0.0;

using clock_type = std::chrono::steady_clock;

inline double seconds_since(clock_type::time_point t0) {
    return std::chrono::duration<double>(clock_type::now() - t0).count();
}

/// Median of a vector (copied).
inline double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

struct TimingResult {
    double median_s = 0.0;   ///< median time per call [s]
    double min_s    = 0.0;   ///< fastest sample, per call [s]
    double max_s    = 0.0;   ///< slowest sample, per call [s]
    long   reps     = 0;     ///< calls per sample
    int    samples  = 0;
};

/// Time `kernel(i)` (i = running call index, so the kernel can cycle through
/// a pool of inputs).  The kernel returns a double that is sunk.
template <class Kernel>
TimingResult time_kernel(Kernel&& kernel,
                         double min_sample_seconds = 0.05,
                         int n_samples = 7,
                         long max_reps = 100000000L) {
    // Warm-up call (also first-touch of caches / lazy precomputation).
    double acc = kernel(0L);

    // Calibrate the repetition count.
    long reps = 1;
    long idx = 1;
    for (;;) {
        auto t0 = clock_type::now();
        for (long r = 0; r < reps; ++r) acc += kernel(idx++);
        const double dt = seconds_since(t0);
        if (dt >= min_sample_seconds || reps >= max_reps) break;
        const double grow = (dt > 0) ? 1.3 * min_sample_seconds / dt : 10.0;
        reps = std::min(max_reps,
                        std::max(reps + 1, static_cast<long>(reps * std::min(grow, 10.0))));
    }

    std::vector<double> per_call;
    per_call.reserve(n_samples);
    for (int s = 0; s < n_samples; ++s) {
        auto t0 = clock_type::now();
        for (long r = 0; r < reps; ++r) acc += kernel(idx++);
        per_call.push_back(seconds_since(t0) / static_cast<double>(reps));
    }
    g_sink = g_sink + acc;

    TimingResult tr;
    tr.median_s = median(per_call);
    tr.min_s = *std::min_element(per_call.begin(), per_call.end());
    tr.max_s = *std::max_element(per_call.begin(), per_call.end());
    tr.reps = reps;
    tr.samples = n_samples;
    return tr;
}

/// Simple CSV writer helper: opens a file or aborts.
inline FILE* open_csv(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "cannot open %s for writing\n", path.c_str());
        std::exit(1);
    }
    return f;
}

}  // namespace bench

#endif  // DDEF_BENCH_UTIL_HPP
