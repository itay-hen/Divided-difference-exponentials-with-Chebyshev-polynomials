#!/usr/bin/env bash
# Build and run the performance experiments E5/E6, then make figures.
#   ./build_and_run.sh [build_dir]
# Single-threaded runs; total wall time ~30 min on an Apple M3 Max
# (dominated by MNP at q = 800 and by the beta = 16 QMC-like chain).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."                                  # header folder
PROJ="$ROOT/../.."                                  # "DDEF acceleration Chevyshev"
DATA=${DATA:-"$PROJ/Data and figures/v18"}
FIGS=${FIGS:-"$PROJ/tex/figs_v18"}
BUILD="${1:-$HERE/build}"
mkdir -p "$BUILD" "$DATA" "$FIGS"

CXX="${CXX:-g++}"
FLAGS="-std=c++17 -O3 -march=native"
INC=(-I"$ROOT" -I/opt/homebrew/include)            # Boost.Math for cyl_bessel_i

for p in e5_runtime e5_refusal_stats e5_interval_sensitivity e6_stack_micro e6_qmc_chain \
         e6_incremental_micro e6_incremental_order e6_qmc_chain_legacy; do
  $CXX $FLAGS "${INC[@]}" "$HERE/$p.cpp" -o "$BUILD/$p"
done

"$BUILD/e5_runtime"              "$DATA"
"$BUILD/e5_refusal_stats"        "$DATA"
"$BUILD/e5_interval_sensitivity" "$DATA"
"$BUILD/e6_stack_micro"          "$DATA"
"$BUILD/e6_qmc_chain"            "$DATA" 100000 20000 2> "$DATA/e6_qmc_failures.log"
# Legacy IncrementalExpDivDiff comparisons (optional):
"$BUILD/e6_incremental_micro"    "$DATA"
"$BUILD/e6_incremental_order"    "$DATA"
"$BUILD/e6_qmc_chain_legacy"     "$DATA" 100000 20000 2> "$DATA/e6_qmc_failures_legacy.log"
python3 -I "$HERE/make_reference.py" "$DATA"
python3 -I "$HERE/make_figures.py"   "$DATA" "$FIGS"
