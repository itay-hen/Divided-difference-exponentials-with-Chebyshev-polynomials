#!/bin/sh
# Reproduce experiments E2 (accuracy over node configurations / orderings) and
# E3 (truncation error vs. a-priori bound).  Run from this directory.
#   DATA : CSV output directory      FIGS : PDF / LaTeX output directory
set -e
set -o pipefail 2>/dev/null || true
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)                                   # the "DDEF c++" folder
DATA=${DATA:-"$ROOT/../../Data and figures/v18"}
FIGS=${FIGS:-"$ROOT/../../tex/figs_v18"}
BIN=${BIN:-"$HERE/bin"}
mkdir -p "$DATA" "$FIGS" "$BIN"
cxx() { g++ -std=c++17 -O2 -I"$ROOT" -I/opt/homebrew/include "$@"; }   # paths contain spaces

cxx "$HERE/e2_accuracy.cpp" -o "$BIN/e2_accuracy"
cxx "$HERE/e3_code_N.cpp" -o "$BIN/e3_code_N"
cxx "$HERE/repro_ascending_order.cpp" -o "$BIN/repro_ascending_order"

python3 -I "$HERE/mp_reference.py" --selftest
"$BIN/e2_accuracy" "$DATA"                                       # ~1 min
python3 -I "$HERE/mp_reference.py" --nodes "$DATA/accuracy_e2_nodes.csv" \
        --out "$DATA/accuracy_e2_reference.csv" --jobs 8          # ~2 min on 8 cores
python3 -I "$HERE/crosscheck_reference.py" --nodes "$DATA/accuracy_e2_nodes.csv" \
        --ref "$DATA/accuracy_e2_reference.csv" --qmax 20
python3 -I "$HERE/e2_analyze.py" --data "$DATA" --figs "$FIGS"
python3 -I "$HERE/e3_truncation.py" --cheb-bin "$BIN/e3_code_N" \
        --csv "$DATA/accuracy_e3_curves.csv" --summary "$DATA/accuracy_e3_summary.csv" \
        --fig "$FIGS/fig_truncation_bounds.pdf"
"$BIN/repro_ascending_order"
