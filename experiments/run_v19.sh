#!/usr/bin/env bash
# Regenerate every data set and figure of the manuscript (version 19).
# Total ~40 min single-threaded on an Apple M3 Max.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
PROJ="$ROOT/../.."
export DATA="$PROJ/Data and figures/v19"
export FIGS="$PROJ/tex/figs_v19"
export SKIP_LEGACY=${SKIP_LEGACY:-1}
mkdir -p "$DATA" "$FIGS" "$HERE/bin"
if [ "$SKIP_LEGACY" = "1" ]; then
  cp "$PROJ/Data and figures/v18/"*legacy* "$PROJ/Data and figures/v18/e6_incremental_order.csv" "$DATA/" 2>/dev/null || true
fi
cxx() { g++ -std=c++17 -O3 -march=native -I"$ROOT" -I/opt/homebrew/include "$@"; }

if [ "${RESUME_FROM:-}" = "" ]; then
echo "== scaling"; cxx "$HERE/scaling/n_kappa_sweep.cpp" -o "$HERE/bin/n_kappa_sweep"
for m in fixed tight pinned; do "$HERE/bin/n_kappa_sweep" "$DATA/n_kappa_sweep_$m.csv" $m; done
python3 -I "$HERE/scaling/plot_scaling.py" "$DATA/n_kappa_sweep_pinned.csv" "$FIGS/fig_scaling.pdf"

echo "== stopping rule (E4)"; cxx "$HERE/accuracy/e4_stopping_rule.cpp" -o "$HERE/bin/e4_stopping_rule"
"$HERE/bin/e4_stopping_rule" > "$DATA/e4_stopping_rule.csv"

echo "== accuracy"; bash "$HERE/accuracy/run_all.sh"
fi
if [ "${RESUME_FROM:-}" = "e3" ]; then
  echo "== truncation (resume)"
  cxx "$HERE/accuracy/e3_code_N.cpp" -o "$HERE/bin/e3_code_N"
  python3 -I "$HERE/accuracy/e3_truncation.py" --cheb-bin "$HERE/bin/e3_code_N" \
    --csv "$DATA/accuracy_e3_curves.csv" --summary "$DATA/accuracy_e3_summary.csv" \
    --fig "$FIGS/fig_truncation_bounds.pdf"
  cxx "$HERE/accuracy/repro_ascending_order.cpp" -o "$HERE/bin/repro_ascending_order"
  "$HERE/bin/repro_ascending_order"
fi

echo "== GBH"; cxx "$HERE/gbh_compare/gbh_vs_stack.cpp" -o "$HERE/bin/gbh_vs_stack"
"$HERE/bin/gbh_vs_stack" > "$DATA/gbh_vs_stack.csv"

echo "== performance"; bash "$HERE/performance/build_and_run.sh" "$HERE/performance/build"
echo "== done"
