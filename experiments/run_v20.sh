#!/usr/bin/env bash
# Regenerate every data set and figure of the manuscript (version 20).
# Total ~55 min (mostly single-threaded) on an Apple M3 Max.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
PROJ="$ROOT/../.."
export DATA="$PROJ/Data and figures/v20"
export FIGS="$PROJ/tex/figs_v20"
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
python3 -I "$HERE/scaling/plot_Rn.py" "$FIGS/fig_Rn.pdf"

echo "== stopping rule (E4)"; cxx "$HERE/accuracy/e4_stopping_rule.cpp" -o "$HERE/bin/e4_stopping_rule"
"$HERE/bin/e4_stopping_rule" > "$DATA/e4_stopping_rule.csv"

echo "== accuracy"; bash "$HERE/accuracy/run_all.sh"

echo "== ratios vs MNP (E7)"; cxx "$HERE/accuracy/e7_ratio.cpp" -o "$HERE/bin/e7_ratio"
"$HERE/bin/e7_ratio" "$DATA/e7_ratio" 20
python3 -I "$HERE/accuracy/mp_reference.py" --nodes "$DATA/e7_ratio_nodes.csv" --out "$DATA/e7_ratio_ref.csv" --jobs 12
python3 -I "$HERE/accuracy/e7_analyze.py" "$DATA/e7_ratio_results.csv" "$DATA/e7_ratio_ref.csv" \
  "$FIGS/fig_ratio_heatmap.pdf" "$DATA/e7_ratio_summary.csv"
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
python3 -I "$HERE/gbh_compare/make_table.py" "$DATA/gbh_vs_stack.csv" "$FIGS/table_gbh.tex"

echo "== performance"; bash "$HERE/performance/build_and_run.sh" "$HERE/performance/build"
echo "== PMR-QMC traces"; cxx "$HERE/pmr_traces/replay.cpp" -I"$HERE/pmr_traces" -o "$HERE/bin/replay"
TR="$DATA/pmr_traces"; mkdir -p "$TR"
ls "$TR"/*.txt.gz >/dev/null 2>&1 || cp "$HERE/pmr_traces/traces/"*.txt.gz "$TR/"
[ -d "$TR" ] && ls "$TR"/*.txt.gz >/dev/null 2>&1 && for f in "$TR"/*.txt.gz; do gunzip -kf "$f"; done
"$HERE/bin/replay" --header "$TR"/trace_*.txt > "$DATA/pmr_replay.csv"
cxx "$HERE/pmr_traces/exact_eval.cpp" -I"$HERE/pmr_traces" -o "$HERE/bin/exact_eval"
python3 -I "$HERE/pmr_traces/exact_check.py" "$HERE/bin/exact_eval" "$DATA/pmr_exact.csv" "$TR"/trace_*.txt
python3 -I "$HERE/pmr_traces/make_table.py" "$DATA/pmr_replay.csv" "$FIGS/table_pmr.tex"
rm -f "$TR"/trace_*.txt
echo "== done"
