# PMR-QMC operation traces (Section 6.4 of the manuscript)

* `generator/` -- how the traces were recorded from the PMR-QMC code of
  Barash, Babakhani and Hen (see its README).
* `replay.cpp` -- replays a trace with GBH (the divdiff class used by
  PMR-QMC, `divdiff_pmrqmc.hpp`, unmodified), the batch Chebyshev method,
  the incremental stack (own sorted order, adaptive interval), and the
  naive stack (fixed interval, PMR's order); one CSV row per trace.
* `exact_eval.cpp`, `exact_check.py` -- 200 sampled reads per trace
  against exact values (sum of residues at the distinct node values,
  mpmath, 300 and 600 digits).
* `make_table.py` -- Table 6.3 from the replay CSV.

`traces/` holds the seven recorded traces (gzip). `../run_v20.sh` runs all
of this on them.
