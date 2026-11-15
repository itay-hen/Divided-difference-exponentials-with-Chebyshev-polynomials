# PMR-QMC trace generator

Records the divided-difference operations that the PMR-QMC code of
Barash, Babakhani and Hen (Phys. Rev. Research 6, 013281 (2024)) performs
on its weight object, for use by `../replay.cpp`.

* `divdiff_trace.hpp` -- the `divdiff` class shipped with PMR-QMC
  (Gupta, Barash, Hen, CPC 254, 107385 (2020); CC BY 4.0), with logging
  added: `A x` (AddElement), `R` (RemoveElement), `V x` (RemoveValue),
  `C` (CurrentLength reset), `Q i` (read of `divdiffs[i]`).  Internal calls
  (AddAll, the re-additions inside RemoveValue) are not logged.  The
  arithmetic is unchanged.
* `pmr_trace.cpp` -- driver: equilibrates for `Tsteps` updates, then
  records `steps` updates.  It needs `replica_core.hpp`, the `Replica`
  class (configuration, `GetEnergies`, `UpdateWeight*`, `update()`) of the
  PMR-QMC code, which is not redistributed here, and `hamiltonian.hpp`,
  produced by PMR-QMC's `prepare.cpp` from an `H_*.txt` file.
* `make_hamiltonians.py` -- the four Hamiltonians used (2D Heisenberg
  4x4 and 8x8, periodic, J = 1; transverse-field Ising chains of 32 and 64
  spins, periodic, J = h = 1).

Runs used for the manuscript (`pmr_trace beta Tsteps steps seed file`,
seed 7):

| model | beta | Tsteps | steps |
|---|---|---|---|
| heis4x4 | 2, 4 | 50000 | 20000 |
| heis8x8 | 1 | 30000 | 10000 |
| heis8x8 | 2 | 100000 | 5000 |
| tfim32 | 4, 8 | 50000 | 20000 |
| tfim64 | 8 | 60000 | 5000 |

The recorded traces are in `../traces/`.
