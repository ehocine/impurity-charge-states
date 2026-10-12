#!/bin/zsh
# Hardening: three-electron addition energy up to the complete basis (57 states per channel at
# E_max = 6m), D = 10 nm, 0.95 Z_c; checks the 1/n_max correction of the n_max = 20 values.
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6
for n in 36 44 57; do
  /usr/bin/time -l python3 scripts/m3_three_electron_scan.py 30 10 $n 2.056347187 >> results/m3/three_D10_nmax_convergence.jsonl 2>> results/m3/run_nmax_full.log
done
echo all done >> results/m3/run_nmax_full.log
