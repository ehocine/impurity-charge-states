#!/bin/zsh
# Hardening: three-electron addition energy in the complete basis at E_max = 4m and 8m
# (6m: results/m3/three_D10_nmax_convergence.jsonl, n_max = 57), D = 10 nm, 0.95 Z_c.
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8
/usr/bin/time -l python3 scripts/m10_three_complete.py results/m10/complete_D10_x0.95.jsonl 30 10 0.95 4 8 > results/m10/run_complete.log 2>&1
echo all done >> results/m10/run_complete.log
