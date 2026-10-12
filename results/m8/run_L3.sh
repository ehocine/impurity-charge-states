#!/bin/zsh
# M8: multipole convergence of mu3, mu4 at 0.95 Z_c, n_max = 12, L = 3 (channels -5/2 ... 7/2);
# compare conv_D*_x0.95.jsonl (L = 2).  M9: n_max = 16 check of the all-in-one-valley gap.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
until grep -q "all done" results/m9/run_gaps.log; do sleep 30; done
python3 scripts/m8_four_electron_scan.py results/m8/L3_x0.95.jsonl 30 10 12 0.95 --ground-only --L 3
python3 scripts/m8_four_electron_scan.py results/m8/L3_x0.95.jsonl 30 3 12 0.95 --ground-only --L 3
python3 scripts/m9_sector_gaps.py results/m9/gaps3_D10_n16.jsonl 30 10 16 0.95
echo all done
