#!/bin/zsh
# M9: lowest energy of every three-electron sector (all-in-one-valley gap), n_max = 12.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
S=scripts/m9_sector_gaps.py; O=results/m9
python3 $S $O/gaps3_D10.jsonl 30 10 12 0.7 0.8 0.85 0.9 0.95 0.99
python3 $S $O/gaps3_D3.jsonl 30 3 12 0.7 0.8 0.85 0.9 0.95 0.99
python3 $S $O/gaps3_nogate.jsonl 30 inf 12 0.8 0.9 0.95 0.99
echo all done
