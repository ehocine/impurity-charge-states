#!/bin/zsh
# Recheck of the three- and four-electron sector searches after the Davidson start-vector fix
until [ -f ~/Desktop/charge-states-in-field/results/harden/recheck.done ]; do sleep 30; done
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
( python3 scripts/m9_sector_gaps.py results/m9/recheck_D10.jsonl 30 10 12 0.95 0.99
  python3 scripts/m9_sector_gaps.py results/m9/recheck_nogate.jsonl 30 inf 12 0.99
  python3 scripts/m8_four_electron_scan.py results/m8/recheck_nogate_n6.jsonl 30 inf 6 0.99 ) > results/m8/recheck_a.log 2>&1 &
( python3 scripts/m8_four_electron_scan.py results/m8/recheck_D10_n12.jsonl 30 10 12 0.95 0.99
  python3 scripts/m8_four_electron_scan.py results/m8/recheck_D3_n12.jsonl 30 3 12 0.99 ) > results/m8/recheck_b.log 2>&1 &
wait
echo all done >> results/m8/recheck_b.log
