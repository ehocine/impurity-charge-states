#!/bin/zsh
# M8: four-electron scans (Table/Fig. 1 N = 4 boundary).  Run from the repository root.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
S=scripts/m8_four_electron_scan.py; O=results/m8
( python3 $S $O/four_30meV_D10_n6.jsonl 30 10 6 0.7 0.8 0.85 0.9 0.95 0.99 ) > $O/A.log 2>&1 &
( python3 $S $O/four_30meV_D3_n6.jsonl 30 3 6 0.7 0.8 0.85 0.9 0.95 0.99;
  python3 $S $O/four_30meV_nogate_n6.jsonl 30 inf 6 0.96 0.99 ) > $O/B.log 2>&1 &
( python3 $S $O/four_30meV_D10_n8.jsonl 30 10 8 0.7 0.8 0.85 ) > $O/C.log 2>&1 &
( python3 $S $O/four_30meV_D10_n8.jsonl 30 10 8 0.9 0.95 0.99 ) > $O/D.log 2>&1 &
( python3 $S $O/four_30meV_D3_n8.jsonl 30 3 8 0.7 0.8 0.85 ) > $O/E.log 2>&1 &
( python3 $S $O/four_30meV_D3_n8.jsonl 30 3 8 0.9 0.95 0.99 ) > $O/F.log 2>&1 &
( python3 $S $O/four_30meV_D10_n10.jsonl 30 10 10 0.95;
  python3 $S $O/four_30meV_D3_n10.jsonl 30 3 10 0.95 ) > $O/G.log 2>&1 &
wait
echo all done
