#!/bin/zsh
# mu3 at n_max = 24, 28 where the n_max dependence is largest (D = 3 nm at 0.95, 0.99 Z_c;
# D = 10 nm at 0.99 Z_c), to test the 1/n_max extrapolation from 12, 16, 20.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
Zc3=2.6185650097041218; Zc10=2.1645759868774976
( for n in 24 28; do python3 scripts/m3_three_electron_scan.py 30 3 $n $(python3 -c "print(0.95*$Zc3, 0.99*$Zc3)"); done > results/m3/three_30meV_D3_n24_28.jsonl 2> results/m3/three_30meV_D3_n24_28.err ) &
( for n in 24 28; do python3 scripts/m3_three_electron_scan.py 30 10 $n $(python3 -c "print(0.99*$Zc10)"); done > results/m3/three_30meV_D10_n24_28.jsonl 2> results/m3/three_30meV_D10_n24_28.err ) &
wait
