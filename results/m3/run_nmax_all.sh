#!/bin/zsh
# mu3 at n_max = 12 and 16 for every charge of the n_max = 20 scans, for the extrapolation of
# mu3 in 1/n_max point by point (same charges as results/m3/three_*_n20.jsonl).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
zs() { python3 -c "import json; print(' '.join(repr(json.loads(l)['Z']) for l in open('results/m3/three_$1_n20.jsonl')))"; }
for n in 12 16; do
  python3 scripts/m3_three_electron_scan.py 30 inf $n ${=$(zs 30meV_nogate)} > results/m3/three_30meV_nogate_n$n.jsonl 2> results/m3/three_30meV_nogate_n$n.err &
  python3 scripts/m3_three_electron_scan.py 15 inf $n ${=$(zs 15meV_nogate)} > results/m3/three_15meV_nogate_n$n.jsonl 2> results/m3/three_15meV_nogate_n$n.err &
  python3 scripts/m3_three_electron_scan.py 30 10 $n ${=$(zs 30meV_D10)} > results/m3/three_30meV_D10_n$n.jsonl 2> results/m3/three_30meV_D10_n$n.err &
  python3 scripts/m3_three_electron_scan.py 30 3 $n ${=$(zs 30meV_D3)} > results/m3/three_30meV_D3_n$n.jsonl 2> results/m3/three_30meV_D3_n$n.err &
  wait
done
