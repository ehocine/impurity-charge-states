#!/bin/zsh
# Three-electron scans (n_max = 20 per channel) after the two-electron scans; then an
# n_max convergence check at the D = 10 nm reference point (Z = 0.95 Z_c).
cd "$(dirname "$0")/../.."
while pgrep -f "m3/run_scans.sh" >/dev/null; do sleep 15; done
export VECLIB_MAXIMUM_THREADS=4
Zc0=1.6713866023; Zc10=2.1645759869
z0=(); z10=()
for f in 0.7 0.8 0.85 0.9 0.95 0.99 1.02 1.05 1.08 1.1; do
  z0+=$(python3 -c "print($f*$Zc0)"); z10+=$(python3 -c "print($f*$Zc10)")
done
python3 scripts/m3_three_electron_scan.py 30 inf 20 $z0 > results/m3/three_30meV_nogate_n20.jsonl 2> results/m3/three_30meV_nogate_n20.err &
python3 scripts/m3_three_electron_scan.py 30 10 20 $z10 > results/m3/three_30meV_D10_n20.jsonl 2> results/m3/three_30meV_D10_n20.err &
wait
for n in 12 16 20 24 28; do
  VECLIB_MAXIMUM_THREADS=8 python3 scripts/m3_three_electron_scan.py 30 10 $n 2.056347187 >> results/m3/three_D10_nmax_convergence.jsonl 2>> results/m3/three_D10_nmax_convergence.err
done
