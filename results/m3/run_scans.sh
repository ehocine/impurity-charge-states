#!/bin/zsh
# Two-electron scans in Z (extended Furry basis of h): 2Delta = 30 meV, no gate and D = 10 nm.
cd "$(dirname "$0")/../.."
export VECLIB_MAXIMUM_THREADS=4 CI_DTYPE=float32
Zc0=1.6713866023; Zc10=2.1645759869
z0=(); z10=()
for f in 0.5 0.6 0.7 0.8 0.85 0.9 0.95 0.99 1.02 1.05 1.08 1.1 1.15; do
  z0+=$(python3 -c "print($f*$Zc0)"); z10+=$(python3 -c "print($f*$Zc10)")
done
python3 scripts/m3_two_electron_scan.py 30 inf $z0 > results/m3/scan_30meV_nogate.jsonl 2> results/m3/scan_30meV_nogate.err &
python3 scripts/m3_two_electron_scan.py 30 10 $z10 > results/m3/scan_30meV_D10.jsonl 2> results/m3/scan_30meV_D10.err &
wait
