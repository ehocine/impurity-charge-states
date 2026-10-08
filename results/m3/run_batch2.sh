#!/bin/zsh
# Batch 2 (2026-10-08): 2Delta = 15 meV without gate; 2Delta = 30 meV with D = 3 nm.
# Two- and three-electron scans as in batch 1.
cd "$(dirname "$0")/../.."
export VECLIB_MAXIMUM_THREADS=4 CI_DTYPE=float32
Zc15=1.6167816162; Zc3=2.6185697
z15=(); z3=(); z15b=(); z3b=()
for f in 0.5 0.6 0.7 0.8 0.85 0.9 0.95 0.99 1.02 1.05 1.08 1.1 1.12 1.15; do
  z15+=$(python3 -c "print($f*$Zc15)"); z3+=$(python3 -c "print($f*$Zc3)")
done
for f in 0.7 0.8 0.85 0.9 0.95 0.99 1.02 1.05 1.08 1.1 1.12 1.15; do
  z15b+=$(python3 -c "print($f*$Zc15)"); z3b+=$(python3 -c "print($f*$Zc3)")
done
python3 scripts/m3_two_electron_scan.py 15 inf $z15 > results/m3/scan_15meV_nogate.jsonl 2> results/m3/scan_15meV_nogate.err &
python3 scripts/m3_two_electron_scan.py 30 3 $z3 > results/m3/scan_30meV_D3.jsonl 2> results/m3/scan_30meV_D3.err &
wait
python3 scripts/m3_three_electron_scan.py 15 inf 20 $z15b > results/m3/three_15meV_nogate_n20.jsonl 2> results/m3/three_15meV_nogate_n20.err &
python3 scripts/m3_three_electron_scan.py 30 3 20 $z3b > results/m3/three_30meV_D3_n20.jsonl 2> results/m3/three_30meV_D3_n20.err &
wait
