#!/bin/zsh
# KK'-KK splitting versus the gap (2Delta = 100, 300 meV; hBN, no gate), Z/Z_c = 0.95, 0.99.
cd "$(dirname "$0")/../.."
export VECLIB_MAXIMUM_THREADS=4 CI_DTYPE=float32
python3 scripts/m3_two_electron_scan.py 100 inf $(python3 -c "print(0.95*1.8205975830, 0.99*1.8205975830)") > results/m3/scan_100meV_nogate.jsonl 2> results/m3/scan_100meV_nogate.err
python3 scripts/m3_two_electron_scan.py 300 inf $(python3 -c "print(0.95*2.07524, 0.99*2.07524)") > results/m3/scan_300meV_nogate.jsonl 2> results/m3/scan_300meV_nogate.err
