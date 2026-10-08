#!/bin/zsh
# Joint L / cut-off convergence, one job at a time (float32 form factors; error ~1e-8 Delta).
cd "$(dirname "$0")/../.."
export VECLIB_MAXIMUM_THREADS=6 CI_DTYPE=float32
python3 scripts/m2_emax_series.py graphene 3 8 > results/m2_series/graphene_L3_E8.json 2>&1
python3 scripts/m2_emax_series.py graphene 4 8 > results/m2_series/graphene_L4_E8.json 2>&1
python3 scripts/m2_emax_series.py dminus 4 11 > results/m2_series/dminus_L4_k11.json 2>&1
python3 scripts/m2_emax_series.py dminus 6 11 > results/m2_series/dminus_L6_k11.json 2>&1
