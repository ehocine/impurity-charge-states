#!/bin/zsh
# Energy cut-off series (2026-10-08); two jobs at a time to stay within memory.
cd "$(dirname "$0")/../.."
export VECLIB_MAXIMUM_THREADS=4
python3 scripts/m2_emax_series.py graphene 2 5 > results/m2_series/graphene_L2_E5.json 2>&1 &
python3 scripts/m2_emax_series.py dminus 2 11 > results/m2_series/dminus_L2_k11.json 2>&1 &
wait
python3 scripts/m2_emax_series.py graphene 2 6 > results/m2_series/graphene_L2_E6.json 2>&1 &
python3 scripts/m2_emax_series.py graphene 2 7 > results/m2_series/graphene_L2_E7.json 2>&1 &
wait
python3 scripts/m2_emax_series.py graphene 2 8 > results/m2_series/graphene_L2_E8.json 2>&1 &
python3 scripts/m2_emax_series.py dminus 2 14 > results/m2_series/dminus_L2_k14.json 2>&1 &
wait
# lower cut-offs of the D- check (dense solver):
python3 scripts/m2_v3_dminus.py 0.05 3 5 20 > results/m2_series/dminus_L3_k5.json
python3 scripts/m2_v3_dminus.py 0.05 5 5 20 > results/m2_series/dminus_L5_k5.json
python3 scripts/m2_v3_dminus.py 0.05 7 5 20 > results/m2_series/dminus_L7_k5.json
python3 scripts/m2_v3_dminus.py 0.05 3 8 20 > results/m2_series/dminus_L3_k8.json
python3 scripts/m2_v3_dminus.py 0.05 3 5 30 > results/m2_series/dminus_L3_k5_R30.json
