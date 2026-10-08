#!/bin/zsh
# High cut-offs of the series (Davidson): E_max = 10m and 12m at L = 2.
cd "$(dirname "$0")/../.."
export VECLIB_MAXIMUM_THREADS=4
for E in 10 12; do python3 scripts/m2_emax_series.py graphene 2 $E > results/m2_series/graphene_L2_E$E.json 2>&1 & done; wait
