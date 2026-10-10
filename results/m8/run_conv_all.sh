#!/bin/zsh
# M8: mu4 at N_MAX = 12, 16 (ground sectors) for all charges, two lanes (one per gate).
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
S=scripts/m8_four_electron_scan.py; O=results/m8
for D in 10 3; do
  ( for n in 12 16; do python3 $S $O/conv_D${D}_all.jsonl 30 $D $n 0.7 0.8 0.85 0.9 0.99 --ground-only; done ) > $O/conv_all_D$D.log 2>&1 &
done
wait
echo all done
