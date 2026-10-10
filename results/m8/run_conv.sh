#!/bin/zsh
# M8: N_MAX convergence of mu4 at 0.95 Z_c (n = 12, 16, 20) (ground sectors only), one run at a time (memory).
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
S=scripts/m8_four_electron_scan.py; O=results/m8
for n in 12 16 20; do
  python3 $S $O/conv_D10_x0.95.jsonl 30 10 $n 0.95 --ground-only
  python3 $S $O/conv_D3_x0.95.jsonl 30 3 $n 0.95 --ground-only
done
# no gate: fourth electron unbound (mu4 > Delta) at 0.99 Z_c
python3 $S $O/conv_nogate.jsonl 30 inf 12 0.99 --ground-only
echo all done
