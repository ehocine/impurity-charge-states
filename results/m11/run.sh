#!/bin/zsh
# Gate-kernel sensitivity: homogeneous kappa vs exact vacuum / hBN / metal kernel
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
python3 scripts/m11_gate_kernel.py results/m11/gate_kernel.jsonl 30 10 12 0.923969 0.95
python3 scripts/m11_gate_kernel.py results/m11/gate_kernel.jsonl 30 3 12 0.95
echo all done
