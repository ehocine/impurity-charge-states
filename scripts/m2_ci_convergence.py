"""M2: convergence of the two-electron CI (graphene/hBN, 2Delta = 30 meV, Z = 0.95 Z_c).

Usage: python3 scripts/m2_ci_convergence.py L E_max/Delta R*Delta   (prints one JSON line)
Results of 2026-10-08: results/m2_ci_convergence_30meV.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from chargestates.furry import FurryBasis
from chargestates.interaction import Interaction, interaction_weight
from chargestates.ci import TwoElectronCI
from collapse2d.analysis import make_solver
from collapse2d.materials import gapped_graphene
L, emax, Rf = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
mat = gapped_graphene(0.015); m = mat.m
S = make_solver(mat, 0.3, "rpa"); Z = 0.95 * 1.6713866023
js = [0.5 + l for l in range(-L, L + 1)]
t = time.time()
B = FurryBasis(S.v, Z, m, R=Rf / m, js=js, E_max=emax * m, r_core=0.3, n_scan=400)
I = Interaction(B, interaction_weight(mat))
out = {"L": L, "Emax": emax, "R": Rf, "n_states": len(B.E), "E1": B.E[B.select(0.5)][0] / m}
for sector, J in (("singlet", 1.0), ("intervalley", 0.0)):
    C = TwoElectronCI(B, I, sector, J)
    w = C.solve(2)
    out[sector] = {"size": C.size, "E2": w[0] / m, "first_order": C.first_order() / m}
out["time"] = time.time() - t
print(json.dumps(out))
