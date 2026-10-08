"""V3 (M2 validation): weak-coupling limit, point charge Z = 1, constant kappa: 2D D- binding -> 0.48054 Ry*."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from chargestates.furry import FurryBasis
from chargestates.interaction import Interaction, q_nodes
from chargestates.ci import TwoElectronCI
from collapse2d.potentials import PointCoulomb
alpha, L, kfac, Rfac = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
m = 1.0; aB = 1 / (alpha * m); Ry = alpha**2 * m / 2
R = Rfac * aB; k_max = kfac / aB; E_max = np.sqrt(m * m + k_max * k_max)
t = time.time()
js = [0.5 + l for l in range(-L, L + 1)]
B = FurryBasis(PointCoulomb(1.0), alpha, m, R=R, js=js, E_max=E_max, r_core=aB, L=min(2 / k_max, R / 10), n_scan=400)
q, qw = q_nodes(R, k_max, aB / 20)
I = Interaction(B, lambda q: alpha * np.ones_like(q), q=q, qw=qw)
E1 = B.E[B.select(0.5)][0]
C = TwoElectronCI(B, I, "singlet", 1.0); w = C.solve(1)
out = {"alpha": alpha, "L": L, "kfac": kfac, "Rfac": Rfac, "n_states": len(B.E), "size": C.size,
       "E1_binding_Ry": (m - E1) / Ry, "Dminus_binding_Ry": (E1 + m - w[0]) / Ry, "time": time.time() - t}
print(json.dumps(out))
