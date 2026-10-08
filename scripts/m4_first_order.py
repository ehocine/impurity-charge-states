"""M4: first-order addition energy mu2 = E1 + U (both electrons in the unrelaxed j = 1/2 orbital),
for comparison with the CI of m3_two_electron_scan.py.

    python3 scripts/m4_first_order.py      -> results/m4/first_order_30meV.json

U = <a0 a0|W|a0 a0> with the screened interaction (and gate) of the scans; same Z values.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from chargestates.furry import FurryBasis  # noqa: E402
from chargestates.interaction import Interaction, interaction_weight, q_nodes  # noqa: E402
from collapse2d.analysis import make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def first_order(D, Zc, fracs):
    mat = gapped_graphene(0.015)
    m = mat.m
    S = make_solver(mat, 0.3, "rpa", D=D)
    w = interaction_weight(mat, D=D)
    R = 30 / m
    q, qw = q_nodes(R, np.sqrt(35) * m, 0.3, dense_panel=8.0, n_log_panels=30)
    out = []
    for f in fracs:
        B = FurryBasis(S.v, f * Zc, m, R=R, js=[0.5], E_max=2 * m, r_core=0.3, n_scan=300, E_floor=-3.5 * m)
        a0 = B.select(0.5)[0]
        I = Interaction(B, w, q=q, qw=qw)
        U = I.matrix_elements([a0], [a0], [a0], [a0], 0)[0, 0, 0, 0]
        out.append({"Z": f * Zc, "Zc": Zc, "E1": B.E[a0] / m, "U": U / m, "mu2_first_order": (B.E[a0] + U) / m})
        print(json.dumps(out[-1]), flush=True)
    return out


if __name__ == "__main__":
    fr = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    res = {"nogate": first_order(np.inf, 1.6713866023, fr + [1.02, 1.05, 1.08, 1.10, 1.12, 1.15]),
           "D10": first_order(10.0, 2.1645759869, fr)}
    os.makedirs(os.path.join(ROOT, "results", "m4"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", "m4", "first_order_30meV.json"), "w"), indent=1)
