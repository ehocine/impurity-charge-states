"""M2: energy cut-off series of the two-electron CI (matrix-free Davidson).

    python3 scripts/m2_emax_series.py graphene L E_max/Delta [sectors...]
    python3 scripts/m2_emax_series.py dminus   L k_max*a_B

graphene: gapped graphene on hBN, 2Delta = 30 meV, kappa = 2.5, d = 0.3 nm, Dirac-sea RPA,
          Z = 0.95 Z_c, box R = 30/m.  Prints E_2 in units of Delta for each sector.
dminus:   validation V3, point charge Z = 1, constant kappa, alpha = 0.05, R = 20 a_B;
          prints the D- binding energy in Ry* (exact non-relativistic value 0.48054).
Set CI_DTYPE=float32 to store the form factors in single precision (halves memory).
Coarser q panels than the defaults (8/R, 30 log panels) keep the form-factor cache small;
their effect on matrix elements is ~1e-6 relative (tested 2026-10-08).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from chargestates.ci import TwoElectronCI  # noqa: E402
from chargestates.furry import FurryBasis  # noqa: E402
from chargestates.interaction import Interaction, interaction_weight, q_nodes  # noqa: E402
from collapse2d.analysis import make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402
from collapse2d.potentials import PointCoulomb  # noqa: E402

DTYPE = np.dtype(os.environ.get("CI_DTYPE", "float64")).type


def graphene(L, emax, sectors):
    mat = gapped_graphene(0.015)
    m = mat.m
    S = make_solver(mat, 0.3, "rpa")
    js = [0.5 + l for l in range(-L, L + 1)]
    t = time.time()
    B = FurryBasis(S.v, 0.95 * 1.6713866023, m, R=30 / m, js=js, E_max=emax * m, r_core=0.3, n_scan=500)
    k_max = np.sqrt(emax ** 2 - 1) * m
    q, qw = q_nodes(B.R, k_max, 0.3, dense_panel=8.0, n_log_panels=30)
    I = Interaction(B, interaction_weight(mat), q=q, qw=qw)
    out = {"system": "graphene", "dtype": DTYPE.__name__, "L": L, "Emax": emax, "n_states": len(B.E), "n_q": len(q),
           "E1": B.E[B.select(0.5)][0] / m}
    for sec in sectors:
        J = 0.0 if sec == "intervalley" else 1.0
        C = TwoElectronCI(B, I, sec, J, dtype=DTYPE)
        w = C.solve(1, method="davidson")
        out[sec] = {"size": C.size, "E2": float(w[0]) / m, **C.davidson_info}
    out["time"] = time.time() - t
    return out


def dminus(L, kfac, alpha=0.05, Rfac=20.0):
    m = 1.0
    aB, Ry = 1 / (alpha * m), alpha ** 2 * m / 2
    R, k_max = Rfac * aB, kfac / aB
    t = time.time()
    js = [0.5 + l for l in range(-L, L + 1)]
    B = FurryBasis(PointCoulomb(1.0), alpha, m, R=R, js=js, E_max=np.sqrt(m * m + k_max * k_max),
                   r_core=aB, L=min(2 / k_max, R / 10), n_scan=500)
    q, qw = q_nodes(R, k_max, aB / 20, dense_panel=8.0, n_log_panels=30)
    I = Interaction(B, lambda q: alpha * np.ones_like(q), q=q, qw=qw)
    E1 = B.E[B.select(0.5)][0]
    C = TwoElectronCI(B, I, "singlet", 1.0, dtype=DTYPE)
    w = C.solve(1, method="davidson")
    return {"system": "dminus", "dtype": DTYPE.__name__, "alpha": alpha, "L": L, "kfac": kfac, "Rfac": Rfac, "n_states": len(B.E),
            "size": C.size, "E1_binding_Ry": (m - E1) / Ry, "Dminus_binding_Ry": float(E1 + m - w[0]) / Ry,
            **C.davidson_info, "time": time.time() - t}


if __name__ == "__main__":
    kind, L, cut = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    if kind == "graphene":
        res = graphene(L, cut, sys.argv[4:] or ["singlet", "intervalley"])
    else:
        res = dminus(L, cut)
    print(json.dumps(res), flush=True)
