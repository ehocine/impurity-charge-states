"""M3: one- and two-electron energies of the impurity versus its charge.

    python3 scripts/m3_two_electron_scan.py GAP_meV D_nm Z1 [Z2 ...]      (D = inf: no gate)

Graphene on hBN (kappa = 2.5), impurity at d = 0.3 nm, Dirac-sea RPA screening, optional
metallic gate at distance D.  Furry basis of h (impurity potential incl. RPA screening), the
projector consistent with the RPA-screened interaction (NOTES, 2026-10-08: a mean-field
projector double-counts the sea's response and collapses into the sea at low Z).  Above Z_c
the dived level is kept as long as it is a narrow resonance, i.e. a box state below -m with
most of its weight inside r < 3/m (FurryBasis E_floor); otherwise E1 is null and no
two-electron energy is computed.

One JSON line per Z: E1 (lowest j = 1/2 level), E2 for the same-valley singlet (J = 1) and the
K-K' pair (M = 0), all in units of Delta; 'dived' = E1 below -Delta.  Settings L = 2,
E_max = 6m, R = 30/m (M2 study; cut-off/L corrections ~ -0.003 Delta, applied in the analysis).
Set CI_DTYPE=float32 for large runs.
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
from collapse2d.analysis import critical_charge, make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

DTYPE = np.dtype(os.environ.get("CI_DTYPE", "float64")).type
L_CH, EMAX, RFAC, D_IMP = 2, 6.0, 30.0, 0.3


def scan(gap_meV, D, Zs):
    mat = gapped_graphene(gap_meV / 2000.0)
    m = mat.m
    S = make_solver(mat, D_IMP, "rpa", D=D)
    w = interaction_weight(mat, D=D)
    R = RFAC / m
    js = [0.5 + l for l in range(-L_CH, L_CH + 1)]
    q, qw = q_nodes(R, np.sqrt(EMAX ** 2 - 1) * m, D_IMP, dense_panel=8.0, n_log_panels=30)
    Zc = critical_charge(S)
    for Z in Zs:
        t = time.time()
        rec = {"gap_meV": gap_meV, "D": D, "Z": Z, "Zc": Zc, "L": L_CH, "Emax": EMAX, "R": RFAC}
        B = FurryBasis(S.v, Z, m, R=R, js=js, E_max=EMAX * m, r_core=D_IMP, n_scan=500, E_floor=-3 * m)
        i12 = B.select(0.5)
        lowest_is_level = len(i12) and (B.E[i12[0]] > -m or B.inner_weight[i12[0]] > 0.5)
        if Z >= Zc and not (len(B.dived()) and np.isclose(B.j[B.dived()[0]], 0.5)):
            rec["E1"] = None                        # broad resonance: no localised level
        else:
            rec["E1"] = float(B.E[i12[0]]) / m
            rec["dived"] = bool(B.E[i12[0]] <= -m)
            if rec["dived"]:
                rec["inner_weight"] = float(B.inner_weight[i12[0]])
            I = Interaction(B, w, q=q, qw=qw)
            for sec, J in (("singlet", 1.0), ("intervalley", 0.0)):
                C = TwoElectronCI(B, I, sec, J, dtype=DTYPE)
                rec[sec] = float(C.solve(1, method="davidson")[0]) / m
        assert lowest_is_level or rec["E1"] is None
        rec["time"] = time.time() - t
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    gap, D = float(sys.argv[1]), float(sys.argv[2])
    scan(gap, D, [float(z) for z in sys.argv[3:]])
