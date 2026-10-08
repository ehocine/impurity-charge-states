"""M3: three-electron energies (two electrons in K, one in K') versus the impurity charge.

    python3 scripts/m3_three_electron_scan.py GAP_meV D_nm N_MAX Z1 [Z2 ...]

Same model and basis as m3_two_electron_scan.py (extended Furry basis of h, L = 2 channels,
R = 30/m), but truncated to the lowest N_MAX states per channel so that the n^3 configuration
space stays tractable.  E2 is recomputed with the same truncation, so that the addition energy
mu3 = E3 - E2 is taken at consistent truncation; the N_MAX dependence is checked separately.
E3: K pair spin singlet (spatially symmetric), M = j1 + j2 - j~3 = 1/2 (all three electrons in
their lowest orbitals).  Energies in units of Delta, one JSON line per Z.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from chargestates.ci import TwoElectronCI  # noqa: E402
from chargestates.ci3 import ThreeElectronCI  # noqa: E402
from chargestates.furry import FurryBasis  # noqa: E402
from chargestates.interaction import Interaction, interaction_weight, q_nodes  # noqa: E402
from collapse2d.analysis import critical_charge, make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

L_CH, EMAX, RFAC, D_IMP = 2, 6.0, 30.0, 0.3


def scan(gap_meV, D, n_max, Zs):
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
        rec = {"gap_meV": gap_meV, "D": D, "Z": Z, "Zc": Zc, "n_max": n_max, "L": L_CH, "Emax": EMAX}
        B = FurryBasis(S.v, Z, m, R=R, js=js, E_max=EMAX * m, r_core=D_IMP, n_scan=500, E_floor=-3 * m)
        if Z >= Zc and not (len(B.dived()) and np.isclose(B.j[B.dived()[0]], 0.5)):
            rec["E1"] = None
            print(json.dumps(rec), flush=True)
            continue
        rec["E1"] = float(B.E[B.select(0.5)][0]) / m
        I = Interaction(B, w, q=q, qw=qw)
        channels = {j: B.select(j)[:n_max] for j in js}
        rec["ranks"] = I.compress(channels, tol=1e-8)
        for sec, J in (("singlet", 1.0), ("intervalley", 0.0)):
            C = TwoElectronCI(B, I, sec, J, n_max=n_max)
            rec["E2_" + sec] = float(C.davidson(tol=1e-10)[0]) / m
        C3 = ThreeElectronCI(B, I, "singlet", 0.5, n_max=n_max)
        e3, _ = C3.davidson(tol=1e-9)
        rec["E3"], rec["size3"], rec["dav3"] = float(e3) / m, C3.size, C3.davidson_info
        rec["time"] = time.time() - t
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    gap, D, n_max = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    scan(gap, D, n_max, [float(z) for z in sys.argv[4:]])
