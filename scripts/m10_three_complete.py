"""M10: three-electron addition energy in the complete basis at several energy cutoffs.

    python3 scripts/m10_three_complete.py OUT.jsonl GAP_meV D_nm X EMAX1 [EMAX2 ...]     (X = Z/Z_c)

Basis as in M3 (Furry basis of h, L = 2, R = 30/m) with cutoff E_max = EMAX m and all states of
every channel (no n_max truncation).  Records E_1, the opposite-valley pair E_2 and the
three-electron ground state E_3 (K pair + K' electron, M = 1/2), all in units of m = Delta, and
mu_3 = E_3 - E_2, for the extrapolation of mu_3 in E_max.  The Davidson subspace is kept small
(max_space = 12) so that E_max = 8m (about 8e6 configurations) fits in memory.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chargestates.deps  # noqa: E402,F401
import numpy as np  # noqa: E402

from chargestates.ci import TwoElectronCI  # noqa: E402
from chargestates.ci3 import ThreeElectronCI  # noqa: E402
from chargestates.furry import FurryBasis  # noqa: E402
from chargestates.interaction import Interaction, interaction_weight, q_nodes  # noqa: E402
from collapse2d.analysis import critical_charge, make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

L_CH, RFAC, D_IMP = 2, 30.0, 0.3


def main():
    out, gap_meV, D, x = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    mat = gapped_graphene(gap_meV / 2000.0)
    m = mat.m
    S = make_solver(mat, D_IMP, "rpa", D=D)
    w = interaction_weight(mat, D=D)
    R = RFAC / m
    js = [0.5 + l for l in range(-L_CH, L_CH + 1)]
    Zc = critical_charge(S)
    for emax in (float(e) for e in sys.argv[5:]):
        t = time.time()
        q, qw = q_nodes(R, np.sqrt(emax ** 2 - 1) * m, D_IMP, dense_panel=8.0, n_log_panels=30)
        B = FurryBasis(S.v, x * Zc, m, R=R, js=js, E_max=emax * m, r_core=D_IMP, n_scan=500, E_floor=-3 * m)
        I = Interaction(B, w, q=q, qw=qw)
        channels = {j: B.select(j) for j in js}
        I.compress(channels, tol=1e-8)
        e1 = float(B.E[B.select(0.5)][0]) / m
        e2 = float(TwoElectronCI(B, I, "intervalley", 0.0).davidson(tol=1e-10)[0]) / m
        C3 = ThreeElectronCI(B, I, "singlet", 0.5)
        e3, _ = C3.davidson(tol=1e-9, max_space=12)
        rec = {"gap_meV": gap_meV, "D": D, "x": x, "Z": x * Zc, "Zc": Zc, "E_max": emax,
               "n_per_channel": {str(j): int(len(i)) for j, i in channels.items()},
               "E1": e1, "E2": e2, "E3": float(e3) / m, "mu3": float(e3) / m - e2, "size3": C3.size,
               "time": time.time() - t}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"E_max={emax}m: n={rec['n_per_channel']} mu3={rec['mu3']:.5f} Delta size={C3.size} "
              f"t={rec['time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
