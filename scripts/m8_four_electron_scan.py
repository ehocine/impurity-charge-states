"""M8: four-electron energies and the fourth addition energy mu4 = E4 - E3 versus the impurity charge.

    python3 scripts/m8_four_electron_scan.py OUT.jsonl GAP_meV D_nm N_MAX X1 [X2 ...] [--ground-only]
                                                                          [--L 3]   (X = Z/Z_c)

Same model and basis as M3 (Furry basis of h, L = 2 channels, R = 30/m, E_max = 6m), truncated to
the lowest N_MAX states per channel.  E3 and E4 are the lowest energies over all three- and
four-electron sectors (ci4.lowest_state: every valley content and exchange symmetry, all M), at the
same truncation, so that mu4 is taken consistently; E2 (same truncation) gives mu3 = E3 - E2.
The truncation dependence is removed by extrapolating mu4 in 1/N_MAX (scripts/m5_figures.py).
--ground-only: only the ground sectors found at N_MAX = 6-10 for every gated case (three electrons:
K singlet pair + K' electron, M = 1/2; four electrons: closed shell, singlet pair in each valley,
M = 0), for the N_MAX convergence of mu4 at large N_MAX.
--L: channels j = 1/2 - L ... 1/2 + L (default 2), for the multipole convergence.
Energies in units of Delta, one JSON line per Z.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from chargestates.ci import TwoElectronCI  # noqa: E402
from chargestates.ci4 import ManyElectronCI, ValleyInteraction, lowest_state  # noqa: E402
from chargestates.furry import FurryBasis  # noqa: E402
from chargestates.interaction import Interaction, interaction_weight, q_nodes  # noqa: E402
from collapse2d.analysis import critical_charge, make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

L_CH, EMAX, RFAC, D_IMP = 2, 6.0, 30.0, 0.3


def main():
    out, gap_meV, D, n_max = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    ground_only = "--ground-only" in sys.argv
    args = sys.argv[5:]
    L_ch = L_CH
    if "--L" in args:
        k = args.index("--L")
        L_ch = int(args[k + 1])
        args = args[:k] + args[k + 2:]
    xs = [float(x) for x in args if not x.startswith("--")]
    mat = gapped_graphene(gap_meV / 2000.0)
    m = mat.m
    S = make_solver(mat, D_IMP, "rpa", D=D)
    w = interaction_weight(mat, D=D)
    R = RFAC / m
    js = [0.5 + l for l in range(-L_ch, L_ch + 1)]
    q, qw = q_nodes(R, np.sqrt(EMAX ** 2 - 1) * m, D_IMP, dense_panel=8.0, n_log_panels=30)
    Zc = critical_charge(S)
    for x in xs:
        t = time.time()
        Z = x * Zc
        B = FurryBasis(S.v, Z, m, R=R, js=js, E_max=EMAX * m, r_core=D_IMP, n_scan=500)
        I = Interaction(B, w, q=q, qw=qw)
        ch = {j: B.select(j)[:n_max] for j in js}
        ti = ValleyInteraction({1: (B, I), -1: (B, I)}, {1: ch, -1: ch})
        e2 = min(TwoElectronCI(B, I, sec, J, n_max=n_max).davidson(tol=1e-10)[0]
                 for sec, J in (("singlet", 1.0), ("intervalley", 0.0)))
        if ground_only:
            e3 = ManyElectronCI(ti, (1, 1, -1), 1, 0.5).davidson(tol=1e-9)[0]
            e4 = ManyElectronCI(ti, (1, 1, -1, -1), 1, 0.0, sym23=1).davidson(tol=1e-9)[0]
            g3 = {"valleys": [1, 1, -1], "sym": 1, "sym23": 0, "M": 0.5}
            g4 = {"valleys": [1, 1, -1, -1], "sym": 1, "sym23": 1, "M": 0.0}
            s4 = [g4]
        else:
            e3, g3, _ = lowest_state(ti, 3, js)
            e4, g4, s4 = lowest_state(ti, 4, js)
        rec = {"gap_meV": gap_meV, "D": D, "Z": Z, "Zc": Zc, "x": x, "n_max": n_max, "L": L_ch, "Emax": EMAX,
               "E1": float(B.E[B.select(0.5)[0]]) / m, "E2": float(e2) / m, "E3": float(e3) / m, "E4": float(e4) / m,
               "mu3": float(e3 - e2) / m, "mu4": float(e4 - e3) / m, "ground3": g3, "ground4": g4,
               "n_solved4": len(s4), "ground_only": ground_only, "time": time.time() - t}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"x={x} n={n_max}: mu3={rec['mu3']:.4f} mu4={rec['mu4']:.4f} Delta  ground4={g4['valleys']} "
              f"sym={g4['sym']},{g4['sym23']} M={g4['M']}  t={rec['time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
