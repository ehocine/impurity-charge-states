"""M9: lowest energy of every three-electron sector versus the impurity charge (zero field).

    python3 scripts/m9_sector_gaps.py OUT.jsonl GAP_meV D_nm N_MAX X1 [X2 ...]      (X = Z/Z_c)

Basis as in M8 (Furry basis of h, L = 2, R = 30/m, E_max = 6m, lowest N_MAX states per channel).
For each sector of ci4.sectors(3) (a pair in K, symmetric or antisymmetric, plus a K' electron; an
up-up pair plus a down electron all in K) the lowest energy over all M is found; within a sector an
M is skipped only when its noninteracting minimum, a rigorous lower bound (W > 0), lies above the
lowest energy of that sector found so far.  Records the sector energies (units of Delta) and the
gap of the all-in-one-valley sector above the three-electron ground state.
"""
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from chargestates.ci4 import ManyElectronCI, ValleyInteraction, sectors  # noqa: E402
from chargestates.furry import FurryBasis  # noqa: E402
from chargestates.interaction import Interaction, interaction_weight, q_nodes  # noqa: E402
from collapse2d.analysis import critical_charge, make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

L_CH, EMAX, RFAC, D_IMP = 2, 6.0, 30.0, 0.3


def sector_lowest(ti, valleys, sym, sym23, js):
    """Lowest energy of one sector over all M, with the rigorous per-M lower bound."""
    Ms = sorted({round(sum(v * j for v, j in zip(valleys, t)), 1) for t in itertools.product(*[js] * len(valleys))})
    cands = []
    for M in Ms:
        ci = ManyElectronCI(ti, valleys, sym, M, sym23=sym23)
        if ci.size == 0:
            continue
        lb = ci.min_one_body()
        if np.isfinite(lb):
            cands.append((ci.min_diagonal(), lb, M, ci))
    best, arg = np.inf, None
    for _, lb, M, ci in sorted(cands, key=lambda c: c[0]):
        if lb > best:
            continue
        e, _ = ci.davidson(tol=1e-9)
        if e < best:
            best, arg = e, M
    return best, arg


def main():
    out, gap_meV, D, n_max = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    xs = [float(x) for x in sys.argv[5:]]
    mat = gapped_graphene(gap_meV / 2000.0)
    m = mat.m
    S = make_solver(mat, D_IMP, "rpa", D=D)
    w = interaction_weight(mat, D=D)
    R = RFAC / m
    js = [0.5 + l for l in range(-L_CH, L_CH + 1)]
    q, qw = q_nodes(R, np.sqrt(EMAX ** 2 - 1) * m, D_IMP, dense_panel=8.0, n_log_panels=30)
    Zc = critical_charge(S)
    for x in xs:
        t = time.time()
        B = FurryBasis(S.v, x * Zc, m, R=R, js=js, E_max=EMAX * m, r_core=D_IMP, n_scan=500)
        I = Interaction(B, w, q=q, qw=qw)
        ch = {j: B.select(j)[:n_max] for j in js}
        ti = ValleyInteraction({1: (B, I), -1: (B, I)}, {1: ch, -1: ch})
        secs = []
        for valleys, sym, sym23 in sectors(3):
            e, M = sector_lowest(ti, valleys, sym, sym23, js)
            secs.append({"valleys": list(valleys), "sym": sym, "M": M, "E": float(e) / m})
        secs.sort(key=lambda s: s["E"])
        one = min(s["E"] for s in secs if len(set(s["valleys"])) == 1)
        rec = {"gap_meV": gap_meV, "D": D, "x": x, "Z": x * Zc, "Zc": Zc, "n_max": n_max, "sectors": secs,
               "ground3": secs[0], "gap_one_valley": one - secs[0]["E"], "time": time.time() - t}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"x={x} n={n_max}: ground {secs[0]['valleys']} sym={secs[0]['sym']} M={secs[0]['M']}; "
              f"all-in-one-valley gap {rec['gap_one_valley']:.4f} Delta  t={rec['time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
