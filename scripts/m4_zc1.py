"""M4: Z_c(1), the charge at which the second electron's addition level reaches -Delta.

    python3 scripts/m4_zc1.py           -> results/m4/zc1_30meV_nogate.json

Beyond ~1.15 Z_c the CI electron count is not defined (broad resonance), so mu2 is continued as
    mu2(Z) ~ eps'(Z) + delta(Z),
eps' = lowest j = 1/2 level of h + V_H[n_a0] (a second electron in the Hartree field of the first,
which occupies the j = 1/2 orbital a0, quasi-bound above Z_c), and delta = mu2(CI) - eps' the
correlation correction, known from the M3 scan up to 1.15 Z_c and extrapolated in Z (linear
and quadratic fits; their spread is the uncertainty).
2Delta = 30 meV, no gate.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from chargestates.furry import FurryBasis  # noqa: E402
from chargestates.interaction import Interaction, interaction_weight, q_nodes  # noqa: E402
from chargestates.meanfield import mean_field_potential  # noqa: E402
from collapse2d.analysis import make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZC = 1.6713866023


def main(fracs):
    mat = gapped_graphene(0.015)
    m = mat.m
    S = make_solver(mat, 0.3, "rpa")
    w = interaction_weight(mat)
    R = 30 / m
    q, qw = q_nodes(R, np.sqrt(8) * m, 0.3, dense_panel=8.0, n_log_panels=30)
    ci = {round(d["Z"] / d["Zc"], 3): min(d["singlet"], d["intervalley"]) - d["E1"]
          for d in map(json.loads, open(os.path.join(ROOT, "results", "m3", "scan_30meV_nogate.jsonl")))
          if d.get("E1") is not None}
    rows = []
    for f in fracs:
        Z = f * ZC
        rec = {"Z_over_Zc": f}
        for w_loc in (0.9, 0.8, 0.7, 0.6, 0.5):              # accept broader resonances for n_a0
            B = FurryBasis(S.v, Z, m, R=R, js=[0.5], E_max=3 * m, r_core=0.3, n_scan=400,
                           E_floor=-4 * m, w_loc=w_loc)
            a0 = B.select(0.5)[0]
            # below Z_c the lowest level is a0; above Z_c a0 must be the dived (quasi-bound)
            # orbital, not the next bound level
            if Z < ZC or B.E[a0] <= -m:
                break
        if Z >= ZC and B.E[a0] > -m:
            rec["E1"] = None
            rows.append(rec)
            print(json.dumps(rec), flush=True)
            continue
        rec["E1"], rec["w_loc_used"] = B.E[a0] / m, w_loc
        rec["inner_weight_a0"] = float(B.inner_weight[a0]) if B.E[a0] <= -m else 1.0
        P = mean_field_potential(S.v, Z, Interaction(B, w, q=q, qw=qw), [a0], w, 1e-7, R)
        Bp = FurryBasis(P, 1.0, m, R=R, js=[0.5], E_max=3 * m, r_core=0.3, n_scan=400, E_floor=-4 * m, w_loc=0.6)
        b0 = Bp.select(0.5)[0]
        rec["eps_prime"] = Bp.E[b0] / m
        if round(f, 3) in ci:
            rec["mu2_CI"] = ci[round(f, 3)]
            rec["delta"] = rec["mu2_CI"] - rec["eps_prime"]
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    # correlation correction delta(Z) from the CI points (Z >= 0.9 Z_c), linear and quadratic fits;
    # the spread of the resulting Z_c(1) is quoted as its uncertainty
    pts = np.array([(r["Z_over_Zc"], r["delta"]) for r in rows if "delta" in r and r["Z_over_Zc"] >= 0.9])
    good = [r for r in rows if r.get("E1") is not None]
    x = np.array([r["Z_over_Zc"] for r in good])
    eps = np.array([r["eps_prime"] for r in good])
    res = {"rows": rows, "Zc": ZC, "fits": {}}
    for deg in (1, 2):
        c = np.polyfit(pts[:, 0], pts[:, 1], deg)
        y = eps + np.polyval(c, x)
        zc1 = None
        below = np.flatnonzero(y < -1)
        if len(below) and below[0] > 0:
            i = below[0]
            zc1 = float(x[i - 1] + (-1 - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))
        res["fits"][f"deg{deg}"] = {"coef": c.tolist(), "mu2_estimate": y.tolist(), "Zc1_over_Zc": zc1}
        print(f"delta fit degree {deg}: Z_c(1)/Z_c = {zc1}", flush=True)
    os.makedirs(os.path.join(ROOT, "results", "m4"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", "m4", "zc1_30meV_nogate.json"), "w"), indent=1)


if __name__ == "__main__":
    main([0.9, 0.95, 0.99, 1.02, 1.05, 1.08, 1.10, 1.12, 1.15, 1.18, 1.2, 1.22, 1.25, 1.3, 1.35, 1.4])
