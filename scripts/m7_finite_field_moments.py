"""M7: finite-field moments M_N = -dE_N/dB of Table IV, with the field in the Dirac equation and in
the Furry projector (chargestates.field).

    python3 scripts/m7_finite_field_moments.py         -> results/m6/finite_field_moments.json

Settings of M6 (Table IV): 2Delta = 30 meV, Z = 0.95 Z_c, no gate and D = 10 nm, L = 2
(j = -3/2 ... 5/2), E_max = 6m, R = 30/m.  In the favourable valley (the one whose level moves down):
E_1, the j = 1/2 level; E_2, the same-valley singlet (full basis, and n_max = 20); E_3, the
singlet pair plus an electron of the other valley (n_max = 20, M = 1/2), the sector of M_3^(20).
Orbital part only (no spin Zeeman term).  Fields 0, 2 and 4 mT: the moments from the two
finite differences agree to < 3 mu_B (linear regime); the 4 mT values are reported.

For N >= 2 these moments are 8-10 % smaller than the zero-field expectation values <P mu P> of M6
(estimates.json): moving the projector with the field adds a negative-energy-state contribution.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from chargestates import observables as ob  # noqa: E402
from chargestates.ci import TwoElectronCI  # noqa: E402
from chargestates.ci3 import ThreeElectronCI  # noqa: E402
from chargestates.field import FieldFurryBasis, magnetic_length  # noqa: E402
from chargestates.interaction import Interaction, compress_shared, interaction_weight, q_nodes  # noqa: E402
from collapse2d.analysis import make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZC = {np.inf: 1.6713866023, 10.0: 2.1645759869}          # 2Delta = 30 meV (M6)
JS = [-1.5, -0.5, 0.5, 1.5, 2.5]
FIELDS = (0.0, 0.002, 0.004)
N_MAX = 20


def energies(mat, v, Z, D, B, valley):
    """E_1, E_2, E_2^(20), E_3^(20) (meV) at field B with the pair in `valley`."""
    m = mat.m
    R, E_max = 30.0 / m, 6.0 * m
    q, qw = q_nodes(R, np.sqrt(35.0) * m, 0.3, dense_panel=8.0, n_log_panels=30)
    w = interaction_weight(mat, D=D)
    bases = {s: FieldFurryBasis(v, Z, m, R=R, js=JS, E_max=E_max, r_core=0.3, n_scan=500,
                                l_B=magnetic_length(B), valley=s) for s in (valley, -valley)}
    bf, bo = bases[valley], bases[-valley]
    If, Io = Interaction(bf, w, q=q, qw=qw), Interaction(bo, w, q=q, qw=qw)
    mev = mat.hbar_v * 1000.0
    out = {"E1": float(bf.E[bf.select(0.5)[0]]) * mev}
    out["E2"] = TwoElectronCI(bf, If, "singlet", 1.0).davidson(tol=1e-11)[0] * mev
    compress_shared([(If, {j: bf.select(j)[:N_MAX] for j in JS}),
                     (Io, {j: bo.select(j)[:N_MAX] for j in JS})], tol=1e-9)
    out["E2_20"] = TwoElectronCI(bf, If, "singlet", 1.0, n_max=N_MAX).davidson(tol=1e-11)[0] * mev
    C3 = ThreeElectronCI(bf, If, "singlet", 0.5, n_max=N_MAX, basis3=bo, inter3=Io)
    out["E3_20"] = C3.davidson(tol=1e-10)[0] * mev
    return out


def case(D):
    mat = gapped_graphene(0.015)
    S = make_solver(mat, 0.3, "rpa", D=D)
    Z = 0.95 * ZC[D]
    # favourable valley: dE_1/dB = -valley * mu_K mu_B, mu_K the moment of the K level at B = 0
    b0 = FieldFurryBasis(S.v, Z, mat.m, R=30 / mat.m, js=[0.5], E_max=2 * mat.m, r_core=0.3, n_scan=200)
    valley = int(np.sign(ob.moment_one(b0, b0.select(0.5)[0])))
    E = {}
    for B in FIELDS:
        t = time.time()
        E[B] = energies(mat, S.v, Z, D, B, valley)
        print(f"D={D} B={B}: {E[B]}  t={time.time() - t:.0f}s", flush=True)
    mom = {k: {f"{B * 1e3:.0f}mT": -(E[B][k] - E[0.0][k]) / B / ob.MU_B_MEV_PER_T for B in FIELDS[1:]}
           for k in E[0.0]}
    M = {k: v["4mT"] for k, v in mom.items()}
    return {"Z": Z, "D": D, "favourable_valley": valley,
            "M1": M["E1"], "M2_KK_singlet": M["E2"], "M2_KK_singlet_n20": M["E2_20"], "M3_n20": M["E3_20"],
            "moments_2mT": {k: v["2mT"] for k, v in mom.items()},
            "energies_meV": {f"{B * 1e3:.0f}mT": e for B, e in E.items()}}


def main():
    res = {"source": "finite-field no-pair energies at B = 0, 2, 4 mT with the field in the Dirac equation "
                     "and the Furry projector (chargestates.field, scripts/m7_finite_field_moments.py); "
                     "settings of M6: L = 2, E_max = 6m, R = 30/m, n_max = 20 for M2^(20), M3^(20); "
                     "moments M_N = -dE_N/dB at 4 mT (units of mu_B)",
           "note": "no gate, 0.95 Z_c: the third electron is not bound (mu_3 > Delta); M3 is that of a box "
                   "continuum state and is not reported (set to null)"}
    for key, D in (("30_nogate", np.inf), ("30_D10", 10.0)):
        res[key] = case(D)
    res["30_nogate"]["M3_n20_continuum"] = res["30_nogate"]["M3_n20"]
    res["30_nogate"]["M3_n20"] = None
    os.makedirs(os.path.join(ROOT, "results", "m6"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", "m6", "finite_field_moments.json"), "w"), indent=1)
    print(json.dumps({k: {kk: res[k][kk] for kk in ("M1", "M2_KK_singlet", "M2_KK_singlet_n20", "M3_n20")}
                      for k in ("30_nogate", "30_D10")}, indent=1))


if __name__ == "__main__":
    main()
