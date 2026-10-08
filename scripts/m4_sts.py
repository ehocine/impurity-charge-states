"""M4: tunnelling observables: Dyson orbitals, quasiparticle weights and radial profiles.

    python3 scripts/m4_sts.py           -> results/m4/sts_30meV.json, figures/fig_dyson_30meV.*

At Z = 0.95 Z_c (2Delta = 30 meV), without gate and with D = 10 nm:
  0 -> 1   the j = 1/2 orbital a0 (removal/addition peak at mu1)
  1 -> 2   Dyson orbitals into the same-valley singlet (adding K, opposite spin: one final
           state) and into the K-K' pair (adding K', either spin: two final states), at
           mu2(S) and mu2(KK'); weights Z_qp = sum |d|^2; intensity ratio 1 : 2 per flavour count
  2 -> 3   (gate) Dyson orbital for adding the K' electron to the K-pair singlet (n_max = 20)
Radial profiles |phi_D|^2/(2 pi r) per nm^2, normalised to the weight; <r> of each orbital.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from chargestates.ci import TwoElectronCI  # noqa: E402
from chargestates.ci3 import ThreeElectronCI  # noqa: E402
from chargestates.furry import FurryBasis  # noqa: E402
from chargestates.interaction import Interaction, interaction_weight, q_nodes  # noqa: E402
from collapse2d.analysis import make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L_CH, EMAX, RFAC, D_IMP = 2, 6.0, 30.0, 0.3


def profile(g, F, G):
    rho = (F ** 2 + G ** 2) / (2 * np.pi * g.r)
    w = float((F ** 2 + G ** 2) @ g.w)
    rbar = float(((F ** 2 + G ** 2) * g.r) @ g.w / w) if w > 0 else np.nan
    return rho, w, rbar


def case(D, Zc, frac, with_three):
    mat = gapped_graphene(0.015)
    m = mat.m
    S = make_solver(mat, D_IMP, "rpa", D=D)
    w = interaction_weight(mat, D=D)
    R = RFAC / m
    js = [0.5 + l for l in range(-L_CH, L_CH + 1)]
    q, qw = q_nodes(R, np.sqrt(EMAX ** 2 - 1) * m, D_IMP, dense_panel=8.0, n_log_panels=30)
    B = FurryBasis(S.v, frac * Zc, m, R=R, js=js, E_max=EMAX * m, r_core=D_IMP, n_scan=500, E_floor=-3 * m)
    g = B.grid
    a0 = B.select(0.5)[0]
    I = Interaction(B, w, q=q, qw=qw)
    out = {"D": D, "Z": frac * Zc, "Zc": Zc, "mu1": B.E[a0] / m}
    r_out = np.geomspace(0.3, 300, 120)
    rho0, w0, r0 = profile(g, B.F[a0], B.G[a0])
    out["orbital"] = {"r_mean_nm": r0, "profile": np.interp(r_out, g.r, rho0).tolist()}
    for sec, J in (("singlet", 1.0), ("intervalley", 0.0)):
        C = TwoElectronCI(B, I, sec, J)
        e, v = C.davidson(tol=1e-10)
        D_ = C.dyson(v, a0)
        rho, wt, rb = profile(g, D_["F"], D_["G"])
        out[sec] = {"mu2": (e - B.E[a0]) / m, "Z_qp": D_["Z_qp"], "r_mean_nm": rb,
                    "profile": np.interp(r_out, g.r, rho).tolist()}
    if with_three:
        n_max = 20
        I.compress({j: B.select(j)[:n_max] for j in js}, tol=1e-8)
        C2 = TwoElectronCI(B, I, "singlet", 1.0, n_max=n_max)
        e2, v2 = C2.davidson(tol=1e-11)
        C3 = ThreeElectronCI(B, I, "singlet", 0.5, n_max=n_max)
        e3, v3 = C3.davidson(tol=1e-10)
        d, wt = C3.dyson_from_pair(v3, C2.full_tensor(v2))
        Fd = sum(d[j] @ B.F[C3.channels[j]] for j in d)
        Gd = sum(d[j] @ B.G[C3.channels[j]] for j in d)
        rho, wt2, rb = profile(g, Fd, Gd)
        out["three"] = {"mu3_n20": (e3 - e2) / m, "Z_qp": wt, "r_mean_nm": rb,
                        "channel_weights": {str(j): float(v @ v) for j, v in d.items()},
                        "profile": np.interp(r_out, g.r, rho).tolist()}
    out["r_nm"] = r_out.tolist()
    return out


def main():
    res = {"nogate": case(np.inf, 1.6713866023, 0.95, False),
           "D10": case(10.0, 2.1645759869, 0.95, True)}
    os.makedirs(os.path.join(ROOT, "results", "m4"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", "m4", "sts_30meV.json"), "w"), indent=1)
    for k, c in res.items():
        print(k, "mu1=%+.4f <r>=%.1f nm" % (c["mu1"], c["orbital"]["r_mean_nm"]))
        for sec in ("singlet", "intervalley"):
            s = c[sec]
            print("   1->2 %-11s mu2=%+.4f Z_qp=%.4f <r>=%.1f nm" % (sec, s["mu2"], s["Z_qp"], s["r_mean_nm"]))
        if "three" in c:
            t = c["three"]
            print("   2->3 mu3(n20)=%+.4f Z_qp=%.4f <r>=%.1f nm  channels %s" % (
                t["mu3_n20"], t["Z_qp"], t["r_mean_nm"], {k: round(v, 4) for k, v in t["channel_weights"].items()}))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, (k, title) in zip(axs, (("nogate", "no gate"), ("D10", "gate, $D=10$ nm"))):
        c = res[k]
        r = np.array(c["r_nm"])
        ax.loglog(r, c["orbital"]["profile"], "k-", lw=1, label="$0\\to1$ ($a_0$)")
        ax.loglog(r, c["singlet"]["profile"], "-", color="#0072B2", lw=1, label="$1\\to2$, KK singlet")
        ax.loglog(r, c["intervalley"]["profile"], "--", color="#D55E00", lw=1, label="$1\\to2$, KK$'$")
        if "three" in c:
            ax.loglog(r, c["three"]["profile"], ":", color="#009E73", lw=1.3, label="$2\\to3$")
        ax.set_xlabel("$r$ (nm)")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(1e-8, 1e-2)
    axs[0].set_ylabel("$|\\phi_D|^2/2\\pi r$ (nm$^{-2}$)")
    axs[1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "figures", f"fig_dyson_30meV.{ext}"), dpi=200)


if __name__ == "__main__":
    main()
