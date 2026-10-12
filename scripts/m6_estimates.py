"""M6: estimates for the manuscript (review rounds 1-2), all from chargestates.observables.

    python3 scripts/m6_estimates.py       -> results/m6/estimates.json

Graphene on hBN (kappa = 2.5, d = 0.3 nm, Dirac-sea RPA), Z = 0.95 Z_c.
  2Delta = 30 meV, no gate and D = 10 nm:
    - orbital moments (units of mu_B) of the one-electron level, the same-valley two-electron
      singlet and the three-electron state (K pair + K'); low-field slopes of the addition energies
      d mu_N/dB = -(M_N - M_{N-1}) mu_B in the favourable valley (meV/T); magnetic length vs orbit
    - slow-sea self-energies of a0 and of the 1->2 Dyson orbital; K-weighted mean excitation
      energy w_bar of a0 (input of the single-pole interpolation, Appendix C); the same estimate
      with the RPA electron-hole continuum of the sea (observables.sigma_dynamic_rpa)
    - moire form factors (scalar, mass, current) of a0 and of the Dyson orbital
    - contact estimate of the intervalley exchange; gap from j = 1/2 to the lowest j = -1/2 level
  2Delta = 100, 300 meV (no gate): moment, band-edge moment, B*, J_iv, <r>

Table IV (--tables) reports the moments M_N = -dE_N/dB of the no-pair energies at 4 mT, with the field
in the Dirac equation and in the Furry projector, read from results/m6/finite_field_moments.json
(scripts/m7_finite_field_moments.py, chargestates.field; run it first); for
N >= 2 they are 8-10 % smaller than the zero-field expectation values <mu> computed here
(estimates.json), which are quoted in the caption.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from chargestates import observables as ob  # noqa: E402
from chargestates.ci import TwoElectronCI  # noqa: E402
from chargestates.ci3 import ThreeElectronCI  # noqa: E402
from chargestates.furry import FurryBasis  # noqa: E402
from chargestates.interaction import Interaction, interaction_weight, q_nodes  # noqa: E402
from collapse2d.analysis import make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZC = {(30, np.inf): 1.6713866023, (30, 10.0): 2.1645759869, (100, np.inf): 1.8205975830, (300, np.inf): 2.07524}
SPLIT_EXTRAP = 1.4          # cut-off/multipole extrapolation factor of the KK'-KK splitting (Table II)


def setup(gap, D, emax, js):
    mat = gapped_graphene(gap / 2000.0)
    m = mat.m
    S = make_solver(mat, 0.3, "rpa", D=D)
    w = interaction_weight(mat, D=D)
    R = 30 / m
    B = FurryBasis(S.v, 0.95 * ZC[(gap, D)], m, R=R, js=js, E_max=emax * m, r_core=0.3, n_scan=500)
    q, qw = q_nodes(R, np.sqrt(emax ** 2 - 1) * m, 0.3, dense_panel=8.0, n_log_panels=30)
    return mat, B, Interaction(B, w, q=q, qw=qw), w, interaction_weight(mat, D=D, rpa=False)


def full_case(gap, D):
    js = [-1.5, -0.5, 0.5, 1.5, 2.5]
    mat, B, I, w, w_env = setup(gap, D, 6.0, js)
    m, g = B.m, B.grid
    a0 = B.select(0.5)[0]
    F, G = B.F[a0], B.G[a0]
    out = {"gap_meV": gap, "D": D, "Delta_meV": gap / 2, "E1": B.E[a0] / m,
           "gap_to_jm12": (B.E[B.select(-0.5)[0]] - B.E[a0]) / m,
           "r_mean_a0_nm": float(((F ** 2 + G ** 2) * g.r) @ g.w)}
    m1 = ob.moment_one(B, a0)
    s = np.sign(m1)                                            # favourable valley: moment along B
    out["band_edge_moment_muB"] = ob.band_edge_moment(mat)
    # two electrons (full basis): same-valley singlet, K-K' pair
    res = {}
    for sec, J in (("singlet", 1.0), ("intervalley", 0.0)):
        C = TwoElectronCI(B, I, sec, J)
        e, v = C.davidson(tol=1e-10)
        res[sec] = (e, v, C)
    E2S, vS, CS = res["singlet"]
    E2K, vK, CK = res["intervalley"]
    D12 = CK.dyson(vK, a0)
    nD = np.sqrt(D12["Z_qp"])
    FD, GD = D12["F"] / nD, D12["G"] / nD
    M1 = s * m1
    M2 = s * ob.moment_two(CS, vS)
    out["moments_muB"] = {"M1": M1, "M2_KK_singlet": M2, "M2_KKprime": s * ob.moment_two(CK, vK)}
    # three electrons (n_max = 20) with a consistently truncated pair
    n_max = 20
    I.compress({j: B.select(j)[:n_max] for j in js}, tol=1e-8)
    C2 = TwoElectronCI(B, I, "singlet", 1.0, n_max=n_max)
    e2, v2 = C2.davidson(tol=1e-11)
    C3 = ThreeElectronCI(B, I, "singlet", 0.5, n_max=n_max)
    e3, v3 = C3.davidson(tol=1e-10)
    M2t, M3 = s * ob.moment_two(C2, v2), s * ob.moment_three(C3, v3)
    out["moments_muB"].update({"M2_KK_singlet_n20": M2t, "M3_n20": M3})
    k = ob.MU_B_MEV_PER_T
    out["slopes_meV_per_T"] = {"mu1": -M1 * k, "mu2": -(M2 - M1) * k, "mu3": -(M3 - M2t) * k}
    # (the all-in-one-valley three-electron sector has no bound state at B = 0: scripts/m9_sector_gaps.py)
    # magnetic length l_B = 25.66 nm / sqrt(B[T]) equals the orbital radius at B_l = (25.66/<r>)^2
    rD = float(((FD ** 2 + GD ** 2) * g.r) @ g.w)
    out["r_mean_dyson12_nm"] = rD
    out["B_lB_equals_r_T"] = {"a0": (25.66 / out["r_mean_a0_nm"]) ** 2, "dyson12": (25.66 / rD) ** 2}
    # self-energy inputs
    sig_a0, wbar = ob.slow_sea(I, mat, a0, w_env)
    out["sigma_slow_meV"] = {"a0": sig_a0, "dyson12": ob.sigma_slow_orbital(I, mat, FD, GD, w_env)}
    out["w_bar_over_Delta"] = wbar
    # single-pole estimate (Appendix C) for w_bar from the lowest transition (j = -1/2) to the
    # K-weighted mean
    out["sigma_dynamic_meV"] = {}
    for wb in (out["gap_to_jm12"], 2.0, wbar):
        s0 = ob.sigma_dynamic(I, mat, F, G, wb, w_env)
        s1 = ob.sigma_dynamic(I, mat, FD, GD, wb, w_env)
        out["sigma_dynamic_meV"]["%.2f" % wb] = {"a0": s0, "dyson12": s1,
                                                 "mu1_shift_Delta": s0 / (gap / 2),
                                                 "mu2_minus_mu1_shift_Delta": (s1 - s0) / (gap / 2)}
    # the same with the RPA electron-hole continuum of the sea instead of a single mode at its edge
    out["sigma_dynamic_rpa_meV"] = {}
    for wb in (out["gap_to_jm12"], 2.0, wbar):
        s0, _, rel0 = ob.sigma_dynamic_rpa(I, mat, F, G, wb, w_env, D)
        s1, _, rel1 = ob.sigma_dynamic_rpa(I, mat, FD, GD, wb, w_env, D)
        out["sigma_dynamic_rpa_meV"]["%.2f" % wb] = {"a0": s0, "dyson12": s1,
                                                     "mu1_shift_Delta": s0 / (gap / 2),
                                                     "mu2_minus_mu1_shift_Delta": (s1 - s0) / (gap / 2),
                                                     "sum_rule_rel_err": max(rel0, rel1)}
    # moire and intervalley exchange
    out["moire"] = {"a0": ob.moire_form_factors(B, F, G), "dyson12": ob.moire_form_factors(B, FD, GD)}
    out["J_iv_meV"] = ob.intervalley_exchange(B, F, G, w, mat)
    out["split_KKp_minus_KK_Delta"] = (E2K - E2S) / m
    return out


def light_case(gap):
    mat, B, I, w, w_env = setup(gap, np.inf, 2.0, [0.5])
    a0 = B.select(0.5)[0]
    F, G = B.F[a0], B.G[a0]
    g = B.grid
    mu = abs(ob.moment_one(B, a0))
    return {"gap_meV": gap, "moment_muB": mu, "band_edge_moment_muB": ob.band_edge_moment(mat),
            "J_iv_meV": ob.intervalley_exchange(B, F, G, w, mat), "r_mean_a0_nm": float(((F ** 2 + G ** 2) * g.r) @ g.w)}


def main():
    res = {"30_nogate": full_case(30, np.inf), "30_D10": full_case(30, 10.0),
           "100": light_case(100), "300": light_case(300)}
    # B* from the M3 scans (L = 2, E_max = 6m) times the extrapolation factor
    for key, gap, fname in (("30_nogate", 30, "scan_30meV_nogate.jsonl"), ("100", 100, "scan_100meV_nogate.jsonl"),
                            ("300", 300, "scan_300meV_nogate.jsonl")):
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "results", "m3", fname))]
        r = [d for d in rows if d.get("E1") is not None and abs(d["Z"] / d["Zc"] - 0.95) < 1e-3][0]
        split = abs(r["intervalley"] - r["singlet"]) * gap / 2 * SPLIT_EXTRAP
        mu = res[key]["moments_muB"]["M1"] if key == "30_nogate" else res[key]["moment_muB"]
        res[key]["split_extrapolated_meV"] = split
        res[key]["B_star_T"] = split / (2 * mu * ob.MU_B_MEV_PER_T)
    os.makedirs(os.path.join(ROOT, "results", "m6"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", "m6", "estimates.json"), "w"), indent=1)
    tables(res)
    print(json.dumps(res, indent=1))


def tables(res):
    """LaTeX bodies of the field and estimates tables of the manuscript (paper/tables/)."""
    tdir = os.path.join(ROOT, "paper", "tables")
    os.makedirs(tdir, exist_ok=True)
    # Table IV: moments M_N = -dE_N/dB of the no-pair energies at finite field (the field also enters
    # the Furry projector), from results/m6/finite_field_moments.json (see its "source"); the
    # zero-field expectation values <mu> of this script (estimates.json) are quoted in the caption.
    ff = json.load(open(os.path.join(ROOT, "results", "m6", "finite_field_moments.json")))
    k = ob.MU_B_MEV_PER_T
    rows = []
    for key, lab in (("30_nogate", "no gate"), ("30_D10", "$D=10$\\,nm")):
        e, M = res[key], ff[key]
        s1 = -M["M1"] * k
        s2 = -(M["M2_KK_singlet"] - M["M1"]) * k
        if M["M3_n20"] is None:
            m3, s3 = "--", "--"
        else:
            m3 = f"{M['M3_n20']:.0f}"
            s3 = f"${-(M['M3_n20'] - M['M2_KK_singlet_n20']) * k:+.1f}$"
        rows.append(f"{lab} & {M['M1']:.0f} & {M['M2_KK_singlet']:.0f} & {M['M2_KK_singlet_n20']:.0f} & "
                    f"{m3} & ${s1:+.1f}$ & ${s2:+.1f}$ & {s3} & "
                    f"{e['B_lB_equals_r_T']['dyson12']:.2f}\\\\")
    open(os.path.join(tdir, "tab_field_body.tex"), "w").write(
        "% generated by scripts/m6_estimates.py -- do not edit by hand\n"
        "\\begin{tabular}{lcccccccc}\n"
        " & $M_1$ & $M_2$ & $M_2^{(20)}$ & $M_3^{(20)}$ & $\\dd\\mu_1/\\dd B$ & $\\dd\\mu_2/\\dd B$ & $\\dd\\mu_3/\\dd B$ & $B_l$\\\\\n"
        " & \\multicolumn{4}{c}{($\\mu_B$)} & \\multicolumn{3}{c}{(meV/T)} & (T)\\\\\n\\hline\n"
        + "\n".join(rows) + "\n\\end{tabular}\n")
    rows = []
    for key, gap in (("30_nogate", 30), ("100", 100), ("300", 300)):
        e = res[key]
        mu = e["moments_muB"]["M1"] if key == "30_nogate" else e["moment_muB"]
        rows.append(f"{gap} & {mu:.0f} & {e['band_edge_moment_muB']:.0f} & {e['split_extrapolated_meV']:.2f} & "
                    f"{e['B_star_T']:.3f} & {e['J_iv_meV']:.2f} & {e['r_mean_a0_nm']:.1f}\\\\")
    open(os.path.join(tdir, "tab_est_body.tex"), "w").write(
        "% generated by scripts/m6_estimates.py -- do not edit by hand\n"
        "\\begin{tabular}{lcccccc}\n"
        "$2\\Delta$ & $\\mu$ & $\\mu_Bm_e/m^*$ & $|\\delta E|$ & $B^*$ & $J_{\\rm iv}$ & $\\langle r\\rangle$\\\\\n"
        "(meV) & ($\\mu_B$) & ($\\mu_B$) & (meV) & (T) & (meV) & (nm)\\\\\n\\hline\n"
        + "\n".join(rows) + "\n\\end{tabular}\n")


if __name__ == "__main__":
    if "--tables" in sys.argv:
        tables(json.load(open(os.path.join(ROOT, "results", "m6", "estimates.json"))))
    else:
        main()
