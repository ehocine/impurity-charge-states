"""M11: sensitivity of the gated results to the dielectric model of the gate.

    python3 scripts/m11_gate_kernel.py OUT.jsonl GAP_meV D_nm N_MAX3 X1 [X2 ...]     (X = Z/Z_c of the
                                                                                    homogeneous model)

Paper 1 (and this work) combines the gate with the mean dielectric constant kappa = (e_t + e_b)/2:
U, W ~ G(q)/eps(q), G = 1 - exp(-2qD), which is exact only for e_t = e_b.  For vacuum above the layer
(e_t = 1) and hBN between layer and gate (e_b = 4, kappa = 2.5) the exact electrostatics replaces
kappa/G by kappa_eff(q) = (e_t + e_b coth qD)/2 in both the impurity potential and the interaction.
For each charge (given as Z/Z_c of the homogeneous model) this script computes, with both kernels,
Z_c, E_1, the lowest two-electron energy (all basis states) and E_3 (n_max = N_MAX3 per channel),
in units of Delta, so that the addition energies can be compared at equal Z and at equal Z/Z_c.
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
from collapse2d.dirac import RadialDirac  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402
from collapse2d.potentials import ScreenedImpurity, polarization  # noqa: E402

L_CH, EMAX, RFAC, D_IMP = 2, 6.0, 30.0, 0.3
E_TOP, E_BOT = 1.0, 4.0


def kappa_eff(q, D):
    x = np.asarray(q, float) * D
    return 0.5 * (E_TOP + E_BOT / np.tanh(np.where(x > 0, x, 1e-300)))


class TwoMediumImpurity(ScreenedImpurity):
    """Paper 1's screened impurity with the exact vacuum / dielectric-spacer / metal kernel."""

    def f(self, q):
        q = np.asarray(q, float)
        e = kappa_eff(q, self.D) + self.rK * q
        if self.N:
            e = e + 2 * np.pi * self.alpha0 * self.N * polarization(q, self.m) / np.where(q > 0, q, 1.0)
        return np.exp(-q * self.d) / e


def two_medium_weight(mat, D):
    """w(q) = q W/2 pi with the exact kernel."""
    def w(q):
        q = np.asarray(q, float)
        return mat.alpha0 / (kappa_eff(q, D) + 2 * np.pi * mat.alpha0 * mat.N * polarization(q, mat.m)
                             / np.where(q > 0, q, 1.0))
    return w


def energies(v, w, Z, m, n_max3):
    R = RFAC / m
    js = [0.5 + l for l in range(-L_CH, L_CH + 1)]
    q, qw = q_nodes(R, np.sqrt(EMAX ** 2 - 1) * m, D_IMP, dense_panel=8.0, n_log_panels=30)
    B = FurryBasis(v, Z, m, R=R, js=js, E_max=EMAX * m, r_core=D_IMP, n_scan=500)
    I = Interaction(B, w, q=q, qw=qw)
    I.compress({j: B.select(j) for j in js}, tol=1e-8)
    e1 = float(B.E[B.select(0.5)][0]) / m
    e2 = min(float(TwoElectronCI(B, I, sec, J).davidson(tol=1e-10)[0]) for sec, J in
             (("singlet", 1.0), ("intervalley", 0.0))) / m
    I3 = Interaction(B, w, q=q, qw=qw)
    I3.compress({j: B.select(j)[:n_max3] for j in js}, tol=1e-8)
    e2t = float(TwoElectronCI(B, I3, "intervalley", 0.0, n_max=n_max3).davidson(tol=1e-10)[0]) / m
    e3 = float(ThreeElectronCI(B, I3, "singlet", 0.5, n_max=n_max3).davidson(tol=1e-9)[0]) / m
    return {"E1": e1, "E2": e2, "mu1": e1, "mu2": e2 - e1, "mu3_trunc": e3 - e2t}


def main():
    out, gap_meV, D, n_max3 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    xs = [float(x) for x in sys.argv[5:]]
    mat = gapped_graphene(gap_meV / 2000.0)
    m = mat.m
    S0 = make_solver(mat, D_IMP, "rpa", D=D)
    v1 = TwoMediumImpurity(1.0, mat, D_IMP, D=D)
    S1 = RadialDirac(v1, m, r_core=D_IMP)
    Zc0, Zc1 = critical_charge(S0), critical_charge(S1)
    print(f"Z_c: homogeneous {Zc0:.4f}, exact {Zc1:.4f}", flush=True)
    w0, w1 = interaction_weight(mat, D=D), two_medium_weight(mat, D)
    for x in xs:
        t = time.time()
        rec = {"gap_meV": gap_meV, "D": D, "x_homogeneous": x, "Zc_homogeneous": Zc0, "Zc_exact": Zc1,
               "n_max3": n_max3}
        Z = x * Zc0
        rec["homogeneous"] = energies(S0.v, w0, Z, m, n_max3)
        rec["exact_same_Z"] = energies(v1, w1, Z, m, n_max3)
        rec["exact_same_x"] = energies(v1, w1, x * Zc1, m, n_max3)
        rec["time"] = time.time() - t
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        h, a, b = rec["homogeneous"], rec["exact_same_Z"], rec["exact_same_x"]
        print(f"x={x}: mu1 {h['mu1']:.4f} / {a['mu1']:.4f} / {b['mu1']:.4f}; mu2 {h['mu2']:.4f} / {a['mu2']:.4f} / "
              f"{b['mu2']:.4f}; mu3 {h['mu3_trunc']:.4f} / {a['mu3_trunc']:.4f} / {b['mu3_trunc']:.4f} "
              f"(homogeneous / exact same Z / exact same Z/Z_c)  t={rec['time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
