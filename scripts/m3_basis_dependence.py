"""V6: dependence of the two-electron energy on the basis (Furry projector).

    python3 scripts/m3_basis_dependence.py E_max/Delta [L] [n_scf]

Graphene on hBN, 2Delta = 30 meV, Z = 0.95 Z_c (as m2_*).  E2 (KK singlet, J = 1) from
  bare  : eigenstates of h (impurity potential only)
  mf    : eigenstates of h + V_H[n_a], a = lowest j = 1/2 orbital of h
  scf   : eigenstates of h + V_H[n_a] with a the lowest orbital of that same operator
          (Hartree self-consistency for the singlet = restricted Hartree-Fock; n_scf iterations)
The CI always uses the bare one-body Hamiltonian h (non-diagonal in the mf/scf bases).
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
from chargestates.meanfield import mean_field_potential, one_body_matrix  # noqa: E402
from collapse2d.analysis import make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

DTYPE = np.dtype(os.environ.get("CI_DTYPE", "float64")).type


def main(emax, L=2, n_scf=6):
    mat = gapped_graphene(0.015)
    m = mat.m
    S = make_solver(mat, 0.3, "rpa")
    Z, R = 0.95 * 1.6713866023, 30 / m
    js = [0.5 + l for l in range(-L, L + 1)]
    w = interaction_weight(mat)
    q, qw = q_nodes(R, np.sqrt(emax ** 2 - 1) * m, 0.3, dense_panel=8.0, n_log_panels=30)
    out = {"Emax": emax, "L": L}

    def ci(B, one_body=None):
        I = Interaction(B, w, q=q, qw=qw)
        C = TwoElectronCI(B, I, "singlet", 1.0, dtype=DTYPE, one_body=one_body)
        return float(C.solve(1, method="davidson")[0]) / m, C, I

    t = time.time()
    B0 = FurryBasis(S.v, Z, m, R=R, js=js, E_max=emax * m, r_core=0.3, n_scan=500)
    out["bare"], C0, I0 = ci(B0)
    out["E1_bare"] = B0.E[B0.select(0.5)][0] / m
    # mean field of the bare orbital
    a0 = B0.select(0.5)[0]
    P = mean_field_potential(S.v, Z, I0, [a0], w, 1e-7, R)
    B1 = FurryBasis(P, 1.0, m, R=R, js=js, E_max=emax * m, r_core=0.3, n_scan=500)
    out["mf"], C1, I1 = ci(B1, one_body_matrix(B1, P))
    # Hartree self-consistency (j = 1/2 channel only for the iterations)
    Bs, Is, history = B1, I1, []
    for it in range(n_scf):
        a = Bs.select(0.5)[0]
        Ps = mean_field_potential(S.v, Z, Is, [a], w, 1e-7, R)
        Bj = FurryBasis(Ps, 1.0, m, R=R, js=[0.5], E_max=2 * m, r_core=0.3, n_scan=200)
        e_orb = Bj.E[Bj.select(0.5)][0] / m
        history.append(e_orb)
        if len(history) > 1 and abs(history[-1] - history[-2]) < 1e-7:
            break
        Is = Interaction(Bj, w, q=q, qw=qw)
        Bs = Bj
    Bs = FurryBasis(Ps, 1.0, m, R=R, js=js, E_max=emax * m, r_core=0.3, n_scan=500)
    out["scf"], _, _ = ci(Bs, one_body_matrix(Bs, Ps))
    out["scf_orbital_energies"] = history
    out["time"] = time.time() - t
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(main(float(a[0]), int(a[1]) if len(a) > 1 else 2, int(a[2]) if len(a) > 2 else 6)),
          flush=True)
