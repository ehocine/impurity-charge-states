"""First-order charging energy U of the j = 1/2 bound level in gapped graphene on hBN.

Direct Coulomb energy of two electrons in the same (unrelaxed) j = 1/2 orbital of the screened
impurity potential of paper 1 (Dirac-sea RPA, impurity at d = 0.3 nm, kappa = 2.5, no gate).
The interaction between electrons in the layer is W(q) = 2 pi alpha0 / (q eps(q)).

This is a go/no-go estimate only: first-order U is NOT reliable here (see NOTES.md).

Needs collapse2d (see chargestates/deps.py).  Run:  python3 scripts/u_estimate.py  (a few minutes).
"""
import os
import sys

import numpy as np
from scipy.special import j0

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chargestates.deps  # noqa: E402,F401  (locates collapse2d)
from collapse2d.analysis import bound_density, critical_charge, make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402
from collapse2d.potentials import ScreenedImpurity  # noqa: E402

HV = 658.2  # hbar v in meV nm
trap = getattr(np, "trapezoid", None) or np.trapz

for gap in (0.03, 0.1, 0.3):                      # 2 Delta in eV
    mat = gapped_graphene(gap / 2)
    S = make_solver(mat, 0.3, "rpa")
    Zc = critical_charge(S, n_scan=60)
    P = ScreenedImpurity(1.0, mat, 0.3)           # only eps(q) is used; it does not depend on d
    for f in (0.8, 0.95):
        Z = f * Zc
        E = S.bound_states(Z, 0.5, n_scan=100, edge=1e-6)[0]
        r = np.geomspace(1e-3, 60 * mat.compton, 4000)
        w = bound_density(S, E, Z, 0.5, list(r))  # F^2 + G^2; 2D density n = w / (2 pi r)
        norm = trap(w, r)
        rmean = trap(w * r, r) / norm
        q = np.geomspace(1e-4 / mat.compton, 200.0, 1500)
        nq = np.array([trap(w * j0(qq * r), r) for qq in q]) / norm   # Fourier transform of n
        W_rpa = 2 * np.pi * mat.alpha0 / (q * P.eps(q))
        W_k = 2 * np.pi * mat.alpha0 / (q * mat.kappa)
        U_rpa = trap(q * nq**2 * W_rpa, q) / (2 * np.pi) * HV
        U_k = trap(q * nq**2 * W_k, q) / (2 * np.pi) * HV
        print(f"2D={gap*1e3:4.0f} meV Zc={Zc:.3f} Z/Zc={f} E/D={E/mat.m:+.3f} <r>={rmean:6.2f} nm "
              f"(lamC={mat.compton:5.1f})  U_rpa={U_rpa:6.1f} meV  U_kappa={U_k:6.1f} meV  "
              f"U_rpa/2D={U_rpa/(gap*1e3):.2f}  room below level E+D={(E+mat.m)*HV:5.1f} meV")
