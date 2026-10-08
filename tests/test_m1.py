"""Milestone M1 regression tests: Furry basis (V1) and interaction matrix elements (V2)."""
import os
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import jv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chargestates.furry import FurryBasis, Grid  # noqa: E402
from chargestates.interaction import Interaction, interaction_weight, q_nodes  # noqa: E402
from collapse2d import exact  # noqa: E402
from collapse2d.analysis import make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402
from collapse2d.potentials import PointCoulomb  # noqa: E402

MAT = gapped_graphene(0.015, kappa=2.5)
ZC_30MEV = 1.6713866023          # paper 1, Table III (Dirac-sea RPA, d = 0.3 nm)


class _Radial:
    """Minimal stand-in for a basis: given radial functions F, G = 0 on a grid."""

    def __init__(self, grid, Fs):
        self.grid, self.F = grid, np.array(Fs)
        self.G = np.zeros_like(self.F)
        self.E_max, self.m = 0.0, 1.0


@pytest.mark.parametrize("beta", [0.2, 0.45])
def test_v1_point_coulomb_levels(beta):
    # R = 120/m: the j = -1/2 level at beta = 0.2 (decay length 7.4/m) is box-converged to 1e-12
    B = FurryBasis(PointCoulomb(1.0), beta, 1.0, R=120.0, js=[-0.5, 0.5], E_max=2.0, r_core=1.0, n_scan=200)
    for j in (-0.5, 0.5):
        ex = exact.dirac_coulomb_levels(beta, j, n_max=3)
        ex = ex[1:] if j < 0 else ex
        assert abs(B.E[B.select(j)][0] - ex[0]) < 1e-8
        assert np.max(np.abs(B.overlap(j) - np.eye(len(B.select(j))))) < 1e-9


def test_v1_screened_bound_states_match_collapse2d():
    m = MAT.m
    S = make_solver(MAT, 0.3, "rpa")
    Z = 0.95 * ZC_30MEV
    B = FurryBasis(S.v, Z, m, R=30 / m, js=[-0.5, 0.5], E_max=3 * m, r_core=0.3, n_scan=200)
    for j in (-0.5, 0.5):
        ref = S.bound_states(Z, j, n_scan=240)
        assert abs(B.E[B.select(j)][0] - ref[0]) < 1e-8 * m


def test_v2_hankel_gaussians():
    g = Grid(1e-6, 60.0, 5.0, 0.01)
    r, sig = g.r, 1.3
    R0 = r * np.exp(-r ** 2 / (2 * sig ** 2)) / sig ** 2               # 2 pi r n(r), Gaussian n
    R1 = 2 * np.pi * r ** 2 * np.exp(-r ** 2 / 2)                      # A^1(q) = 2 pi q e^{-q^2/2}
    q, qw = q_nodes(60.0, 0.0, 0.3, q_max=40.0)
    I = Interaction(_Radial(g, [np.sqrt(R0), np.sqrt(R1)]), interaction_weight(MAT, kappa_only=True), q=q, qw=qw)
    c = MAT.alpha0 / MAT.kappa
    assert I.matrix_elements([0], [0], [0], [0], 0)[0, 0, 0, 0] == pytest.approx(c * np.sqrt(np.pi) / (2 * sig), rel=1e-10)
    assert I.matrix_elements([1], [1], [1], [1], 1)[0, 0, 0, 0] == pytest.approx(c * np.pi ** 2.5, rel=1e-10)


def test_v2_mixed_multipole_against_quadrature():
    g = Grid(1e-6, 60.0, 5.0, 0.01)
    sig = 1.3
    Fa = lambda r: np.sqrt(r * np.exp(-r ** 2 / (2 * sig ** 2)) / sig ** 2)  # noqa: E731
    Fc = lambda r: r * np.exp(-r ** 2 / 4)                                  # noqa: E731
    q, qw = q_nodes(60.0, 0.0, 0.3, q_max=40.0)
    I = Interaction(_Radial(g, [Fa(g.r), Fc(g.r)]), interaction_weight(MAT, kappa_only=True), q=q, qw=qw)
    V = I.matrix_elements([0], [1], [1], [0], 1)[0, 0, 0, 0]
    A = lambda k: quad(lambda r: Fa(r) * Fc(r) * jv(1, k * r), 0, 40, limit=400)[0]  # noqa: E731
    ref = MAT.alpha0 / MAT.kappa * quad(lambda k: A(k) ** 2, 0, 30, limit=200)[0]
    assert V == pytest.approx(ref, rel=1e-5)


def test_v2_charging_energy_matches_first_order_estimate():
    """U of the j = 1/2 orbital at Z/Z_c = 0.95, 2Delta = 30 meV (scripts/u_estimate.py: 21.0 meV)."""
    m = MAT.m
    S = make_solver(MAT, 0.3, "rpa")
    B = FurryBasis(S.v, 0.95 * ZC_30MEV, m, R=30 / m, js=[0.5], E_max=3 * m, r_core=0.3, n_scan=150)
    I = Interaction(B, interaction_weight(MAT))
    i0 = B.select(0.5)[:1]
    U = I.matrix_elements(i0, i0, i0, i0, 0)[0, 0, 0, 0] * MAT.hbar_v * 1000
    assert U == pytest.approx(21.0, abs=0.06)
