"""Tests of chargestates.observables (magnetic moments, self-energy inputs, form factors)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chargestates import observables as ob  # noqa: E402
from chargestates.ci import TwoElectronCI  # noqa: E402
from chargestates.furry import FurryBasis  # noqa: E402
from chargestates.interaction import Interaction, interaction_weight  # noqa: E402
from collapse2d.analysis import make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

MAT = gapped_graphene(0.015, kappa=2.5)


def test_moment_tends_to_band_edge_value_for_shallow_level():
    """A weakly bound level inherits the band-edge valley moment mu_B m_e/m* (Xiao et al. 2007)."""
    m = MAT.m
    S = make_solver(MAT, 0.3, "rpa")
    ratios = []
    for Z in (0.6, 0.8):
        B = FurryBasis(S.v, Z, m, R=120 / m, js=[0.5], E_max=1.5 * m, r_core=0.3, n_scan=300)
        a0 = B.select(0.5)[0]
        ratios.append(abs(ob.moment_one(B, a0)) / ob.band_edge_moment(MAT))
    assert ratios[0] > ratios[1]               # shallower level: closer to the band-edge value
    assert 0.85 < ratios[0] < 1.0


def test_two_electron_moments():
    """Without interaction the singlet is a0 x a0: moment exactly 2 m1.  With interaction the K-K'
    ground state is symmetric under exchanging the roles of the two electrons (same one-body
    matrices, symmetric W), so its moment cancels exactly; the same-valley pair has more than
    twice m1 because the outer orbital is more extended."""
    m = MAT.m
    S = make_solver(MAT, 0.3, "rpa")
    B = FurryBasis(S.v, 1.5, m, R=12 / m, js=[-0.5, 0.5, 1.5], E_max=2.5 * m, r_core=0.3, n_scan=150)
    a0 = B.select(0.5)[0]
    m1 = ob.moment_one(B, a0)
    I0 = Interaction(B, lambda q: 0.0 * q)
    C0 = TwoElectronCI(B, I0, "singlet", 1.0, n_max=6)
    w, v = np.linalg.eigh(C0.hamiltonian())
    assert ob.moment_two(C0, v[:, 0]) == pytest.approx(2 * m1, rel=1e-10)
    I = Interaction(B, interaction_weight(MAT))
    C = TwoElectronCI(B, I, "intervalley", 0.0, n_max=6)
    w, v = np.linalg.eigh(C.hamiltonian())
    assert abs(ob.moment_two(C, v[:, 0])) < 1e-8 * abs(m1)
    C = TwoElectronCI(B, I, "singlet", 1.0, n_max=6)
    w, v = np.linalg.eigh(C.hamiltonian())
    assert ob.moment_two(C, v[:, 0]) / m1 > 2.0


def test_dynamic_self_energy_limits():
    """sigma_dynamic interpolates between 0 (instantaneous sea, w_bar -> 0) and the slow-sea bound
    (w_bar -> infinity), and is monotonic in w_bar."""
    m = MAT.m
    S = make_solver(MAT, 0.3, "rpa")
    B = FurryBasis(S.v, 1.5, m, R=12 / m, js=[0.5], E_max=2.5 * m, r_core=0.3, n_scan=150)
    w, w_env = interaction_weight(MAT), interaction_weight(MAT, rpa=False)
    I = Interaction(B, w)
    a0 = B.select(0.5)[0]
    F, G = B.F[a0], B.G[a0]
    slow = ob.sigma_slow_orbital(I, MAT, F, G, w_env)
    assert slow < 0
    assert ob.sigma_dynamic(I, MAT, F, G, 1e-9, w_env) == pytest.approx(0.0, abs=1e-6 * abs(slow))
    assert ob.sigma_dynamic(I, MAT, F, G, 1e9, w_env) == pytest.approx(slow, rel=1e-6)
    vals = [ob.sigma_dynamic(I, MAT, F, G, x, w_env) for x in (0.5, 1.0, 2.0, 4.0)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert ob.slow_sea(I, MAT, a0, w_env)[0] == pytest.approx(slow, rel=1e-10)
