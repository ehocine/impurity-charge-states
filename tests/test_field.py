"""Tests of the field code (chargestates.field) and of the two-basis three-electron CI."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chargestates.ci3 import ThreeElectronCI  # noqa: E402
from chargestates.field import FieldFurryBasis, magnetic_length  # noqa: E402
from chargestates.furry import FurryBasis  # noqa: E402
from chargestates.interaction import Interaction, compress_shared, interaction_weight  # noqa: E402
from chargestates.observables import MU_B_MEV_PER_T, moment_one, moment_three  # noqa: E402
from collapse2d.analysis import make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

MAT = gapped_graphene(0.015)
M = MAT.m
Z = 0.95 * 1.6713866023
JS = [-0.5, 0.5, 1.5]


@pytest.fixture(scope="module")
def shape():
    return make_solver(MAT, 0.3, "rpa").v


def basis(shape, B=0.0, valley=1, Z=Z, E_max=3.0, R=15.0, bc=None):
    return FieldFurryBasis(shape, Z, M, R=R / M, js=JS, E_max=E_max * M, r_core=0.3, n_scan=200,
                           l_B=magnetic_length(B), valley=valley, bc=bc)


def test_zero_field_equals_furry(shape):
    """At B = 0 the Pruefer-index Furry space is E > -m (FurryBasis), state by state."""
    a = FurryBasis(shape, Z, M, R=15 / M, js=JS, E_max=3 * M, r_core=0.3, n_scan=200)
    b = basis(shape)
    assert len(a.E) == len(b.E)
    assert np.max(np.abs(np.sort(a.E) - np.sort(b.E))) < 1e-10 * M


def test_landau_levels(shape):
    """Without impurity: +-sqrt(m^2 + 2n/l^2); the n = 0 level at +m is an electron state of K' only
    (the K zero modes at -m belong to the sea)."""
    l = magnetic_length(1.0)
    exact = np.sqrt(M * M + 2 * np.arange(0, 5) / l ** 2)
    for valley in (1, -1):
        b = basis(shape, B=1.0, valley=valley, Z=0.0, E_max=5.0, R=12.0)
        for j in JS:
            E = b.E[b.select(j)]
            assert np.all(np.min(np.abs(E[:, None] - exact[None, :]), axis=1) < 1e-8 * M)
        has_zero = any(np.isclose(b.E[b.select(j)][0], M, atol=1e-8 * M) for j in JS)
        assert has_zero == (valley == -1)


def test_low_field_slope_is_orbital_moment(shape):
    """One electron: -dE_1/dB at B -> 0 equals +-<mu> of the B = 0 level in valley K / K'."""
    b0 = basis(shape)
    E0 = b0.E[b0.select(0.5)[0]]
    mu = moment_one(b0, b0.select(0.5)[0])
    dB = 2e-3
    for valley in (1, -1):
        b = basis(shape, B=dB, valley=valley)
        slope = (b.E[b.select(0.5)[0]] - E0) * MAT.hbar_v * 1000 / dB / MU_B_MEV_PER_T
        assert abs(slope + valley * mu) < 0.02 * abs(mu)


def test_three_electron_second_basis(shape):
    """ThreeElectronCI with basis3 = an identical copy of the basis reproduces the one-basis CI;
    with two different bases (K' pair, K electron in a field) H stays symmetric."""
    w = interaction_weight(MAT)
    b = basis(shape, bc="F")
    c = basis(shape, bc="F")
    Ib = Interaction(b, w)
    Ic = Interaction(c, w, q=Ib.q, qw=Ib.qw)
    ch = {j: b.select(j)[:4] for j in JS}
    compress_shared([(Ib, ch), (Ic, {j: c.select(j)[:4] for j in JS})], tol=1e-10)
    e1, _ = ThreeElectronCI(b, Ib, "singlet", 0.5, n_max=4).davidson(tol=1e-11)
    e2, _ = ThreeElectronCI(b, Ib, "singlet", 0.5, n_max=4, basis3=c, inter3=Ic).davidson(tol=1e-11)
    assert abs(e1 - e2) < 1e-10 * M
    # field: K' pair, K third electron
    bp, bk = basis(shape, B=0.5, valley=-1, E_max=2.5, R=10.0), basis(shape, B=0.5, valley=1, E_max=2.5, R=10.0)
    Ip = Interaction(bp, w)
    Ik = Interaction(bk, w, q=Ip.q, qw=Ip.qw)
    compress_shared([(Ip, {j: bp.select(j)[:4] for j in JS}), (Ik, {j: bk.select(j)[:4] for j in JS})], tol=1e-10)
    C = ThreeElectronCI(bp, Ip, "singlet", 0.5, n_max=4, basis3=bk, inter3=Ik)
    rng = np.random.default_rng(3)
    x, y = C.symmetrize(rng.normal(size=C.size)), C.symmetrize(rng.normal(size=C.size))
    assert abs(x @ C.matvec(y) - y @ C.matvec(x)) < 1e-10 * np.linalg.norm(x) * np.linalg.norm(y)
    e, v = C.davidson(tol=1e-10)
    assert np.isfinite(moment_three(C, v))
