"""Tests of the generic two- to four-electron CI (chargestates.ci4)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chargestates.ci import TwoElectronCI  # noqa: E402
from chargestates.ci3 import ThreeElectronCI  # noqa: E402
from chargestates.ci4 import ManyElectronCI, ValleyInteraction, lowest_state  # noqa: E402
from chargestates.furry import FurryBasis  # noqa: E402
from chargestates.interaction import Interaction, interaction_weight  # noqa: E402
from collapse2d.analysis import make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

MAT = gapped_graphene(0.015, kappa=2.5)
JS = [-0.5, 0.5, 1.5]
NMAX = 3


@pytest.fixture(scope="module")
def small():
    m = MAT.m
    S = make_solver(MAT, 0.3, "rpa")
    B = FurryBasis(S.v, 1.5, m, R=12 / m, js=JS, E_max=2.5 * m, r_core=0.3, n_scan=150)
    I = Interaction(B, interaction_weight(MAT))
    ch = {j: B.select(j)[:NMAX] for j in JS}
    ti = ValleyInteraction({1: (B, I), -1: (B, I)}, {1: ch, -1: ch}, tol=1e-12)
    return B, I, ti


def test_two_and_three_electrons_match(small):
    """The generic CI reproduces TwoElectronCI (singlet, K-K') and ThreeElectronCI (pair + K')."""
    B, I, ti = small
    for sec, val, sym, M in (("singlet", (1, 1), 1, 1.0), ("triplet", (1, 1), -1, 1.0), ("intervalley", (1, -1), 0, 0.0)):
        ref = TwoElectronCI(B, I, sec, M, n_max=NMAX).solve(1, method="dense")[0]
        e, _ = ManyElectronCI(ti, val, sym, M).davidson(tol=1e-11)
        assert abs(e - ref) < 1e-9 * B.m
    for pair, sym in (("singlet", 1), ("triplet", -1)):
        ref, _ = ThreeElectronCI(B, I, pair, 0.5, n_max=NMAX).davidson(tol=1e-11)
        e, _ = ManyElectronCI(ti, (1, 1, -1), sym, 0.5).davidson(tol=1e-11)
        assert abs(e - ref) < 1e-9 * B.m


def test_four_electrons(small):
    """N = 4: symmetric H, exact noninteracting limit, rigorous lower bound, and valley mirror symmetry
    at B = 0 (the (K, K, K, K') and (K', K', K', K) sectors are degenerate)."""
    B, I, ti = small
    rng = np.random.default_rng(4)
    for val, sym, s23, M in (((1, 1, 1, 1), -1, -1, 2.0), ((1, 1, -1, -1), 1, 1, 0.0), ((1, 1, -1, -1), -1, 1, 0.0),
                             ((1, 1, 1, -1), -1, 0, 1.0)):
        c = ManyElectronCI(ti, val, sym, M, sym23=s23)
        x, y = c.symmetrize(rng.normal(size=c.size)), c.symmetrize(rng.normal(size=c.size))
        assert abs(x @ c.matvec(y) - y @ c.matvec(x)) < 1e-10 * np.linalg.norm(x) * np.linalg.norm(y)
        off = {p: 0.0 for p in c.pairs}
        e0, _ = ManyElectronCI(ti, val, sym, M, terms=off, sym23=s23).davidson(tol=1e-11)
        e, _ = c.davidson(tol=1e-10)
        assert abs(c.min_one_body() - e0) < 1e-9 * B.m and c.min_one_body() < e
    a, _ = ManyElectronCI(ti, (1, 1, 1, -1), -1, 1.0).davidson(tol=1e-11)
    b, _ = ManyElectronCI(ti, (-1, -1, -1, 1), -1, -1.0).davidson(tol=1e-11)
    assert abs(a - b) < 1e-9 * B.m
    # the skipping by the lower bound does not change the ground state: compare with all sectors
    import itertools
    from chargestates.ci4 import sectors
    E4, g4, _ = lowest_state(ti, 4, JS)
    full = []
    for val, sym, s23 in sectors(4):
        for M in sorted({round(sum(v * j for v, j in zip(val, t)), 1) for t in itertools.product(*[JS] * 4)}):
            c = ManyElectronCI(ti, val, sym, M, sym23=s23)
            if c.size and np.isfinite(c.min_one_body()):
                full.append(c.davidson(tol=1e-10)[0])
    assert abs(E4 - min(full)) < 1e-9 * B.m
