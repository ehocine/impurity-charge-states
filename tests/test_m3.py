"""Milestone M3 tests: three-electron CI, extended Furry basis, Hartree potential."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chargestates.ci import TwoElectronCI  # noqa: E402
from chargestates.ci3 import ThreeElectronCI  # noqa: E402
from chargestates.furry import FurryBasis  # noqa: E402
from chargestates.interaction import Interaction, interaction_weight, q_nodes  # noqa: E402
from chargestates.meanfield import hartree_potential  # noqa: E402
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
    return B, I


def brute_force(B, I, pair, M):
    """Dense H from uncompressed matrix elements, restricted to the pair-symmetric sector."""
    ch = {j: B.select(j)[:NMAX] for j in JS}
    conf = [(ja, a, jb, b, jc, c) for ja in JS for jb in JS for jc in JS if ja + jb - jc == M
            for a in range(NMAX) for b in range(NMAX) for c in range(NMAX)]
    n = len(conf)
    pos = {x: k for k, x in enumerate(conf)}
    H = np.zeros((n, n))

    def V(j1, i1, j2, i2, j3, i3, j4, i4):            # <12|W|34>, electron 1: 1->3, 2: 2->4
        l = int(round(abs(j1 - j3)))
        return I.matrix_elements([ch[j1][i1]], [ch[j2][i2]], [ch[j3][i3]], [ch[j4][i4]], l)[0, 0, 0, 0]

    for k, (ja, a, jb, b, jc, c) in enumerate(conf):
        H[k, k] += B.E[ch[ja][a]] + B.E[ch[jb][b]] + B.E[ch[jc][c]]
        for kk, (ja2, a2, jb2, b2, jc2, c2) in enumerate(conf):
            if (jc, c) == (jc2, c2) and ja + jb == ja2 + jb2:
                H[k, kk] += V(ja, a, jb, b, ja2, a2, jb2, b2)
            if (jb, b) == (jb2, b2) and ja - ja2 == jc - jc2:
                H[k, kk] += V(ja, a, jc, c, ja2, a2, jc2, c2)
            if (ja, a) == (ja2, a2) and jb - jb2 == jc - jc2:
                H[k, kk] += V(jb, b, jc, c, jb2, b2, jc2, c2)
    s = 1.0 if pair == "singlet" else -1.0
    P = np.zeros((n, n))
    for k, (ja, a, jb, b, jc, c) in enumerate(conf):
        P[k, k] += 0.5
        P[k, pos[(jb, b, ja, a, jc, c)]] += 0.5 * s
    ev, U = np.linalg.eigh(P)
    Q = U[:, ev > 0.5]
    return np.linalg.eigvalsh(Q.T @ H @ Q)


@pytest.mark.parametrize("pair", ["singlet", "triplet"])
def test_three_electron_matches_brute_force(small, pair):
    B, I = small
    C3 = ThreeElectronCI(B, _compressed(B, I), pair, 0.5, js=JS, n_max=NMAX)
    ref = brute_force(B, I, pair, 0.5)
    w, _ = C3.davidson(tol=1e-11)
    assert w == pytest.approx(ref[0], abs=1e-8 * abs(ref[0]))


def _compressed(B, I):
    Ic = Interaction(B, interaction_weight(MAT), q=I.q, qw=I.qw)
    Ic.compress({j: B.select(j)[:NMAX] for j in JS}, tol=1e-10)
    return Ic


def test_only_same_valley_interaction_gives_pair_plus_free_electron(small):
    B, I = small
    Ic = _compressed(B, I)
    C3 = ThreeElectronCI(B, Ic, "singlet", 0.5, js=JS, n_max=NMAX, terms=(1.0, 0.0, 0.0))
    w, _ = C3.davidson(tol=1e-11)
    pair = TwoElectronCI(B, I, "singlet", 1.0, js=JS, n_max=NMAX).solve(1)[0]
    assert w == pytest.approx(pair + B.E[B.select(0.5)[0]], abs=1e-9 * abs(w))


def test_hartree_tail_is_one_electron():
    """V_H of one bound electron -> alpha0/(kappa r) far away (the Dirac sea has no net charge),
    smoothly across the switch from fixed q nodes to Ogata's quadrature at 20 nm."""
    m = MAT.m
    S = make_solver(MAT, 0.3, "rpa")
    R = 30 / m
    B = FurryBasis(S.v, 0.95 * 1.6713866023, m, R=R, js=[0.5], E_max=3 * m, r_core=0.3, n_scan=200)
    w = interaction_weight(MAT)
    q, qw = q_nodes(R, np.sqrt(8) * m, 0.3, dense_panel=8.0, n_log_panels=30)
    I = Interaction(B, w, q=q, qw=qw)
    r = np.array([10.0, 19.999, 20.001, 50.0, 1000.0, 1300.0])
    x = hartree_potential(I, [B.select(0.5)[0]], r, w) * MAT.kappa * r / MAT.alpha0
    assert x[0] == pytest.approx(0.4420538, abs=2e-5)       # converged reference (480 log panels)
    assert x[3] == pytest.approx(0.8774649, abs=2e-5)
    assert abs(x[2] - x[1]) < 1e-4
    assert x[4] == pytest.approx(1.0, abs=1e-3) and x[5] == pytest.approx(1.0, abs=1e-3)


def test_extended_basis_finds_dived_level():
    """Above Z_c the dived j = 1/2 level is a narrow resonance; paper 1, Table VIII:
    E_r = -1.23955 Delta at Z/Z_c = 1.05 (2Delta = 30 meV, Dirac-sea RPA)."""
    m = MAT.m
    S = make_solver(MAT, 0.3, "rpa")
    B = FurryBasis(S.v, 1.05 * 1.6713866023, m, R=30 / m, js=[0.5], E_max=2 * m, r_core=0.3, n_scan=300,
                   E_floor=-3 * m)
    d = B.dived()
    assert len(d) == 1
    assert B.E[d[0]] / m == pytest.approx(-1.23955, abs=3e-4)  # within ~Gamma of the Siegert pole
    assert B.inner_weight[d[0]] > 0.9


def test_three_electron_dyson_without_intervalley_interaction(small):
    """With W_13 = W_23 = 0 the three-electron ground state is (KK pair) x (lowest K' orbital):
    the Dyson amplitude for adding the K' electron is the lowest j~ = 1/2 orbital, weight one."""
    B, I = small
    Ic = _compressed(B, I)
    C3 = ThreeElectronCI(B, Ic, "singlet", 0.5, js=JS, n_max=NMAX, terms=(1.0, 0.0, 0.0))
    e3, v3 = C3.davidson(tol=1e-12)
    C2 = TwoElectronCI(B, I, "singlet", 1.0, js=JS, n_max=NMAX)
    w2, v2 = np.linalg.eigh(C2.hamiltonian())
    full = C2.full_tensor(v2[:, 0])
    assert sum(float((T ** 2).sum()) for T in full.values()) == pytest.approx(1.0, abs=1e-12)
    d, weight = C3.dyson_from_pair(v3, full)
    assert weight == pytest.approx(1.0, abs=1e-8)
    assert abs(d[0.5][0]) == pytest.approx(1.0, abs=1e-6)
