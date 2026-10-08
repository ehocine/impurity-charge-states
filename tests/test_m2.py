"""Milestone M2 tests: internal consistency of the two-electron CI."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chargestates.ci import TwoElectronCI  # noqa: E402
from chargestates.furry import FurryBasis  # noqa: E402
from chargestates.interaction import Interaction, interaction_weight  # noqa: E402
from collapse2d.analysis import make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

MAT = gapped_graphene(0.015, kappa=2.5)


@pytest.fixture(scope="module")
def small():
    m = MAT.m
    S = make_solver(MAT, 0.3, "rpa")
    B = FurryBasis(S.v, 1.5, m, R=12 / m, js=[-0.5, 0.5, 1.5], E_max=2.5 * m, r_core=0.3, n_scan=150)
    return B, Interaction(B, interaction_weight(MAT))


def test_single_configuration_is_first_order(small):
    B, I = small
    C = TwoElectronCI(B, I, "singlet", 1.0, js=[0.5], n_max=1)
    assert C.size == 1
    assert C.solve(1)[0] == pytest.approx(C.first_order(), rel=1e-13)


def test_hamiltonian_symmetric(small):
    B, I = small
    for sector, J in (("singlet", 1.0), ("triplet", 1.0), ("intervalley", 0.0)):
        H = TwoElectronCI(B, I, sector, J, n_max=6).hamiltonian()
        assert np.max(np.abs(H - H.T)) < 1e-14 * np.max(np.abs(H))


def test_intervalley_one_channel_contains_singlet_and_triplet(small):
    """With only j = 1/2 (one channel pair), the K-K' problem is the same-valley problem without
    the symmetry constraint, so its spectrum is the union of singlet and triplet spectra."""
    B, I = small
    kw = dict(js=[0.5], n_max=5)
    ev = np.sort(np.linalg.eigvalsh(TwoElectronCI(B, I, "intervalley", 0.0, **kw).hamiltonian()))
    s = np.linalg.eigvalsh(TwoElectronCI(B, I, "singlet", 1.0, **kw).hamiltonian())
    t = np.linalg.eigvalsh(TwoElectronCI(B, I, "triplet", 1.0, **kw).hamiltonian())
    assert np.allclose(ev, np.sort(np.concatenate([s, t])), rtol=0, atol=1e-12 * np.max(np.abs(ev)))


def test_no_interaction_gives_sums_of_orbital_energies(small):
    B, I = small
    I0 = Interaction(B, lambda q: 0.0 * q)
    C = TwoElectronCI(B, I0, "singlet", 1.0, n_max=4)
    E = B.E
    sums = np.sort([E[a] + E[b] for _, _, sa, sb in C.blocks for a, b in zip(sa, sb)])
    assert np.allclose(np.sort(np.linalg.eigvalsh(C.hamiltonian())), sums, atol=1e-15)


@pytest.mark.parametrize("sector,J", [("singlet", 1.0), ("triplet", 1.0), ("intervalley", 0.0)])
def test_matrix_free_matches_dense(small, sector, J):
    B, I = small
    C = TwoElectronCI(B, I, sector, J, n_max=8)
    H = C.hamiltonian()
    v = np.random.default_rng(1).standard_normal(C.size)
    assert np.allclose(C.matvec(v), H @ v, rtol=0, atol=1e-12 * np.max(np.abs(H)) * np.linalg.norm(v))
    assert np.allclose(C.diagonal(), np.diag(H), rtol=0, atol=1e-13 * np.max(np.abs(H)))
    w_dense = np.linalg.eigvalsh(H)[0]
    w_dav, _ = C.davidson(tol=1e-12)
    assert w_dav == pytest.approx(w_dense, abs=1e-11 * abs(w_dense))


@pytest.mark.parametrize("sector,J", [("singlet", 1.0), ("triplet", 1.0), ("intervalley", 0.0)])
def test_nondiagonal_one_body(small, sector, J):
    """A non-diagonal one-body matrix (as in a mean-field basis): matrix-free = dense, and the
    no-interaction spectrum equals the sums of the one-body eigenvalues."""
    B, I = small
    rng = np.random.default_rng(7)
    pert = {}

    def one_body(j, idx):
        if j not in pert:
            X = rng.standard_normal((len(idx), len(idx))) * 0.05 * B.m
            pert[j] = 0.5 * (X + X.T)
        return np.diag(B.E[idx]) + pert[j]

    C = TwoElectronCI(B, I, sector, J, n_max=6, one_body=one_body)
    H = C.hamiltonian()
    v = rng.standard_normal(C.size)
    assert np.allclose(C.matvec(v), H @ v, rtol=0, atol=1e-12 * np.max(np.abs(H)) * np.linalg.norm(v))
    assert np.allclose(C.diagonal(), np.diag(H), rtol=0, atol=1e-13 * np.max(np.abs(H)))
    # without interaction: eigenvalues are e_i + e_k of the one-body matrices
    C0 = TwoElectronCI(B, Interaction(B, lambda q: 0.0 * q), sector, J, n_max=6, one_body=one_body)
    ev = np.sort(np.linalg.eigvalsh(C0.hamiltonian()))
    eps = {j: np.linalg.eigvalsh(t) for j, t in C0.t.items()}
    sums = []
    for ja, jb, sa, sb in C0.blocks:
        ea, eb = eps[ja], eps[jb]
        if sector == "intervalley" or ja != jb:
            sums += [x + y for x in ea for y in eb]
        else:
            n = len(ea)
            sums += [ea[i] + ea[k] for i in range(n) for k in range(n) if (k >= i if sector == "singlet" else k > i)]
    assert np.allclose(ev, np.sort(sums), rtol=0, atol=1e-13)


@pytest.mark.parametrize("sector,J", [("singlet", 1.0), ("intervalley", 0.0)])
def test_compressed_q_matches(small, sector, J):
    """Compressing the q dimension (Interaction.compress) leaves the lowest energy unchanged."""
    B, _ = small
    I = Interaction(B, interaction_weight(MAT))
    C = TwoElectronCI(B, I, sector, J, n_max=8)
    ref = np.linalg.eigvalsh(C.hamiltonian())[0]
    ranks = I.compress(C.channels, tol=1e-9)
    assert max(ranks.values()) < len(I.q) / 2
    Cc = TwoElectronCI(B, I, sector, J, n_max=8)
    w, _ = Cc.davidson(tol=1e-11)
    assert w == pytest.approx(ref, abs=1e-9 * abs(ref))


@pytest.mark.parametrize("sector,J", [("singlet", 1.0), ("intervalley", 0.0)])
def test_dyson_orbital(small, sector, J):
    """Without interaction the two-electron ground state is a0 x a0: the Dyson orbital is a0 itself
    with unit weight; with interaction the weight is below one and its sum rule
    sum_b |d_b|^2 <= 1 holds."""
    B, I = small
    a0 = B.select(0.5)[0]
    C0 = TwoElectronCI(B, Interaction(B, lambda q: 0.0 * q), sector, J, n_max=6)
    w, v = np.linalg.eigh(C0.hamiltonian())
    D0 = C0.dyson(v[:, 0], a0)
    assert D0["Z_qp"] == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(np.abs(D0["F"]), np.abs(B.F[a0]), atol=1e-10)
    C = TwoElectronCI(B, I, sector, J, n_max=6)
    w, v = np.linalg.eigh(C.hamiltonian())
    D = C.dyson(v[:, 0], a0)
    assert 0.5 < D["Z_qp"] < 1.0
