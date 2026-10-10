"""Electron-electron interaction in the Furry basis.

For orbitals psi = (2 pi r)^{-1/2} (F e^{i(j-1/2)theta}, i G e^{i(j+1/2)theta}) of one valley,
the pair density is psi_a^dag psi_c = R_ac(r) e^{i(j_c - j_a)theta} / (2 pi r) with
R_ac = F_a F_c + G_a G_c.  With the multipole expansion of an isotropic interaction,
W(|r1 - r2|) = sum_l e^{il(theta1 - theta2)} (1/2pi) int q dq W(q) J_l(q r1) J_l(q r2),
the two-electron matrix element (electron 1: a -> c, electron 2: b -> d) is

    <ab|W|cd> = int_0^inf dq  w(q) A^l_ac(q) A^l_bd(q),     w(q) = q W(q) / (2 pi),
    A^l_ac(q) = int_0^R dr R_ac(r) J_l(q r),                 l = |j_a - j_c|,

with the selection rule j_a - j_c = j_d - j_b for two electrons in the same valley.
A K' orbital is the complex conjugate of the K orbital with the same radial functions,
so its pair density carries e^{-i(j_c - j_a)theta}: for electron 2 in the other valley
the selection rule becomes j_a - j_c = j_b - j_d, with the same formula for the value.
Intervalley exchange and scattering (lattice-scale oscillating densities) are neglected.

The in-plane interaction is W(q) = 2 pi alpha0 G(q) / (q eps(q)), with eps(q) and the gate
factor G(q) = 1 - exp(-2qD) of paper 1; the Dirac sea's linear response is inside eps.
"""
import numpy as np
from scipy.special import jv

from .deps import collapse2d  # noqa: F401
from collapse2d.potentials import ScreenedImpurity


def interaction_weight(mat, D=np.inf, rpa=True, kappa_only=False):
    """w(q) = q W(q)/(2 pi) in units hbar v = 1 (finite as q -> 0)."""
    if kappa_only:
        return lambda q: mat.alpha0 / mat.kappa * np.ones_like(np.asarray(q, float))
    # eps(q) and G(q) exactly as in paper 1 (no table of the potential is needed)
    P = ScreenedImpurity(1.0, mat, 1.0, D=D, rpa=rpa, r_max=10.0, n_r=16)
    return lambda q: mat.alpha0 * P.G(np.asarray(q, float)) / P.eps(np.asarray(q, float))


def q_nodes(R, k_max, r_core, q_max=None, dense_panel=4.0, n_dense=8, n_log_panels=60, n_log=16):
    """Quadrature nodes and weights on [0, q_max).

    Form factors of box states oscillate in q on the scale 1/R up to q ~ 2 k_max, so
    the first part is covered with Gauss-Legendre panels of width dense_panel/R (8 points
    each; widths 1/R to 8/R change matrix elements by < 5e-10 nm^-1, i.e. ~1e-7 relative,
    for 2Delta = 30 meV); the tail up to q_max = 60/r_core with logarithmically spaced panels.
    """
    q_max = q_max if q_max is not None else 60.0 / r_core
    xg, wg = np.polynomial.legendre.leggauss(n_dense)
    q_a = 3.0 * k_max + 20.0 / R
    edges = np.arange(0.0, q_a + dense_panel / R, dense_panel / R)
    edges = np.concatenate([edges, np.geomspace(edges[-1], q_max, n_log_panels + 1)[1:]])
    qs, ws = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = n_dense if hi <= q_a + dense_panel / R else n_log
        x, w = (xg, wg) if n == n_dense else np.polynomial.legendre.leggauss(n)
        qs.append(0.5 * (hi + lo) + 0.5 * (hi - lo) * x)
        ws.append(0.5 * (hi - lo) * w)
    return np.concatenate(qs), np.concatenate(ws)


class Interaction:
    """Form factors and two-electron matrix elements for a FurryBasis."""

    def __init__(self, basis, weight, q=None, qw=None, r_core=0.3, resolve=1.0):
        self.basis = basis
        g = basis.grid
        if q is None:
            k_max = np.sqrt(max(basis.E_max ** 2 - basis.m ** 2, 0.0))
            q, qw = q_nodes(g.R, k_max, r_core)
        self.q, self.qw = q, qw
        self.wq = weight(q) * qw                 # includes the quadrature weights
        # local grid spacing in r; J_l(q r) is only sampled where q dr < resolve, so
        # unresolved oscillations at large r (where the integrand is negligible for the
        # q concerned) do not alias into the form factors
        dr = np.gradient(g.r)
        self._mask = (q[:, None] * dr[None, :]) < resolve
        self._J = {}

    def _bessel(self, l):
        if l not in self._J:
            g = self.basis.grid
            self._J[l] = (jv(l, self.q[:, None] * g.r[None, :]) * self._mask * g.w[None, :]).T
        return self._J[l]                        # (n_r, n_q), includes Simpson weights

    def form_factors(self, ia, ic, l):
        """A^l_ac(q) for all a in ia, c in ic: array (len(ia), len(ic), n_q)."""
        B = self.basis
        Rac = B.F[ia][:, None, :] * B.F[ic][None, :, :] + B.G[ia][:, None, :] * B.G[ic][None, :, :]
        na, nc, nr = Rac.shape
        return (Rac.reshape(na * nc, nr) @ self._bessel(l)).reshape(na, nc, -1)

    def matrix_elements(self, ia, ib, ic, id_, l):
        """<ab|W|cd> for index lists ia, ib, ic, id_: array (a, b, c, d) for multipole l."""
        Aac = self.form_factors(ia, ic, l)
        Abd = self.form_factors(ib, id_, l)
        na, nc, nq = Aac.shape
        nb, nd, _ = Abd.shape
        V = (Aac.reshape(na * nc, nq) * self.wq) @ Abd.reshape(nb * nd, nq).T
        return V.reshape(na, nc, nb, nd).transpose(0, 2, 1, 3)

    # -- compression of the q dimension ------------------------------------------------
    def compress(self, channels, tol=1e-9):
        """Project sqrt(w) A^l onto an orthonormal basis of q-profiles, one per multipole l.

        channels: dict j -> state indices (the channels a CI will use).  For each l the Gram
        matrix G = sum (A sqrt w)^T (A sqrt w) over all channel pairs with |j - j'| = l is
        diagonalised and eigenvectors with eigenvalue > tol^2 * max are kept.  Then
        <ab|W|cd> = sum_k C_ac,k C_bd,k with C = (A sqrt w) Phi_l, to relative accuracy ~tol.
        Typical ranks: ~130 of ~1000 q nodes at tol = 1e-8 (graphene, E_max = 6m).
        """
        sw = np.sqrt(self.wq)
        js = sorted(channels)
        grams = {}
        for ja in js:
            for jc in js:
                l = int(round(abs(ja - jc)))
                A = self.form_factors(channels[ja], channels[jc], l).reshape(-1, len(self.q)) * sw
                grams[l] = grams.get(l, 0.0) + A.T @ A
        self._phi = {}
        for l, G in grams.items():
            lam, U = np.linalg.eigh(G)
            keep = lam > tol ** 2 * lam[-1]
            self._phi[l] = U[:, keep]
        self.ranks = {l: P.shape[1] for l, P in self._phi.items()}
        return self.ranks

    def form_factors_compressed(self, ia, ic, l):
        """(A^l_ac sqrt w) Phi_l: array (len(ia), len(ic), rank_l); requires compress()."""
        A = self.form_factors(ia, ic, l)
        return (A * np.sqrt(self.wq)) @ self._phi[l]


def compress_shared(pairs, tol=1e-9):
    """Compress several Interaction objects on the same q nodes with one set of q-profiles.

    pairs: list of (Interaction, channels) with channels a dict j -> state indices.  Needed when
    the electrons of a CI live in different bases (valleys K and K' in a field, field.py): the
    factorisation <ab|W|cd> = sum_k C_ac,k C_bd,k requires the same Phi_l for both form factors.
    Sets _phi on every Interaction; returns the ranks.
    """
    q = pairs[0][0].q
    grams = {}
    for inter, channels in pairs:
        if inter.q.shape != q.shape or np.any(inter.q != q):
            raise ValueError("compress_shared needs the same q nodes for all interactions")
        sw = np.sqrt(inter.wq)
        js = sorted(channels)
        for ja in js:
            for jc in js:
                l = int(round(abs(ja - jc)))
                A = inter.form_factors(channels[ja], channels[jc], l).reshape(-1, len(q)) * sw
                grams[l] = grams.get(l, 0.0) + A.T @ A
    phi = {}
    for l, G in grams.items():
        lam, U = np.linalg.eigh(G)
        phi[l] = U[:, lam > tol ** 2 * lam[-1]]
    for inter, _ in pairs:
        inter._phi = phi
        inter.ranks = {l: P.shape[1] for l, P in phi.items()}
    return {l: P.shape[1] for l, P in phi.items()}
