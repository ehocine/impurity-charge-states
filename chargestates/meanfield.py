"""Mean-field (Hartree) potentials and the matching one-body matrices.

The Furry projector depends on the one-body potential that defines the basis.  Near threshold
(and above Z_c of the empty level) the natural choice for the added electrons is the impurity
potential plus the Hartree field of the electrons already bound:

    h' = h + V_H,     V_H(r) = sum_occ int d^2r' n_a(r') W(|r - r'|)
                             = sum_occ int dq w(q) J_0(q r) A^0_aa(q),

with w(q) = q W(q)/(2 pi) and the form factors of `interaction.py`.  The many-body Hamiltonian
is still built from the bare h; in the eigenbasis of h' its one-body matrix is
<a|h|b> = E'_a delta_ab - <a|V_H|b>  (`one_body_matrix`).
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import j0


class TabulatedPotential:
    """lam * v(r) + extra(r), with extra tabulated on a log grid (and constant beyond it)."""

    def __init__(self, v, lam, r_tab, extra_tab):
        self.v, self.lam = v, float(lam)
        self._spl = CubicSpline(np.log(r_tab), extra_tab)
        self._r0, self._r1 = r_tab[0], r_tab[-1]
        self._e0, self._e1 = extra_tab[0], extra_tab[-1]
        self.kinks = tuple(getattr(v, "kinks", ()))
        self.point = getattr(v, "point", False)
        # coupling seen by the regular start of a point charge (FurryBasis is then used with lam = 1)
        self.beta = self.lam * v.beta if getattr(v, "beta", None) is not None else None

    def extra(self, r):
        r = np.asarray(r, float)
        out = np.empty_like(r)
        lo, hi = r < self._r0, r > self._r1
        mid = ~(lo | hi)
        out[lo], out[hi] = self._e0, self._e1 * self._r1 / r[hi]   # Coulomb-like beyond the table
        out[mid] = self._spl(np.log(r[mid]))
        return out

    def __call__(self, r):
        return self.lam * self.v(r) + self.extra(r)


def hartree_potential(inter, occupied, r, weight, r_switch=20.0, ogata=None):
    """V_H(r) of electrons in the basis states `occupied` (global indices, one electron each).

    V_H(r) = int dq w(q) A(q) J_0(q r) with A(q) = int R(r') J_0(q r') dr'.  For r < r_switch
    the fixed q nodes of `inter` are used; they alias at large r, where J_0(q r) oscillates many
    times per logarithmic panel.  For r >= r_switch Ogata's quadrature (nodes at the zeros of
    J_0, collapse2d.hankel) is used with A evaluated at its nodes; it is accurate once r exceeds
    ~0.25 of the length on which w A varies (the orbital size), checked in tests/test_m3.py.
    weight: w(q) = q W(q)/(2 pi), as passed to Interaction.
    """
    from collapse2d.hankel import OgataJ0
    B, g = inter.basis, inter.basis.grid
    r = np.asarray(r, float)
    Rd = sum(B.F[a] ** 2 + B.G[a] ** 2 for a in occupied)    # radial density (summed)
    VH = np.zeros_like(r)
    small = r < r_switch
    if np.any(small):
        # own q nodes with a finer logarithmic tail: the default 30 panels of the matrix-element
        # grid give V_H with ~2e-4 relative error at 10-20 nm (480 panels: converged)
        from .interaction import q_nodes
        k_max = np.sqrt(max(B.E_max ** 2 - B.m ** 2, (4 * B.m) ** 2))
        qf, qwf = q_nodes(g.R, k_max, 60.0 / inter.q.max(), q_max=inter.q.max(), dense_panel=8.0,
                          n_log_panels=240)
        dr = np.gradient(g.r)
        Jf = j0(qf[:, None] * g.r[None, :]) * ((qf[:, None] * dr[None, :]) < 1.0)
        A = Jf @ (Rd * g.w)
        VH[small] = (weight(qf) * qwf * A) @ j0(qf[:, None] * r[None, small])
    if np.any(~small):
        O = ogata or OgataJ0(n=300, h=0.01)
        dr = np.gradient(g.r)

        def f(q):
            qf = q.ravel()
            out = np.empty_like(qf)
            for k in range(0, len(qf), 2000):
                qq = qf[k:k + 2000]
                J = j0(qq[:, None] * g.r[None, :]) * ((qq[:, None] * dr[None, :]) < 1.0)
                out[k:k + 2000] = weight(qq) * (J @ (Rd * g.w))
            return out.reshape(q.shape)

        VH[~small] = O(f, r[~small])
    return VH


def mean_field_potential(v, lam, inter, occupied, weight, r_min, r_max, n_r=200):
    """lam * v + V_H[occupied] as a callable potential for FurryBasis (use lam = 1 there)."""
    r_tab = np.geomspace(r_min, r_max, n_r)
    return TabulatedPotential(v, lam, r_tab, hartree_potential(inter, occupied, r_tab, weight))


def one_body_matrix(basis_mf, potential_mf):
    """Callable (j, idx) -> <a|h|b> = E'_a delta_ab - <a|V_H|b> for a basis of h' = h + V_H."""
    g = basis_mf.grid
    VH = potential_mf.extra(g.r)

    def fn(j, idx):
        F, G = basis_mf.F[idx], basis_mf.G[idx]
        M = (F * (VH * g.w)) @ F.T + (G * (VH * g.w)) @ G.T
        return np.diag(basis_mf.E[idx]) - M

    return fn


def scf_hartree(v, lam, m, R, weight, q, qw, r_core, mix=0.4, tol=1e-8, maxit=60, E_orb=2.0, L=None,
                n_tab=200):
    """Self-consistent Hartree potential for two electrons in one spatial orbital (restricted
    Hartree-Fock for the singlet, and the mean field of a K-K' pair): the lowest j = 1/2 state
    of h' = h + V_H[n_a] is the orbital a itself.  Linear mixing of V_H.

    Returns (potential, history of orbital energies in units of m).  The potential is a
    TabulatedPotential for FurryBasis with lam = 1.
    """
    from .furry import FurryBasis
    from .interaction import Interaction
    r_tab = np.geomspace(1e-7, R, n_tab)
    B = FurryBasis(v, lam, m, R=R, js=[0.5], E_max=E_orb * m, r_core=r_core, n_scan=200, L=L)
    VH = hartree_potential(Interaction(B, weight, q=q, qw=qw), [B.select(0.5)[0]], r_tab, weight)
    history = []
    for _ in range(maxit):
        P = TabulatedPotential(v, lam, r_tab, VH)
        B = FurryBasis(P, 1.0, m, R=R, js=[0.5], E_max=E_orb * m, r_core=r_core, n_scan=200, L=L)
        history.append(B.E[B.select(0.5)][0] / m)
        VH_new = hartree_potential(Interaction(B, weight, q=q, qw=qw), [B.select(0.5)[0]], r_tab, weight)
        change = np.max(np.abs(VH_new - VH)) / m
        VH = (1 - mix) * VH + mix * VH_new
        if change < tol:
            break
    return TabulatedPotential(v, lam, r_tab, VH), history
