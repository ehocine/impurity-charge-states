"""One-body observables and order-of-magnitude estimates in the Furry basis.

Orbital magnetic moment.  The current operator of h is j = e v sigma (valley K), so
    mu_z = (1/2)(x j_y - y j_x)  ->  <a|mu_z|b> = -(e v/2) int (F_a G_b + G_a F_b) r dr   (same j),
in units of mu_B = e hbar/2m_e:  mu_ab / mu_B = -(m_e v/hbar) int (F_a G_b + G_a F_b) r dr.
For K' (h_K' = h_K^*) the current is e v (-sigma_x, sigma_y) and <a'|mu_z|b'> = -<a|mu_z|b>.

Polarization self-energy (slow-sea limit) and the coupling to excitations.  With the part of
the interaction induced by the Dirac sea, W - v_env (v_env: environment and gate only, i.e.
interaction_weight(..., rpa=False); the metal's image response is instantaneous and its
self-image energy is the same everywhere in the layer), an electron in orbital a couples to
transitions a -> n with strength
    K_an = int dq [w_env(q) - w(q)] A^l_an(q)^2,   w = q W/2 pi,  l = |j_a - j_n|,
and the slow-sea self-energy (polarization built on the averaged density) is -K_aa/2.  The
K-weighted mean excitation energy of the other transitions sets the dynamical reduction
(Appendix of the manuscript): Sigma - Sigma_inst ~ Sigma_slow * w_bar / (omega_p + w_bar).

Moire form factors.  A moire perturbation with wave vector G couples through the orbital's
scalar density (scalar potential), its sigma_z density F^2 - G^2 (mass modulation) and its
current density (pseudo-gauge field); their isotropic Fourier components at |G|:
    n(G) = int (F^2+G^2) J0(Gr) dr,  n_z(G) = int (F^2-G^2) J0(Gr) dr,  n_j(G) = int 2FG J1(Gr) dr.

Intervalley exchange (contact estimate): J_iv ~ W(|K - K'|) int |phi|^4 d^2r.
"""
import numpy as np
from scipy.special import j0, j1

MEV_HV = None          # set from the material: energies in nm^-1 times hbar v (meV nm)
ME_V_OVER_HBAR = 9.1093837e-31 * 1.0e6 / 1.054571817e-34 * 1e-9    # m_e v / hbar in nm^-1 (v = 1e6 m/s)
MU_B_MEV_PER_T = 0.0578838
A_LATTICE = 0.246                                                  # nm
K_KPRIME = 4 * np.pi / (3 * A_LATTICE) * np.sqrt(3)                # |K - K'| in nm^-1


# ---------------------------------------------------------------------------------------
# magnetic moment
# ---------------------------------------------------------------------------------------

def moment_matrix(basis, idx):
    """<a|mu_z|b>/mu_B for the states idx (one channel), valley K."""
    g = basis.grid
    F, G = basis.F[idx], basis.G[idx]
    M = (F * (g.r * g.w)) @ G.T
    return -ME_V_OVER_HBAR * (M + M.T)


def moment_one(basis, a):
    return float(moment_matrix(basis, [a])[0, 0])


def moment_two(ci, vec):
    """<mu_z>/mu_B of a TwoElectronCI eigenvector; for 'intervalley' electron 2 is in K'."""
    full = ci.full_tensor(vec)
    s2 = -1.0 if ci.sector == "intervalley" else 1.0
    Mj = {j: moment_matrix(ci.basis, idx) for j, idx in ci.channels.items()}
    tot = 0.0
    for (ja, jb), T in full.items():
        tot += np.sum(T * (Mj[ja] @ T)) + s2 * np.sum(T * (T @ Mj[jb].T))
    return float(tot)


def moment_three(ci3, vec):
    """<mu_z>/mu_B of a ThreeElectronCI eigenvector (electrons 1, 2 in K, electron 3 in K')."""
    Mj = {j: moment_matrix(ci3.basis, idx) for j, idx in ci3.channels.items()}
    tot = 0.0
    for (ja, jb, jc), P in ci3.full_tensor(vec).items():
        tot += np.sum(P * np.einsum("aA,Abc->abc", Mj[ja], P))
        tot += np.sum(P * np.einsum("bB,aBc->abc", Mj[jb], P))
        tot -= np.sum(P * np.einsum("cC,abC->abc", Mj[jc], P))
    return float(tot)


def band_edge_moment(mat):
    """mu_B m_e/m*, m* = Delta/v^2 (valley moment at the band edge, Xiao et al. 2007)."""
    return 9.1093837e-31 * 1e12 / (mat.Delta * 1.602176634e-19)


# ---------------------------------------------------------------------------------------
# polarization self-energy and coupling to excitations
# ---------------------------------------------------------------------------------------

def _kernel(inter, w_env):
    """[w_env(q) - w(q)] dq on the q nodes of inter."""
    return (w_env(inter.q) - inter.wq / inter.qw) * inter.qw


def slow_sea(inter, mat, a, w_env):
    """Sigma_slow (meV) of orbital a and the K-weighted mean excitation energy (units of m) of
    its transitions to the other basis states (all channels of the basis)."""
    B = inter.basis
    kern = _kernel(inter, w_env)
    K_aa = None
    num = den = 0.0
    for j in np.unique(B.j):
        idx = B.select(j)
        l = int(round(abs(j - B.j[a])))
        A = inter.form_factors([a], idx, l)[0]                            # (n_j, n_q)
        K = (A ** 2) @ kern
        for k, n in enumerate(idx):
            if n == a:
                K_aa = K[k]
                continue
            num += K[k] * (B.E[n] - B.E[a])
            den += K[k]
    return -0.5 * K_aa * mat.hbar_v * 1000, num / den / B.m


# ---------------------------------------------------------------------------------------
# moire form factors and intervalley exchange
# ---------------------------------------------------------------------------------------

def moire_form_factors(basis, F, G, lam_M=14.0):
    """Scalar, mass and current form factors of the orbital (F, G) at G_M = 4 pi/(sqrt3 lambda_M)."""
    g = basis.grid
    GM = 4 * np.pi / (np.sqrt(3) * lam_M)
    x = GM * g.r
    return {"G_M": GM,
            "scalar": float(((F ** 2 + G ** 2) * j0(x)) @ g.w),
            "mass": float(((F ** 2 - G ** 2) * j0(x)) @ g.w),
            "current": float((2 * F * G * j1(x)) @ g.w)}


def intervalley_exchange(basis, F, G, weight, mat):
    """Contact estimate W(|K-K'|) int |phi|^4 d^2r in meV (upper estimate: no atomic form factors)."""
    g = basis.grid
    W = 2 * np.pi * weight(np.array([K_KPRIME]))[0] / K_KPRIME
    n4 = float(((F ** 2 + G ** 2) ** 2 / (2 * np.pi * g.r)) @ g.w)
    return W * n4 * mat.hbar_v * 1000


def sigma_slow_orbital(inter, mat, F, G, w_env):
    """Slow-sea self-energy (meV) of an arbitrary normalised orbital (F, G) on the basis grid."""
    kern = _kernel(inter, w_env)
    A = (F ** 2 + G ** 2) @ inter._bessel(0)
    return -0.5 * float(kern @ A ** 2) * mat.hbar_v * 1000


def sigma_dynamic(inter, mat, F, G, w_bar, w_env):
    """Single-pole estimate of the localisation-dependent polarization self-energy (meV):
        Sigma = -(1/2) int dq [w_env(q) - w(q)] A(q)^2 * w_bar / (omega_p(q) + w_bar),
    with omega_p(q) = sqrt(q^2 + 4 m^2), the edge of the electron-hole continuum of the gapped sea
    at wave vector q, and w_bar (units of m) the mean excitation energy of the bound electron.
    omega_p -> 0 gives the slow-sea bound, omega_p -> infinity the instantaneous limit (0)."""
    m = inter.basis.m
    kern = _kernel(inter, w_env)
    A = (F ** 2 + G ** 2) @ inter._bessel(0)
    wp = np.sqrt(inter.q ** 2 + 4 * m * m)
    fac = w_bar * m / (wp + w_bar * m)
    return -0.5 * float((kern * fac) @ A ** 2) * mat.hbar_v * 1000
