"""Furry basis in a perpendicular magnetic field (low-field moments of Table IV).

Valley K, field B > 0, symmetric gauge: the vector potential only adds r/2l^2 to the centrifugal
term of the radial equations of furry.py,

    F' = (j/r + b r) F - (E - U + m) G,      G' = -(j/r + b r) G + (E - U - m) F,     b = 1/(2 l^2),

with l = sqrt(hbar/eB) = 25.66 nm/sqrt(B[T]); the Pruefer angle obeys
theta' = E - U - m cos 2theta - (j/r + b r) sin 2theta and is still monotone in E.  Valley K' is
the complex conjugate of K at -B: its orbitals are those of b -> -b, labelled by the K quantum
number j~ (physical angular momentum -j~), as everywhere in this package.  The field enters the
one-body problem only; the screening of the Dirac sea is unchanged at the fields used here
(m l > 5, where the Landau quantisation changes the sea polarization by less than 1e-4).

Box.  Valley K uses F(R) = 0 as at B = 0.  For K' this condition admits an artefact state localised
at the wall (F = 0, G ~ r^-j exp(r^2/4l^2) at E = -m + U(R)), so K' uses G(R) = 0.  The two
conditions differ only in the discretised continuum.

Furry space.  In a field the criterion E > -m fails (the K zero modes, j <= -1/2, sit exactly at
-m).  The electron states are defined by the Pruefer index: in each channel the box eigenstates are
labelled by n with theta(R; E_n) = phi_R + n pi (phi_R = pi/2 for F(R) = 0, 0 for G(R) = 0); the
electron states are n >= n0(j), where n0(j) is the index of the lowest state above -m without
impurity (U = 0, same field and box).  At B = 0 and below Z_c this is E > -m (FurryBasis).  Moving
the projector with the field is what makes the finite-field moment -dE/dB of N >= 2 electrons
differ from the zero-field expectation value <P mu P> (observables.moment_two/three).

This module is the low-field core of the field code of the companion paper (fieldstates,
github.com/ehocine/charge-states-in-field), which adds the Landau-quantized screening.
"""
import numpy as np

from .furry import FurryBasis, _theta_start

HBAR_V = 0.6582            # eV nm (collapse2d.materials.gapped_graphene)
LB_NM_SQRT_T = 25.6556     # magnetic length sqrt(hbar/eB) in nm times sqrt(B[T])


def magnetic_length(B_T):
    return np.inf if B_T == 0 else LB_NM_SQRT_T / np.sqrt(abs(B_T))


def _pruefer_field(grid, U_all, m, E, j, theta0, b, reverse=False, store=False):
    """furry._pruefer with the field term b r added to j/r."""
    E = np.asarray(E, float)
    j = np.asarray(j, float)
    dr, jf, n = grid.drdu_all, grid.jfac_all, grid.n
    cen = j[..., None] * jf if j.ndim else j * jf              # (j/r) dr/du on all points
    cen = cen + b * grid.r_all * dr                            # + b r dr/du
    th = np.array(theta0, float) * np.ones_like(E)
    lr = np.zeros_like(th)
    if store:
        TH = np.empty((n + 1,) + th.shape)
        LR = np.empty((n + 1,) + th.shape)
        k0 = n if reverse else 0
        TH[k0], LR[k0] = th, lr

    def rhs(t, i):
        c, s = np.cos(2 * t), np.sin(2 * t)
        ci = cen[..., i]
        return dr[i] * (E - U_all[i] - m * c) - ci * s, ci * c - dr[i] * m * s

    h = -grid.h if reverse else grid.h
    for step in range(n):
        if reverse:
            i0, i1, i2 = 2 * (n - step), 2 * (n - step) - 1, 2 * (n - step) - 2
        else:
            i0, i1, i2 = 2 * step, 2 * step + 1, 2 * step + 2
        a1, b1 = rhs(th, i0)
        a2, b2 = rhs(th + 0.5 * h * a1, i1)
        a3, b3 = rhs(th + 0.5 * h * a2, i1)
        a4, b4 = rhs(th + h * a3, i2)
        th = th + h / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        lr = lr + h / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)
        if store:
            k = n - step - 1 if reverse else step + 1
            TH[k], LR[k] = th, lr
    return (TH, LR) if store else (th, lr)


class FieldFurryBasis(FurryBasis):
    """Electron (Furry) states of U = lam v(r) in a field, valley K (valley=+1) or K' (valley=-1).

    l_B   : magnetic length in nm (np.inf: no field; then identical to FurryBasis below Z_c)
    bc    : box condition, 'F' (F(R) = 0) or 'G' (G(R) = 0); default 'F' for K, 'G' for K'
    Other arguments as FurryBasis (E_floor and the dived-level treatment are not
    supported: the index criterion keeps the bound level wherever it lies).
    After construction also: valley, l_B, b, n0 (dict j -> lowest electron index), and
    sea_gap (dict j -> distance in energy from the lowest electron state to the highest sea state
    of the channel, a measure of how far the channel is from a level crossing with the sea).
    """

    def __init__(self, v, lam, m, R, js, E_max, r_core, l_B=np.inf, valley=+1, bc=None, **kw):
        if kw.get("E_floor") is not None:
            raise ValueError("FieldBasis selects the electron states by their Pruefer index")
        self.valley = int(valley)
        self.bc = bc or ("F" if self.valley > 0 else "G")
        self.phi_R = np.pi / 2 if self.bc == "F" else 0.0         # theta(R) = phi_R + n pi
        self.l_B = float(l_B)
        self.b = 0.0 if np.isinf(l_B) else self.valley / (2.0 * l_B * l_B)
        super().__init__(v, lam, m, R, js, E_max, r_core, **kw)

    # -- Pruefer with the field -------------------------------------------------------
    def _theta_R(self, E, j, lam=None):
        lam = self.lam if lam is None else lam
        U = self.U_all if lam == self.lam else lam * self.v(self.grid.r_all)
        t0 = _theta_start(self.v, lam, self.m, E, j, self.grid.r_s)
        return _pruefer_field(self.grid, U, self.m, E, j, t0, self.b)[0]

    def _first_index(self):
        """n0(j): index of the lowest state above -m without impurity."""
        m = self.m
        E_star = np.full(len(self.js), -m * (1 - 1e-6))
        th0 = self._theta_R(E_star, self.js, lam=0.0)
        return {j: int(np.floor((t - self.phi_R) / np.pi)) + 1 for j, t in zip(self.js, th0)}

    def _find_energies(self, n_scan, tol=1e-14, maxit=200):
        m = self.m
        self.n0 = self._first_index()
        n0 = np.array([self.n0[j] for j in self.js])
        # lower end of the scan: below the lowest electron state of every channel, and below the
        # highest sea state (to measure the distance to the sea)
        E_lo = -m * (1 - 1e-13)
        for _ in range(60):
            th = self._theta_R(np.full(len(self.js), E_lo), self.js)
            if np.all(th < self.phi_R + (n0 - 1) * np.pi):
                break
            E_lo -= 0.25 * m
        else:
            raise RuntimeError("could not bracket the lowest electron states")
        self.E_lo = E_lo
        Es = np.linspace(E_lo, self.E_max, n_scan)
        EE, JJ = np.meshgrid(Es, self.js)
        TH = self._theta_R(EE.ravel(), JJ.ravel()).reshape(EE.shape)
        if np.any(np.diff(TH, axis=1) < -1e-9):
            raise RuntimeError("Pruefer angle not monotone in E: refine the grid")
        targets, a, b, fa, fb, jt = [], [], [], [], [], []
        for ij, j in enumerate(self.js):
            th = TH[ij]
            n_lo = self.n0[j] - 1                                   # highest sea state, for sea_gap
            n_hi = int(np.floor((th[-1] - self.phi_R) / np.pi))
            for n in range(n_lo, n_hi + 1):
                t = self.phi_R + n * np.pi
                k = int(np.searchsorted(th, t))
                if k == 0:
                    raise RuntimeError("scan does not bracket the sea state")
                targets.append(t)
                jt.append(j)
                a.append(Es[k - 1]), b.append(Es[k])
                fa.append(th[k - 1] - t), fb.append(th[k] - t)
        t, jt = np.array(targets), np.array(jt)
        a, b, fa, fb = map(np.array, (a, b, fa, fb))
        side = np.zeros_like(a)
        for it in range(maxit):
            act = (b - a) > tol * max(m, 1.0)
            if not np.any(act):
                break
            if it % 2 == 0:
                c_try = np.where(np.abs(fb - fa) > 0, (a * fb - b * fa) / (fb - fa), 0.5 * (a + b))
                c = np.where((c_try > a) & (c_try < b), c_try, 0.5 * (a + b))
            else:
                c = 0.5 * (a + b)
            fc = np.zeros_like(c)
            fc[act] = self._theta_R(c[act], jt[act]) - t[act]
            left = act & (fc < 0)
            right = act & (fc >= 0)
            fb = np.where(left & (side == -1), 0.5 * fb, fb)
            fa = np.where(right & (side == 1), 0.5 * fa, fa)
            a, fa = np.where(left, c, a), np.where(left, fc, fa)
            b, fb = np.where(right, c, b), np.where(right, fc, fb)
            side = np.where(left, -1, np.where(right, 1, side))
        E = 0.5 * (a + b)
        nidx = np.round((t - self.phi_R) / np.pi).astype(int)
        sea = nidx == np.array([self.n0[j] for j in jt]) - 1
        self.sea_top = {j: float(E[sea & (jt == j)][0]) for j in self.js}
        keep = ~sea
        self.E, self.j, self.n_target = E[keep], jt[keep], t[keep]
        self.sea_gap = {}
        for j in self.js:
            sel = self.j == j
            self.sea_gap[j] = float(self.E[sel].min() - self.sea_top[j]) if np.any(sel) else np.nan

    def _eigenfunctions(self):
        g, m = self.grid, self.m
        t0 = _theta_start(self.v, self.lam, m, self.E, self.j, g.r_s)
        TO, LO = _pruefer_field(g, self.U_all, m, self.E, self.j, t0, self.b, store=True)
        TI, LI = _pruefer_field(g, self.U_all, m, self.E, self.j, np.full_like(self.E, self.phi_R), self.b,
                                reverse=True, store=True)
        n_st = TO.shape[1]
        mismatch = np.abs(np.sin(TO - TI))
        lnrho = np.empty_like(LO)
        theta = np.empty_like(TO)
        self.r_match = np.empty(n_st)
        self.max_mismatch = np.empty(n_st)
        for s in range(n_st):
            good = mismatch[:, s] < 1e-7
            if not np.any(good):
                k = int(np.argmin(mismatch[:, s]))
            else:
                idx = np.flatnonzero(good)
                runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                run = max(runs, key=len)
                k = int(run[len(run) // 2])
            self.r_match[s] = g.r[k]
            self.max_mismatch[s] = mismatch[k, s]
            shift = np.round((TI[k, s] - TO[k, s]) / np.pi) * np.pi
            theta[:k + 1, s] = TO[:k + 1, s]
            theta[k + 1:, s] = TI[k + 1:, s] - shift
            lnrho[:k + 1, s] = LO[:k + 1, s]
            lnrho[k + 1:, s] = LI[k + 1:, s] + (LO[k, s] - LI[k, s])
        lnrho -= lnrho.max(axis=0)
        rho = np.exp(lnrho)
        F, G = (rho * np.cos(theta)).T, (rho * np.sin(theta)).T
        norm = np.sqrt(g.integrate(F * F + G * G))
        self.F, self.G = F / norm[:, None], G / norm[:, None]

    # -- per-channel truncation ----------------------------------------------------------
    def truncate(self, n_max):
        """Keep the lowest n_max electron states of every channel (in place); returns self."""
        keep = np.zeros(len(self.E), bool)
        for j in self.js:
            keep[self.select(j)[:n_max]] = True
        self._keep(keep)
        return self
