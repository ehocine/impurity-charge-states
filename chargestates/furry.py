"""Single-particle Furry basis: eigenstates of the radial Dirac equation in a box.

For H = sigma.p + m sigma_z + U(r) (hbar v = 1) and total angular momentum j the radial
equations are (as in collapse2d)

    F' =  (j/r) F - (E - U + m) G,      G' = -(j/r) G + (E - U - m) F,

with |psi|^2 = (F^2 + G^2)/(2 pi r).  In a box of radius R with F(R) = 0 the radial
operator is self-adjoint and its spectrum is discrete.  The positive-energy (Furry)
basis consists of the box eigenstates with E > -m, i.e. the bound states and the
discretised conduction continuum.

Eigenvalues are found from the Pruefer angle theta (F = rho cos theta, G = rho sin theta),

    theta' = E - U - m cos 2theta - (j/r) sin 2theta,
    (ln rho)' = (j/r) cos 2theta - m sin 2theta,

integrated outward from the regular solution.  theta(R; E) increases monotonically
with E, and E is an eigenvalue exactly when theta(R; E) = pi/2 + n pi.  The eigenvalues
in an energy window are therefore counted exactly, without scanning for sign changes.
Eigenfunctions are assembled from an outward and an inward (F(R) = 0) integration.

The integration uses fourth-order Runge-Kutta on a uniform grid in u = ln r + r/L,
like collapse2d.dirac, and is vectorised over (E, j) pairs.
"""
import numpy as np

from .deps import collapse2d  # noqa: F401  (makes the potentials importable)


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------


def _u_of_r(r, L):
    return np.log(r) + r / L


def _r_of_u(u, L):
    """Invert u = ln r + r/L by Newton iteration (vectorised)."""
    u = np.asarray(u, dtype=float)
    r = np.where(u < np.log(L), np.exp(np.minimum(u, 700.0)), np.maximum(L * (u - np.log(L)), L))
    for _ in range(80):
        rn = r - (np.log(r) + r / L - u) / (1.0 / r + 1.0 / L)
        rn = np.where(rn <= 0, 0.5 * r, rn)
        done = np.max(np.abs(rn - r) / r) < 1e-15
        r = rn
        if done:
            break
    return r


class Grid:
    """Uniform grid in u = ln r + r/L from r_s to R.

    `r_all` holds the full and half points (2n + 1 values) used by RK4, `r` the n + 1
    full points, and `w` Simpson weights for integrals over r on the full points.
    """

    def __init__(self, r_s, R, L, h=0.01):
        u0, u1 = _u_of_r(r_s, L), _u_of_r(R, L)
        n = 2 * int(np.ceil((u1 - u0) / (2 * h)))          # even, for Simpson's rule
        self.n, self.h, self.L, self.R, self.r_s = n, (u1 - u0) / n, float(L), float(R), float(r_s)
        u = u0 + 0.5 * self.h * np.arange(2 * n + 1)
        r = _r_of_u(u, L)
        r[0], r[-1] = r_s, R
        self.r_all = r
        self.drdu_all = r * L / (L + r)                     # dr/du
        self.jfac_all = L / (L + r)                         # (1/r) dr/du
        self.r = r[::2]
        simpson = np.ones(n + 1)
        simpson[1:-1:2], simpson[2:-1:2] = 4.0, 2.0
        self.w = self.h / 3.0 * simpson * self.drdu_all[::2]

    def integrate(self, f):
        """Integral over r of f given on the full points (last axis)."""
        return np.asarray(f) @ self.w


# ---------------------------------------------------------------------------
# Pruefer integration
# ---------------------------------------------------------------------------


def _pruefer(grid, U_all, m, E, j, theta0, reverse=False, store=False):
    """RK4 for (theta, ln rho) along the grid; E, j, theta0 are arrays of one shape.

    Returns (theta, lnrho) at the far end, or arrays of shape (n + 1, ...) on the full
    points (in grid order) if store is True.
    """
    E = np.asarray(E, float)
    j = np.asarray(j, float)
    dr, jf, n = grid.drdu_all, grid.jfac_all, grid.n
    th = np.array(theta0, float) * np.ones_like(E)
    lr = np.zeros_like(th)
    if store:
        TH = np.empty((n + 1,) + th.shape)
        LR = np.empty((n + 1,) + th.shape)
        k0 = n if reverse else 0
        TH[k0], LR[k0] = th, lr

    def rhs(t, i):
        c, s = np.cos(2 * t), np.sin(2 * t)
        return dr[i] * (E - U_all[i] - m * c) - j * jf[i] * s, j * jf[i] * c - dr[i] * m * s

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


def _theta_start(v, lam, m, E, j, r_s):
    """Pruefer angle of the regular solution at r_s (cf. collapse2d.dirac.regular_start)."""
    E = np.asarray(E, float)
    j = np.asarray(j, float)
    if getattr(v, "point", False):
        g = lam * v.beta                                    # U = -g/r near the origin
        s = np.sqrt(j * j - g * g)
        return np.arctan2((j - s) / g, 1.0)
    w = E - lam * v(np.array([r_s]))[0]
    aj = np.abs(j)
    F = np.where(j > 0, 1.0, -(w + m) * r_s / (2 * aj + 1))
    G = np.where(j > 0, (w - m) * r_s / (2 * aj + 1), 1.0)
    return np.arctan2(G, F)


# ---------------------------------------------------------------------------
# basis
# ---------------------------------------------------------------------------


class FurryBasis:
    """Positive-energy box eigenstates of U = lam * v(r) for several j.

    v       : potential shape from collapse2d.potentials (callable, vectorised)
    lam     : coupling (the impurity charge Z for the shapes of collapse2d.analysis)
    m       : Dirac mass (nm^-1, hbar v = 1)
    R       : box radius (nm)
    js      : total angular momenta
    E_max   : keep states with -m < E <= E_max
    r_core  : length scale of the core (d or r0), sets the start radius
    L, h    : grid parameters (default L from the largest momentum kept)

    After construction: E, j (one entry per state), F, G (states x full grid points,
    normalised to int (F^2 + G^2) dr = 1), and grid.
    """

    def __init__(self, v, lam, m, R, js, E_max, r_core, L=None, h=0.01, n_scan=400, r_s=None,
                 E_floor=None, r_loc=None, w_loc=0.9):
        self.v, self.lam, self.m, self.R = v, float(lam), float(m), float(R)
        self.js = np.array(sorted(js), float)
        self.E_max = float(E_max)
        k_max = np.sqrt(max(E_max * E_max - m * m, (4 * m) ** 2))
        if L is None:
            L = min(1.0 / m, 2.0 / k_max, R / 10)
        r_s = r_s if r_s is not None else 1e-6 * r_core
        self.grid = Grid(r_s, R, L, h)
        self.U_all = self.lam * v(self.grid.r_all)
        self.E_lo = -m * (1 - 1e-13) if E_floor is None else float(E_floor)
        self._find_energies(n_scan)
        self._eigenfunctions()
        if E_floor is not None:
            self._dived_orbitals(r_loc if r_loc is not None else 3.0 / m, w_loc)

    # -- eigenvalues ------------------------------------------------------------
    def _theta_R(self, E, j):
        t0 = _theta_start(self.v, self.lam, self.m, E, j, self.grid.r_s)
        return _pruefer(self.grid, self.U_all, self.m, E, j, t0)[0]

    def _find_energies(self, n_scan, tol=1e-14, maxit=200):
        m = self.m
        # coarse scan, uniform in E (brackets need not isolate single roots: theta is monotone)
        Es = np.linspace(self.E_lo, self.E_max, n_scan)
        EE, JJ = np.meshgrid(Es, self.js)
        TH = self._theta_R(EE.ravel(), JJ.ravel()).reshape(EE.shape)
        if np.any(np.diff(TH, axis=1) < -1e-9):
            raise RuntimeError("Pruefer angle not monotone in E: refine the grid")
        targets, a, b, fa, fb, jt = [], [], [], [], [], []
        for ij, j in enumerate(self.js):
            th = TH[ij]
            n_lo = int(np.floor((th[0] - np.pi / 2) / np.pi)) + 1
            n_hi = int(np.floor((th[-1] - np.pi / 2) / np.pi))
            for n in range(n_lo, n_hi + 1):
                t = np.pi / 2 + n * np.pi
                k = int(np.searchsorted(th, t))              # th[k-1] < t <= th[k]
                targets.append(t)
                jt.append(j)
                a.append(Es[k - 1]), b.append(Es[k])
                fa.append(th[k - 1] - t), fb.append(th[k] - t)
        t, jt = np.array(targets), np.array(jt)
        a, b, fa, fb = map(np.array, (a, b, fa, fb))
        # bracketed root finding: Illinois steps alternating with bisection
        c = 0.5 * (a + b)
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
            # Illinois: halve the retained end's function value if the same end stays twice
            fb = np.where(left & (side == -1), 0.5 * fb, fb)
            fa = np.where(right & (side == 1), 0.5 * fa, fa)
            a, fa = np.where(left, c, a), np.where(left, fc, fa)
            b, fb = np.where(right, c, b), np.where(right, fc, fb)
            side = np.where(left, -1, np.where(right, 1, side))
        self.E = 0.5 * (a + b)
        self.j = jt
        self.n_target = t

    # -- eigenfunctions ---------------------------------------------------------
    def _eigenfunctions(self):
        g, m = self.grid, self.m
        t0 = _theta_start(self.v, self.lam, m, self.E, self.j, g.r_s)
        TO, LO = _pruefer(g, self.U_all, m, self.E, self.j, t0, store=True)
        TI, LI = _pruefer(g, self.U_all, m, self.E, self.j, np.full_like(self.E, np.pi / 2),
                          reverse=True, store=True)
        n_pts, n_st = TO.shape
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
                # middle of the longest run of agreement
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

    def _dived_orbitals(self, r_loc, w_loc):
        """Replace the box states below -m by the quasi-bound orbitals of dived levels.

        Above Z_c the empty level is a resonance in the valence continuum.  In the box its weight
        is shared by the box states within ~Gamma of E_r, which are otherwise extended (sea)
        states.  For each j, the combination of box states below -m with the largest weight
        inside r < r_loc (leading eigenvectors of the inner-weight matrix, kept if that weight
        exceeds w_loc) is the quasi-bound orbital; its energy <h> lies within ~Gamma of E_r.
        It is orthogonal to all states above -m (it is built from eigenstates below -m), and
        the rest of the window, the sea, is dropped.  Valid while Gamma is small compared with
        the energies of interest; Gamma is the uncertainty of E1 above Z_c.
        """
        m, g = self.m, self.grid
        inside = g.r <= r_loc
        keep = self.E > -m
        newF, newG, newE, newj, neww = [], [], [], [], []
        for j in np.unique(self.j):
            idx = np.flatnonzero((self.j == j) & ~keep)
            if len(idx) < 2:
                continue
            idx = idx[np.argsort(self.E[idx])]
            w1 = (self.F[idx][:, inside] ** 2 + self.G[idx][:, inside] ** 2) @ g.w[inside]
            k0 = int(np.argmax(w1))
            # every sea state has some weight inside r_loc, so a wide window can localise
            # arbitrary packets: use only the states within ~1.5 level spacings of the
            # most localised one, where the resonance's spectral weight sits
            spacing = np.median(np.diff(self.E[idx][max(k0 - 3, 0):k0 + 4]))
            win = idx[np.abs(self.E[idx] - self.E[idx[k0]]) <= 1.5 * spacing]
            F, G = self.F[win], self.G[win]
            Mw = (F[:, inside] * g.w[inside]) @ F[:, inside].T + (G[:, inside] * g.w[inside]) @ G[:, inside].T
            lam, U = np.linalg.eigh(Mw)
            if lam[-1] > w_loc:
                c = U[:, -1]
                newF.append(c @ F)
                newG.append(c @ G)
                newE.append(float(c ** 2 @ self.E[win]))
                newj.append(j)
                neww.append(float(lam[-1]))
        self.inner_weight = np.zeros(len(self.E))
        self._keep(keep)
        if newE:
            self.F = np.vstack([self.F, np.array(newF)])
            self.G = np.vstack([self.G, np.array(newG)])
            self.E = np.concatenate([self.E, newE])
            self.j = np.concatenate([self.j, newj])
            self.inner_weight = np.concatenate([self.inner_weight, neww])
            for name in ("n_target", "r_match", "max_mismatch"):
                setattr(self, name, np.concatenate([getattr(self, name), np.full(len(newE), np.nan)]))

    def _keep(self, mask):
        for name in ("E", "j", "n_target", "F", "G", "r_match", "max_mismatch", "inner_weight"):
            if hasattr(self, name):
                setattr(self, name, getattr(self, name)[mask])

    def dived(self):
        """Indices of the kept states below -m (dived levels)."""
        return np.flatnonzero(self.E <= -self.m)

    # -- conveniences -------------------------------------------------------------
    def select(self, j):
        """Indices of the states with angular momentum j, in ascending energy."""
        idx = np.flatnonzero(np.isclose(self.j, j))
        return idx[np.argsort(self.E[idx])]

    def overlap(self, j):
        """Overlap matrix of the states with angular momentum j (should be the identity)."""
        idx = self.select(j)
        Fj, Gj = self.F[idx], self.G[idx]
        return (Fj * self.grid.w) @ Fj.T + (Gj * self.grid.w) @ Gj.T
