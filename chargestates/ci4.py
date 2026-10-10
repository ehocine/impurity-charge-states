"""Configuration interaction for two to four electrons with arbitrary valley content.

    H = sum_i h(i) + sum_{i<k} W(i, k),   no-pair: each electron in the Furry space of its valley.

Electron i lives in valley v_i (+1: K, -1: K'; a K' orbital is labelled by the K orbital j~ whose
complex conjugate it is).  Spin enters through the exchange symmetry of the spatial wave function:
electrons 0, 1 may be same-valley partners with a symmetric (sym = +1, spin singlet) or
antisymmetric (sym = -1) wave function, likewise electrons 2, 3 (sym23); all other electrons are
distinguishable (other valley or other spin projection).  Sectors (lowest_state):

    three electrons  (v, v, v') sym +-1   pair in v + one electron in the other valley
                     (v, v, v)  sym -1    up-up pair + down electron in one valley (S_z = 1/2)
    four electrons   (v, v, v, v) sym -1, sym23 -1   (up-up)(down-down) in one valley (S_z = 0)
                     (v, v, v', v') sym +-1, sym23 +-1   a pair in each valley
                     (v, v, v, v') sym -1                up-up-down in v + one electron in v'

These spin-projection sectors contain every spin multiplet, so the lowest state over them is the
ground state.  Conserved: M = sum_i s_{v_i} j_i (selection rules of interaction.py).  The vector is
stored as product tensors per channel tuple (both orders of each symmetric pair) with the
exchange symmetry imposed on the vector; two-body terms are applied as two tensor contractions of
the jointly compressed form factors (interaction.compress_shared).  ci3.ThreeElectronCI is the
special case of a pair plus one electron of the other valley.

The generic contraction is that of the field code of the companion paper (fieldstates.ci); in a
field each valley has its own basis (field.FieldFurryBasis), at B = 0 both use the same one.
"""
import itertools

import numpy as np


class ValleyInteraction:
    """Compressed form factors for electrons of valleys +1 and -1.

    pairs: {valley: (basis, Interaction)} (the same objects for both valleys at B = 0);
    channels: {valley: {j: state indices}}.  Compresses all Interactions jointly
    (interaction.compress_shared) and serves C(v, ja, jc) as arrays (k, a, c)."""

    def __init__(self, pairs, channels, tol=1e-8):
        from .interaction import compress_shared
        self.bases = {v: b for v, (b, _) in pairs.items()}
        self.inter = {v: I for v, (_, I) in pairs.items()}
        self.channels = channels
        uniq = {}
        for v, (b, I) in pairs.items():
            uniq.setdefault(id(I), (I, {}))[1].update(channels[v])
        self.ranks = compress_shared(list(uniq.values()), tol=tol)
        self._C = {}

    def C(self, v, ja, jc):
        key = (v, ja, jc)
        if key not in self._C:
            ch = self.channels[v]
            l = int(round(abs(ja - jc)))
            Cf = self.inter[v].form_factors_compressed(ch[ja], ch[jc], l)
            self._C[key] = np.ascontiguousarray(Cf.transpose(2, 0, 1))
        return self._C[key]


class ManyElectronCI:
    def __init__(self, ti, valleys, sym, M, terms=None, sym23=0):
        """ti: ValleyInteraction (compressed); valleys: tuple of +-1 per electron (2 to 4);
        sym: exchange symmetry of electrons 0, 1 (+1, -1, or 0 for distinguishable);
        sym23: the same for electrons 2, 3 (four electrons only)."""
        self.ti, self.valleys, self.sym, self.M = ti, tuple(int(v) for v in valleys), int(sym), float(M)
        self.sym23 = int(sym23)
        self.N = len(valleys)
        if self.N not in (2, 3, 4):
            raise ValueError("two to four electrons")
        if self.sym and self.valleys[0] != self.valleys[1]:
            raise ValueError("exchange symmetry needs two electrons of one valley")
        if self.sym23 and (self.N != 4 or self.valleys[2] != self.valleys[3]):
            raise ValueError("sym23 needs four electrons with electrons 2, 3 in one valley")
        self.pairs = [(i, k) for i in range(self.N) for k in range(i + 1, self.N)]
        self.terms = terms or {p: 1.0 for p in self.pairs}
        ch = ti.channels
        self.ch = ch
        labels = [sorted(ch[v]) for v in self.valleys]
        self.tuples = [t for t in itertools.product(*labels)
                       if np.isclose(sum(v * j for v, j in zip(self.valleys, t)), self.M)]
        self.shapes = [tuple(len(ch[v][j]) for v, j in zip(self.valleys, t)) for t in self.tuples]
        self.offs = np.cumsum([0] + [int(np.prod(s)) for s in self.shapes])
        self.size = int(self.offs[-1])
        self.index = {t: k for k, t in enumerate(self.tuples)}
        self.E = [{j: ti.bases[v].E[idx] for j, idx in ch[v].items()} for v in self.valleys]
        # couplings: for each pair (i, k), the (target, source) tuples differing only there
        self.links = {}
        for (i, k) in self.pairs:
            L = []
            for T, tT in enumerate(self.tuples):
                for S, tS in enumerate(self.tuples):
                    if all(tT[p] == tS[p] for p in range(self.N) if p not in (i, k)):
                        L.append((T, S))
            self.links[(i, k)] = L

    # -- helpers ---------------------------------------------------------------------------
    def _blocks(self, v):
        return [v[self.offs[k]:self.offs[k + 1]].reshape(self.shapes[k]) for k in range(len(self.tuples))]

    def _sym_pair(self, v, i, sign):
        out = np.empty_like(v)
        P = self._blocks(v)
        perm = list(range(self.N))
        perm[i], perm[i + 1] = i + 1, i
        for k, t in enumerate(self.tuples):
            tt = list(t)
            tt[i], tt[i + 1] = t[i + 1], t[i]
            X = P[self.index[tuple(tt)]].transpose(perm)
            out[self.offs[k]:self.offs[k + 1]] = (0.5 * (P[k] + sign * X)).ravel()
        return out

    def symmetrize(self, v):
        if self.sym:
            v = self._sym_pair(v, 0, self.sym)
        if self.sym23:
            v = self._sym_pair(v, 2, self.sym23)
        return v

    def _Cpair(self, i, jT, jS):
        return self.ti.C(self.valleys[i], jT, jS)

    def _contract(self, Ci, Ck, X, i, k):
        """sum_q sum_{A,B} Ci[q,a,A] Ck[q,b,B] X[..A(i)..B(k)..] -> array with a at i, b at k."""
        N = self.N
        Y = np.tensordot(Ci, X, axes=([2], [i]))               # (q, a, X axes without i)
        kk = 2 + (k if k < i else k - 1)
        Y = np.moveaxis(Y, kk, -1)                             # (q, a, rest..., B)
        Z = np.tensordot(Ck, Y, axes=([0, 2], [0, Y.ndim - 1]))  # (b, a, rest...)
        rest = [p for p in range(N) if p not in (i, k)]
        src = {k: 0, i: 1}
        src.update({p: 2 + n for n, p in enumerate(rest)})
        return Z.transpose([src[p] for p in range(N)])

    # -- H v -----------------------------------------------------------------------------
    def matvec(self, v):
        P = self._blocks(v)
        out = [np.zeros(s) for s in self.shapes]
        for k, t in enumerate(self.tuples):
            for i in range(self.N):
                e = self.E[i][t[i]]
                sh = [1] * self.N
                sh[i] = len(e)
                out[k] += e.reshape(sh) * P[k]
        for (i, k), L in self.links.items():
            s = self.terms[(i, k)]
            if not s:
                continue
            for T, S in L:
                tT, tS = self.tuples[T], self.tuples[S]
                Ci = self._Cpair(i, tT[i], tS[i])                  # (q, a, A)
                Ck = self._Cpair(k, tT[k], tS[k])                  # (q, b, B)
                out[T] += s * self._contract(Ci, Ck, P[S], i, k)
        return np.concatenate([o.ravel() for o in out])

    def diagonal(self):
        d = []
        for t, shp in zip(self.tuples, self.shapes):
            D = np.zeros(shp)
            dg = []
            for i in range(self.N):
                e = self.E[i][t[i]]
                sh = [1] * self.N
                sh[i] = len(e)
                D = D + e.reshape(sh)
                dg.append(np.einsum("qaa->qa", self._Cpair(i, t[i], t[i])))
            for (i, k) in self.pairs:
                s = self.terms[(i, k)]
                if not s:
                    continue
                V = dg[i].T @ dg[k]                                 # (a, b)
                sh = [1] * self.N
                sh[i], sh[k] = V.shape
                D = D + s * V.reshape(sh)
            d.append(D.ravel())
        return np.concatenate(d)

    def min_one_body(self):
        """Lowest noninteracting energy of the sector: the smallest sum of orbital energies over the
        allowed configurations (Pauli: an antisymmetric pair in one channel needs two different
        orbitals).  Since W(q) > 0, the projected interaction is positive semidefinite, so this is a
        rigorous lower bound on the lowest eigenvalue of the sector."""
        best = np.inf
        for t in self.tuples:
            lows = [self.E[i][t[i]] for i in range(self.N)]
            if any(len(e) == 0 for e in lows):
                continue
            tot, ok = 0.0, True
            used = set()
            for (i, sgn) in ((0, self.sym), (2, self.sym23)):
                if sgn == -1 and t[i] == t[i + 1]:
                    e = np.sort(lows[i])
                    if len(e) < 2:
                        ok = False
                        break
                    tot += e[0] + e[1]
                    used.update((i, i + 1))
            if not ok:
                continue
            tot += sum(float(np.min(lows[i])) for i in range(self.N) if i not in used)
            best = min(best, tot)
        return best

    def min_diagonal(self):
        """Lowest diagonal element among the configurations allowed by the symmetry (Pauli: no
        doubly occupied orbital in an antisymmetric pair)."""
        dg = self.diagonal()[self.allowed()]
        return float(dg.min()) if len(dg) else np.inf

    def allowed(self):
        """Mask of the product configurations that survive the exchange symmetry (Pauli)."""
        mask = np.ones(self.size, bool)
        for (i, sgn) in ((0, self.sym), (2, self.sym23)):
            if sgn != -1:
                continue
            for k, t in enumerate(self.tuples):
                if t[i] == t[i + 1]:
                    blk = mask[self.offs[k]:self.offs[k + 1]].reshape(self.shapes[k])
                    n = self.shapes[k][i]
                    idx = [slice(None)] * self.N
                    idx[i], idx[i + 1] = np.arange(n), np.arange(n)
                    blk[tuple(idx)] = False
        return mask

    # -- lowest eigenpair ------------------------------------------------------------------
    def davidson(self, tol=1e-9, max_iter=400, max_space=12, x0=None):
        """Lowest eigenpair by Davidson's method (diagonal preconditioner).  Memory-lean: the
        projected matrix is updated in place and the subspace is restarted from the current Ritz
        vector after max_space vectors, so at most 2 max_space vectors of the full size are stored
        (four electrons with 20 states per channel: 1.4e7 amplitudes, 110 MB each)."""
        diag = self.diagonal()
        if x0 is None:
            # start from the lowest configuration allowed by the exchange symmetry
            ok = np.flatnonzero(self.allowed())
            if len(ok) == 0:
                raise ValueError("no configuration allowed by the exchange symmetry")
            x = np.zeros(self.size)
            x[ok[np.argmin(diag[ok])]] = 1.0
            x = self.symmetrize(x)
        else:
            x = self.symmetrize(np.array(x0, float))
        x /= np.linalg.norm(x)
        Vs, AVs = [x], [self.matvec(x)]
        Hs = np.array([[x @ AVs[0]]])
        res = np.inf
        for it in range(max_iter):
            ev, ey = np.linalg.eigh(0.5 * (Hs + Hs.T))
            theta, yv = ev[0], ey[:, 0]
            u = sum(c * v for c, v in zip(yv, Vs))
            Au = sum(c * v for c, v in zip(yv, AVs))
            r = Au - theta * u
            res = np.linalg.norm(r)
            if res < tol * max(1.0, abs(theta)):
                break
            den = theta - diag
            t = self.symmetrize(r / np.where(np.abs(den) < 1e-12, 1e-12, den))
            del r
            if len(Vs) >= max_space:
                nu = np.linalg.norm(u)
                Vs, AVs = [u / nu], [Au / nu]
                Hs = np.array([[Vs[0] @ AVs[0]]])
            for _ in range(2):
                for vv in Vs:
                    t -= (vv @ t) * vv
            nt = np.linalg.norm(t)
            if nt < 1e-14:
                break
            t /= nt
            At = self.matvec(t)
            k = len(Vs)
            H2 = np.empty((k + 1, k + 1))
            H2[:k, :k] = Hs
            H2[:k, k] = [v @ At for v in Vs]
            H2[k, :k] = [t @ Av for Av in AVs]
            H2[k, k] = t @ At
            Hs = H2
            Vs.append(t)
            AVs.append(At)
        self.info = {"iterations": it + 1, "residual": float(res)}
        self.energy, self.vector = float(theta), u
        return self.energy, u

    # -- analysis --------------------------------------------------------------------------
    def full_tensor(self, vec):
        return {t: P for t, P in zip(self.tuples, self._blocks(vec))}


def sectors(N, field=False):
    """(valleys, sym, sym23) of all N-electron sectors (N = 3, 4); at B = 0 (field=False) the
    valley-mirrored copies are omitted (same spectra)."""
    vs = (1, -1) if field else (1,)
    if N == 3:
        return [((v, v, -v), s, 0) for v in vs for s in (1, -1)] + [((v, v, v), -1, 0) for v in vs]
    if N == 4:
        out = [((v,) * 4, -1, -1) for v in vs]
        out += [((1, 1, -1, -1), s1, s2) for s1 in (1, -1) for s2 in (1, -1)]
        out += [((v, v, v, -v), -1, 0) for v in vs]
        return out
    raise ValueError("N = 3 or 4")


def lowest_state(ti, N, js, field=False, tol=1e-9):
    """Lowest N-electron energy over all sectors and M values (units of the basis energies).

    Sectors are solved in the order of their lowest diagonal element and skipped only when a
    rigorous lower bound, their noninteracting minimum (W > 0), exceeds the lowest energy found.
    Returns (E, record of the lowest sector, list of the solved sectors)."""
    cands = []
    for valleys, sym, sym23 in sectors(N, field):
        Ms = sorted({round(sum(v * j for v, j in zip(valleys, t)), 1)
                     for t in itertools.product(*[js] * len(valleys))})
        for M in Ms:
            ci = ManyElectronCI(ti, valleys, sym, M, sym23=sym23)
            if ci.size == 0:
                continue
            lb = ci.min_one_body()
            if np.isfinite(lb):
                cands.append((ci.min_diagonal(), lb, valleys, sym, sym23, M, ci))
    best, solved = np.inf, []
    for d, lb, valleys, sym, sym23, M, ci in sorted(cands, key=lambda c: c[0]):
        if lb > best:
            continue
        e, _ = ci.davidson(tol=tol)
        solved.append({"valleys": list(valleys), "sym": sym, "sym23": sym23, "M": M, "E": float(e),
                       "size": ci.size, "residual": ci.info["residual"]})
        best = min(best, e)
    solved.sort(key=lambda r: r["E"])
    return best, solved[0], solved
