"""Two-electron configuration interaction in the Furry basis (no-pair Hamiltonian).

    H = sum_i h_1(i) + W(1, 2),   restricted to products of positive-energy orbitals,

where h_1 is diagonal in the basis (energies E_a) and W is the screened interaction of
`interaction.py`.  Sectors:

  'singlet'      both electrons in one valley, spin singlet: spatially symmetric,
                 conserved J = j_1 + j_2
  'triplet'      one valley, spin triplet: spatially antisymmetric, J = j_1 + j_2
  'intervalley'  one electron in K, one in K' (any spin): distinguishable, conserved
                 M = j_1 - j~_2, where j~ labels the K orbital whose complex conjugate is the
                 K' orbital (its angular momentum is -j~)

For a symmetric (+) or antisymmetric (-) configuration |S_ab> the matrix element is
<S_ab|W|S_cd> = (<ab|W|cd> +- <ab|W|dc>) n_ab n_cd with n_ab = 1/sqrt(2) if a = b, else 1.
"""
from functools import lru_cache

import numpy as np
from scipy.linalg import eigh


class TwoElectronCI:
    def __init__(self, basis, inter, sector, J, js=None, n_max=None, dtype=np.float64, one_body=None):
        """js: channels to use (default: all of the basis); n_max: states per channel;
        dtype: storage of the form factors in the matrix-free path (float32 halves memory;
        products are accumulated in float64);
        one_body: optional callable (j, idx) -> matrix <a|h|b> of the one-body Hamiltonian
        between the basis states idx of channel j.  Default: diagonal, the basis energies
        (the basis diagonalises h).  Needed when the basis is built from another potential,
        e.g. h' = h + V_H (mean-field basis): then <a|h|b> = E'_a delta_ab - <a|V_H|b>."""
        self.dtype = dtype
        self._one_body_fn = one_body
        if sector not in ("singlet", "triplet", "intervalley"):
            raise ValueError(sector)
        self.basis, self.inter, self.sector, self.J = basis, inter, sector, float(J)
        js = np.array(sorted(set(np.round(basis.j, 1)) if js is None else js), float)
        self.channels = {j: basis.select(j)[:n_max] for j in js}
        if one_body is None:
            self.t = {j: np.diag(basis.E[idx]) for j, idx in self.channels.items()}
        else:
            self.t = {j: np.asarray(one_body(j, idx), float) for j, idx in self.channels.items()}
        self._build_configurations()
        self._ff = lru_cache(maxsize=64)(self._form_factors)

    # -- configurations -----------------------------------------------------------
    def _build_configurations(self):
        ch, J = self.channels, self.J
        blocks = []
        for ja in ch:
            jb = ja - J if self.sector == "intervalley" else J - ja
            if jb not in ch:
                continue
            if self.sector != "intervalley" and jb < ja:
                continue                          # unordered pair of channels, listed once
            ia, ib = ch[ja], ch[jb]
            A, Bi = np.meshgrid(np.arange(len(ia)), np.arange(len(ib)), indexing="ij")
            A, Bi = A.ravel(), Bi.ravel()
            if self.sector != "intervalley" and ja == jb:
                keep = A <= Bi if self.sector == "singlet" else A < Bi
                A, Bi = A[keep], Bi[keep]
            if len(A):
                blocks.append((ja, jb, ia[A], ib[Bi]))
        self.blocks = blocks
        self.size = sum(len(b[2]) for b in blocks)

    # -- matrix elements ------------------------------------------------------------
    def _form_factors(self, ja, jc):
        """A^l_ac(q) for all states of channels ja, jc (cached)."""
        l = int(round(abs(ja - jc)))
        return self.inter.form_factors(self.channels[ja], self.channels[jc], l)

    def _two_body(self, ja, jb, jc, jd):
        """<ab|W|cd> for all states of the four channels: array (a, b, c, d)."""
        Aac = self._ff(ja, jc)
        Abd = self._ff(jb, jd)
        na, nc, nq = Aac.shape
        nb, nd, _ = Abd.shape
        V = (Aac.reshape(na * nc, nq) * self.inter.wq) @ Abd.reshape(nb * nd, nq).T
        return V.reshape(na, nc, nb, nd).transpose(0, 2, 1, 3)

    def _local(self, j, idx):
        """Positions of global state indices idx within channel j."""
        pos = {s: k for k, s in enumerate(self.channels[j])}
        return np.array([pos[s] for s in idx])

    def hamiltonian(self):
        E = self.basis.E
        H = np.zeros((self.size, self.size))
        offs = np.cumsum([0] + [len(b[2]) for b in self.blocks])
        sign = {"singlet": 1.0, "triplet": -1.0}.get(self.sector, 0.0)
        for P, (ja, jb, sa, sb) in enumerate(self.blocks):
            la, lb = self._local(ja, sa), self._local(jb, sb)
            nP = np.where(sa == sb, 1 / np.sqrt(2), 1.0)
            for Q, (jc, jd, sc, sd) in enumerate(self.blocks):
                if Q < P:
                    continue
                lc, ld = self._local(jc, sc), self._local(jd, sd)
                D = self._two_body(ja, jb, jc, jd)[la[:, None], lb[:, None], lc[None, :], ld[None, :]]
                if sign:
                    X = self._two_body(ja, jb, jd, jc)[la[:, None], lb[:, None], ld[None, :], lc[None, :]]
                    nQ = np.where(sc == sd, 1 / np.sqrt(2), 1.0)
                    blk = (D + sign * X) * nP[:, None] * nQ[None, :]
                else:
                    blk = D
                H[offs[P]:offs[P + 1], offs[Q]:offs[Q + 1]] = blk
                H[offs[Q]:offs[Q + 1], offs[P]:offs[P + 1]] = blk.T
            H[offs[P]:offs[P + 1], offs[P]:offs[P + 1]] += self._one_body_block(P)
        return H

    def _one_body_block(self, P):
        """<S_i|h_1 + h_2|S_j> within block P (one-body terms do not change the channels)."""
        ja, jb, sa, sb = self.blocks[P]
        la, lb = self._local(ja, sa), self._local(jb, sb)
        ta, tb = self.t[ja], self.t[jb]
        T = ta[la[:, None], la[None, :]] * (lb[:, None] == lb[None, :]) \
            + (la[:, None] == la[None, :]) * tb[lb[:, None], lb[None, :]]
        sign = {"singlet": 1.0, "triplet": -1.0}.get(self.sector, 0.0)
        if sign and ja == jb:
            T = T + sign * (ta[la[:, None], lb[None, :]] * (lb[:, None] == la[None, :])
                            + (la[:, None] == lb[None, :]) * ta[lb[:, None], la[None, :]])
            n = np.where(sa == sb, 1 / np.sqrt(2), 1.0)
            T = T * n[:, None] * n[None, :]
        return T

    def solve(self, n_roots=4, method=None):
        """Lowest eigenvalues.  method 'dense' (default below 9000 configurations) or
        'davidson' (matrix-free, for the lowest root only)."""
        method = method or ("dense" if self.size < 9000 else "davidson")
        if method == "davidson":
            w, v = self.davidson()
            self.energies, self.vectors = np.array([w]), v[:, None]
            return self.energies
        H = self.hamiltonian()
        n = min(n_roots, self.size)
        w, v = eigh(H, subset_by_index=[0, n - 1])
        self.energies, self.vectors = w, v
        return w

    # -- matrix-free application of H -----------------------------------------------
    # <ab|W|cd> = sum_q w_q A_ac(q) A_bd(q), so for a configuration vector V_cd of block Q
    # the direct term is (sum_q w_q A^{ja,jc}(q) V A^{jb,jd}(q)^T)_ab and the exchange term
    # (sum_q w_q A^{ja,jd}(q) V^T A^{jb,jc}(q)^T)_ab; each is two matrix products.
    def _A_pair(self, ja, jc):
        """Form factors of channels ja, jc as an array (q, a, c), cached."""
        key = (ja, jc)
        if not hasattr(self, "_Acache"):
            self._Acache = {}
        if key not in self._Acache:
            l = int(round(abs(ja - jc)))
            if getattr(self.inter, "_phi", None) is not None:
                # compressed q dimension, sqrt(w) absorbed (Interaction.compress)
                A = self.inter.form_factors_compressed(self.channels[ja], self.channels[jc], l)
            else:
                A = self.inter.form_factors(self.channels[ja], self.channels[jc], l)
            self._Acache[key] = np.ascontiguousarray(A.transpose(2, 0, 1), dtype=self.dtype)
        return self._Acache[key]

    def _weights(self, ja, jc):
        """q weights for the cached form factors of channels (ja, jc) (ones if compressed)."""
        if getattr(self.inter, "_phi", None) is not None:
            return np.ones(self._A_pair(ja, jc).shape[0])
        return self.inter.wq

    def _apply_W(self, ja, jb, jc, jd, V):
        """sum_cd <ab|W|cd> V_cd for full channel blocks: array (n_ja, n_jb)."""
        Aq = self._A_pair(ja, jc)                          # (q, a, c)
        Bq = self._A_pair(jb, jd)                          # (q, b, d)
        T = Aq @ V.astype(self.dtype)                      # (q, a, d)
        T *= self._weights(ja, jc).astype(self.dtype)[:, None, None]
        return np.matmul(T, Bq.transpose(0, 2, 1)).sum(axis=0, dtype=np.float64)

    def _layout(self):
        if not hasattr(self, "_lay"):
            offs = np.cumsum([0] + [len(b[2]) for b in self.blocks])
            lay = []
            for P, (ja, jb, sa, sb) in enumerate(self.blocks):
                lay.append((offs[P], offs[P + 1], self._local(ja, sa), self._local(jb, sb),
                            # 1/sqrt(2) only for a symmetrised pair of identical orbitals; in the
                            # intervalley sector a K and a K' orbital may share an index
                            np.where((sa == sb) & (self.sector != "intervalley"), 1 / np.sqrt(2), 1.0)))
            self._lay = lay
        return self._lay

    def matvec(self, v):
        E = self.basis.E
        sign = {"singlet": 1.0, "triplet": -1.0}.get(self.sector, 0.0)
        lay = self._layout()
        out = np.zeros_like(v)
        for P, (ja, jb, sa, sb) in enumerate(self.blocks):
            o0, o1, la, lb, nP = lay[P]
            acc = np.zeros((len(self.channels[ja]), len(self.channels[jb])))
            for Q, (jc, jd, sc, sd) in enumerate(self.blocks):
                q0, q1, lc, ld, nQ = lay[Q]
                V = np.zeros((len(self.channels[jc]), len(self.channels[jd])))
                V[lc, ld] = v[q0:q1] * nQ
                acc += self._apply_W(ja, jb, jc, jd, V)
                if sign:
                    acc += sign * self._apply_W(ja, jb, jd, jc, V.T)
            # one-body: R = t_a V + V t_b^T (+- the same with V^T for two identical channels)
            V = np.zeros_like(acc)
            V[la, lb] = v[o0:o1] * nP
            R = self.t[ja] @ V + V @ self.t[jb].T
            if sign and ja == jb:
                R += sign * (self.t[ja] @ V.T + V.T @ self.t[jb].T)
            out[o0:o1] = (acc[la, lb] + R[la, lb]) * nP
        return out

    def diagonal(self):
        E = self.basis.E
        sign = {"singlet": 1.0, "triplet": -1.0}.get(self.sector, 0.0)
        d = np.empty(self.size)
        for (o0, o1, la, lb, nP), (ja, jb, sa, sb) in zip(self._layout(), self.blocks):
            Aaa = self._A_pair(ja, ja)
            Abb = self._A_pair(jb, jb)
            Da = np.einsum("qaa->qa", Aaa)[:, la]
            Db = np.einsum("qbb->qb", Abb)[:, lb]
            val = self._weights(ja, ja) @ (Da * Db).astype(np.float64)
            if sign:
                Aab = self._A_pair(ja, jb)
                val = val + sign * (self._weights(ja, jb) @ (Aab[:, la, lb].astype(np.float64) ** 2))
            d[o0:o1] = np.diag(self.t[ja])[la] + np.diag(self.t[jb])[lb] + val * nP ** 2
        return d

    def davidson(self, tol=1e-10, max_iter=300, max_space=40):
        """Lowest eigenpair by Davidson's method with the diagonal as preconditioner."""
        diag = self.diagonal()
        x = np.zeros(self.size)
        x[np.argmin(diag)] = 1.0
        Vs, AVs = [x], [self.matvec(x)]
        theta, y = None, None
        for it in range(max_iter):
            Vm, AVm = np.array(Vs).T, np.array(AVs).T
            Hs = Vm.T @ AVm
            Hs = 0.5 * (Hs + Hs.T)
            ev, ey = np.linalg.eigh(Hs)
            theta, y = ev[0], ey[:, 0]
            u, Au = Vm @ y, AVm @ y
            r = Au - theta * u
            res = np.linalg.norm(r)
            if res < tol * max(1.0, abs(theta)):
                break
            den = theta - diag
            den = np.where(np.abs(den) < 1e-12, 1e-12, den)
            t = r / den
            if len(Vs) >= max_space:                          # restart from the current vector
                Vs, AVs = [u / np.linalg.norm(u)], [Au / np.linalg.norm(u)]
            for _ in range(2):                                # orthogonalise twice
                for vv in Vs:
                    t -= (vv @ t) * vv
            nt = np.linalg.norm(t)
            if nt < 1e-14:
                break
            t /= nt
            Vs.append(t)
            AVs.append(self.matvec(t))
        self.davidson_info = {"iterations": it + 1, "residual": float(res)}
        return theta, u

    def first_order(self):
        """First-order energy <S|H|S> of the configuration with the lowest E_a + E_b."""
        E = self.basis.E
        P, k = min(((P, int(np.argmin(E[b[2]] + E[b[3]]))) for P, b in enumerate(self.blocks)),
                   key=lambda t: E[self.blocks[t[0]][2][t[1]]] + E[self.blocks[t[0]][3][t[1]]])
        ja, jb, sa, sb = self.blocks[P]
        a, b = sa[k], sb[k]
        la, lb = self._local(ja, [a])[0], self._local(jb, [b])[0]
        D = self._two_body(ja, jb, ja, jb)[la, lb, la, lb]
        if self.sector == "intervalley":
            return E[a] + E[b] + D
        X = self._two_body(ja, jb, jb, ja)[la, lb, lb, la]
        sign = 1.0 if self.sector == "singlet" else -1.0
        n2 = 0.5 if a == b else 1.0
        return E[a] + E[b] + (D + sign * X) * n2

    # -- Dyson orbital ----------------------------------------------------------------------
    def dyson(self, vec, a0):
        """Amplitudes d_b = <Psi_2| c^dag_b |a0> for adding an electron to the one-electron state
        in orbital a0 (global index) and the radial functions of the Dyson orbital.

        'intervalley': d_b = C_{a0 b} (the added electron is the K' one).
        'singlet': d_b = C_{(a0 b)} / sqrt(2) for b != a0 and C_{(a0 a0)} for b = a0 (spin
        bookkeeping of |S_ab> x singlet against c^dag_{b,down} c^dag_{a0,up}|0>).
        Returns dict with global indices 'b', amplitudes 'd', quasiparticle weight 'Z_qp' =
        sum |d_b|^2, and 'F', 'G' of the Dyson orbital on the basis grid.
        """
        B = self.basis
        amps = {}
        for (o0, o1, la, lb, nP), (ja, jb, sa, sb) in zip(self._layout(), self.blocks):
            c = vec[o0:o1]
            for k in range(len(sa)):
                a, b = sa[k], sb[k]
                if self.sector == "intervalley":
                    if a == a0:
                        amps[b] = amps.get(b, 0.0) + c[k]
                else:
                    f = 1.0 if a == b else 1 / np.sqrt(2)
                    if a == a0:
                        amps[b] = amps.get(b, 0.0) + c[k] * f
                    elif b == a0:
                        amps[a] = amps.get(a, 0.0) + c[k] * f
        idx = np.array(sorted(amps))
        d = np.array([amps[i] for i in idx])
        return {"b": idx, "d": d, "Z_qp": float(d @ d), "F": d @ B.F[idx], "G": d @ B.G[idx]}

    def full_tensor(self, vec):
        """The configuration vector as full product amplitudes Psi[a, b] for every ordered
        channel pair (ja, jb): dict (ja, jb) -> array (n_ja, n_jb), normalised so that
        sum |Psi|^2 over all ordered pairs is 1.  Same valley: Psi[a,b] = +-Psi[b,a]
        (C/sqrt(2) for a != b, C for a = b in the singlet); intervalley: Psi = C, a in K, b in K'."""
        out = {}
        sign = {"singlet": 1.0, "triplet": -1.0}.get(self.sector, 0.0)
        for (o0, o1, la, lb, nP), (ja, jb, sa, sb) in zip(self._layout(), self.blocks):
            T = out.setdefault((ja, jb), np.zeros((len(self.channels[ja]), len(self.channels[jb]))))
            c = vec[o0:o1]
            if self.sector == "intervalley":
                T[la, lb] += c
                continue
            f = np.where(sa == sb, 1.0, 1 / np.sqrt(2))
            T[la, lb] += c * f
            Tt = out.setdefault((jb, ja), np.zeros((len(self.channels[jb]), len(self.channels[ja]))))
            off = sa != sb
            Tt[lb[off], la[off]] += sign * c[off] * f[off]
        return out
