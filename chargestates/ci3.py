"""Three-electron configuration interaction: two electrons in valley K, one in K'.

    H = sum_i h(i) + W_12 + W_13 + W_23,   no-pair (positive-energy orbitals only).

Electrons 1, 2 are in K: their spatial wave function is symmetric ('singlet' pair) or
antisymmetric ('triplet' pair) under 1 <-> 2.  Electron 3 is in K', labelled by the K
orbital j~ whose complex conjugate it is (actual angular momentum -j~); with intervalley
exchange neglected it is distinguishable.  Conserved: M = j_1 + j_2 - j~_3.

Selection rules (see interaction.py): W_12 (same valley) conserves j_1 + j_2; W_13 and W_23
(opposite valleys) conserve j_1 - j~_3 and j_2 - j~_3 respectively.  All matrix elements are
sum_k C_ac,k C_bd,k with the q-compressed form factors of Interaction.compress, so every
term is applied as two tensor contractions.  The vector is stored as full product tensors
Psi[a, b, c] per channel triple, including both orders of the K pair; the 1 <-> 2 symmetry is
imposed on the vector (H commutes with the exchange), so W_12 needs no explicit exchange term.
"""
import numpy as np


class ThreeElectronCI:
    def __init__(self, basis, inter, pair, M, js=None, n_max=None, one_body=None, terms=(1.0, 1.0, 1.0)):
        """pair: 'singlet' or 'triplet' (symmetry of the K pair); terms: scale factors of
        (W_12, W_13, W_23), for tests.  inter must be compressed (Interaction.compress)."""
        if getattr(inter, "_phi", None) is None:
            raise ValueError("call inter.compress(channels) first")
        self.basis, self.inter, self.M = basis, inter, float(M)
        self.sign = {"singlet": 1.0, "triplet": -1.0}[pair]
        self.terms = terms
        js = np.array(sorted(set(np.round(basis.j, 1)) if js is None else js), float)
        self.channels = {j: basis.select(j)[:n_max] for j in js}
        if one_body is None:
            self.t = {j: np.diag(basis.E[idx]) for j, idx in self.channels.items()}
        else:
            self.t = {j: np.asarray(one_body(j, idx), float) for j, idx in self.channels.items()}
        self.n = {j: len(idx) for j, idx in self.channels.items()}
        self.triples = [(ja, jb, ja + jb - self.M) for ja in js for jb in js if (ja + jb - self.M) in self.channels]
        self.shapes = [(self.n[a], self.n[b], self.n[c]) for a, b, c in self.triples]
        self.offs = np.cumsum([0] + [int(np.prod(s)) for s in self.shapes])
        self.size = int(self.offs[-1])
        self.index = {tr: k for k, tr in enumerate(self.triples)}
        self._C = {}

    # -- helpers -------------------------------------------------------------------------
    def _Cf(self, ja, jc):
        """Compressed form factors of channels (ja, jc) as (k, a, c), cached."""
        key = (ja, jc)
        if key not in self._C:
            l = int(round(abs(ja - jc)))
            C = self.inter.form_factors_compressed(self.channels[ja], self.channels[jc], l)
            self._C[key] = np.ascontiguousarray(C.transpose(2, 0, 1))
        return self._C[key]

    def _blocks(self, v):
        return [v[self.offs[k]:self.offs[k + 1]].reshape(self.shapes[k]) for k in range(len(self.triples))]

    def symmetrize(self, v):
        """Project onto the sector: Psi[a,b,c] -> (Psi[a,b,c] + sign Psi[b,a,c]) / 2."""
        out = np.empty_like(v)
        P = self._blocks(v)
        for k, (ja, jb, jc) in enumerate(self.triples):
            kk = self.index[(jb, ja, jc)]
            out[self.offs[k]:self.offs[k + 1]] = (0.5 * (P[k] + self.sign * P[kk].transpose(1, 0, 2))).ravel()
        return out

    # -- H v -------------------------------------------------------------------------------
    def matvec(self, v):
        P = self._blocks(v)
        out = [np.zeros(s) for s in self.shapes]
        s12, s13, s23 = self.terms
        for k, (ja, jb, jc) in enumerate(self.triples):
            R = out[k]
            # one-body
            R += np.einsum("aA,Abc->abc", self.t[ja], P[k])
            R += np.einsum("bB,aBc->abc", self.t[jb], P[k])
            R += np.einsum("cC,abC->abc", self.t[jc], P[k])
            for kk, (ja2, jb2, jc2) in enumerate(self.triples):
                X = P[kk]
                # W_12: same jc, j_a + j_b conserved (automatic: both triples have the same M)
                if s12 and jc2 == jc:
                    C1, C2 = self._Cf(ja, ja2), self._Cf(jb, jb2)       # (k,a,A), (k,b,B)
                    T = np.tensordot(C1, X, axes=([2], [0]))           # (k, a, B, c)
                    R += s12 * np.einsum("kaBc,kbB->abc", T, C2, optimize=True)
                # W_13: same jb, j_a - j~_c conserved
                if s13 and jb2 == jb and (ja - ja2) == (jc - jc2):
                    C1, C3 = self._Cf(ja, ja2), self._Cf(jc, jc2)       # (k,a,A), (k,c,Cc)
                    T = np.tensordot(C1, X, axes=([2], [0]))           # (k, a, b, Cc)
                    R += s13 * np.einsum("kabC,kcC->abc", T, C3, optimize=True)
                # W_23: same ja, j_b - j~_c conserved
                if s23 and ja2 == ja and (jb - jb2) == (jc - jc2):
                    C2, C3 = self._Cf(jb, jb2), self._Cf(jc, jc2)
                    T = np.tensordot(C2, X, axes=([2], [1]))           # (k, b, a, Cc)
                    R += s23 * np.einsum("kbaC,kcC->abc", T, C3, optimize=True)
        return np.concatenate([o.ravel() for o in out])

    def diagonal(self):
        d = []
        s12, s13, s23 = self.terms
        for (ja, jb, jc) in self.triples:
            ta, tb, tc = np.diag(self.t[ja]), np.diag(self.t[jb]), np.diag(self.t[jc])
            Da = np.einsum("kaa->ka", self._Cf(ja, ja))
            Db = np.einsum("kbb->kb", self._Cf(jb, jb))
            Dc = np.einsum("kcc->kc", self._Cf(jc, jc))
            D = ta[:, None, None] + tb[None, :, None] + tc[None, None, :]
            D = D + s12 * np.einsum("ka,kb->ab", Da, Db)[:, :, None]
            D = D + s13 * np.einsum("ka,kc->ac", Da, Dc)[:, None, :]
            D = D + s23 * np.einsum("kb,kc->bc", Db, Dc)[None, :, :]
            d.append(D.ravel())
        return np.concatenate(d)

    def dense(self):
        """Dense H in the sector-agnostic product space (small tests only)."""
        I = np.eye(self.size)
        return np.array([self.matvec(I[:, i]) for i in range(self.size)]).T

    # -- lowest eigenpair --------------------------------------------------------------------
    def davidson(self, tol=1e-9, max_iter=200, max_space=30):
        diag = self.diagonal()
        # start: lowest diagonal configuration, symmetrised
        x = np.zeros(self.size)
        order = np.argsort(diag)
        for i in order[:4]:
            x[i] = 1.0
            y = self.symmetrize(x)
            if np.linalg.norm(y) > 1e-8:
                x = y
                break
            x[i] = 0.0
        x /= np.linalg.norm(x)
        Vs, AVs = [x], [self.matvec(x)]
        for it in range(max_iter):
            Vm, AVm = np.array(Vs).T, np.array(AVs).T
            Hs = Vm.T @ AVm
            ev, ey = np.linalg.eigh(0.5 * (Hs + Hs.T))
            theta, yv = ev[0], ey[:, 0]
            u, Au = Vm @ yv, AVm @ yv
            r = Au - theta * u
            res = np.linalg.norm(r)
            if res < tol * max(1.0, abs(theta)):
                break
            den = theta - diag
            t = self.symmetrize(r / np.where(np.abs(den) < 1e-12, 1e-12, den))
            if len(Vs) >= max_space:
                nu = np.linalg.norm(u)
                Vs, AVs = [u / nu], [Au / nu]
            for _ in range(2):
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

    def full_tensor(self, vec):
        """dict (ja, jb, jc) -> Psi[a, b, c] (views into vec), sum |Psi|^2 = |vec|^2."""
        return {tr: P for tr, P in zip(self.triples, self._blocks(vec))}

    def dyson_from_pair(self, vec3, pair_full):
        """Amplitudes d_c = <Psi_3| c^dag_{c,K'} |Psi_2> for adding the K' electron to a two-electron
        K-pair state given as TwoElectronCI.full_tensor (same channels and n_max).  The pair's spin
        state is unchanged, so only the spatial overlap enters.
        Returns dict j~ -> amplitudes over that K' channel, and the total weight."""
        d = {}
        for (ja, jb, jc), P in self.full_tensor(vec3).items():
            T = pair_full.get((ja, jb))
            if T is None:
                continue
            d[jc] = d.get(jc, 0.0) + np.einsum("abc,ab->c", P, T)
        return d, float(sum(v @ v for v in d.values()))
