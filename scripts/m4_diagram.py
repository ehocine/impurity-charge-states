"""M4: charge-state diagram (Z, E_F) of the impurity from the M3 scans.

    python3 scripts/m4_diagram.py

Reads results/m3/scan_<gap>meV_<case>.jsonl (one and two electrons, L = 2, E_max = 6m) and
results/m3/three_<gap>meV_<case>_n<n>.jsonl (three electrons, n_max = 12, 16, 20 per channel) and
draws, for E_F inside the gap, the ground-state electron number N: the grand-canonical ground
state has N electrons when mu_N < E_F < mu_{N+1}, with the addition energies
  mu_1 = E_1,  mu_2 = E_2 - E_1 (E_2 = lowest of the K-K singlet and K-K' pair),
  mu_3 = E_3 - E_2 (both at the same truncation n_max = 12, 16, 20 states per channel,
         results/m3/three_<gap>meV_<case>_n<n>.jsonl), extrapolated point by point as a + b/n_max
         (least squares over the three); its uncertainty is the larger deviation of the 12-16 and
         16-20 two-point extrapolations, at least 0.005 Delta.  At D = 10 nm, Z = 0.95 Z_c this
         gives 0.711, against 0.707-0.711 from the complete basis (all states up to E_max = 4m, 6m,
         8m: 0.7218, 0.7176, 0.7154) extrapolated in E_max^-p (results/m3/
         three_D10_nmax_convergence.jsonl, results/m10).  The truncation dependence varies strongly
         with the charge and the gate (none without a gate, up to 0.15 Delta from n_max = 20 at
         D = 3 nm near Z_c), so a single shift calibrated at one point does not apply.
  mu_4 = E_4 - E_3 (results/m8, ground sectors: closed shell / K pair + K' electron) at
         n_max = 12 and 16, extrapolated as a + b/n_max (checked against n_max = 20 at 0.95 Z_c:
         within 0.002 Delta).  With a gate the fourth electron completes the fourfold level near
         the conduction-band edge; without a gate it is unbound (mu_4 > Delta, results/m8/
         four_30meV_nogate_n6.jsonl).
Cases: 2Delta = 30 and 15 meV without gate, 30 meV with D = 10 and 3 nm.
Writes results/m4/diagram.json and figures/fig_diagram.{pdf,png}.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M3 = os.path.join(ROOT, "results", "m3")
OUT = os.path.join(ROOT, "results", "m4")
FIG = os.path.join(ROOT, "figures")


def load(name):
    path = os.path.join(M3, name)
    if not os.path.exists(path):
        return []
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return sorted(rows, key=lambda d: d["Z"])


def mu3_extrapolated(gap, case):
    """{round(Z, 6): (mu_3 extrapolated in 1/n_max, uncertainty, mu_3 at n_max = 20)}.

    a + b/n_max is fitted to the three largest n_max available (12, 16, 20, and 24, 28 where
    results/m3/three_<gap>meV_<case>_n24_28.jsonl has them).  The uncertainty is the largest
    deviation from it of the two-point extrapolations of consecutive n_max and of the fit to
    12, 16, 20, and at least 0.005 Delta."""
    runs = {n: {round(d["Z"], 6): d for d in load(f"three_{gap}meV_{case}_n{n}.jsonl")
                if d.get("E1") is not None and "E3" in d} for n in (12, 16, 20)}
    for d in load(f"three_{gap}meV_{case}_n24_28.jsonl"):           # same charges to 1e-5
        near = [z for z in runs[20] if abs(z - d["Z"]) < 1e-4]
        if "E3" in d and near:
            runs.setdefault(d["n_max"], {})[near[0]] = d
    out = {}
    for z, d20 in runs[20].items():
        ns = [n for n in sorted(runs) if z in runs[n]]
        if not {12, 16, 20} <= set(ns):
            continue
        mu = {n: runs[n][z]["E3"] - min(runs[n][z]["E2_singlet"], runs[n][z]["E2_intervalley"]) for n in ns}

        def fit(sel):
            b, a = np.polyfit(1.0 / np.array(sel, float), [mu[n] for n in sel], 1)
            return a
        a = fit(ns[-3:])
        cand = [fit([12, 16, 20])] + [mu[n2] - (mu[n1] - mu[n2]) / (1 / n1 - 1 / n2) / n2 for n1, n2 in zip(ns, ns[1:])]
        out[z] = (float(a), float(max(0.005, *(abs(c - a) for c in cand[-3:] + cand[:1]))), float(mu[20]))
    return out


def mu4_extrapolated(case):
    """{round(Z/Z_c, 3): mu_4} from the n_max = 12, 16 runs of M8 (a + b/n extrapolation)."""
    M8 = os.path.join(ROOT, "results", "m8")
    rows = []
    for name in (f"conv_{case}_all.jsonl", f"conv_{case}_x0.95.jsonl"):
        path = os.path.join(M8, name)
        if os.path.exists(path):
            rows += [json.loads(l) for l in open(path) if l.strip()]
    out = {}
    for x in sorted({round(r["x"], 3) for r in rows}):
        mu = {r["n_max"]: r["mu4"] for r in rows if round(r["x"], 3) == x}
        if 12 in mu and 16 in mu:
            b = (mu[12] - mu[16]) / (1 / 12 - 1 / 16)
            out[x] = mu[16] - b / 16
    return out


def addition_energies(case, gap=30):
    two = [d for d in load(f"scan_{gap}meV_{case}.jsonl") if d.get("E1") is not None]
    mu3 = mu3_extrapolated(gap, case)
    mu4 = mu4_extrapolated(case) if gap == 30 and case != "nogate" else {}
    rows = []
    for d in two:
        E2 = min(d["singlet"], d["intervalley"])
        rec = {"Z": d["Z"], "Zc": d["Zc"], "mu1": d["E1"], "mu2": E2 - d["E1"],
               "split_KKp_minus_KK": d["intervalley"] - d["singlet"], "dived": d.get("dived", False)}
        t = mu3.get(round(d["Z"], 6))
        if t is not None:
            rec["mu3"], rec["mu3_err"], rec["mu3_n20"] = t
        x = round(d["Z"] / d["Zc"], 3)
        if x in mu4:
            rec["mu4"] = mu4[x]
        rows.append(rec)
    return rows



def draw(ax, x, mus, ylim, x_min=None, small=False, n_label=5):
    """Charge-state regions between the addition energies mus = (mu_1 ... mu_4) on axes ax."""
    from scipy.interpolate import PchipInterpolator
    cols = ["#f2f2f2", "#cfe3f3", "#9ecae1", "#6baed6", "#3182bd"]
    lo_x = round(float(x.min()), 3) if x_min is None else x_min
    xs = np.linspace(x.min(), x.max(), 400)
    lines = []
    for mu in mus:
        ok = np.isfinite(mu)
        if ok.sum() >= 2:
            f = PchipInterpolator(x[ok], mu[ok], extrapolate=False)
            y = f(xs)
            y[(xs < x[ok].min()) | (xs > x[ok].max())] = np.nan
        else:
            y = np.full_like(xs, np.nan)
        lines.append(y)
    # region N lies between mu_N (below) and mu_{N+1} (above), clipped to the gap
    bounds = [np.full_like(xs, -1.0)] + [np.where(np.isfinite(y), np.clip(y, -1, 1), 1.0) for y in lines] \
        + [np.full_like(xs, 1.0)]
    span = ylim[1] - ylim[0]
    vis = xs >= lo_x
    for n in range(5):
        lo, hi = bounds[n], bounds[n + 1]
        ax.fill_between(xs, lo, np.maximum(hi, lo), color=cols[n], lw=0)
        mid = 0.5 * (lo + hi)
        sel = vis & ((hi - lo) > 0.09 * span) & (mid > ylim[0]) & (mid < ylim[1])
        if n < n_label and np.any(sel):
            i = np.flatnonzero(sel)[int(0.6 * np.sum(sel))] if n < 2 else np.flatnonzero(sel)[-max(1, np.sum(sel) // 4)]
            i = min(i, int(np.searchsorted(xs, xs.max() - 0.06 * (xs.max() - lo_x))))   # keep the label inside
            ax.text(xs[i], mid[i], f"$N={n}$", ha="center", va="center", fontsize=6 if small else 8,
                    color="w" if n == 4 else "k")
    for y, mu in zip(lines, mus):
        ax.plot(xs, y, "k-", lw=0.7 if small else 0.9)
        ok = np.isfinite(mu)
        ax.plot(x[ok], mu[ok], "k.", ms=2 if small else 3)
    ax.axvline(1.0, color="0.4", lw=0.6, ls=":")
    ax.set_ylim(*ylim)
    ax.set_xlim(lo_x, max(x.max(), 1.0) + 0.01)

def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    CASES = (("30meV_nogate", 30, "nogate", "$2\\Delta=30$ meV, no gate"),
             ("15meV_nogate", 15, "nogate", "$2\\Delta=15$ meV, no gate"),
             ("30meV_D10", 30, "D10", "30 meV, gate $D=10$ nm"),
             ("30meV_D3", 30, "D3", "30 meV, gate $D=3$ nm"))
    data = {key: addition_energies(case, gap) for key, gap, case, _ in CASES}
    json.dump(data, open(os.path.join(OUT, "diagram.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), sharey=True)
    axs = axs.ravel()
    for ax, (key, _, _, title) in zip(axs, CASES):
        rows = data[key]
        if not rows:
            continue
        Z = np.array([r["Z"] for r in rows])
        Zc = rows[0]["Zc"]
        mus = [np.array([r.get(k, np.nan) for r in rows]) for k in ("mu1", "mu2", "mu3", "mu4")]
        gated = key.endswith(("D10", "D3"))
        draw(ax, Z / Zc, mus, (-1, 1), n_label=3 if gated else 5)
        if gated:
            # zoom on the narrow three- and four-electron strips below the band edge
            ins = ax.inset_axes([0.07, 0.07, 0.42, 0.42])
            draw(ins, Z / Zc, mus, (0.6, 1.0), x_min=0.8, small=True)
            ins.tick_params(labelsize=6, length=2, pad=1)
            ins.set_xticks([0.8, 0.9, 1.0])
            ins.set_yticks([0.6, 0.8, 1.0])
            ax.add_patch(plt.Rectangle((0.8, 0.6), 0.21, 0.4, fill=False, ec="0.3", lw=0.6, ls="--"))
        ax.set_xlabel("$Z/Z_c$")
        ax.set_title(title, fontsize=9)
    axs[0].set_ylabel("$E_F/\\Delta$")
    axs[2].set_ylabel("$E_F/\\Delta$")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_diagram.{ext}"), dpi=200)
    for case, rows in data.items():
        print(case)
        for r in rows:
            print("  Z/Zc=%.3f mu1=%+.4f mu2=%+.4f mu3=%s mu4=%s split=%+.5f%s" % (
                r["Z"] / r["Zc"], r["mu1"], r["mu2"], ("%+.4f" % r["mu3"]) if "mu3" in r else "  --  ",
                ("%+.4f" % r["mu4"]) if "mu4" in r else "  --  ",
                r["split_KKp_minus_KK"], " (dived)" if r["dived"] else ""))


if __name__ == "__main__":
    main()
