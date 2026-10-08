"""M4: charge-state diagram (Z, E_F) of the impurity from the M3 scans.

    python3 scripts/m4_diagram.py

Reads results/m3/scan_<gap>meV_<case>.jsonl (one and two electrons, L = 2, E_max = 6m) and
results/m3/three_<gap>meV_<case>_n20.jsonl (three electrons, n_max = 20 per channel) and
draws, for E_F inside the gap, the ground-state electron number N: the grand-canonical ground
state has N electrons when mu_N < E_F < mu_{N+1}, with the addition energies
  mu_1 = E_1,  mu_2 = E_2 - E_1 (E_2 = lowest of the K-K singlet and K-K' pair),
  mu_3 = E_3 - E_2 (both at n_max = 20, where the truncation largely cancels) plus the
         correction MU3_SHIFT to n_max -> inf, calibrated at D = 10 nm, Z = 0.95 Z_c
         (results/m3/three_D10_nmax_convergence.jsonl: n_max = 12..28 -> 0.7433 .. 0.7255,
         1/n extrapolation 0.712; shift -0.018 +- 0.01 Delta).
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


MU3_SHIFT = -0.018


def load(name):
    path = os.path.join(M3, name)
    if not os.path.exists(path):
        return []
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return sorted(rows, key=lambda d: d["Z"])


def addition_energies(case, gap=30):
    two = [d for d in load(f"scan_{gap}meV_{case}.jsonl") if d.get("E1") is not None]
    three = {round(d["Z"], 6): d for d in load(f"three_{gap}meV_{case}_n20.jsonl")
             if d.get("E1") is not None and "E3" in d}
    rows = []
    for d in two:
        E2 = min(d["singlet"], d["intervalley"])
        rec = {"Z": d["Z"], "Zc": d["Zc"], "mu1": d["E1"], "mu2": E2 - d["E1"],
               "split_KKp_minus_KK": d["intervalley"] - d["singlet"], "dived": d.get("dived", False)}
        t = three.get(round(d["Z"], 6))
        if t is not None:
            E2t = min(t["E2_singlet"], t["E2_intervalley"])
            rec["mu3"] = t["E3"] - E2t + MU3_SHIFT
        rows.append(rec)
    return rows


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
    cols = ["#f2f2f2", "#cfe3f3", "#9ecae1", "#6baed6"]
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), sharey=True)
    axs = axs.ravel()
    for ax, (key, _, _, title) in zip(axs, CASES):
        rows = data[key]
        if not rows:
            continue
        Z = np.array([r["Z"] for r in rows])
        Zc = rows[0]["Zc"]
        mus = [np.array([r.get(k, np.nan) for r in rows]) for k in ("mu1", "mu2", "mu3")]
        x = Z / Zc
        from scipy.interpolate import PchipInterpolator
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
        for n in range(4):
            lo, hi = bounds[n], bounds[n + 1]
            ax.fill_between(xs, lo, np.maximum(hi, lo), color=cols[n], lw=0)
            sel = (hi - lo) > 0.18
            if np.any(sel):
                i = np.flatnonzero(sel)[int(0.6 * np.sum(sel))] if n < 2 else np.flatnonzero(sel)[-max(1, np.sum(sel) // 4)]
                ax.text(xs[i], 0.5 * (lo[i] + hi[i]), f"$N={n}$", ha="center", va="center", fontsize=8)
        for y, mu in zip(lines, mus):
            ax.plot(xs, y, "k-", lw=0.9)
            ok = np.isfinite(mu)
            ax.plot(x[ok], mu[ok], "k.", ms=3)
        ax.axvline(1.0, color="0.4", lw=0.6, ls=":")
        ax.set_ylim(-1, 1)
        ax.set_xlim(x.min(), max(x.max(), 1.0) + 0.01)
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
            print("  Z/Zc=%.3f mu1=%+.4f mu2=%+.4f mu3=%s split=%+.5f%s" % (
                r["Z"] / r["Zc"], r["mu1"], r["mu2"], ("%+.4f" % r["mu3"]) if "mu3" in r else "  --  ",
                r["split_KKp_minus_KK"], " (dived)" if r["dived"] else ""))


if __name__ == "__main__":
    main()
