"""M5: figures for the manuscript from the saved results (seconds).

    python3 scripts/m5_figures.py

fig_addition : mu1, mu2 (CI) and mu2 (first order) vs Z/Z_c, 2Delta = 30 meV, no gate and D = 10 nm
fig_splitting: E2(K-K') - E2(K-K singlet) vs Z/Z_c for all cases, and in meV vs the gap
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
M3 = os.path.join(ROOT, "results", "m3")
M4 = os.path.join(ROOT, "results", "m4")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm", "axes.linewidth": 0.6,
                     "lines.linewidth": 1.1, "legend.fontsize": 7, "savefig.bbox": "tight"})
C = {"blue": "#0072B2", "red": "#D55E00", "green": "#009E73", "orange": "#E69F00", "grey": "#7f7f7f"}


def rows(name):
    p = os.path.join(M3, name)
    return sorted([json.loads(l) for l in open(p) if l.strip()], key=lambda d: d["Z"]) if os.path.exists(p) else []


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200)
    plt.close(fig)


def fig_addition():
    fo = json.load(open(os.path.join(M4, "first_order_30meV.json")))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, (case, title) in zip(axs, (("nogate", "no gate"), ("D10", "gate, $D=10$ nm"))):
        two = [d for d in rows(f"scan_30meV_{case}.jsonl") if d.get("E1") is not None]
        x = np.array([d["Z"] / d["Zc"] for d in two])
        mu1 = np.array([d["E1"] for d in two])
        mu2 = np.array([min(d["singlet"], d["intervalley"]) - d["E1"] for d in two])
        f = fo[case]
        xf = np.array([d["Z"] / d["Zc"] for d in f])
        ax.plot(x, mu1, "k-o", ms=2.5, label="$\\mu_1$")
        ax.plot(x, mu2, "-o", color=C["blue"], ms=2.5, label="$\\mu_2$ (CI)")
        ax.plot(xf, [d["mu2_first_order"] for d in f], "--", color=C["red"], label="$\\mu_2$ (first order)")
        ax.axhspan(-1, 1, color=C["grey"], alpha=0.08, lw=0)
        ax.axhline(1, color=C["grey"], lw=0.5)
        ax.axhline(-1, color=C["grey"], lw=0.5)
        ax.axvline(1, color=C["grey"], lw=0.5, ls=":")
        ax.set_xlabel("$Z/Z_c$")
        ax.set_ylim(-2.0, 1.3)
        ax.set_title(title, fontsize=9)
    axs[0].set_ylabel("addition energy $/\\Delta$")
    axs[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_addition")


def fig_splitting():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axs[0]
    for (name, lab, col) in (("scan_30meV_nogate.jsonl", "30 meV", C["blue"]),
                             ("scan_15meV_nogate.jsonl", "15 meV", C["green"]),
                             ("scan_30meV_D10.jsonl", "30 meV, $D=10$ nm", C["orange"]),
                             ("scan_30meV_D3.jsonl", "30 meV, $D=3$ nm", C["red"])):
        r = [d for d in rows(name) if d.get("E1") is not None]
        x = [d["Z"] / d["Zc"] for d in r]
        y = [1e3 * (d["intervalley"] - d["singlet"]) for d in r]
        ax.plot(x, y, "-o", ms=2.5, color=col, label=lab)
    ax.axvline(1, color=C["grey"], lw=0.5, ls=":")
    ax.set_xlabel("$Z/Z_c$")
    ax.set_ylabel("$[E_2(KK')-E_2(KK)]/\\Delta$ ($\\times10^{-3}$)")
    ax.legend(frameon=False, loc="lower left")
    ax = axs[1]
    gaps, vals = [], []
    for g in (15, 30, 100, 300):
        r = [d for d in rows(f"scan_{g}meV_nogate.jsonl") if d.get("E1") is not None
             and abs(d["Z"] / d["Zc"] - 0.95) < 1e-3]
        if r:
            gaps.append(g)
            vals.append((r[0]["intervalley"] - r[0]["singlet"]) * g / 2)
    ax.loglog(gaps, np.abs(vals), "o-", color=C["blue"], ms=3, label="$L=2$, $E_{\\max}=6m$")
    ax.loglog(gaps, 1.4 * np.abs(vals), "--", color=C["blue"], lw=0.8, label="extrapolated ($\\times1.4$)")
    ax.set_xlabel("gap $2\\Delta$ (meV)")
    ax.set_ylabel("$|E_2(KK')-E_2(KK)|$ (meV)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_splitting")


if __name__ == "__main__":
    fig_addition()
    fig_splitting()
    print("ok")
