"""Abbildungen zur Auswertung.

Breiten: 3,45 Zoll fuer eine Spalte, 7,1 Zoll fuer beide Spalten.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

COND_ORDER = ["baseline", "rot05", "rot10", "rot20"]
COND_LABEL = {"baseline": "Baseline", "rot05": "±5°", "rot10": "±10°", "rot20": "±20°"}
COMPS = [("H1a", "rot05", "baseline"), ("H1b", "rot10", "baseline"),
         ("H1c", "rot20", "baseline"), ("H2", "rot10", "rot20")]

COL_W, FULL_W = 3.45, 7.1          # Zoll: eine Spalte / beide Spalten

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "figure.dpi": 300, "savefig.dpi": 300,
})


def _pairs(df, a, b, metric="test_acc"):
    A = df[df.condition == a].set_index("seed")[metric]
    B = df[df.condition == b].set_index("seed")[metric]
    idx = sorted(set(A.index) & set(B.index))
    return A.loc[idx].to_numpy(), B.loc[idx].to_numpy()


def plot_accuracy_by_condition(df, out, metric="test_acc"):
    conds = [c for c in COND_ORDER if c in set(df.condition)]
    data = [df[df.condition == c].sort_values("seed")[metric].to_numpy() for c in conds]
    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    seeds = sorted(set.intersection(*[set(df[df.condition == c].seed) for c in conds]))
    for s in seeds:
        ys = [float(df[(df.condition == c) & (df.seed == s)][metric].iloc[0]) for c in conds]
        ax.plot(range(1, len(conds)+1), ys, color="grey", alpha=0.2, lw=0.5, zorder=1)
    ax.boxplot(data, positions=range(1, len(conds)+1), widths=0.5,
               showfliers=False, zorder=2)
    for i, y in enumerate(data, 1):
        ax.scatter(np.full_like(y, i, dtype=float)
                   + np.random.default_rng(0).uniform(-0.08, 0.08, len(y)),
                   y, s=7, alpha=0.65, zorder=3)
    ax.set_xticks(range(1, len(conds)+1))
    ax.set_xticklabels([COND_LABEL[c] for c in conds])
    ax.set_ylabel("Top-1-Accuracy (Test)")
    ax.grid(axis="y", alpha=0.2); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3); fig.savefig(out); plt.close(fig)


def plot_qq(df, out, metric="test_acc"):
    fig, axes = plt.subplots(1, len(COMPS), figsize=(FULL_W, 2.1))
    for ax, (name, a, b) in zip(np.atleast_1d(axes), COMPS):
        A, B = _pairs(df, a, b, metric)
        stats.probplot((A - B) * 100, dist="norm", plot=ax)
        ax.set_title(name)
        ax.set_xlabel("theoretische Quantile")
        ax.set_ylabel("beobachtete Differenz (pp)" if ax is axes[0] else "")
        ax.get_lines()[0].set_markersize(2.5)
        ax.get_lines()[1].set_linewidth(1.0)
        ax.tick_params(pad=1.5)
    fig.tight_layout(pad=0.3); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def plot_forest(res, out, res_f1=None):
    """Mittlere gepaarte Differenzen mit 95%-Bootstrap-KI, in Spaltenbreite.
    Mit res_f1 werden Accuracy und Macro-F1 versetzt nebeneinander gezeigt;
    Marker unterscheiden sich zusaetzlich zur Farbe (Graustufendruck)."""
    series = [(res, "Top-1-Accuracy", "#1f77b4", "o")]
    if res_f1 is not None:
        series.append((res_f1, "Macro-F1", "#d95f02", "s"))
    n = len(res)
    y = np.arange(n)[::-1].astype(float)
    y[-1] -= 0.4                                  # H2 von den H1-Vergleichen absetzen
    off = 0.14 if len(series) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    for i, (r, label, color, marker) in enumerate(series):
        yy = y + (off if i == 0 else -off)
        m = r.mean_diff.to_numpy() * 100
        ax.errorbar(m, yy,
                    xerr=[m - r.ci95_lo.to_numpy() * 100, r.ci95_hi.to_numpy() * 100 - m],
                    fmt=marker, ms=3.8, lw=1.3, capsize=2, capthick=1.0,
                    color=color, label=label)
    ax.axvline(0, color="0.3", ls="--", lw=0.8, zorder=0)
    ax.axhline((y[-2] + y[-1]) / 2, color="0.8", lw=0.6, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.comparison}: {COND_LABEL[r.cond_a]} – {COND_LABEL[r.cond_b]}"
                        for _, r in res.iterrows()])
    ax.set_ylim(y[-1] - 0.5, y[0] + 0.5)
    ax.set_xlabel("mittlere gepaarte Differenz (pp)")
    ax.grid(axis="x", alpha=0.25, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if len(series) > 1:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                  frameon=False, handletextpad=0.3, columnspacing=1.2,
                  borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
