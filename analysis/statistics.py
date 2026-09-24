"""Statistische Auswertung der gepaarten Differenzen.

H1a: rot05 > baseline, H1b: rot10 > baseline, H1c: rot20 > baseline
(einseitig, Holm-korrigiert); H2: rot10 > rot20 (einseitig, separat).
Dazu Cohen's dz, 95%-Bootstrap-KI (B=10000), Wilcoxon und gepaarter
Permutationstest als Robustheitspruefung sowie TOST (Grenze dz=+-0.50) fuer
nicht signifikante Vergleiche.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

BOOTSTRAP_B = 10_000
PERMUTATION_B = 10_000
ANALYSIS_SEED = 12_345
ALPHA = 0.05
TOST_BOUND_DZ = 0.50      # Aequivalenzgrenze fuer nicht signifikante Vergleiche

COMPARISONS = [("H1a", "rot05", "baseline"), ("H1b", "rot10", "baseline"),
               ("H1c", "rot20", "baseline"), ("H2", "rot10", "rot20")]
HOLM_FAMILY = ["H1a", "H1b", "H1c"]


def paired_diffs(df, cond_a, cond_b, metric="test_acc"):
    a = df[df.condition == cond_a].set_index("seed")[metric]
    b = df[df.condition == cond_b].set_index("seed")[metric]
    common = sorted(set(a.index) & set(b.index))
    return (a.loc[common] - b.loc[common]).to_numpy()


def cohens_dz(d):
    return float(d.mean() / d.std(ddof=1))


def bootstrap_ci(d, rng, b=BOOTSTRAP_B, level=0.95):
    idx = rng.integers(0, len(d), size=(b, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1-level)/2*100, (1+level)/2*100])
    return float(lo), float(hi)


def permutation_p_greater(d, rng, b=PERMUTATION_B):
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(b, len(d)))
    perm = (signs * d).mean(axis=1)
    return float((np.sum(perm >= obs) + 1) / (b + 1))


def tost_p(d, bound_dz=TOST_BOUND_DZ):
    """TOST (Lakens 2018) mit standardisierter Grenze +-bound_dz.
    In Rohwerten ist die Grenze bound_dz * s_D; zurueck: (p, Grenze)."""
    bound = bound_dz * d.std(ddof=1)
    p_lo = stats.ttest_1samp(d, -bound, alternative="greater").pvalue
    p_hi = stats.ttest_1samp(d, bound, alternative="less").pvalue
    return float(max(p_lo, p_hi)), float(bound)


def holm_correction(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[name] = running
    return adj


def analyze(df, metric="test_acc", n_test=None, tost_comparisons=None):
    """Gepaarte Vergleiche fuer eine Zielgroesse.

    n_test: Groesse des Testsplits; damit wird die mittlere Differenz in
    zusaetzlich korrekt klassifizierte Testsequenzen umgerechnet.
    tost_comparisons: Vergleiche, fuer die TOST gerechnet wird. Ohne Angabe
    die in dieser Zielgroesse nicht signifikanten Vergleiche.
    """
    rng = np.random.default_rng(ANALYSIS_SEED)
    rows = []
    for name, a, b in COMPARISONS:
        d = paired_diffs(df, a, b, metric)
        t, p = stats.ttest_rel(d, np.zeros_like(d), alternative="greater")
        try:
            w_p = stats.wilcoxon(d, alternative="greater").pvalue
        except ValueError:
            w_p = 1.0
        lo, hi = bootstrap_ci(d, rng)
        rows.append({"comparison": name, "cond_a": a, "cond_b": b,
                     "metric": metric, "n_pairs": len(d),
                     "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
                     "ci95_lo": lo, "ci95_hi": hi, "cohens_dz": cohens_dz(d),
                     "t": float(t), "df": len(d) - 1, "p_one_sided": float(p),
                     "p_wilcoxon": float(w_p),
                     "p_permutation": permutation_p_greater(d, rng),
                     "n_positive": int((d > 0).sum())})
    res = pd.DataFrame(rows)
    in_family = res.comparison.isin(HOLM_FAMILY)
    res["p_holm"] = res.comparison.map(holm_correction(
        dict(zip(res.comparison[in_family], res.p_one_sided[in_family]))))
    res["p_permutation_holm"] = res.comparison.map(holm_correction(
        dict(zip(res.comparison[in_family], res.p_permutation[in_family]))))
    if n_test is not None:
        res["extra_correct_per_run"] = res.mean_diff * n_test
    if tost_comparisons is None:
        # in der Holm-Familie nach p_holm, sonst nach dem unkorrigierten p
        p_dec = res.p_holm.fillna(res.p_one_sided)
        tost_comparisons = set(res.comparison[p_dec >= ALPHA])
    tost = [tost_p(paired_diffs(df, a, b, metric)) if name in tost_comparisons
            else (np.nan, np.nan)
            for name, a, b in zip(res.comparison, res.cond_a, res.cond_b)]
    res["p_tost"] = [t[0] for t in tost]
    res["tost_bound"] = [t[1] for t in tost]
    return res


def not_significant(res):
    """Vergleiche ohne signifikantes Ergebnis im primaeren t-Test."""
    p_dec = res.p_holm.fillna(res.p_one_sided)
    return set(res.comparison[p_dec >= ALPHA])


def sensitivity_line(n, alpha=0.05):
    """Power fuer dz=0.50 bei n Paaren und kleinster Effekt mit 80% Power."""
    from scipy.optimize import brentq
    from scipy.stats import nct, t as tdist

    def power(dz, a):
        df = n - 1
        return 1 - nct.cdf(tdist.ppf(1 - a, df), df, dz * np.sqrt(n))

    p_full = power(0.50, alpha)
    p_holm = power(0.50, alpha / 3)
    dz80 = brentq(lambda z: power(z, alpha) - 0.80, 0.01, 2.0)
    return (f"Sensitivitaet (n={n}, einseitig, alpha={alpha}): Power fuer "
            f"dz=0.50 betraegt {p_full:.2f} (unkorrigiert) bzw. {p_holm:.2f} "
            f"(Holm-strengster Vergleich, alpha={alpha/3:.4f}); der kleinste "
            f"mit 80% Power detektierbare Effekt liegt bei dz~{dz80:.2f}.")


def to_markdown(res, alpha=0.05):
    lines = ["# Statistische Auswertung (Top-1-Accuracy)", "",
             "Alle Tests einseitig ('greater'), gepaart. H1a-H1c Holm-korrigiert; "
             f"H2 separat. Bootstrap B={BOOTSTRAP_B:,}, Permutation B={PERMUTATION_B:,}.",
             "",
             "| Vergleich | n | Diff. > 0 | mittl. Diff. | 95%-KI | dz | t(df) | p | p(Holm) "
             "| p(Wilcoxon) | p(Perm) | p(Perm, Holm) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: f"{v:.4f}" if pd.notna(v) else "--"
    for _, r in res.iterrows():
        lines.append(f"| {r.comparison} ({r.cond_a} vs {r.cond_b}) | {r.n_pairs} "
                     f"| {r.n_positive} | {r.mean_diff:+.4f} "
                     f"| [{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}] "
                     f"| {r.cohens_dz:+.3f} | {r.t:.3f} ({r.df}) | {r.p_one_sided:.4f} "
                     f"| {fmt(r.p_holm)} | {r.p_wilcoxon:.4f} | {r.p_permutation:.4f} "
                     f"| {fmt(r.p_permutation_holm)} |")
    if "extra_correct_per_run" in res:
        lines += ["", "Mittlere Differenz in zusaetzlich korrekt klassifizierten "
                  "Testsequenzen je Lauf: "
                  + ", ".join(f"{r.comparison} {r.extra_correct_per_run:+.1f}"
                              for _, r in res.iterrows())]
    for _, r in res[res.p_tost.notna()].iterrows():
        lines += ["", f"TOST {r.comparison} (nicht signifikant): Grenze dz=+-{TOST_BOUND_DZ:.2f} "
                  f"= +-{r.tost_bound:.4f}, p(TOST) = {r.p_tost:.4f}"]
    n = int(res.n_pairs.iloc[0])
    lines += ["", f"alpha = {alpha}; Interpretation primaer ueber Effektgroessen und KIs.",
              "", sensitivity_line(n, alpha)]
    return "\n".join(lines)
