"""Statistische Auswertung und Abbildungen aus <results>/all_results.csv.

    python -m analysis.run_analysis --results results_aslc --figures figures_aslc
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis import plots
from analysis.statistics import analyze, not_significant, to_markdown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_aslc")
    ap.add_argument("--figures", default="figures_aslc")
    a = ap.parse_args()
    res_dir, fig_dir = Path(a.results), Path(a.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(res_dir / "all_results.csv")
    print(f"Geladen: {len(df)} Laeufe, Bedingungen: {sorted(df.condition.unique())}")
    first_run = next(res_dir.glob("*/seed_*.json"))
    n_test = json.loads(first_run.read_text())["n_test"]

    # TOST fuer beide Zielgroessen bei den Vergleichen, die in der
    # Top-1-Accuracy nicht signifikant sind
    res_acc = analyze(df, "test_acc", n_test=n_test)
    res_f1 = analyze(df, "test_macro_f1", tost_comparisons=not_significant(res_acc))
    res_acc.to_csv(res_dir / "statistics.csv", index=False)
    res_f1.to_csv(res_dir / "statistics_f1.csv", index=False)
    (res_dir / "statistics.md").write_text(to_markdown(res_acc), encoding="utf-8")

    plots.plot_qq(df, fig_dir / "qq_differences.png")
    plots.plot_accuracy_by_condition(df, fig_dir / "accuracy_by_condition.png")
    for ext in ("png", "pdf"):
        plots.plot_forest(res_acc, fig_dir / f"forest_effects.{ext}", res_f1=res_f1)
    print("Statistik und Diagramme geschrieben.")


if __name__ == "__main__":
    main()
