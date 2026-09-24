"""Fuehrt alle Laeufe aus: 4 Bedingungen x 45 Seeds = 180 Trainingslaeufe.

Bereits vorhandene Ergebnis-JSONs werden uebersprungen. Fehlgeschlagene
Laeufe werden mit demselben Seed wiederholt. Am Ende entsteht
<results_dir>/all_results.csv fuer die Auswertung.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd
import yaml

from src.seeds import SEEDS
from src.train import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    res_dir = Path(cfg["experiment"]["results_dir"])
    seeds = SEEDS[: int(cfg["experiment"]["n_seeds"])]
    conditions = list(cfg["conditions"].keys())
    total = len(conditions) * len(seeds)
    print(f"Geplant: {len(conditions)} x {len(seeds)} = {total} Laeufe")

    done = 0
    for cond in conditions:
        for seed in seeds:
            out = res_dir / cond / f"seed_{seed:03d}.json"
            if out.exists():
                done += 1; continue
            for attempt in range(a.retries + 1):
                try:
                    r = run(cfg, cond, seed); done += 1
                    print(f"[{done}/{total}] {cond} seed {seed}: acc={r['test_acc']:.4f}")
                    break
                except Exception:
                    print(f"FEHLER {cond} seed {seed} (Versuch {attempt+1}):")
                    traceback.print_exc()
            else:
                print(f"AUSSCHLUSS dokumentieren: {cond} seed {seed}")

    rows = []
    for cond in conditions:
        for f in sorted((res_dir / cond).glob("seed_*.json")):
            j = json.loads(f.read_text())
            rows.append({k: j[k] for k in
                         ("condition", "seed", "max_rotation_deg",
                          "test_acc", "test_macro_f1", "best_val_acc")})
    df = pd.DataFrame(rows).sort_values(["condition", "seed"])
    df.to_csv(res_dir / "all_results.csv", index=False)
    print(f"Geschrieben: {res_dir/'all_results.csv'} ({len(df)} Zeilen)")


if __name__ == "__main__":
    main()
