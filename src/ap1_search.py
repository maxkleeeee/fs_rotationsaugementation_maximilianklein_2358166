"""AP 1: Festlegung von Modellarchitektur und Hyperparametern.

Trainiert die Kandidatenkonfigurationen ohne Rotationsaugmentation und
bewertet sie auf dem Validierungssplit. Der Testsplit wird nicht geladen.
Kriterium ist die Val-Accuracy nach der letzten Epoche, gemittelt ueber
die Seeds.

    python -m src.ap1_search --seeds 3
    python -m src.ap1_search --seeds 8 --only C_klein_subset
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy import stats
from torch.utils.data import DataLoader

from src.data.dataset import LandmarkSequenceDataset
from src.model import SignLSTM
from src.seeds import SHUFFLE_STREAM_OFFSET
from src.train import resolve_landmark_subset

# Abweichungen von configs/default.yaml je Kandidat. landmark_subset ist
# ueberall explizit gesetzt, damit die Suche nicht vom Wert der finalen
# Config abhaengt.
CANDIDATES: dict[str, dict] = {
    "A_gross_2layer":   {"model": {"landmark_subset": "all", "hidden_dim": 256,
                                   "num_layers": 2, "dropout": 0.3},
                         "train": {"batch_size": 32, "weight_decay": 1e-4}},
    "B_klein_1layer":   {"model": {"landmark_subset": "all", "hidden_dim": 128,
                                   "num_layers": 1, "dropout": 0.4},
                         "train": {"batch_size": 16, "weight_decay": 1e-2}},
    "C_klein_subset":   {"model": {"landmark_subset": "upper_body", "hidden_dim": 128,
                                   "num_layers": 1, "dropout": 0.4},
                         "train": {"batch_size": 16, "weight_decay": 1e-2}},
    "D_mittel_2layer":  {"model": {"landmark_subset": "all", "hidden_dim": 192,
                                   "num_layers": 2, "dropout": 0.4},
                         "train": {"batch_size": 16, "weight_decay": 1e-2}},
    "E_klein_bs32":     {"model": {"landmark_subset": "all", "hidden_dim": 128,
                                   "num_layers": 1, "dropout": 0.4},
                         "train": {"batch_size": 32, "weight_decay": 1e-2}},
    "F_gross_bs16":     {"model": {"landmark_subset": "all", "hidden_dim": 256,
                                   "num_layers": 2, "dropout": 0.3},
                         "train": {"batch_size": 16, "weight_decay": 1e-2}},
    "G_bidir_klein":    {"model": {"landmark_subset": "all", "hidden_dim": 128,
                                   "num_layers": 1, "dropout": 0.4,
                                   "bidirectional": True},
                         "train": {"batch_size": 16, "weight_decay": 1e-2}},
}


def merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for section, values in override.items():
        out[section] = {**out.get(section, {}), **values}
    return out


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device, non_blocking=True)).argmax(1).cpu()
            correct += int((pred == y).sum()); total += len(y)
    return correct / max(1, total)


def train_one(cfg, seed, cache: dict) -> dict:
    """Ein Trainingslauf ohne Rotation; gibt Val-Kennwerte zurueck."""
    device = torch.device(cfg["experiment"]["device"]
                          if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        det = bool(cfg["experiment"].get("deterministic", True))
        torch.backends.cudnn.benchmark = not det
        torch.backends.cudnn.deterministic = det

    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    subset = resolve_landmark_subset(m.get("landmark_subset"))
    key = ("all" if subset is None else tuple(subset))
    if key not in cache:
        manifest = Path(d["landmarks_dir"]) / "manifest.json"
        cache[key] = (
            LandmarkSequenceDataset(manifest, d["landmarks_dir"], "train",
                                    d["seq_len"], landmark_subset=subset),
            LandmarkSequenceDataset(manifest, d["landmarks_dir"], "val",
                                    d["seq_len"], landmark_subset=subset))
    train_ds, val_ds = cache[key]

    torch.manual_seed(seed)
    shuffle_gen = torch.Generator().manual_seed(seed + SHUFFLE_STREAM_OFFSET)
    pin = device.type == "cuda"
    tl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                    generator=shuffle_gen, num_workers=0, pin_memory=pin)
    vl = DataLoader(val_ds, batch_size=64, num_workers=0, pin_memory=pin)

    model = SignLSTM(train_ds.n_landmarks * 6, m["hidden_dim"],
                     m["num_layers"], len(train_ds.classes), m["dropout"],
                     m.get("bidirectional", False)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"],
                            weight_decay=t["weight_decay"])
    epochs, warmup = t["epochs"], int(t.get("warmup_epochs", 0))
    spe = max(1, len(tl))

    def lr_factor(step):
        ep = step / spe
        if warmup > 0 and ep < warmup:
            return ep / warmup
        prog = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_factor)
    crit = nn.CrossEntropyLoss(label_smoothing=t["label_smoothing"])
    clip = float(t.get("grad_clip", 0.0))

    hist, t0 = [], time.time()
    for _ in range(epochs):
        model.train()
        for x, y in tl:
            opt.zero_grad()
            crit(model(x.to(device, non_blocking=True)),
                 y.to(device, non_blocking=True)).backward()
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); sched.step()
        hist.append(evaluate(model, vl, device))
    train_acc = evaluate(model, DataLoader(train_ds, batch_size=64), device)
    return {"val_final": hist[-1], "val_last10": float(np.mean(hist[-10:])),
            "val_best": float(max(hist)), "train_acc": train_acc,
            "n_params": sum(p.numel() for p in model.parameters()),
            "secs": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--only", default=None, help="Nur diesen Kandidaten.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base = yaml.safe_load(Path(a.config).read_text())
    out_path = Path(a.out or Path(base["experiment"]["results_dir"])
                    / "ap1_search.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line); done.add((r["candidate"], r["seed"]))

    cache: dict = {}
    for name, override in CANDIDATES.items():
        if a.only and name != a.only:
            continue
        cfg = merge(base, override)
        cfg["model"]["input_dim"] = None
        for seed in range(1, a.seeds + 1):
            if (name, seed) in done:
                continue
            r = train_one(cfg, seed, cache)
            r.update({"candidate": name, "seed": seed,
                      "model": cfg["model"], "train": cfg["train"]})
            with open(out_path, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{name:18s} s{seed} train={r['train_acc']:.4f} "
                  f"val_final={r['val_final']:.4f} "
                  f"val_last10={r['val_last10']:.4f} "
                  f"({r['n_params']/1e6:.2f}M, {r['secs']}s)", flush=True)

    rows = [json.loads(l) for l in out_path.read_text().splitlines()]
    agg: dict[str, list] = {}
    for r in rows:
        agg.setdefault(r["candidate"], []).append(r)
    print(f"\n{'Kandidat':18s} {'n':>2s} {'train':>7s} {'val_final':>10s} "
          f"{'val_last10':>11s} {'Params':>8s}")
    ranked = sorted(agg.items(),
                    key=lambda kv: -np.mean([x["val_final"] for x in kv[1]]))
    for name, rs in ranked:
        print(f"{name:18s} {len(rs):2d} "
              f"{np.mean([x['train_acc'] for x in rs]):7.4f} "
              f"{np.mean([x['val_final'] for x in rs]):10.4f} "
              f"{np.mean([x['val_last10'] for x in rs]):11.4f} "
              f"{rs[0]['n_params']/1e6:7.2f}M")
    print(f"\nBester Kandidat: {ranked[0][0]}")
    (best, rb), (second, rs) = ranked[0], ranked[1]
    vb = {x["seed"]: x["val_final"] for x in rb}
    vs = {x["seed"]: x["val_final"] for x in rs}
    common = sorted(set(vb) & set(vs))
    if len(common) > 1:
        p = stats.ttest_rel([vb[k] for k in common], [vs[k] for k in common]).pvalue
        print(f"{best} vs. {second} ueber {len(common)} gemeinsame Seeds: "
              f"{np.mean([vb[k] for k in common]):.4f} vs. "
              f"{np.mean([vs[k] for k in common]):.4f} "
              f"(gepaarter t-Test, zweiseitig, p = {p:.3f})")
    print("Diese Werte in configs/default.yaml uebertragen und anschliessend "
          "src.run_experiments starten.")


if __name__ == "__main__":
    main()
