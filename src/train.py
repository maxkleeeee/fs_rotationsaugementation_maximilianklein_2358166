"""Ein Trainingslauf = eine Bedingung x ein Seed.

Drei getrennte RNG-Streams pro Seed:
  * seed                       -> Gewichtsinitialisierung und Dropout (torch)
  * seed + SHUFFLE_OFFSET      -> Reihenfolge der Mini-Batches (DataLoader)
  * seed + AUG_OFFSET          -> Rotationswinkel (numpy, im Dataset)
Baseline und Rotationsbedingungen unterscheiden sich bei gleichem Seed
nur in der Rotation.

    python -m src.train --condition rot10 --seed 7
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LandmarkSequenceDataset
from src.model import SignLSTM
from src.seeds import AUG_STREAM_OFFSET, SHUFFLE_STREAM_OFFSET


# MediaPipe-Pose: 0-10 Gesicht, 11/12 Schultern, 13/14 Ellenbogen,
# 15/16 Handgelenke, 17-22 Handpunkte der Pose, 23/24 Huefte, 25-32 Beine;
# 33-74 die beiden Haende.
LANDMARK_PRESETS = {
    "all": None,
    "upper_body": list(range(11, 25)) + list(range(33, 75)),   # 56 Landmarks
}


def resolve_landmark_subset(spec):
    """Config-Wert -> Indexliste. None/'all' = alle 75 Landmarks."""
    if spec is None or spec == "all":
        return None
    if isinstance(spec, str):
        if spec not in LANDMARK_PRESETS:
            raise ValueError(f"Unbekanntes landmark_subset {spec!r}; "
                             f"bekannt: {sorted(LANDMARK_PRESETS)}")
        return LANDMARK_PRESETS[spec]
    return list(spec)


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            ps.append(logits.argmax(1).cpu().numpy())
            ys.append(y.numpy())
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    return (float((y_true == y_pred).mean()),
            float(f1_score(y_true, y_pred, average="macro")))


def run(cfg, condition, seed, progress: bool = False) -> dict:
    device = torch.device(cfg["experiment"]["device"]
                          if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        det = bool(cfg["experiment"].get("deterministic", True))
        torch.backends.cudnn.benchmark = not det
        torch.backends.cudnn.deterministic = det
    max_rot = float(cfg["conditions"][condition])
    d = cfg["data"]

    torch.manual_seed(seed)
    shuffle_gen = torch.Generator().manual_seed(seed + SHUFFLE_STREAM_OFFSET)
    aug_rng = np.random.default_rng(seed + AUG_STREAM_OFFSET)

    manifest = Path(d["landmarks_dir"]) / "manifest.json"
    subset = resolve_landmark_subset(cfg["model"].get("landmark_subset"))
    mk = lambda split, aug: LandmarkSequenceDataset(
        manifest, d["landmarks_dir"], split, d["seq_len"],
        augment=aug, max_rotation_deg=max_rot if aug else 0.0,
        aug_rng=aug_rng if aug else None, landmark_subset=subset)
    train_ds, val_ds, test_ds = mk("train", True), mk("val", False), mk("test", False)

    t = cfg["train"]
    checkpoint = str(t.get("checkpoint", "final"))
    if checkpoint != "final":
        raise ValueError(f"train.checkpoint muss 'final' sein, nicht {checkpoint!r}")
    pin = device.type == "cuda"
    # num_workers=0: Worker-Prozesse bekaemen je eine Kopie von aug_rng und
    # wuerden in jeder Epoche dieselben Winkel ziehen.
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                              generator=shuffle_gen, num_workers=0,
                              pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=t["batch_size"], pin_memory=pin)

    m = cfg["model"]
    # Pro Landmark x, y, z und dx, dy, dz
    input_dim = train_ds.n_landmarks * 6
    if m.get("input_dim") not in (None, input_dim):
        raise ValueError(f"model.input_dim={m['input_dim']} passt nicht zur "
                         f"Landmark-Auswahl ({train_ds.n_landmarks} x 6 = "
                         f"{input_dim}).")
    model = SignLSTM(input_dim, m["hidden_dim"], m["num_layers"],
                     len(train_ds.classes), m["dropout"], m["bidirectional"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"],
                            weight_decay=t["weight_decay"])
    epochs = t["epochs"]
    warmup = int(t.get("warmup_epochs", 0))
    steps_per_epoch = max(1, len(train_loader))

    def lr_factor(step):
        ep = step / steps_per_epoch
        if warmup > 0 and ep < warmup:
            return ep / warmup
        prog = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_factor)
    grad_clip = float(t.get("grad_clip", 0.0))
    crit = nn.CrossEntropyLoss(label_smoothing=t["label_smoothing"])

    best_val, history = -1.0, []
    epoch_iter = tqdm(range(epochs), desc=f"{condition} s{seed}",
                      disable=not progress)
    for epoch in epoch_iter:
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = crit(model(x.to(device, non_blocking=True)),
                        y.to(device, non_blocking=True))
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step(); sched.step()
        val_acc, val_f1 = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "val_acc": val_acc, "val_f1": val_f1})
        best_val = max(best_val, val_acc)
        if progress:
            epoch_iter.set_postfix(val_acc=f"{val_acc:.3f}", best=f"{best_val:.3f}")

    # Alle Bedingungen werden nach derselben Anzahl Trainingsschritte evaluiert;
    # der Validierungssplit wird nicht zur Modellauswahl genutzt.
    test_acc, test_f1 = evaluate(model, test_loader, device)

    out_dir = Path(cfg["experiment"]["results_dir"]) / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {"torch": torch.__version__,
           "cuda": torch.version.cuda if torch.cuda.is_available() else None,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)}
    result = {"condition": condition, "seed": seed, "max_rotation_deg": max_rot,
              "test_acc": test_acc, "test_macro_f1": test_f1,
              "best_val_acc": best_val, "final_val_acc": history[-1]["val_acc"],
              "checkpoint": checkpoint, "epochs": t["epochs"],
              "n_landmarks": train_ds.n_landmarks,
              "landmark_subset": cfg["model"].get("landmark_subset", "all"),
              "n_classes": len(train_ds.classes),
              "n_train": len(train_ds), "n_val": len(val_ds),
              "n_test": len(test_ds),
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": env, "history": history}
    (out_dir / f"seed_{seed:03d}.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--progress", action="store_true",
                    help="Fortschrittsbalken pro Epoche anzeigen.")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    r = run(cfg, a.condition, a.seed, progress=a.progress)
    print(f"[{a.condition} | seed {a.seed}] "
          f"test_acc={r['test_acc']:.4f} macro_f1={r['test_macro_f1']:.4f}")
