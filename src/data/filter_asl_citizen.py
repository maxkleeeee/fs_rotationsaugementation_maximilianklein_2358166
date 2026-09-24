"""Wendet die Ausschlussregeln an und erstellt das Manifest.

ASL Citizen (Desai et al., 2023) gibt die Split-Zuordnung ueber drei CSVs
mit den Spalten ``participant, video_file, gloss`` vor. Die Splits sind
signer-disjunkt; das wird hier geprueft und im Manifest protokolliert.

Schritte:
  1. Klassenauswahl aus data/aslc_class_selection.json lesen
  2. Zeilen der drei Split-CSVs auf diese Klassen einschraenken
  3. Abgleich mit extraction_stats.json (Key = Dateiname ohne .mp4)
  4. Ausschluss: Extraktionsstatus != ok oder missing_hand_frac > Schwelle
  5. Ausschluss von Klassen mit zu wenigen Trainingssequenzen

Ausgabe: <landmarks_dir>/manifest.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def safe_dirname(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip()).strip("_") or "unknown"


def read_splits(splits_dir: Path) -> list[dict]:
    rows = []
    for split in ("train", "val", "test"):
        with open(splits_dir / f"{split}.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({"participant": r["participant"],
                             "video_file": r["video_file"],
                             "gloss": r["gloss"], "split": split})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--selection", default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    d = cfg["data"]
    landmarks_dir = Path(d["landmarks_dir"])
    max_missing = d["max_missing_hand_frac"]
    min_train = d["min_train_sequences_per_class"]
    splits_dir = Path(a.splits_dir or d["splits_dir"])
    selection = Path(a.selection or d["class_selection"])

    wanted = set(json.loads(selection.read_text())["all"])
    stats = json.loads((landmarks_dir / "extraction_stats.json").read_text())

    items, excluded = [], Counter()
    for r in read_splits(splits_dir):
        if r["gloss"] not in wanted:
            continue
        vid = Path(r["video_file"]).stem
        st = stats.get(vid)
        if st is None:
            excluded["not_extracted"] += 1
            continue
        if st.get("status") != "ok":
            excluded[st.get("status", "unknown")] += 1
            continue
        if st.get("missing_hand_frac", 0) > max_missing:
            excluded["hand_frac_gt_threshold"] += 1
            continue
        items.append({"video_id": vid, "gloss": r["gloss"],
                      "gloss_dir": safe_dirname(r["gloss"]),
                      "split": r["split"], "participant": r["participant"],
                      "left_hand_missing": st.get("left_hand_frac") == 0,
                      "right_hand_missing": st.get("right_hand_frac") == 0})

    train_counts = Counter(i["gloss"] for i in items if i["split"] == "train")
    keep = {g for g, c in train_counts.items() if c >= min_train}
    dropped_classes = sorted({i["gloss"] for i in items} - keep)
    items_before = len(items)
    items = [i for i in items if i["gloss"] in keep]

    classes = sorted(keep)
    label = {g: k for k, g in enumerate(classes)}
    for i in items:
        i["label"] = label[i["gloss"]]

    # Signer-Ueberschneidung zwischen den Splits (erwartet: leer)
    signers = defaultdict(set)
    for i in items:
        signers[i["split"]].add(i["participant"])
    overlap = {f"{a_}/{b_}": sorted(signers[a_] & signers[b_])
               for a_, b_ in (("train", "val"), ("train", "test"),
                              ("val", "test"))}

    # Sequenzen, in denen eine Hand in keinem Frame erkannt wurde; ihre
    # Koordinaten stehen nach der Interpolation auf dem Bildursprung
    left = [i.pop("left_hand_missing") for i in items]
    right = [i.pop("right_hand_missing") for i in items]
    hand_never_detected = {"left": sum(left), "right": sum(right),
                           "left_or_right": sum(l or r for l, r in zip(left, right))}

    splits = Counter(i["split"] for i in items)
    per_class = {s: Counter(i["gloss"] for i in items if i["split"] == s)
                 for s in ("train", "val", "test")}
    manifest = {
        "classes": classes,
        "items": items,
        "excluded": dict(excluded),
        "dropped_by_class_min": items_before - len(items),
        "dropped_classes": dropped_classes,
        "source": "ASL Citizen (Desai et al., 2023), offizielle signer-disjunkte Splits",
        "signers_per_split": {s: len(v) for s, v in signers.items()},
        "signer_overlap": overlap,
        "split_sizes": dict(splits),
        "hand_never_detected": hand_never_detected,
        "min_sequences_per_class": {s: (min(c.values()) if c else 0)
                                    for s, c in per_class.items()},
    }
    out = landmarks_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))

    print(f"Manifest: {len(classes)} Klassen, {len(items)} Sequenzen -> {out}")
    print(f"  Splits: {dict(splits)}")
    print(f"  Signer je Split: {manifest['signers_per_split']}")
    print(f"  Signer-Ueberschneidung: "
          f"{ {k: len(v) for k, v in overlap.items()} }")
    print(f"  Ausgeschlossen: {dict(excluded)}")
    print(f"  Hand in keinem Frame erkannt: {hand_never_detected}")
    if dropped_classes:
        print(f"  Klassen unter min_train={min_train}: {dropped_classes}")


if __name__ == "__main__":
    main()
