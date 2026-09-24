"""Legt die Videos der ausgewaehlten ASL-Citizen-Klassen nach Klassen ab.

ASL Citizen (Desai et al., 2023) wird als ZIP-Archiv ausgeliefert:
    ASL_Citizen/videos/<video_file>.mp4
    ASL_Citizen/splits/{train,val,test}.csv   (participant, video_file, gloss)

Das Skript entpackt die Videos der Klassen aus data/aslc_class_selection.json
nach
    data/videos_aslc/<gloss>/<video_file>.mp4
Bereits vorhandene Dateien werden uebersprungen.

    python -m src.data.prepare_asl_citizen --zip <pfad>/ASL_Citizen.zip
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

from tqdm import tqdm

ZIP_VIDEO_PREFIX = "ASL_Citizen/videos/"


def safe_dirname(s: str) -> str:
    """Ordnername einer Klasse; identisch zu filter_asl_citizen.safe_dirname."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip()).strip("_") or "unknown"


def read_splits(splits_dir: Path) -> list[dict]:
    rows = []
    for split in ("train", "val", "test"):
        with open(splits_dir / f"{split}.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["split"] = split
                rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Pfad zu ASL_Citizen.zip")
    ap.add_argument("--splits-dir", default="data/aslc_splits")
    ap.add_argument("--selection", default="data/aslc_class_selection.json")
    ap.add_argument("--out-dir", default="data/videos_aslc")
    args = ap.parse_args()

    wanted = set(json.loads(Path(args.selection).read_text())["all"])
    print(f"Klassen: {len(wanted)}")

    rows = [r for r in read_splits(Path(args.splits_dir)) if r["gloss"] in wanted]
    print(f"Videos laut Splits: {len(rows)}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    todo, skipped = [], 0
    for r in rows:
        dest = out_root / safe_dirname(r["gloss"]) / r["video_file"]
        if dest.exists() and dest.stat().st_size > 4096:
            skipped += 1
            continue
        todo.append((r, dest))
    print(f"Bereits vorhanden: {skipped}, zu entpacken: {len(todo)}")

    written, missing = 0, []
    with zipfile.ZipFile(args.zip) as z:
        names = set(z.namelist())
        for r, dest in tqdm(todo, desc="Entpacken"):
            member = ZIP_VIDEO_PREFIX + r["video_file"]
            if member not in names:
                missing.append(r["video_file"])
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(dest, "wb") as f:
                while chunk := src.read(1 << 20):
                    f.write(chunk)
            written += 1

    meta = {"zip": str(args.zip), "classes": sorted(wanted),
            "videos_written": written, "videos_skipped": skipped,
            "missing_in_zip": missing}
    (out_root / "_prepare_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Fertig. Geschrieben: {written}, im ZIP nicht gefunden: {len(missing)}")


if __name__ == "__main__":
    main()
