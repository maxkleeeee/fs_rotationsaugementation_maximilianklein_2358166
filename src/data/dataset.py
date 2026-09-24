"""PyTorch-Dataset fuer Landmark-Sequenzen.

Alle Sequenzen eines Splits werden beim Initialisieren geladen, normalisiert
und auf seq_len Frames resampelt. In __getitem__ folgen dann (nur im
Training) die Rotation und das Anhaengen der Geschwindigkeiten.

Der Rotationswinkel wird pro __getitem__ aus aug_rng gezogen. Das Dataset
muss daher im Hauptprozess laufen (num_workers=0), sonst erhielte jeder
Worker eine Kopie des RNG-Zustands.

Resampling vor der Rotation ist gleichwertig zur umgekehrten Reihenfolge,
da beide Operationen linear sind und die Rotation fuer alle Frames gleich ist.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.augmentation import draw_angle, rotate_sequence
from src.data.preprocessing import add_velocity, normalize_sequence, resample_time


class LandmarkSequenceDataset(Dataset):
    def __init__(self, manifest_path, landmarks_dir, split, seq_len,
                 augment=False, max_rotation_deg=0.0, aug_rng=None,
                 landmark_subset=None):
        manifest = json.loads(Path(manifest_path).read_text())
        self.items = [it for it in manifest["items"] if it["split"] == split]
        self.classes = manifest["classes"]
        self.landmarks_dir = Path(landmarks_dir)
        self.seq_len = seq_len
        # Auswahl nach der Normalisierung, damit der Ankerpunkt (Pose 11/12)
        # unabhaengig von der Auswahl bleibt
        self.landmark_subset = (None if landmark_subset is None
                                else np.asarray(landmark_subset, dtype=int))
        self.augment = augment
        self.max_rotation_deg = max_rotation_deg
        self.aug_rng = aug_rng
        if augment and max_rotation_deg > 0 and aug_rng is None:
            raise ValueError("Augmentation aktiv, aber kein aug_rng uebergeben.")

        self._cache = [self._load_prepared(it) for it in self.items]

    def _load_prepared(self, item: dict) -> np.ndarray:
        path = self.landmarks_dir / item["gloss_dir"] / f"{item['video_id']}.npz"
        arr = np.load(path)["landmarks"]
        arr = resample_time(normalize_sequence(arr), self.seq_len)
        if self.landmark_subset is not None:
            arr = np.ascontiguousarray(arr[:, self.landmark_subset, :])
        return arr

    @property
    def n_landmarks(self) -> int:
        return (75 if self.landmark_subset is None
                else int(len(self.landmark_subset)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        seq = self._cache[i]
        if self.augment and self.max_rotation_deg > 0:
            angle = draw_angle(self.aug_rng, self.max_rotation_deg)
            seq = rotate_sequence(seq, angle)
        else:
            seq = seq.copy()
        seq = add_velocity(seq)
        x = torch.from_numpy(seq.reshape(self.seq_len, -1))
        y = torch.tensor(self.items[i]["label"], dtype=torch.long)
        return x, y
