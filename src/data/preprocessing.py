"""Raeumliche Normalisierung, zeitliches Resampling und Geschwindigkeiten.

Ankerpunkt ist der Mittelpunkt der Schulter-Landmarks (MediaPipe-Pose 11/12),
pro Frame subtrahiert; skaliert wird mit der Schulterbreite (Median ueber die
Sequenz).
"""
from __future__ import annotations

import numpy as np

POSE_L_SHOULDER, POSE_R_SHOULDER = 11, 12
EPS = 1e-6


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Anker-Subtraktion und Skalierung auf die Schulterbreite. seq: (T, L, 3)."""
    ls = seq[:, POSE_L_SHOULDER, :]
    rs = seq[:, POSE_R_SHOULDER, :]
    anchor = (ls + rs) / 2.0
    width = np.linalg.norm((ls - rs)[:, :2], axis=1)
    scale = float(np.median(width))
    scale = scale if scale > EPS else 1.0
    out = (seq - anchor[:, None, :]) / scale
    return out.astype(np.float32)


def resample_time(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Lineare Interpolation auf target_len gleichmaessig verteilte Frames."""
    t = seq.shape[0]
    if t == target_len:
        return seq
    idx = np.linspace(0, t - 1, target_len)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (idx - lo)[:, None, None]
    return ((1 - w) * seq[lo] + w * seq[hi]).astype(np.float32)


def add_velocity(seq: np.ndarray) -> np.ndarray:
    """(T, L, 3) -> (T, L, 6) mit [x, y, z, dx, dy, dz].

    dx, dy, dz sind die Differenzen zum Vorframe (erster Frame: 0). Die
    Funktion wird nach der Rotation aufgerufen, damit Lage und Bewegung
    gleich gedreht sind.
    """
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq, vel], axis=-1).astype(np.float32)
