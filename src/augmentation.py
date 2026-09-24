"""Roll-Rotation von Landmark-Sequenzen in der Bildebene.

Pro Sequenz und Epoche wird ein Winkel theta ~ U(-A, +A) gezogen und auf alle
Frames angewendet. Rotiert werden x und y um den Ankerpunkt, z bleibt
unveraendert. Der Winkel stammt aus einem eigenen RNG-Stream; augmentiert
werden nur Trainingsdaten (siehe dataset.py).
"""
from __future__ import annotations

import numpy as np


def rotate_sequence(seq: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotiert eine normalisierte Sequenz (T, L, 3) um den Ursprung.

    Gibt eine Kopie zurueck; die z-Koordinate bleibt unveraendert.
    """
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    out = seq.copy()
    x, y = seq[..., 0], seq[..., 1]
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out


def draw_angle(rng: np.random.Generator, max_abs_deg: float) -> float:
    """theta ~ U(-max_abs_deg, +max_abs_deg); 0 fuer die Baseline."""
    if max_abs_deg <= 0:
        return 0.0
    return float(rng.uniform(-max_abs_deg, max_abs_deg))
