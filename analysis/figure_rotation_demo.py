"""Abbildung: Wirkung der Rotation auf einen einzelnen Frame.

Zeigt denselben Frame ohne Rotation und mit +5, +10 und +20 Grad. Dargestellt
werden die 56 Landmarks des Modells (Schultern bis Huefte und beide Haende)
nach derselben Transformation wie im Training: normalisieren, rotieren,
zurueck in Bildkoordinaten.

    python -m analysis.figure_rotation_demo --out figures_aslc/fig_rotation_demo.png

Standardmaessig werden nur die Landmarks gezeichnet. --with-frame blendet das
Videobild ein; solche Abbildungen duerfen nach der ASL-Citizen-Lizenz nicht
veroeffentlicht werden.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.augmentation import rotate_sequence
from src.data.preprocessing import POSE_L_SHOULDER, POSE_R_SHOULDER

N_POSE = 33
# Kanten der Pose zwischen den im Modell genutzten Punkten
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (11, 23), (12, 24), (23, 24)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
              (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20),
              (0, 17)]
ANGLES = [0, 5, 10, 20]


def landmarks_from_frame(frame_bgr):
    """MediaPipe Holistic auf einem Einzelframe -> (75,3) bildnormiert."""
    import mediapipe as mp
    arr = np.full((75, 3), np.nan, dtype=np.float32)
    with mp.solutions.holistic.Holistic(static_image_mode=True,
                                        model_complexity=1) as h:
        res = h.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if res.pose_landmarks:
        for i, lm in enumerate(res.pose_landmarks.landmark):
            arr[i] = (lm.x, lm.y, lm.z)
    if res.left_hand_landmarks:
        for i, lm in enumerate(res.left_hand_landmarks.landmark):
            arr[N_POSE + i] = (lm.x, lm.y, lm.z)
    if res.right_hand_landmarks:
        for i, lm in enumerate(res.right_hand_landmarks.landmark):
            arr[N_POSE + 21 + i] = (lm.x, lm.y, lm.z)
    return arr


def to_pixels(norm_lm, anchor, scale, w, h):
    """Normalisierte Modellkoordinaten -> Pixel."""
    xy = norm_lm[:, :2] * scale + anchor[None, :2]
    return np.stack([xy[:, 0] * w, xy[:, 1] * h], axis=1)


def draw_skeleton(ax, px, color, lw, alpha, ms):
    for a, b in POSE_EDGES:
        if not (np.isnan(px[a]).any() or np.isnan(px[b]).any()):
            ax.plot([px[a, 0], px[b, 0]], [px[a, 1], px[b, 1]],
                    color=color, lw=lw, alpha=alpha, solid_capstyle="round")
    for base in (N_POSE, N_POSE + 21):
        for a, b in HAND_EDGES:
            ia, ib = base + a, base + b
            if not (np.isnan(px[ia]).any() or np.isnan(px[ib]).any()):
                ax.plot([px[ia, 0], px[ib, 0]], [px[ia, 1], px[ib, 1]],
                        color=color, lw=lw * 0.7, alpha=alpha,
                        solid_capstyle="round")
    keep = list(range(11, 25)) + list(range(N_POSE, 75))
    pts = px[keep]
    ok = ~np.isnan(pts).any(1)
    ax.plot(pts[ok, 0], pts[ok, 1], "o", color=color, ms=ms, alpha=alpha,
            mec="none")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--stats", default="data/landmarks_aslc/extraction_stats.json")
    ap.add_argument("--video-id", default="04628430080673662-TEACH")
    ap.add_argument("--videos-dir", default="data/videos_aslc")
    ap.add_argument("--out", default="figures_aslc/fig_rotation_demo.png")
    ap.add_argument("--with-frame", action="store_true",
                    help="Videobild als Hintergrund einblenden (nicht zur "
                         "Veroeffentlichung, siehe ASL-Citizen-Lizenz).")
    ap.add_argument("--frame", type=int, default=None,
                    help="Frame-Index. Ohne Angabe die Mitte des Videos "
                         "bzw. des getrimmten Bereichs.")
    a = ap.parse_args()

    if a.video:
        path = Path(a.video)
        if a.frame is not None:
            idx = a.frame
        else:
            cap = cv2.VideoCapture(str(path))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            idx = max(0, n // 2)
    else:
        stats = json.loads(Path(a.stats).read_text())
        entry = stats[a.video_id]
        path = Path(a.videos_dir) / entry["gloss"] / f"{a.video_id}.mp4"
        # Mitte des getrimmten Bereichs
        idx = (a.frame if a.frame is not None
               else int(entry.get("trimmed_start", 0)) + int(entry["n_frames"]) // 2)

    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Frame {idx} aus {path} nicht lesbar.")
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    lm = landmarks_from_frame(frame)
    if np.isnan(lm[[POSE_L_SHOULDER, POSE_R_SHOULDER]]).any():
        raise SystemExit("Schulter-Landmarks fehlen in diesem Frame.")

    # Normalisierung wie in preprocessing.normalize_sequence (Einzelframe)
    anchor = (lm[POSE_L_SHOULDER] + lm[POSE_R_SHOULDER]) / 2.0
    scale = float(np.linalg.norm(
        (lm[POSE_L_SHOULDER] - lm[POSE_R_SHOULDER])[:2]))
    norm = (lm - anchor[None, :]) / scale

    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.3))
    base_px = to_pixels(norm, anchor, scale, w, h)
    # Gemeinsame Achsengrenzen fuer alle vier Panels
    keep = list(range(11, 25)) + list(range(N_POSE, 75))
    allpx = np.stack([to_pixels(rotate_sequence(norm[None], d)[0], anchor,
                                scale, w, h)[keep] for d in ANGLES])
    m = allpx.reshape(-1, 2)
    m = m[~np.isnan(m).any(1)]
    pad = 0.02 * w
    x0, x1 = min(0, m[:, 0].min()) - pad, max(w, m[:, 0].max()) + pad
    y0, y1 = min(0, m[:, 1].min()) - pad, max(h, m[:, 1].max()) + pad
    for ax, deg in zip(axes, ANGLES):
        ax.set_facecolor("#f7f7f7" if a.with_frame else "white")
        if a.with_frame:
            ax.imshow(rgb, alpha=0.45, extent=(0, w, h, 0))
        ax.add_patch(plt.Rectangle((0, 0), w, h, fill=False, ec="0.55",
                                   lw=0.9, ls="--"))
        rot = rotate_sequence(norm[None], deg)[0]
        px = to_pixels(rot, anchor, scale, w, h)
        if deg != 0:
            draw_skeleton(ax, base_px, "0.45", lw=1.3, alpha=0.6, ms=2.2)
        draw_skeleton(ax, px, "#d62728" if deg else "#1f77b4",
                      lw=2.0, alpha=0.95, ms=3.6)
        ax.plot(anchor[0] * w, anchor[1] * h, "+", color="k", ms=11, mew=1.8)
        ax.set_title("ohne Rotation" if deg == 0 else f"+{deg}$^\\circ$",
                     fontsize=12)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_alpha(0.3)
    fig.tight_layout(pad=0.5)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format=out.suffix.lstrip(".") or "pdf", dpi=200,
                bbox_inches="tight")
    print(f"Geschrieben: {out}  (Quelle {path.name}, Frame {idx}, {w}x{h})")


if __name__ == "__main__":
    main()
