"""Extrahiert MediaPipe-Holistic-Landmarks aus den Gebaerdenvideos.

Erwartet die Videos klassenweise unter <videos-dir>/<gloss>/<video_id>.mp4
(siehe prepare_asl_citizen.py). Pro Frame werden 33 Pose- und 2 x 21
Hand-Landmarks gespeichert. Randframes ohne erkannte Hand werden abgeschnitten,
fehlende Werte innerhalb der Sequenz linear interpoliert. Das Einlesen eines
Videos laeuft mit Timeout, damit defekte Dateien den Lauf nicht blockieren.
Bereits erfolgreich extrahierte Videos werden uebersprungen.

Ausgabe:
    <out-dir>/<gloss>/<video_id>.npz      (key "landmarks", (T, 75, 3))
    <out-dir>/extraction_stats.json       ({video_id: {status, ...}, "_meta": ...})
"""
from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

N_POSE, N_HAND = 33, 21
N_LM = N_POSE + 2 * N_HAND        # 75
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
MIN_FRAMES_AFTER_TRIM = 8
MP_VERSION = getattr(mp, "__version__", "unknown")


def _read_opencv(path_str: str, max_frames: int, out: dict):
    try:
        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            out["result"] = None; return
        frames = []
        while len(frames) < max_frames:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        cap.release()
        out["result"] = frames or None
    except Exception:
        out["result"] = None


def _read_pyav(path_str: str, max_frames: int, out: dict):
    try:
        import av
    except ImportError:
        out["result"] = None; return
    try:
        container = av.open(path_str)
        frames = []
        for f in container.decode(video=0):
            frames.append(f.to_ndarray(format="bgr24"))
            if len(frames) >= max_frames:
                break
        container.close()
        out["result"] = frames or None
    except Exception:
        out["result"] = None


def read_frames_with_timeout(path: Path, max_frames: int, timeout_s: int):
    for name, target in (("opencv", _read_opencv), ("pyav", _read_pyav)):
        out: dict = {"result": None}
        t = threading.Thread(target=target, args=(str(path), max_frames, out),
                             daemon=True)
        t.start(); t.join(timeout=timeout_s)
        if t.is_alive():
            return "__timeout__"
        if out["result"] is not None:
            return out["result"], name
    return None


def landmarks_to_array(res):
    arr = np.full((N_LM, 3), np.nan, dtype=np.float32)
    if res.pose_landmarks:
        for i, lm in enumerate(res.pose_landmarks.landmark):
            arr[i] = (lm.x, lm.y, lm.z)
    lh = res.left_hand_landmarks is not None
    rh = res.right_hand_landmarks is not None
    if lh:
        for i, lm in enumerate(res.left_hand_landmarks.landmark):
            arr[N_POSE + i] = (lm.x, lm.y, lm.z)
    if rh:
        for i, lm in enumerate(res.right_hand_landmarks.landmark):
            arr[N_POSE + N_HAND + i] = (lm.x, lm.y, lm.z)
    return arr, lh, rh


def resize_bgr(img, target: int):
    h, w = img.shape[:2]
    s = min(h, w)
    if s <= target:
        return img
    scale = target / s
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def trim_hand_edges(seq, hand_flags):
    T = seq.shape[0]
    first = next((i for i, h in enumerate(hand_flags) if h), None)
    if first is None:
        return None
    last = T - 1 - next(i for i, h in enumerate(reversed(hand_flags)) if h)
    return seq[first:last + 1].copy(), hand_flags[first:last + 1], first, T - 1 - last


def interpolate_nan(seq):
    T = seq.shape[0]
    t = np.arange(T)
    flat = seq.reshape(T, -1)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        ok = ~np.isnan(col)
        n_ok = int(ok.sum())
        if n_ok == 0:
            flat[:, c] = 0.0
        elif n_ok < T:
            flat[:, c] = np.interp(t, t[ok], col[ok])
    return flat.reshape(seq.shape)


def process_video(path, holistic, resize_target, sample_rate,
                  max_frames, read_timeout_s) -> dict:
    read = read_frames_with_timeout(path, max_frames, read_timeout_s)
    if read == "__timeout__":
        return {"status": "read_timeout"}
    if read is None:
        return {"status": "read_failed"}
    frames_raw, reader = read

    frames_all, lh_flags, rh_flags = [], [], []
    for i, img in enumerate(frames_raw):
        if sample_rate > 1 and (i % sample_rate) != 0:
            continue
        img = resize_bgr(img, resize_target)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = holistic.process(rgb)
        arr, lh, rh = landmarks_to_array(res)
        frames_all.append(arr); lh_flags.append(lh); rh_flags.append(rh)

    if len(frames_all) < MIN_FRAMES_AFTER_TRIM:
        return {"status": "too_short", "n_frames_raw": len(frames_raw), "reader": reader}

    seq = np.stack(frames_all)
    hand_flags = [lh or rh for lh, rh in zip(lh_flags, rh_flags)]
    trimmed = trim_hand_edges(seq, hand_flags)
    if trimmed is None:
        return {"status": "no_hand_detected", "reader": reader}
    seq_t, flags_t, cs, ce = trimmed
    if seq_t.shape[0] < MIN_FRAMES_AFTER_TRIM:
        return {"status": "too_short_after_trim", "reader": reader}

    missing_hand_frac = 1.0 - float(np.mean(flags_t))
    seq_interp = interpolate_nan(seq_t)
    T_out = seq_interp.shape[0]
    return {"status": "ok", "sequence": seq_interp, "reader": reader,
            "n_frames_raw": len(frames_raw), "n_frames": int(T_out),
            "trimmed_start": cs, "trimmed_end": ce,
            "missing_hand_frac": round(missing_hand_frac, 4),
            "left_hand_frac": round(float(np.mean(lh_flags[cs:cs + T_out])), 4),
            "right_hand_frac": round(float(np.mean(rh_flags[cs:cs + T_out])), 4)}


def iter_all_videos(videos_root: Path):
    for p in sorted(videos_root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        try:
            parts = p.parent.relative_to(videos_root).parts
        except ValueError:
            parts = ()
        yield p.stem, (parts[0] if parts else None), p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="data/videos_aslc")
    ap.add_argument("--out-dir", default="data/landmarks_aslc")
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--resize", type=int, default=384)
    ap.add_argument("--sample-rate", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--read-timeout", type=int, default=30)
    ap.add_argument("--no-retry-failed", dest="retry_failed", action="store_false",
                    help="Videos mit Fehlerstatus nicht erneut verarbeiten.")
    args = ap.parse_args()

    videos_root = Path(args.videos_dir)
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    stats_path = out_root / "extraction_stats.json"
    stats: dict = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    all_videos = list(iter_all_videos(videos_root))
    print(f"Gefunden: {len(all_videos)} Videos in {videos_root}")

    to_do, skipped_ok = [], 0
    for vid, gloss, path in all_videos:
        if gloss is None:
            stats[vid] = {"status": "no_gloss_folder"}; continue
        out = out_root / gloss / f"{vid}.npz"
        if out.exists() and stats.get(vid, {}).get("status") == "ok":
            skipped_ok += 1; continue
        if (not args.retry_failed and vid in stats
                and stats[vid].get("status") not in (None, "ok")):
            continue
        to_do.append((vid, gloss, path))

    print(f"Bereits ok: {skipped_ok}, zu verarbeiten: {len(to_do)}")
    print(f"Reader-Timeout: {args.read_timeout}s, max_frames: {args.max_frames}")

    FLUSH_EVERY, processed = 25, 0
    with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=args.model_complexity) as holistic:
        for vid, gloss, path in tqdm(to_do, desc="Extract"):
            out_cls = out_root / gloss; out_cls.mkdir(exist_ok=True)
            out = out_cls / f"{vid}.npz"
            res = process_video(path, holistic, args.resize, args.sample_rate,
                                args.max_frames, args.read_timeout)
            if res.get("status") == "ok":
                seq = res.pop("sequence")
                np.savez_compressed(out, landmarks=seq)
            entry = {k: v for k, v in res.items() if k != "sequence"}
            entry["gloss"] = gloss
            stats[vid] = entry
            processed += 1
            if processed % FLUSH_EVERY == 0:
                stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))

    stats["_meta"] = {"mediapipe_version": MP_VERSION,
                      "model_complexity": args.model_complexity,
                      "resize_shorter_side": args.resize,
                      "sample_rate": args.sample_rate,
                      "max_frames": args.max_frames,
                      "read_timeout_s": args.read_timeout,
                      "trim_edges_without_hand": True,
                      "hand_present_definition": "left OR right hand detected",
                      "min_frames_after_trim": MIN_FRAMES_AFTER_TRIM,
                      "layout": "landmarks_dir/<gloss>/<video_id>.npz"}
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))

    by_status: dict[str, int] = {}
    for k, v in stats.items():
        if k == "_meta":
            continue
        by_status[v.get("status", "unknown")] = by_status.get(v.get("status", "unknown"), 0) + 1
    print(f"Fertig. Status-Verteilung: {by_status}")


if __name__ == "__main__":
    main()
