"""Real-video check of Tether's core measurement: deceleration of a sliding object -> friction.

Data: the real split of the IDPP friction benchmark (Ma et al., "Inferring Dynamic Physical Properties from Video
Foundation Models", 2025; Hugging Face Oliver-Ma/IDPP-Data, Apache-2.0): iPhone slow-motion videos of an object
sliding down a short ramp and then across a flat surface until it stops; ground-truth friction for every object-surface
pair from a spring dynamometer (the archive's json calls the field gt_restitution; the values are the friction
coefficients the paper reports, 0.105-0.544).

The videos carry no metric reference and do not state the slow-motion factor, so absolute friction is out of reach:
deceleration is measured in object sizes per (video second)^2, and one constant per object and camera setup (the
object's size and the camera's time base and angle; the clips are named c1/c2 and their constants differ by ~1.5x)
maps it to friction. That constant is fitted on the *other* surfaces
(leave one surface out), then the held-out surface's friction is predicted and compared with the dynamometer.
Also reported: how well constant deceleration (Coulomb sliding, Tether's model) explains each slide (R^2).

Result (2026-10-10, 52 clips): 45 tracked; constant deceleration explains the slides with median R^2 0.9985; predicted
friction correlates with the dynamometer at r 0.70 but its error (0.068) is not better than guessing the mean of the
other surfaces (0.058). The clips confirm the model and the tracker on real footage; pinning the value down needs
what Tether asks for and these clips lack, a reference of known size and a known frame rate.

Which friction law? (scale-free, by BIC on each tracked slide): constant deceleration (dry Coulomb sliding, Tether's
model) is best on 36 of 45 slides; deceleration proportional to speed (viscous) is decisively worse on 35 (median
delta BIC 22.8); a mixed law is decisively better on 5, with a median viscous share of 1% at launch. Real sliding on
these surfaces is Coulomb, which is the model Tether fits and the pattern check tests against.

Run: .venv/bin/python -m eval.real_friction --download   (fetches the two real-split zips, ~310 MB, to runs/real_data)
     .venv/bin/python -m eval.real_friction <unzipped-dir> ...                    -> runs/proof/real_friction.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from studio.video import track

ROOT = Path(__file__).resolve().parent.parent
W = 960


def read(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.resize(f, (W, int(f.shape[0] * W / f.shape[1]))))
    return out


def rest_blob(frames: list[np.ndarray]) -> tuple[float, float, float] | None:
    """Where the object rests at the end: the blob that differs between the first and last frame, nearest the bottom
    of its travel (the start blob sits on the ramp). Returns (x, y, size_px)."""
    d = np.abs(frames[-1].astype(np.int16) - frames[0].astype(np.int16)).max(axis=2).astype(np.uint8)
    m = cv2.morphologyEx((cv2.GaussianBlur(d, (5, 5), 0) > 40).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, c = cv2.connectedComponentsWithStats(m)
    blobs = [(c[i][0], c[i][1], float(np.sqrt(st[i, cv2.CC_STAT_AREA]))) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 300]
    if len(blobs) < 1:
        return None
    big = sorted(blobs, key=lambda b: -b[2])[:2]
    # the resting object is the one the last frame shows: its patch matches the last frame, not the first
    def score(b):
        x, y, s = b
        r = int(s / 2)
        y0, y1, x0, x1 = max(0, int(y) - r), int(y) + r, max(0, int(x) - r), int(x) + r
        return float(np.abs(frames[-1][y0:y1, x0:x1].astype(np.int16) - np.median(np.stack(frames[::10]), axis=0)[y0:y1, x0:x1]).mean())
    return max(big, key=score)


def laws(s: np.ndarray) -> dict:
    """Which friction law explains one slide best, by BIC (n log(RSS/n) + k log n), on distance-to-rest s(t) in frames:
    coulomb  constant deceleration (dry sliding, Tether's model)                  s = s0 - v0 t + a t^2 / 2
    viscous  deceleration proportional to speed (lubricated or drag-dominated)    s = c + b exp(-t / tau)
    mixed    both                                     dv/dt = -a - v / tau, integrated in closed form
    Scale-free: no size reference or frame rate is needed to compare shapes."""
    from scipy.optimize import curve_fit

    t = np.arange(len(s), dtype=float)
    n = len(s)

    def bic(pred, k):
        rss = max(float(np.sum((s - pred) ** 2)), 1e-12)
        return n * np.log(rss / n) + k * np.log(n)

    out = {}
    A = np.stack([np.ones_like(t), t, 0.5 * t * t], axis=1)
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    out["coulomb"] = bic(A @ coef, 3)
    v0 = max(1e-3, -coef[1])

    def visc(t, c, b, tau):
        return c + b * np.exp(-t / tau)

    def mixed(t, s0, v0, a, tau):
        return s0 - ((v0 + a * tau) * tau * (1 - np.exp(-t / tau)) - a * tau * t)

    try:
        pv, _ = curve_fit(visc, t, s, p0=(s[-1], s[0] - s[-1], max(2.0, n / 3)), bounds=([-np.inf, 0, 0.3], [np.inf, np.inf, 1e4]), maxfev=20000)
        out["viscous"] = bic(visc(t, *pv), 3)
    except Exception:  # noqa: BLE001
        out["viscous"] = float("inf")
    try:
        pm, _ = curve_fit(mixed, t, s, p0=(s[0], v0, max(1e-4, coef[2]), 50.0), bounds=([-np.inf, 0, 0, 0.3], [np.inf, np.inf, np.inf, 1e5]), maxfev=20000)
        out["mixed"] = bic(mixed(t, *pm), 4)
        out["mixed_viscous_share"] = round(float((pm[1] / pm[3]) / (pm[2] + pm[1] / pm[3] + 1e-12)), 3)  # at launch
    except Exception:  # noqa: BLE001
        out["mixed"] = float("inf")
    out["best"] = min(("coulomb", "viscous", "mixed"), key=lambda k: out[k])
    return {k: (round(float(v), 2) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def decel(path: Path) -> dict | None:
    frames = read(path)
    rb = rest_blob(frames)
    if rb is None:
        return None
    x, y, size = rb
    rev = frames[::-1]
    uv, _ = track(rev, (x, y), int(np.clip(size * 1.1, 20, 220)))
    uv = np.asarray(uv)[::-1]  # forward time
    # tracking glitches (the object entering, a hand) are single-frame jumps: a 5-frame running median removes them
    pad = np.pad(uv, ((2, 2), (0, 0)), mode="edge")
    uv = np.stack([np.median(pad[i:i + 5], axis=0) for i in range(len(uv))])
    disp = uv - uv[-1]
    speed = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    moving = np.flatnonzero(speed > 0.6)
    if len(moving) < 8:
        return None
    t_stop = int(moving[-1]) + 1
    # decelerating stretch: from the speed peak (bottom of the ramp) to the stop
    sm = np.convolve(speed, np.ones(3) / 3, mode="same")
    t_peak = int(np.argmax(sm[:t_stop]))
    a0, a1 = t_peak + 2, t_stop
    if a1 - a0 < 6:
        return None
    axis = disp[a0] / (np.linalg.norm(disp[a0]) + 1e-9)
    s = (disp[a0:a1 + 1] @ axis) / size  # distance still to go, in object sizes
    t = np.arange(len(s), dtype=float)
    # constant deceleration: s(t) = s0 - v0 t + a t^2 / 2 (distance to the rest point shrinks)
    A = np.stack([np.ones_like(t), t, 0.5 * t * t], axis=1)
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    pred = A @ coef
    r2 = 1 - float(np.sum((s - pred) ** 2) / max(np.sum((s - s.mean()) ** 2), 1e-12))
    return {"decel": float(coef[2]) * 900.0, "r2": round(r2, 4), "frames": int(a1 - a0), "slide_sizes": round(float(s[0]), 2),
            "laws": laws(s)}


def law_summary(used: list[dict]) -> dict:
    """Coulomb vs viscous vs mixed over the tracked slides: wins by BIC, and how decisive (delta BIC > 6 is strong)."""
    rs = [r["laws"] for r in used if "laws" in r]
    if not rs:
        return {}
    d_visc = [r["viscous"] - r["coulomb"] for r in rs if np.isfinite(r["viscous"])]
    d_mix = [r["coulomb"] - r["mixed"] for r in rs if np.isfinite(r["mixed"])]
    return {"slides": len(rs), "best": {k: int(sum(r["best"] == k for r in rs)) for k in ("coulomb", "viscous", "mixed")},
            "coulomb_beats_viscous_strongly": int(sum(d > 6 for d in d_visc)), "median_bic_viscous_minus_coulomb": round(float(np.median(d_visc)), 1),
            "mixed_beats_coulomb_strongly": int(sum(d > 6 for d in d_mix)),
            "median_viscous_share_at_launch": round(float(np.median([r.get("mixed_viscous_share", 0) for r in rs])), 3)}


def main(dirs: list[str]) -> dict:
    gt = {}
    vids = []
    for d in dirs:
        for j in Path(d).rglob("bounce_analysis_results.json"):
            gt.update({k: v["gt_restitution"] for k, v in json.loads(j.read_text()).items()})
        vids += sorted(Path(d).rglob("*_rgb.mp4"))
    rows = []
    for v in vids:
        key = v.name.replace("_rgb.mp4", "")
        obj, surface, cam = key.split("_")[0], key.split("_")[1], key.split("_")[2]
        if key not in gt:
            continue
        r = decel(v)
        if r is None or r["decel"] <= 0 or r["r2"] < 0.9:
            rows.append({"clip": key, "object": obj, "camera": cam, "surface": surface, "mu_true": gt[key], "skipped": True, **(r or {})})
            continue
        rows.append({"clip": key, "object": obj, "camera": cam, "surface": surface, "mu_true": gt[key], "skipped": False, **r})
        print(key, gt[key], round(r["decel"], 3), r["r2"], flush=True)
    used = [r for r in rows if not r["skipped"]]
    # per object, camera and surface: median deceleration over clips
    by = defaultdict(list)
    for r in used:
        by[(r["object"], r["camera"], r["surface"])].append(r)
    pairs = [{"object": o, "camera": c, "surface": s, "mu_true": rs[0]["mu_true"], "decel": float(np.median([r["decel"] for r in rs])), "clips": len(rs)}
             for (o, c, s), rs in by.items()]
    # leave one surface out: k = sum(mu*a)/sum(a^2) on the other surfaces seen by the same object and camera
    for p in pairs:
        others = [q for q in pairs if q["object"] == p["object"] and q["camera"] == p["camera"] and q is not p]
        if not others:
            p["mu_pred"] = None
            continue
        k = sum(q["mu_true"] * q["decel"] for q in others) / sum(q["decel"] ** 2 for q in others)
        p["mu_pred"] = round(k * p["decel"], 4)
    scored = [p for p in pairs if p["mu_pred"] is not None]
    err = [abs(p["mu_pred"] - p["mu_true"]) for p in scored]
    rel = [abs(p["mu_pred"] - p["mu_true"]) / p["mu_true"] for p in scored]
    mt, mp = np.array([p["mu_true"] for p in scored]), np.array([p["mu_pred"] for p in scored])
    base = [abs(np.mean([q["mu_true"] for q in pairs if q["object"] == p["object"] and q["camera"] == p["camera"] and q is not p]) - p["mu_true"])
            for p in scored]  # no measurement: guess the mean friction of the other surfaces
    out = {"source": "IDPP friction real split (Oliver-Ma/IDPP-Data, Apache-2.0)", "clips_total": len(rows), "clips_used": len(used),
           "median_r2_constant_decel": round(float(np.median([r["r2"] for r in used])), 4) if used else None,
           "pairs": pairs, "pairs_scored": len(scored),
           "mae": round(float(np.mean(err)), 4) if err else None, "median_rel_err": round(float(np.median(rel)), 3) if rel else None,
           "pearson_r": round(float(np.corrcoef(mt, mp)[0, 1]), 3) if len(scored) > 2 else None,
           "baseline_mae_no_measurement": round(float(np.mean(base)), 4) if base else None,
           "friction_law": law_summary(used), "rows": rows}
    (ROOT / "runs" / "proof").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "proof" / "real_friction.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "pairs")}, indent=1))
    for p in sorted(scored, key=lambda p: (p["object"], p["camera"], p["mu_true"])):
        print(f"  {p['object']:11s} {p['camera']} {p['surface']:18s} true {p['mu_true']:.3f}  pred {p['mu_pred']:.3f}  ({p['clips']} clips)")
    return out


HF = "https://huggingface.co/datasets/Oliver-Ma/IDPP-Data/resolve/main/video_zips/friction_absolute/"
ZIPS = ("7.18_friction_test3_rgb_seg_videos_152frames_abs_train.zip", "7.18_friction_test3_rgb_seg_videos_152frames_abs_test.zip")


def download() -> list[str]:
    import subprocess
    import zipfile

    root = ROOT / "runs" / "real_data"
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for z in ZIPS:
        dst = root / z
        if not dst.exists():
            subprocess.run(["curl", "-sL", "-o", str(dst), HF + z], check=True)
        d = root / z.replace(".zip", "")
        if not d.exists():
            with zipfile.ZipFile(dst) as f:
                f.extractall(d)
        out.append(str(d))
    return out


if __name__ == "__main__":
    main(download() if sys.argv[1:] == ["--download"] else sys.argv[1:])
