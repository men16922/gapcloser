"""Real objects, real video, independently measured friction: Tether on the EV-RealPhys benchmark.

Data: EV-RealPhys (Kandukuri, Strecke, Stueckler, "Physics-Based Rigid Body Object Tracking and Friction Filtering
From RGB-D Videos", arXiv 2309.15703; MPI, CC BY-SA 4.0): YCB objects pushed by hand across a table and left to
slide to rest, filmed by an Intel RealSense D455 (848x480, 30 Hz after downsampling) with motion-capture poses and
the camera calibrated to the table. The friction of each object on the table was measured separately, by tilting the
table and timing ten slides per object (the paper's Table 5): cracker box 0.280, mustard bottle 0.159, pitcher 0.220,
bleach cleanser 0.169. Those values never enter Tether.

Two ways in, both ending in Tether's own fitter (offline structure search + bootstrap 90% intervals):

  video   Tether's tracker on the RGB frames (the object found as what moved, tracked back from rest), each pixel
          put on the table with the dataset's camera calibration: the role the A4 sheet plays in the Studio. The
          object's centre height (the number the Studio asks for) comes from where the object rests.
  log     the motion-capture centre of mass, as a robot would log it.

Each push is cut at release (peak speed, the hand lets go) and kept to rest; pushes of one object form one session.
Also reported: the paper's own estimator on the same real sequences (mean error 0.082, median 0.025).

Run: .venv/bin/python -m tether.eval.real_benchmark --download   (1.7 GB to runs/real_data/ev-realphys, git-ignored)
     .venv/bin/python -m tether.eval.real_benchmark              -> runs/proof/real_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from tether.paths import REPO

ROOT = REPO
DATA = ROOT / "runs" / "real_data" / "ev-realphys"
OUT = ROOT / "runs" / "proof" / "real_benchmark.json"
URL = "https://keeper.mpdl.mpg.de/d/5ec213b655b44e40a382/files/?p=/ev-realphys.tar.gz&dl=1"
FPS = 30.0
GT = {"2": ("cracker box", 0.280), "5": ("mustard bottle", 0.159), "11": ("pitcher", 0.220), "12": ("bleach cleanser", 0.169),
      # not in the paper's table: the dataset's physics_properties.json stores 5x the tilt-test value for the four above
      # (1.4, 0.794, 1.102, 0.846) and 0.55 for the mug, i.e. 0.110 on the same scale
      "14": ("mug", 0.110)}
PAPER = {"mean_abs_error": 0.082, "median_abs_error": 0.0246}


def sequences() -> list[Path]:
    base = DATA / "ev-realphys"
    return sorted(p for split in ("val_sliding", "test_sliding") for p in (base / split).glob("*") if (p / "scene_gt.json").exists())


def load(seq: Path) -> dict:
    gt = json.loads((seq / "scene_gt.json").read_text())
    cam = json.loads((seq / "scene_camera.json").read_text())
    props = json.loads((seq / "scene_properties.json").read_text())["object_properties"]
    obj = next(iter(props))
    n = len(gt)
    com = np.array([gt[str(i)][0].get("cam_t_inertial_m2w", gt[str(i)][0]["cam_t_m2w"]) for i in range(n)]) / 1000.0
    return {"seq": seq, "obj": obj, "n": n, "com": com, "cam": [cam[str(i)] for i in range(n)]}


def push_from(xy: np.ndarray):
    """One push the way the Studio measures video: positions along the motion, segmented and launched by
    studio.video (release at peak speed, launch speed from a constant-deceleration fit to the next frames)."""
    from tether.studio.session import Push
    from tether.studio.video import launch_state, segment

    d = xy[-1] - xy[0]
    axis = d / (np.linalg.norm(d) + 1e-12)
    s = (xy - xy[0]) @ axis
    segs = segment(s, FPS)
    if not segs:
        return None, None
    r, e = max(segs, key=lambda q: s[q[1]] - s[q[0]])  # the push itself, not a wobble at rest
    rest = float(np.median(s[e:min(len(s), e + 5)]))
    s0, v0 = launch_state(s, r, e, FPS)
    track = [0.0] + [float(x - s0) for x in s[r + 1:e + 1]] + [rest - s0] * 3
    return Push(stop=round(rest - s0, 4), launch_speed=round(float(v0), 4), track=[round(x, 4) for x in track], origin="video"), (r, e)


def constant_decel_mu(track: list[float], stop: float) -> float | None:
    """Per push, without Tether: fit s(t) = v0 t - a t^2 / 2 until rest, mu = a / g."""
    s = np.asarray(track)
    k = int(np.argmax(s >= stop - 1e-3)) if stop > 0 else 0
    if k < 4:
        return None
    t = np.arange(k + 1) / FPS
    A = np.stack([t, -0.5 * t * t], axis=1)
    v0, a = np.linalg.lstsq(A, s[: k + 1], rcond=None)[0]
    return float(a / 9.81)


def video_xy(d: dict) -> tuple[np.ndarray | None, str]:
    """Tether's tracker on the RGB frames, pixels put on the table plane (z = object centre height) with the dataset's
    camera calibration."""
    import cv2

    from tether.studio.video import track

    rgb = d["seq"] / "rgb"
    files = sorted(rgb.glob("*.png")) or sorted(rgb.glob("*.jpg"))
    if not files:
        return None, "no rgb frames"
    frames = [cv2.imread(str(f)) for f in files][: d["n"]]
    # the click: where the object rests (the Studio asks for a click when it cannot find the object itself; these
    # clips start mid-push with the hand and a ruler in view). The motion-capture rest position, projected into
    # the image, stands in for that one click; every other position comes from the RGB tracker.
    c = d["cam"][-1]
    K, R, t = np.asarray(c["cam_K"]).reshape(3, 3), np.asarray(c["cam_R_w2c"]).reshape(3, 3), np.asarray(c["cam_t_w2c"]) / 1000.0
    proj = lambda w: (K @ (R @ w + t))[:2] / (K @ (R @ w + t))[2]  # noqa: E731
    rest = d["com"][-1]
    x, y = proj(rest)
    size = float(np.linalg.norm(proj(rest + [0.05, 0, 0]) - proj(rest - [0.05, 0, 0])))  # ~10 cm footprint
    uv, _ = track(frames[::-1], (x, y), int(np.clip(size * 1.1, 20, 220)))
    uv = np.asarray(uv)[::-1]
    pad = np.pad(uv, ((2, 2), (0, 0)), mode="edge")
    uv = np.stack([np.median(pad[i:i + 5], axis=0) for i in range(len(uv))])
    h = float(np.median(d["com"][-5:, 2]))  # centre height above the table: the Studio's "object height" input
    out = []
    for (u, v), c in zip(uv, d["cam"]):
        K = np.asarray(c["cam_K"]).reshape(3, 3)
        R = np.asarray(c["cam_R_w2c"]).reshape(3, 3)
        t = np.asarray(c["cam_t_w2c"]) / 1000.0
        C = -R.T @ t  # camera centre in the table frame
        ray = R.T @ np.linalg.solve(K, [u, v, 1.0])
        lam = (h - C[2]) / ray[2]
        out.append((C + lam * ray)[:2])
    return np.asarray(out), "ok"


def calibrate_pushes(name: str, pushes: list) -> dict:
    from tether.studio.fit import calibrate
    from tether.studio.session import Session

    s = Session(name, "video", pushes)
    cal = calibrate(s)
    lo, hi = cal.intervals.get("mu_eff", (cal.model["mu_eff"],) * 2)
    return {"mu": round(cal.model["mu_eff"], 4), "interval": [round(lo, 4), round(hi, 4)], "structure": cal.structure,
            "stop_rms_m": cal.residuals["stop_residual_rms_m"]}


def main(with_video: bool = True) -> dict:
    seqs = sequences()
    if not seqs:
        sys.exit("EV-RealPhys not found: run with --download first")
    by_obj: dict[str, dict] = {}
    rows = []
    for seq in seqs:
        d = load(seq)
        if d["obj"] not in GT:
            continue
        row = {"sequence": f"{seq.parent.name}/{seq.name}", "object": GT[d["obj"]][0]}
        p_log, seg = push_from(d["com"][:, :2])
        by_obj.setdefault(d["obj"], {"log": [], "video": []})
        if p_log is None:
            row["log"] = "no push found"
        else:
            row.update(release_frame=seg[0], rest_frame=seg[1], launch_mps=p_log.launch_speed, slide_m=p_log.stop,
                       mu_per_push_log=constant_decel_mu(p_log.track, p_log.stop))
            by_obj[d["obj"]]["log"].append(p_log)
        if with_video:
            xy, status = video_xy(d)
            row["video"] = status
            if xy is not None:
                from tether.studio.video import implausible

                p_vid, _ = push_from(xy)
                why = implausible(p_vid.track, FPS, p_vid.launch_speed) if p_vid is not None else None
                if p_vid is None or why:
                    row["video"] = "no push found" if p_vid is None else f"left out: track {why}"
                else:
                    row.update(slide_m_video=p_vid.stop, launch_mps_video=p_vid.launch_speed,
                               mu_per_push_video=constant_decel_mu(p_vid.track, p_vid.stop))
                    if p_log is not None:
                        row["video_vs_mocap_stop_m"] = round(abs(p_vid.stop - p_log.stop), 4)
                    by_obj[d["obj"]]["video"].append(p_vid)
        rows.append(row)
        print(row, flush=True)
    objects = []
    for oid, ps in by_obj.items():
        name, mu_true = GT[oid]
        o = {"object": name, "mu_tilt_test": mu_true, "pushes": {w: len(ps[w]) for w in ps}}
        for way in ("log", "video"):
            if ps[way]:
                r = calibrate_pushes(f"{name} ({way})", ps[way])
                r["error"] = round(r["mu"] - mu_true, 4)
                r["inside_interval"] = r["interval"][0] <= mu_true <= r["interval"][1]
                o[way] = r
        objects.append(o)
    summary = {}
    for way in ("log", "video"):
        errs = [abs(o[way]["error"]) for o in objects if way in o]
        if errs:
            summary[way] = {"objects": len(errs), "mean_abs_error": round(float(np.mean(errs)), 4),
                            "median_abs_error": round(float(np.median(errs)), 4), "within_0_05": int(sum(e <= 0.05 for e in errs))}
    out = {"source": "EV-RealPhys (Kandukuri et al. 2023, MPI, CC BY-SA 4.0), real split", "fps": FPS,
           "ground_truth": "tilted-table slides, measured separately (paper Table 5)", "paper_estimator_on_real": PAPER,
           "summary": summary, "objects": objects, "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("summary", "objects")}, indent=1, default=str))
    return out


def download() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if (DATA / "ev-realphys").exists():
        return
    tar = DATA / "ev-realphys.tar.gz"
    if not tar.exists():
        subprocess.run(["curl", "-L", "-o", str(tar), URL], check=True)
    subprocess.run(["tar", "-xzf", str(tar), "-C", str(DATA)], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    if a.download:
        download()
    main(with_video=not a.no_video)
    _ = math
