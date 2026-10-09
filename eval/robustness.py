"""Phone-video robustness: the same hidden table filmed clean and filmed badly (hand-held shake, motion blur,
exposure flicker, textured table, heavy compression; studio.video_sample.render(hard=True)), measured end to end
with the automatic sheet detection, the tracker and the offline fit. Reports corner error, launch-speed bias and
whether each hidden value lands inside its 90% interval.

Run: .venv/bin/python -m eval.robustness   (renders two 11-push videos in NVIDIA Newton, ~1-2 min)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from studio.pipeline import analyze
from studio.video import auto_sheet, track_video
from studio.video_sample import MU_EFF, SPEEDS, SPEEDS_2, STRIP, render

TRUTH = {"mu_eff": MU_EFF, "patch_y0": STRIP[0], "patch_mu": STRIP[1]}


def run_condition(name: str, hard: bool, tmp: Path) -> dict:
    path = tmp / f"{name}.mp4"
    meta = render(path, SPEEDS + SPEEDS_2, 5, hard=hard)
    corners = auto_sheet(path)
    true_c = meta["sheet_corners_px"]
    corner_err = None if corners is None else round(float(np.mean([min(np.hypot(a - x, b - y) for x, y in corners) for a, b in true_c])), 2)
    session, info = track_video(path, corners or true_c, object_height_m=0.06)
    starts = sorted(p["start_y_m"] for p in meta["pushes"])
    origin = starts[len(starts) // 2]
    truth = dict(TRUTH, patch_y0=round(TRUTH["patch_y0"] - origin, 4))
    measured = sorted(p.launch_speed for p in session.pushes)
    true_v = sorted(p["launch_speed_mps"] for p in meta["pushes"])
    n = min(len(measured), len(true_v))
    bias = float(np.median([m / t - 1 for m, t in zip(measured[:n], true_v[:n])])) if n else None
    cal = analyze(session, None).calibration
    inside = {k: bool(cal.intervals[k][0] <= truth[k] <= cal.intervals[k][1]) for k in truth if k in cal.intervals}
    return {"condition": name, "corners_auto": corners is not None, "corner_err_px": corner_err, "pushes_found": len(session.pushes),
            "pushes_true": len(meta["pushes"]), "launch_speed_bias": None if bias is None else round(bias, 3),
            "model": {k: cal.model.get(k) for k in truth}, "intervals": {k: [round(x, 3) for x in cal.intervals.get(k, [])] for k in truth},
            "truth": truth, "inside": inside}


def main() -> dict:
    tmp = Path(tempfile.mkdtemp())
    out = [run_condition("clean phone", False, tmp), run_condition("hand-held, blurred, compressed", True, tmp)]
    for r in out:
        print(json.dumps(r))
    return {"conditions": out}


if __name__ == "__main__":
    main()
