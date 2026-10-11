"""Real table, end to end: phone videos of an object flicked across a real surface beside an A4 sheet, calibrated by
the Studio pipeline, against a friction measured independently with a tilt test.

Ground truth without instruments (docs/guide/06): raise one end of the surface under the object and read the angle
with a phone level. theta_k is the angle at which the object, nudged, keeps sliding at a steady speed: tan(theta_k) is
the sliding (kinetic) friction Tether measures. theta_s, where it starts to slide on its own, gives the static friction
tan(theta_s), an upper bound.

Layout (film/real/ is git-ignored: recordings of a room stay local; only the numbers are committed):
  film/real/<clip>.mov|.mp4           one clip per object-surface pair, 6-10 flicks, A4 sheet in view
  film/real/truth.json                {"<clip>": {"tilt_kinetic_deg": 21.5, "tilt_static_deg": 25.0,
                                                   "object_height_cm": 5.0, "corners": [[x, y], ...] (optional)}}

Run: .venv/bin/python -m tether.eval.prove_real      -> runs/proof/real_calibration.json (and a table on stdout)
Pass: Tether's friction within +-0.05 of tan(theta_k) on at least 3 object-surface pairs (docs/plans/2026-10-11-s-level-plan.md).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from tether.paths import REPO

ROOT = REPO
REAL = ROOT / "film" / "real"
OUT = ROOT / "runs" / "proof" / "real_calibration.json"
TOL = 0.05


def one(path: Path, t: dict) -> dict:
    from tether.studio import structure
    from tether.studio.fit import calibrate
    from tether.studio.video import auto_sheet, track_video

    corners = t.get("corners") or auto_sheet(path)
    if corners is None:
        return {"clip": path.stem, "error": "sheet not found: add its four corners to truth.json"}
    session, dbg = track_video(path, corners, t.get("sheet", "a4"), path.stem, None, t.get("object_height_cm", 5.0) / 100)
    cal = calibrate(session)
    mu = cal.model["mu_eff"]
    lo, hi = cal.intervals.get("mu_eff", (mu, mu))
    mu_k = math.tan(math.radians(t["tilt_kinetic_deg"])) if t.get("tilt_kinetic_deg") is not None else None
    mu_s = math.tan(math.radians(t["tilt_static_deg"])) if t.get("tilt_static_deg") is not None else None
    return {"clip": path.stem, "pushes": len(session.pushes), "fps": dbg.get("fps"), "camera_height_m": dbg["camera"].get("height_m"),
            "structure": cal.structure, "mu": round(mu, 4), "mu_interval": [round(lo, 4), round(hi, 4)],
            "region": {"from_m": cal.model.get("patch_y0"), "mu": cal.model.get("patch_mu")} if cal.model.get("patch_y0") else None,
            "stop_residual_rms_m": cal.residuals["stop_residual_rms_m"],
            "left_unexplained": structure.describe(cal.residuals.get("patterns") or {"findings": [], "note": "not checked"}),
            "mu_tilt_kinetic": None if mu_k is None else round(mu_k, 4), "mu_tilt_static": None if mu_s is None else round(mu_s, 4),
            "error_vs_tilt": None if mu_k is None else round(mu - mu_k, 4),
            "within_tol": None if mu_k is None else abs(mu - mu_k) <= TOL}


def main(real: Path = REAL) -> dict:
    truth_path = real / "truth.json"
    if not truth_path.exists():
        sys.exit(f"no {truth_path}: film the pairs and note the tilt angles first (docs/guide/06)")
    truth = json.loads(truth_path.read_text())
    rows = []
    for name, t in truth.items():
        path = next((p for ext in (".mov", ".mp4", ".MOV", ".MP4") if (p := real / f"{name}{ext}").exists()), None)
        if path is None:
            rows.append({"clip": name, "error": "video not found"})
            continue
        try:
            rows.append(one(path, t))
        except Exception as e:  # noqa: BLE001 - one bad clip must not hide the others
            rows.append({"clip": name, "error": f"{type(e).__name__}: {e}"})
    scored = [r for r in rows if r.get("within_tol") is not None]
    out = {"pairs": len(rows), "scored": len(scored), "within_0_05": sum(r["within_tol"] for r in scored),
           "mae_vs_tilt": round(sum(abs(r["error_vs_tilt"]) for r in scored) / len(scored), 4) if scored else None,
           "pass": len(scored) >= 3 and all(r["within_tol"] for r in scored), "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print(f"{'clip':24s} {'pushes':>6s} {'Tether mu (90%)':>22s} {'tilt mu_k':>9s} {'error':>7s}")
    for r in rows:
        if r.get("error"):
            print(f"{r['clip']:24s} {r['error']}")
            continue
        iv = r["mu_interval"]
        print(f"{r['clip']:24s} {r['pushes']:6d} {r['mu']:.3f} ({iv[0]:.3f}-{iv[1]:.3f}) {r['mu_tilt_kinetic'] or float('nan'):9.3f} "
              f"{r['error_vs_tilt'] if r['error_vs_tilt'] is not None else float('nan'):+7.3f}")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REAL)
