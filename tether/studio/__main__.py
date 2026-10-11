"""Tether Studio CLI.

  python -m tether.studio calibrate LOG.csv|LOG.json [--llm tokenfactory|local] [--out DIR]
  python -m tether.studio video CLIP.mp4 --corners x1,y1,x2,y2,x3,y3,x4,y4 [--sheet a4|letter] [--out DIR] [--llm ...]

Writes calibration.json, newton_calibration.py, isaaclab_events.py, report.md (and pushes.csv for videos).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tether.studio import export
from tether.studio.pipeline import analyze
from tether.studio.session import SessionError, load, to_csv


def _llm(name: str | None):
    if not name or name == "none":
        return None
    from tether.agent.llm import make_llm

    return make_llm(name)


def _print_step(e: dict) -> None:
    if e["type"] == "stage":
        print(f"[{e['stage']}]", {k: v for k, v in e.items() if k not in ("type", "stage")} or "")
    elif e["type"] == "agent_step":
        s = e["step"]
        res = s.get("result")
        tail = ""
        if isinstance(res, dict) and "stop_residual_rms_m" in res:
            tail = f" -> stop rms {res['stop_residual_rms_m'] * 100:.1f} cm"
        print(f"  agent: {s.get('tool')} {json.dumps(s.get('args', {}))[:140]}{tail}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tether.studio", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="calibrate from a robot log (CSV/JSON)")
    c.add_argument("log")
    v = sub.add_parser("video", help="track pushes in a phone video, then calibrate")
    v.add_argument("clip")
    v.add_argument("--corners", required=True, help="pixel corners of the sheet: x1,y1,...,x4,y4 (any order)")
    v.add_argument("--sheet", default="a4", choices=["a4", "letter"])
    for p in (c, v):
        p.add_argument("--llm", default=None, help="tokenfactory | local | none (default: offline structure search)")
        p.add_argument("--out", default=None)
        p.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "calibrate":
            path = Path(a.log)
            session = load(path.read_text(), path.name, a.name)
            out = Path(a.out or f"runs/studio/{path.stem}")
        else:
            from tether.studio.video import track_video

            corners = [float(x) for x in a.corners.split(",")]
            session, _ = track_video(Path(a.clip), corners, sheet=a.sheet, name=a.name)
            out = Path(a.out or f"runs/studio/{Path(a.clip).stem}")
    except SessionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    res = analyze(session, _llm(a.llm), on_event=_print_step)
    files = export.write_all(out, session, res.calibration, res.next_experiment, res.agent)
    if a.cmd == "video":
        (out / "pushes.csv").write_text(to_csv(session))
    print()
    print(export.markdown(session, res.calibration, res.next_experiment, res.agent))
    print("wrote:", ", ".join(files.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
