"""A Studio session: the user's own measurements of a push task, ready for calibration.

Sources:
  robot log  CSV or JSON, one row per push: command (or launch_speed), stop, optional target, perceived,
             tipped, track (cube position along the push axis per frame, any fps)
  video      phone video tracked by studio.video (launch speed and track measured, no command)

Everything downstream works on `Session.rollout()`, the same Rollout/Trial contract the simulator uses,
so the agent's tools and the fitter treat real data and simulated data identically.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict, dataclass, field

from sim.params import PARAM_SPACE, ParamSet
from sim.push_task import FRAME_DT, Rollout, Trial

MAX_TRIALS = 200
MAX_TRACK = 400  # frames after resampling to 30 fps (13 s)


class SessionError(ValueError):
    """Input problem, phrased for the person who uploaded the file."""


@dataclass
class Push:
    stop: float  # m along the push axis, from the launch point
    command: float | None = None  # robot command (velocity units); None for hand/phone pushes
    launch_speed: float | None = None  # m/s, measured (video) or None
    target: float | None = None  # true target distance (m), if the robot was aiming
    perceived: float | None = None  # target distance as the robot's camera saw it
    tipped: bool = False
    track: list[float] = field(default_factory=list)  # 30 fps, starting at launch
    origin: str = "log"  # log | video | suggested
    start: float = 0.0  # launch position along the push axis relative to the session origin (video)

    def to_trial(self) -> Trial:
        nan = float("nan")
        cmd = self.command if self.command is not None else self.launch_speed
        return Trial(self.target if self.target is not None else nan, self.perceived if self.perceived is not None else nan,
                     float(cmd), self.stop, list(self.track), self.tipped, self.start, self.command is None)


@dataclass
class Session:
    name: str
    source: str  # log | video | sample
    pushes: list[Push]
    sim: ParamSet = field(default_factory=ParamSet.nominal)  # the user's current simulator
    notes: list[str] = field(default_factory=list)  # warnings from parsing, shown to the user
    meta: dict = field(default_factory=dict)

    @property
    def has_commands(self) -> bool:
        return all(p.command is not None for p in self.pushes)

    @property
    def has_targets(self) -> bool:
        return any(p.target is not None and p.perceived is not None for p in self.pushes)

    @property
    def has_tracks(self) -> bool:
        return any(len(p.track) > 2 for p in self.pushes)

    def rollout(self) -> Rollout:
        return Rollout([p.to_trial() for p in self.pushes])

    def coverage(self) -> tuple[float, float]:
        stops = [p.start + p.stop for p in self.pushes if not p.tipped]  # table coordinates
        return (min(stops), max(stops)) if stops else (0.0, 0.0)

    def to_json(self) -> dict:
        return {"name": self.name, "source": self.source, "notes": self.notes, "meta": self.meta,
                "sim": {k: self.sim[k] for k in PARAM_SPACE}, "pushes": [asdict(p) for p in self.pushes]}

    @classmethod
    def from_json(cls, d: dict) -> Session:
        if not isinstance(d, dict):
            raise SessionError("expected a JSON object with a 'pushes' (or 'trials') list")
        rows = d.get("pushes", d.get("trials"))
        if not isinstance(rows, list):
            raise SessionError("expected a JSON object with a 'pushes' (or 'trials') list")
        fps = _num(d.get("fps"), "fps") or 30.0
        pushes, notes = _parse_rows(rows, fps)
        sim = ParamSet.nominal()
        if isinstance(d.get("sim"), dict):
            known = {k: float(v) for k, v in d["sim"].items() if k in PARAM_SPACE and isinstance(v, (int, float))}
            sim = sim.with_(**known)
        return cls(str(d.get("name") or "uploaded log"), str(d.get("source") or "log"), pushes, sim,
                   notes + list(d.get("notes") or []), dict(d.get("meta") or {}))


ALIASES = {
    "stop": ("stop", "stop_m", "slide", "slide_m", "final", "final_m", "real_stop_m"),
    "command": ("command", "cmd", "u", "action"),
    "launch_speed": ("launch_speed", "launch_speed_mps", "v0"),
    "target": ("target", "target_m", "goal"),
    "perceived": ("perceived", "perceived_m", "observed", "observed_m"),
    "tipped": ("tipped", "tipped_over"),
    "track": ("track", "track_m", "positions"),
    "start": ("start", "start_m", "launch_position"),
}


def _num(v, name: str, row: int | None = None) -> float | None:
    if v is None or v == "" or (isinstance(v, str) and v.strip().lower() in ("nan", "none", "null")):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        where = f" in row {row + 1}" if row is not None else ""
        raise SessionError(f"'{name}'{where} is not a number: {v!r}") from None
    if not math.isfinite(x):
        return None
    return x


def resample(track: list[float], fps: float) -> list[float]:
    """Linear resampling to the simulator's 30 fps camera clock."""
    if not track or abs(fps - 1.0 / FRAME_DT) < 1e-6:
        return list(track)
    dt_in, n_out = 1.0 / fps, int((len(track) - 1) / fps / FRAME_DT) + 1
    out = []
    for k in range(n_out):
        x = k * FRAME_DT / dt_in
        i = min(int(x), len(track) - 2)
        u = x - i
        out.append(track[i] + u * (track[i + 1] - track[i]))
    return out


def _parse_rows(rows: list, fps: float) -> tuple[list[Push], list[str]]:
    if not rows:
        raise SessionError("the file has no pushes")
    if len(rows) > MAX_TRIALS:
        raise SessionError(f"at most {MAX_TRIALS} pushes per session (got {len(rows)})")
    pushes, notes = [], []
    for i, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise SessionError(f"row {i + 1} is not an object")
        low = {str(k).strip().lower(): v for k, v in raw.items()}
        get = {k: next((low[a] for a in al if a in low), None) for k, al in ALIASES.items()}
        stop = _num(get["stop"], "stop", i)
        if stop is None:
            raise SessionError(f"row {i + 1} has no stop distance (column 'stop', metres)")
        if not -0.05 <= stop <= 5.0:
            raise SessionError(f"row {i + 1}: stop {stop} m is outside 0-5 m; is it in metres?")
        cmd, v0 = _num(get["command"], "command", i), _num(get["launch_speed"], "launch_speed", i)
        track = get["track"]
        if isinstance(track, str):
            track = [t for t in track.replace(";", " ").replace(",", " ").split() if t]
        track = [_num(t, "track", i) for t in (track or [])]
        track = resample([t for t in track if t is not None], fps)[:MAX_TRACK]
        if cmd is None and v0 is None:
            if len(track) > 2:
                v0 = (track[1] - track[0]) / FRAME_DT
            else:
                raise SessionError(f"row {i + 1} needs a command, a launch_speed or a track")
        tipped = str(get["tipped"]).strip().lower() in ("1", "true", "yes", "y") if get["tipped"] is not None else False
        pushes.append(Push(stop, cmd, v0, _num(get["target"], "target", i), _num(get["perceived"], "perceived", i), tipped, track,
                           str(low.get("origin") or "log"), _num(get["start"], "start", i) or 0.0))
    if any(p.command is None for p in pushes) and any(p.command is not None for p in pushes):
        raise SessionError("either every push has a command or none does (mixed rows)")
    if not any(len(p.track) > 2 for p in pushes):
        notes.append("no tracks: launch speed and position-dependent friction cannot be separated from stops alone")
    n_tip = sum(p.tipped for p in pushes)
    if n_tip:
        notes.append(f"{n_tip} tipped push(es) are excluded from the fit (the sliding model does not cover tipping)")
    return pushes, notes


def load(text: str, filename: str = "log.csv", name: str | None = None) -> Session:
    """Parse an uploaded log. CSV: header row; `track` as space/semicolon separated positions (30 fps unless
    an `fps` column says otherwise). JSON: {"pushes": [...], "fps": 30, "sim": {...}}."""
    text = text.lstrip("﻿")
    if filename.lower().endswith(".json") or text.lstrip().startswith("{"):
        try:
            d = json.loads(text)
        except json.JSONDecodeError as e:
            raise SessionError(f"not valid JSON: {e.msg} at line {e.lineno}") from None
        s = Session.from_json(d)
    else:
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise SessionError("empty CSV")
        rows = list(reader)
        fps = _num(rows[0].get("fps"), "fps") if rows else None
        pushes, notes = _parse_rows(rows, fps or 30.0)
        s = Session(name or filename.rsplit("/", 1)[-1], "log", pushes, ParamSet.nominal(), notes)
    if name:
        s.name = name
    return s


def to_csv(session: Session) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["command", "launch_speed", "stop", "target", "perceived", "tipped", "start", "origin", "track"])
    f = lambda x: "" if x is None else f"{x:.5f}"  # noqa: E731
    for p in session.pushes:
        w.writerow([f(p.command), f(p.launch_speed), f(p.stop), f(p.target), f(p.perceived), int(p.tipped), f(p.start), p.origin,
                    " ".join(f"{y:.4f}" for y in p.track)])
    return out.getvalue()
