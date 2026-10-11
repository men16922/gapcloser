"""Record Studio sessions for the static page: every sample goes through the live server API (same code path a
visitor uses), with the Nemotron agent when an LLM is configured. Output: runs/studio-demo/<sample>.json
(session, tracked videos, events incl. agent steps, result) + the first frames and videos it needs.

Run: .venv/bin/python -m tether.studio.record [--llm tokenfactory|local|none] [--only flick]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from tether.paths import REPO

ROOT = REPO
OUT = ROOT / "runs" / "studio-demo"


def events(c, sid):
    with c.stream("GET", f"/api/studio/sessions/{sid}/events") as r:
        body = "".join(r.iter_text())
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def stage_extras(c, sid) -> dict:
    """Newton replay and retraining for the calibration on screen, so the shared page has no dead ends per take."""
    verify = c.post(f"/api/studio/sessions/{sid}/verify").json()
    c.post(f"/api/studio/sessions/{sid}/train")
    while (tr := c.get(f"/api/studio/sessions/{sid}/train").json())["status"] == "running":
        time.sleep(1)
    return {"verify": verify, "training": tr.get("result")}


def record(c, sample: dict, use_agent: bool) -> dict:
    stages, extras = [], []
    s = c.post("/api/studio/sessions", data={"sample": sample["id"]}).json()
    sid = s["id"]
    if sample["kind"] == "video":
        parts = sample["parts"]
        for i, part in enumerate(parts):
            if i:
                s = c.post(f"/api/studio/sessions/{sid}/videos", data={"part": part}).json()
            s = c.post(f"/api/studio/sessions/{sid}/videos/{i}/track", json={"corners": s["videos"][i]["corners_hint"]}).json()
            c.post(f"/api/studio/sessions/{sid}/analyze", json={"agent": use_agent})
            ev = events(c, sid)
            st = c.get(f"/api/studio/sessions/{sid}").json()
            stages.append({"videos": len(st["videos"]), "session": st["session"], "events": ev, "result": st["result"]})
            extras.append(stage_extras(c, sid))
    else:
        c.post(f"/api/studio/sessions/{sid}/analyze", json={"agent": use_agent})
        ev = events(c, sid)
        st = c.get(f"/api/studio/sessions/{sid}").json()
        stages.append({"videos": 0, "session": st["session"], "events": ev, "result": st["result"]})
        extras.append(stage_extras(c, sid))
    st = c.get(f"/api/studio/sessions/{sid}").json()
    truth = c.get(f"/api/studio/sessions/{sid}/truth").json()
    return {"sample": sample, "sid": sid, "videos": st["videos"], "stages": stages, "truth": truth,
            "verify": extras[-1]["verify"], "training": extras[-1]["training"], "stage_extras": extras}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="tokenfactory")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    os.environ["TETHER_NO_AUTOAPP"] = "1"
    from fastapi.testclient import TestClient

    from tether.server.app import create_app
    from tether.server.studio_api import SAMPLES

    llm = None
    if a.llm != "none":
        from tether.agent.llm import make_llm

        llm = make_llm(a.llm)
    tmp = Path(tempfile.mkdtemp(prefix="studio-rec-"))
    (tmp / "demo").mkdir()
    os.environ["TETHER_LLM"] = a.llm
    c = TestClient(create_app(llm=llm, env_name="analytic", render=False, data_dir=tmp))
    OUT.mkdir(parents=True, exist_ok=True)
    for sample in SAMPLES:
        if a.only and sample["id"] != a.only:
            continue
        rec = record(c, sample, llm is not None)
        media = OUT / sample["id"]
        media.mkdir(exist_ok=True)
        for v in rec["videos"]:
            for key in ("file", "frame"):
                name = v[key].rsplit("/", 1)[-1]
                shutil.copyfile(tmp / "studio" / rec["sid"] / name, media / name)
                v[key] = f"{sample['id']}/{name}"
        if llm is not None and getattr(llm, "usage", None) is not None:
            u = llm.usage
            rec["usage"] = {"calls": u.calls, "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "by_model": u.by_model}
        (OUT / f"{sample['id']}.json").write_text(json.dumps(rec, indent=1, default=str))
        last = rec["stages"][-1]["result"]
        agent = last.get("agent") or {}
        print(sample["id"], "structure", last["calibration"]["structure"], "by", last["calibration"]["chosen_by"],
              "| agent steps", agent.get("steps"), "err", (agent.get("error") or "")[:80])
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
