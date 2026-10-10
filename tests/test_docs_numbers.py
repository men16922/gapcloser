"""The numbers in README and DEVPOST are the numbers in runs/proof/: every headline claim is checked against the
recorded proof files, so a rerun that changes a result fails here until the text is updated."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROOF = ROOT / "runs" / "proof"
README = (ROOT / "README.md").read_text()
DEVPOST = (ROOT / "docs" / "submission" / "DEVPOST.md").read_text()


def pct(x: float) -> str:
    return f"{int(x * 100 + 0.5 + 1e-9)}%"  # half up, as eval/prove.py and the video


def load(name: str) -> dict:
    path = PROOF / name
    if not path.exists():
        pytest.skip(f"{name} not recorded")
    return json.loads(path.read_text())


ROWS = {"in-menu": ("In-menu effects only", "In-menu effects"), "speed": ("+ friction that depends on speed", "+ speed-dependent friction"),
        "two regions": ("+ a second friction region", "+ a second friction region"), "slope": ("+ the table tilted 1.5–3°", "+ a 1.5–3° tilt")}


@pytest.mark.parametrize("cond", list(ROWS))
def test_cross_engine_tables_match_the_recorded_run(cond):
    r = load("cross_engine.json")["summary"][cond]
    m, best = r["success_mean"], r["best_dr_width"]
    ci = r["success_ci95"]["tether"]
    cells = [pct(m["current"]), pct(m[best]), pct(m["dr oracle"]), pct(m["exact params"]), f"**{pct(m['tether'])}** ({pct(ci[0])[:-1]}–{pct(ci[1])[:-1]})"]
    for doc, label in ((README, ROWS[cond][0]), (DEVPOST, ROWS[cond][1])):
        line = next(ln for ln in doc.splitlines() if ln.strip().startswith(f"| {label} |"))
        got = [c.strip() for c in line.strip().strip("|").split("|")][1:6]
        assert got == cells, (label, got, cells)


def test_cross_engine_summary_sentences():
    s = load("cross_engine.json")["summary"]
    at_least = sum(r["tether_beats_best_dr"] + r["tether_ties_best_dr"] for r in s.values())
    worlds = sum(r["worlds"] for r in s.values())
    assert f"in {at_least} of {worlds} worlds" in README and f"in {at_least} of {worlds} worlds" in DEVPOST
    assert f"{s['speed']['flagged']}/24 as" in README and f"{s['in-menu']['flagged']}/24 (false alarms)" in README
    assert f"in-menu worlds (the old 1 cm rule flagged {s['in-menu']['size_flagged']})" in DEVPOST


def test_real_footage_friction_law():
    law = load("real_friction.json")["friction_law"]
    for doc in (README, DEVPOST):
        assert f"{law['best']['coulomb']} of {law['slides']}" in doc or f"on {law['best']['coulomb']}" in doc
        assert f"decisively worse on {law['coulomb_beats_viscous_strongly']}" in doc or f"viscous\n  law decisively on {law['coulomb_beats_viscous_strongly']}" in doc


def test_retrain_and_coverage_claims():
    p = load("proof.json")
    ex = p["examples"]
    r = {k: ex[k]["retrain_hidden_success"] for k in ex}
    row = lambda name, k: f"| {name} | {pct(r[k]['current'])} | {pct(r[k]['wide'])} | **{pct(r[k]['tether'])}** |"  # noqa: E731
    for name, k in (("Robot phone video", "flick"), ("Robot log", "lab-bench"), ("Pusher log", "press-line"),
                    ("Roadside video", "stop-line"), ("Short pushes only", "short-reach")):
        assert row(name, k) in DEVPOST, row(name, k)
    assert f"truth {pct(p['coverage']['coverage'])} of the time" in DEVPOST
    ce = load("cross_engine.json")["summary"]["in-menu"]
    assert f"{pct(ce['coverage'])} of {ce['values']} values" in DEVPOST


def test_real_benchmark_tables_match_the_recorded_run():
    rb = load("real_benchmark.json")
    for o in rb["objects"]:
        lg, vd = o["log"], o["video"]
        readme = (f"| {o['object']} | {o['mu_tilt_test']:.3f} | {lg['mu']:.3f} ({lg['interval'][0]:.3f}–{lg['interval'][1]:.3f}) | "
                  f"{vd['mu']:.3f} ({vd['interval'][0]:.3f}–{vd['interval'][1]:.3f}) | {o['pushes']['log']} / {o['pushes']['video']} |")
        assert readme in README, readme
        assert f"| {o['object']} | {o['mu_tilt_test']:.3f} | {lg['mu']:.3f} | {vd['mu']:.3f} |" in DEVPOST
    sm = rb["summary"]
    for doc in (README, DEVPOST):
        assert f"{sm['log']['mean_abs_error']:.3f}" in doc and f"{sm['video']['mean_abs_error']:.3f}" in doc
