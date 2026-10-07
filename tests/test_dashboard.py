"""Dashboard v3 data contract: the recorded bundle carries what the lab notebook, strip rendering and Gap-Bench
panel read, and the built page wires those views in (offline; no browser)."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "runs" / "demo"
TOOLS = {"decel_profile", "perception_check", "fit_hypothesis", "test_hypothesis", "probe_real", "commit", "text"}
METHODS = {"nominal", "full_dr", "rule", "sysid", "agent"}


@pytest.fixture(scope="module")
def bundle():
    path = DEMO / "bundle.json"
    if not path.exists():
        pytest.skip("no recorded bundle")
    return json.loads(path.read_text())


def test_open_runs_carry_notebook_trace(bundle):
    open_runs = [r for r in bundle["runs"] if r.get("open")]
    assert open_runs, "dashboard v3 expects open-world runs"
    for run in open_runs:
        assert {"rule", "sysid", "full_dr", "real_trials"} <= set(run["baselines"])
        diags = [e for e in run["events"] if e["type"] == "diagnose"]
        assert diags and all(e.get("trace") for e in diags)
        for e in diags:
            assert {s["tool"] for s in e["trace"]} <= TOOLS
            assert e["trace"][-1]["tool"] == "commit" and isinstance(e["trace"][-1]["args"]["model"], dict)
            for s in e["trace"]:
                if s["tool"] == "decel_profile":
                    assert all("-" in b["y_m"] and "decel_g" in b for b in s["result"] if "y_m" in b)
                if s["tool"] in ("fit_hypothesis", "test_hypothesis"):
                    assert "stop_residual_rms_m" in s["result"]
                if s["tool"] == "probe_real" and "results" in s["result"]:
                    assert "probe_budget_left" in s["result"]
        # notebook pairs each diagnosis with the measurement of the same iteration (sim_params = the "old" model)
        for e in diags:
            assert any(m["type"] == "measure" and m["iter"] == e["iter"] and "sim_params" in m for m in run["events"])


def test_open_replays_describe_friction_strip(bundle):
    for run in (r for r in bundle["runs"] if r.get("open")):
        for e in run["events"]:
            if e["type"] != "measure" or not e.get("clip"):
                continue
            rp = json.loads((DEMO / e["clip"]["replay"]).read_text())
            for k in ("real", "sim"):
                p = rp["trials"][k].get("patch")
                if p is not None:
                    assert set(p) == {"y0", "mu_near", "mu_far"} and 0 < p["y0"] < 1


def test_open_benchmark_summary_shape(bundle):
    ob = bundle["open_benchmark"]
    assert ob["rows"] and {"prompt_tokens", "completion_tokens"} <= set(ob["usage"])
    for tier, ms in ob["summary"].items():
        assert METHODS <= set(ms), tier
        for s in ms.values():
            assert 0 <= s["mean"] <= 1 and s["ci95"] >= 0 and s["real_trials"] > 0


def test_built_page_wires_v3_views(tmp_path):
    from dashboard.build import build

    bundle = {"generated": "2026-10-07T00:00:00+00:00", "task": {"name": "Push-to-line", "success_tol_m": 0.03, "n_targets": 1, "targets": [0.45]},
              "params": {}, "stack": {"physics": "NVIDIA Newton", "diagnoser": "x"}, "benchmark": None,
              "open_benchmark": {"rows": [], "summary": {"open": {"agent": {"mean": 1.0, "ci95": 0.0, "real_trials": 40}}}, "usage": {}},
              "runs": [{"id": "o", "title": "O", "open": True, "events": [
                  {"type": "diagnose", "iter": 0, "trace": [{"tool": "commit", "args": {"model": {}, "explanation": "</script>"}}]}]}]}
    (tmp_path / "bundle.json").write_text(json.dumps(bundle))
    build(tmp_path / "bundle.json", tmp_path / "dist" / "g.html")
    html = (tmp_path / "dist" / "g.standalone.html").read_text()
    assert '"open_benchmark":{' in html and '"open":true' in html
    for hook in ("function renderNotebook", "function renderGapBench", "function buildStrip", '"gap-bench"', "data-reveal"):
        assert hook in html, hook
    assert html.count("</script>") == html.count("<script")  # trace text cannot close a script early
