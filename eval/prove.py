"""One command, every headline number: recomputed from scratch where it can be (no network, no LLM credits), read
from the recorded benchmark files where it needs the Nemotron agent. Writes runs/proof/PROOF.md and proof.json.

  1. Interval coverage     24 random hidden worlds (analytic env): how often the 90% interval contains the truth
  2. Newton replay          the six recorded Studio examples, exported physics vs current sim, rms stop error
  3. Retrain and test       a policy learned in parallel Newton worlds (current / wide / Tether), hidden-world success
  4. Next experiment        studio_bench (50 analytic worlds): real runs to 95% success, suggested vs random vs sweep
  5. Gap-Bench              recorded Nemotron runs (runs/bench/open_newton_super_n15.json), quoted, not re-run

Run: .venv/bin/python -m eval.prove [--quick]     (about 5 minutes on a laptop CPU; --quick: 8 worlds, fewer iterations)
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "proof"
DEMO = ROOT / "runs" / "studio-demo"
SCALE = {"driving": 25.0}


def coverage(n_worlds: int) -> dict:
    from sim.push_task import AnalyticPushEnv
    from studio import samples as S
    from studio.pipeline import analyze

    env, rng = AnalyticPushEnv(), random.Random(7)
    hit = tot = 0
    per: dict[str, list[bool]] = {}
    for i in range(n_worlds):
        mu = rng.uniform(0.35, 0.85)
        h = {"object_mu": mu, "table_mu": mu, "actuator_gain": rng.uniform(0.8, 1.15), "patch_y0": rng.uniform(0.25, 0.45),
             "patch_mu": mu * rng.uniform(0.45, 0.75), "camera_pitch_deg": rng.uniform(-2.5, 2.5)}
        S.SAMPLES[f"_proof{i}"] = {"title": "proof", "hidden": h, "targets": 14, "probes": [1.5, 2.0, 2.5, 2.9]}
        s, _ = S.make(f"_proof{i}", env)
        cal = analyze(s, None).calibration
        truth = {**h, "mu_eff": mu}
        for k, iv in cal.intervals.items():
            if k in truth:
                ok = iv[0] <= truth[k] <= iv[1]
                hit, tot = hit + ok, tot + 1
                per.setdefault(k, []).append(ok)
    return {"worlds": n_worlds, "values": tot, "coverage": round(hit / tot, 3),
            "by_field": {k: round(sum(v) / len(v), 3) for k, v in per.items()}}


def examples(quick: bool) -> dict:
    import studio.train as tr
    from studio.fit import calibrate
    from studio.session import load
    from studio.verify import verify

    if quick:
        tr.N_WORLDS, tr.ITERS, tr.LR = 8, 4, 0.85
    out = {}
    for f in sorted(DEMO.glob("*.json")):
        rec = json.loads(f.read_text())
        st = rec["stages"][-1]
        s = load(json.dumps(st["session"]), "x.json", st["session"]["name"])
        dom = rec["sample"].get("domain", "robot")
        s.meta["domain"] = dom
        cal = calibrate(s, st["result"]["calibration"]["structure"])  # same structure the agent chose, refit here
        v = verify(s, cal)
        t = tr.train(s, cal, tr.hidden_params(rec["truth"]["model"], s))
        t_model = rec["truth"]["model"]
        inside = {k: (iv[0] <= t_model[k] <= iv[1]) for k, iv in cal.intervals.items() if t_model.get(k) is not None}
        out[f.stem] = {"domain": dom, "runs": len(s.pushes), "structure": cal.structure,
                       "truth_inside_interval": inside,
                       "replay_rms_m": {"calibrated": v["rms_calibrated_m"], "current": v["rms_current_m"]},
                       "scale": SCALE.get(dom, 1.0),
                       "retrain_hidden_success": {k: c["final_real"] for k, c in t["conditions"].items()}}
        print(f.stem, out[f.stem]["retrain_hidden_success"], out[f.stem]["replay_rms_m"], flush=True)
    return out


def next_experiment(quick: bool) -> dict:
    path = OUT / "studio_bench.json"
    subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "eval.studio_bench", "--worlds", "12" if quick else "50", "--json", str(path)],
                   check=True, cwd=ROOT, capture_output=True)
    return json.loads(path.read_text())["summary"]


def gap_bench() -> dict:
    d = json.loads((ROOT / "runs/bench/open_newton_super_n15.json").read_text())["summary"]
    return {tier: {m: round(v["mean"], 3) for m, v in row.items()} for tier, row in d.items()}


def pct(x) -> str:
    return "–" if x is None else f"{round(x * 100)}%"


def report(p: dict) -> str:
    L = [f"# Tether: proof run", "", f"{p['when']} · git {p['git']} · {p['machine']} · {p['seconds']} s · {'quick' if p['quick'] else 'full'}", "",
         "Everything below except section 5 was recomputed by `python -m eval.prove` on this machine, with no network.", "",
         "## 1. Interval coverage", "",
         f"Over {p['coverage']['worlds']} random hidden worlds ({p['coverage']['values']} fitted values), the 90% interval contained the "
         f"truth **{pct(p['coverage']['coverage'])}** of the time (target 90%). By field: "
         + ", ".join(f"{k} {pct(v)}" for k, v in p["coverage"]["by_field"].items()) + ".", "",
         "## 2. Newton replay and 3. retrain and test (six Studio examples)", "",
         "| Example | Domain | Runs | Truth inside 90% | Replay rms: exported / current | Retrain, hidden-world success: current / wide / Tether |",
         "|---|---|---|---|---|---|"]
    for k, e in p["examples"].items():
        sc = e["scale"]
        u = (lambda m: f"{m * sc:.2f} m") if sc != 1 else (lambda m: f"{m * 100:.1f} cm")
        ins = sum(e["truth_inside_interval"].values())
        r = e["retrain_hidden_success"]
        L.append(f"| {k} | {e['domain']} | {e['runs']} | {ins}/{len(e['truth_inside_interval'])} | {u(e['replay_rms_m']['calibrated'])} / "
                 f"{u(e['replay_rms_m']['current'])} | {pct(r['current'])} / {pct(r['wide'])} / **{pct(r['tether'])}** |")
    ne = p["next_experiment"]
    g = p["gap_bench"]
    L += ["", "## 4. Next experiment", "",
          f"Real runs to reach 95% success: suggested **{ne['suggested']['mean_pushes']}**, random {ne['random']['mean_pushes']}, "
          f"hand-made sweep {ne['sweep']['mean_pushes']} ({ne['suggested']['worlds']} worlds).", "",
          "## 5. Gap-Bench (recorded Nemotron runs, quoted)", "",
          "| Tier | Nominal | Domain rand. | Rule-based | System ID | Nemotron agent |", "|---|---|---|---|---|---|"]
    for tier, row in g.items():
        L.append(f"| {tier} | {pct(row.get('nominal'))} | {pct(row.get('full_dr'))} | {pct(row.get('rule'))} | {pct(row.get('sysid'))} | {pct(row.get('agent'))} |")
    L += ["", "Limits: all data is synthetic (NVIDIA Newton) so the truth is known; a real phone video has not been validated yet."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    p = {"when": time.strftime("%Y-%m-%d %H:%M"), "git": git, "machine": f"{platform.machine()} {platform.system()}", "quick": a.quick}
    print("1/4 coverage", flush=True)
    p["coverage"] = coverage(8 if a.quick else 24)
    print("2-3/4 replay and retrain", flush=True)
    p["examples"] = examples(a.quick)
    print("4/4 next experiment", flush=True)
    p["next_experiment"] = next_experiment(a.quick)
    p["gap_bench"] = gap_bench()
    p["seconds"] = round(time.time() - t0)
    (OUT / "proof.json").write_text(json.dumps(p, indent=1))
    (OUT / "PROOF.md").write_text(report(p))
    print(report(p))


if __name__ == "__main__":
    main()
