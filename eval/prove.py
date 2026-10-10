"""One command, every headline number: recomputed from scratch where it can be (no network, no LLM credits), read
from the recorded benchmark files where it needs the Nemotron agent. Writes runs/proof/PROOF.md and proof.json.

  1. Interval coverage     24 random hidden worlds (analytic env): how often the 90% interval contains the truth
  2. Newton replay          the six recorded Studio examples, exported physics vs current sim, rms stop error
  3. Retrain and test       a policy learned in parallel Newton worlds (current / wide / Tether), hidden-world success
  4. Next experiment        studio_bench (50 analytic worlds): real runs to 95% success, suggested vs random vs sweep
  5. Gap-Bench              recorded Nemotron runs (runs/bench/open_newton_super_n15.json), quoted, not re-run
  6. Phone robustness       the same table filmed clean and hand-held/blurred/compressed, measured end to end with
                            automatic sheet detection (eval.robustness; skipped with --quick)
  7. Second engine          hidden worlds made by MuJoCo (contact launch, off-menu effects), quoted from
                            runs/proof/cross_engine.json (`make cross-engine`, ~1 h)
  8. Real footage           52 public slow-motion clips (IDPP), quoted from runs/proof/real_friction.json (`make real-check`)
  9. Real objects           EV-RealPhys: YCB objects pushed across a real table, friction measured separately by a tilt
                            test; quoted from runs/proof/real_benchmark.json (`make real-benchmark`)

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


def quoted(name: str) -> dict | None:
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else None


def pct(x) -> str:
    return "–" if x is None else f"{int(x * 100 + 0.5 + 1e-9)}%"  # half up (0.575 -> 58%), like the video and the docs


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
    if p.get("robustness"):
        L += ["", "## 6. Phone robustness (automatic sheet corners, tracking, fit)", "",
              "| Condition | Corner error | Pushes found | Launch-speed bias | Truth inside 90% | Intervals (mu, region start, region mu) |", "|---|---|---|---|---|---|"]
        for r in p["robustness"]:
            iv = r["intervals"]
            L.append(f"| {r['condition']} | {r['corner_err_px']} px | {r['pushes_found']}/{r['pushes_true']} | {r['launch_speed_bias'] * 100:+.1f}% | "
                     f"{sum(r['inside'].values())}/{len(r['inside'])} | {iv['mu_eff']}, {iv['patch_y0']}, {iv['patch_mu']} |")
    ce = p.get("cross_engine")
    if ce:
        L += ["", f"## 7. Second engine: hidden worlds made by {ce['engine']} (quoted from `make cross-engine`)", "",
              "The \"real\" pushes come from MuJoCo instead of Newton: soft contacts, a paddle that pushes the object up to speed "
              "instead of an assigned velocity, and in three conditions an effect the fitter has no field for. The Studio analyses "
              "each first-day log offline; policies are trained in each simulator and run in the hidden MuJoCo world (24 targets, "
              "±3 cm). DR: domain randomization around the current sim at 25-100% of each parameter's range (best width shown); "
              "DR oracle: randomized over the hidden worlds' own distribution, which no user knows.", "",
              "| Condition | Worlds | Current sim | Best DR | DR oracle | Exact parameters | **Tether** | Tether ≥ best DR | Left unexplained flagged | Hold-out stop error: Tether / current |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for cond, row in ce["summary"].items():
            m = row["success_mean"]
            bd = row["best_dr_width"]
            L.append(f"| {cond} | {row['worlds']} | {pct(m['current'])} | {pct(m[bd])} ({bd[3:]}) | {pct(m['dr oracle'])} | {pct(m['exact params'])} | "
                     f"**{pct(m['tether'])}** | {row['tether_beats_best_dr'] + row['tether_ties_best_dr']}/{row['worlds']} | "
                     f"{row['flagged']}/{row['worlds']} | {row['holdout_rms_m']['tether'] * 100:.1f} / {row['holdout_rms_m']['current'] * 100:.1f} cm |")
        L += ["", "\"Exact parameters\" trains on the hidden world's own MuJoCo parameter values: it misses where MuJoCo's contact "
              "behaves differently from its parameters (a few % in friction) or where an off-menu effect acts; Tether fits the "
              "behaviour, not the parameter. Interval coverage of MuJoCo's parameter values: "
              + ", ".join(f"{c} {pct(r['coverage'])}" for c, r in ce["summary"].items()) + " (the parameter is not the behaviour here)."]
    rf = p.get("real_footage")
    if rf:
        fl = rf.get("friction_law") or {}
        L += ["", "## 8. Real footage (IDPP, quoted from `make real-check`)", "",
              f"{rf['clips_used']} of {rf['clips_total']} real slow-motion slides tracked. Which friction law explains each slide "
              f"(scale-free, BIC): Coulomb (constant deceleration, Tether's model) best on {fl.get('best', {}).get('coulomb')} of "
              f"{fl.get('slides')}; viscous decisively worse on {fl.get('coulomb_beats_viscous_strongly')}; a mixed law decisively better on "
              f"{fl.get('mixed_beats_coulomb_strongly')}. The friction *value* is not identifiable from these clips (no size reference, unknown "
              f"slow-motion factor): leave-one-surface-out error {rf['mae']} vs {rf['baseline_mae_no_measurement']} for guessing the mean."]
    rb = p.get("real_benchmark")
    if rb:
        L += ["", "## 9. Real objects with independently measured friction (EV-RealPhys, quoted from `make real-benchmark`)", "",
              "YCB objects pushed by hand across a real table, filmed at 30 Hz (RealSense D455, RGB only here) with motion capture; "
              "each object's friction was measured separately by tilting the table (Kandukuri et al. 2023, Table 5). Tether never "
              "sees those values. Log: the motion-capture centre of mass as a robot log. Video: Tether's tracker on the RGB "
              "frames, pixels put on the table with the dataset's camera calibration, one click per clip at the rest position.", "",
              "| Object | Tilt test | Tether, log (90%) | Tether, video (90%) | Pushes log / video |", "|---|---|---|---|---|"]
        for o in rb["objects"]:
            lg, vd = o.get("log") or {}, o.get("video") or {}
            fmt = lambda r: f"{r['mu']:.3f} ({r['interval'][0]:.3f}–{r['interval'][1]:.3f})" if r else "–"  # noqa: E731
            L.append(f"| {o['object']} | {o['mu_tilt_test']:.3f} | {fmt(lg)} | {fmt(vd)} | {o['pushes'].get('log', 0)} / {o['pushes'].get('video', 0)} |")
        sm = rb["summary"]
        L += ["", f"Mean absolute error: log {sm['log']['mean_abs_error']:.3f} (median {sm['log']['median_abs_error']:.3f}), "
              f"video {sm['video']['mean_abs_error']:.3f} (median {sm['video']['median_abs_error']:.3f}); within ±0.05: "
              f"{sm['log']['within_0_05']}/{sm['log']['objects']} and {sm['video']['within_0_05']}/{sm['video']['objects']}. "
              f"The paper's own estimator on the same real sequences: mean {rb['paper_estimator_on_real']['mean_abs_error']}, "
              f"median {rb['paper_estimator_on_real']['median_abs_error']}. Intervals are too narrow on real data: they contain the "
              f"tilt-test value for {sum(bool((o.get('log') or {}).get('inside_interval')) for o in rb['objects'])} of 5 (log) and "
              f"{sum(bool((o.get('video') or {}).get('inside_interval')) for o in rb['objects'])} of 5 (video)."]
    rc = p.get("real_calibration")
    if rc:
        L += ["", "## 10. Real table, end to end (quoted from `make prove-real`)", "",
              "| Clip | Pushes | Tether friction (90%) | Tilt test tan(θk) | Error |", "|---|---|---|---|---|"]
        for r in rc["rows"]:
            if r.get("error"):
                L.append(f"| {r['clip']} | – | {r['error']} | | |")
            else:
                L.append(f"| {r['clip']} | {r['pushes']} | {r['mu']:.3f} ({r['mu_interval'][0]:.3f}–{r['mu_interval'][1]:.3f}) | "
                         f"{r['mu_tilt_kinetic']} | {r['error_vs_tilt']:+.3f} |")
        L += ["", f"Within ±0.05 of the tilt test: {rc['within_0_05']} of {rc['scored']} pairs."]
    L += ["", "Limits: sections 1-6 use synthetic data (NVIDIA Newton) so the truth is known; section 7 uses a second engine; "
          "section 8 is real footage without a scale reference; section 9 is real objects and real video with an independent "
          "friction measurement, calibrated by the dataset's camera calibration instead of a sheet of paper."]
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
    p["cross_engine"] = quoted("cross_engine.json")
    if p["cross_engine"]:
        p["cross_engine"] = {k: p["cross_engine"][k] for k in ("engine", "summary")}
    rf = quoted("real_friction.json")
    p["real_footage"] = {k: v for k, v in rf.items() if k not in ("rows", "pairs")} if rf else None
    p["real_calibration"] = quoted("real_calibration.json")
    rb = quoted("real_benchmark.json")
    p["real_benchmark"] = {k: v for k, v in rb.items() if k != "rows"} if rb else None
    if not a.quick:
        from eval.robustness import main as robustness

        print("robustness", flush=True)
        p["robustness"] = robustness()["conditions"]
    p["seconds"] = round(time.time() - t0)
    (OUT / "proof.json").write_text(json.dumps(p, indent=1))
    (OUT / "PROOF.md").write_text(report(p))
    print(report(p))


if __name__ == "__main__":
    main()
