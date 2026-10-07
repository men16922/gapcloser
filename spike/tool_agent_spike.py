"""Spike: Nemotron as a tool-using agent on open-world faults (analytic model from open_world_spike).

Tools: decel_profile, perception_check, test_hypothesis (sim only, free), probe_real (costs real
trials), commit. Success is measured by rollout with a policy that inverts the committed sim.
Usage: .venv/bin/python spike/tool_agent_spike.py [tokenfactory|local] [world ...]
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, replace

sys.path.insert(0, "spike")
sys.path.insert(0, ".")
from open_world_spike import DT, G, TOL  # noqa: E402

from agent.llm import BASE_URL, OLLAMA_URL, load_dotenv  # noqa: E402


@dataclass(frozen=True)
class W:
    mu: float = 0.8
    gain: float = 1.0
    patch_y0: float | None = None
    patch_mu: float | None = None
    lens_k: float = 0.0  # perceived = t + lens_k * t^2 (radial distortion, 1/m)

    def launch(self, cmd):
        return self.gain * max(0.0, cmd)

    def slide(self, cmd):
        v = self.launch(cmd)
        d1 = v * v / (2 * self.mu * G)
        if self.patch_y0 is None or d1 <= self.patch_y0:
            return d1
        return self.patch_y0 + (v * v - 2 * self.mu * G * self.patch_y0) / (2 * self.patch_mu * G)

    def track(self, cmd, n=60):
        y, v, out, sub = 0.0, self.launch(cmd), [0.0], 20
        for _ in range(n):
            for _ in range(sub):
                mu = self.patch_mu if (self.patch_y0 is not None and y >= self.patch_y0) else self.mu
                if v <= 0:
                    break
                v = max(0.0, v - mu * G * DT / sub)
                y += v * DT / sub
            out.append(y)
        return out

    def perceive(self, t):
        return t + self.lens_k * t * t


def inverse_cmd(sim: W, target_true: float, observed: float) -> float:
    """Policy: invert the sim. It only knows the observed target; sim's lens model undistorts it."""
    t = observed
    if sim.lens_k:
        t = (-1 + math.sqrt(1 + 4 * sim.lens_k * observed)) / (2 * sim.lens_k)
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if sim.slide(mid) < t else (lo, mid)
    return 0.5 * (lo + hi)


def rollout(real: W, sim: W, targets):
    out = []
    for t in targets:
        obs = real.perceive(t)
        c = inverse_cmd(sim, t, obs)
        out.append({"target": t, "observed": obs, "command": c, "stop": real.slide(c), "track": real.track(c)})
    return out


def success(real: W, sim: W, targets) -> float:
    return sum(abs(r["stop"] - r["target"]) <= TOL for r in rollout(real, sim, targets)) / len(targets)


def rule_fit(real: W, sim: W, targets) -> W:
    """Rule-based baseline (TrajectoryDiagnoser-style): one gain, one mu from early tracks, linear perception."""
    gs, decs = [], []
    trials = rollout(real, sim, targets)
    for r in trials:
        tr = r["track"][:7]
        pts = [(k * DT, y) for k, y in enumerate(tr)]
        s11 = sum(t * t for t, _ in pts); s12 = sum(-0.5 * t**3 for t, _ in pts); s22 = sum(0.25 * t**4 for t, _ in pts)
        b1 = sum(t * y for t, y in pts); b2 = sum(-0.5 * t * t * y for t, y in pts)
        det = s11 * s22 - s12 * s12
        v0, a = (b1 * s22 - b2 * s12) / det, (s11 * b2 - s12 * b1) / det
        gs.append(v0 / r["command"]); decs.append(a / G)
    return replace(sim, gain=sum(gs) / len(gs), mu=sum(decs) / len(decs))


TOOLS = [
    {"type": "function", "function": {"name": "decel_profile", "description": "Deceleration of the real cube (in g) measured from camera tracks, binned by cube position along the push axis.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "perception_check", "description": "Perceived target distance vs the known true target position for each real trial (camera calibration check).", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "test_hypothesis", "description": "Free (sim only): replay every real trial's command in a candidate sim model and return stop-distance residuals (sim - real). Use it to verify a hypothesis before committing.",
        "parameters": {"type": "object", "properties": {"model": {"$ref": "#/$defs/model"}}, "required": ["model"], "$defs": {}}}},
    {"type": "function", "function": {"name": "fit_hypothesis", "description": "Free (sim only): least-squares fit of the chosen free parameters of a model structure to all real data (stops, launch speeds, perception). Give the structure as a starting model and list which fields are free. Returns fitted model and residuals. Prefer this over hand-tuning numbers.",
        "parameters": {"type": "object", "properties": {"model": {"$ref": "#/$defs/model"}, "free": {"type": "array", "items": {"type": "string", "enum": ["mu", "gain", "patch_y0", "patch_mu", "lens_k"]}}}, "required": ["model", "free"]}}},
    {"type": "function", "function": {"name": "probe_real", "description": "Costs real-robot trials (budget limited): push the real cube with the given commands (launch command units, 0.5-4.5) and return stop positions and launch speeds. Use only when existing data cannot separate hypotheses.",
        "parameters": {"type": "object", "properties": {"commands": {"type": "array", "items": {"type": "number"}, "maxItems": 6}}, "required": ["commands"]}}},
    {"type": "function", "function": {"name": "commit", "description": "Final answer: the sim model to retrain the policy on.", "parameters": {"type": "object", "properties": {"model": {"$ref": "#/$defs/model"}, "explanation": {"type": "string"}}, "required": ["model", "explanation"]}}},
]
MODEL_SCHEMA = {"type": "object", "properties": {
    "mu": {"type": "number", "description": "effective friction on the table (mu_eff)"},
    "gain": {"type": "number", "description": "actuator gain: launch speed = gain * command"},
    "patch_y0": {"type": ["number", "null"], "description": "optional: position (m) where a table region with different friction starts"},
    "patch_mu": {"type": ["number", "null"], "description": "optional: mu_eff beyond patch_y0"},
    "lens_k": {"type": "number", "description": "camera radial distortion: perceived = true + lens_k * true^2 (1/m), 0 = none"}},
    "required": ["mu", "gain", "patch_y0", "patch_mu", "lens_k"]}
for t in TOOLS:  # inline the shared model schema (some servers do not resolve $ref)
    props = t["function"]["parameters"].get("properties", {})
    if "model" in props:
        props["model"] = MODEL_SCHEMA
    t["function"]["parameters"].pop("$defs", None)

SYSTEM = """You are GapCloser, an agent that fixes a robot simulator so a policy trained in it works on the real robot.
Task: a robot pushes a cube (launch speed = gain * command); it slides to a stop; the goal is to stop within 3 cm of a target line.
The policy perceives the target through a camera and inverts the simulator to pick the command, so a correct simulator
(dynamics AND perception) means success. The current sim is wrong. Work like a scientist: inspect evidence with tools, form
hypotheses (start with decel_profile and perception_check; the sim model supports global friction, actuator gain, a table region with different friction, and camera
radial distortion), fit each candidate structure with fit_hypothesis (free) and compare residuals, use probe_real only if needed (real trials are expensive),
Targets span 0.2-0.6 m: if the real cubes' tracks do not cover that whole range, the sim is unverified there, so probe_real with commands that reach it before committing. Then commit. Keep tool arguments numeric and brief. Do not explain at length between calls."""


def run_agent(real: W, sim: W, targets, provider: str, max_steps=10, probe_budget=12):
    from openai import OpenAI
    import os
    load_dotenv()
    if provider == "tokenfactory":
        cli, model = OpenAI(base_url=BASE_URL, api_key=os.environ["NEBIUS_API_KEY"], timeout=180), "nvidia/nemotron-3-super-120b-a12b"
    else:
        cli, model = OpenAI(base_url=OLLAMA_URL, api_key="ollama", timeout=300), "nemotron-3-nano:30b"
    trials = rollout(real, sim, targets)
    real_cmds = [(r["command"], r["stop"], r["track"]) for r in trials]
    summary = {"target_range_m": [0.2, 0.6], "real_track_coverage_m": [0.0, round(max(r["stop"] for r in trials), 3)], "sim_current": asdict(sim), "real_success_rate": sum(abs(r["stop"] - r["target"]) <= TOL for r in trials) / len(trials),
               "trials": [{"target": round(r["target"], 3), "observed": round(r["observed"], 3), "command": round(r["command"], 3),
                           "real_stop": round(r["stop"], 3), "sim_predicted_stop": round(sim.slide(r["command"]), 3)} for r in trials[::2]]}
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Iteration evidence:\n" + json.dumps(summary)}]
    usage = [0, 0]
    log, used = [], 0

    def tool(name, args):
        nonlocal used
        if name == "decel_profile":
            bins = {}
            for c, _, tr in real_cmds:
                for k in range(1, len(tr) - 1):
                    v1, v2 = (tr[k] - tr[k - 1]) / DT, (tr[k + 1] - tr[k]) / DT
                    if v2 > 0.05:
                        bins.setdefault(round(int(tr[k] / 0.05) * 0.05, 2), []).append((v1 - v2) / DT / G)
            return [{"y_m": f"{b:.2f}-{b + 0.05:.2f}", "decel_g": round(sum(v) / len(v), 3), "n": len(v)} for b, v in sorted(bins.items())]
        if name == "perception_check":
            return [{"true": round(r["target"], 3), "perceived": round(r["observed"], 3)} for r in trials[::2]]
        if name == "test_hypothesis":
            m = W(**{k: args["model"].get(k) for k in ("mu", "gain", "patch_y0", "patch_mu")}, lens_k=args["model"].get("lens_k") or 0.0)
            if m.patch_y0 is not None and m.patch_mu is None:
                m = replace(m, patch_y0=None)
            res = [m.slide(c) - s for c, s, _ in real_cmds]
            persp = [m.perceive(r["target"]) - r["observed"] for r in trials]
            return {"stop_residual_rms_m": round(math.sqrt(sum(x * x for x in res) / len(res)), 4),
                    "worst_stop_residuals": sorted(((round(c, 2), round(x, 3)) for (c, _, _), x in zip(real_cmds, res)), key=lambda p: -abs(p[1]))[:3],
                    "perception_residual_rms_m": round(math.sqrt(sum(x * x for x in persp) / len(persp)), 4)}
        if name == "fit_hypothesis":
            from scipy.optimize import minimize
            base = dict(args["model"]); free = [f for f in args.get("free", []) if f in ("mu", "gain", "patch_y0", "patch_mu", "lens_k")]
            if ("patch_y0" in free or "patch_mu" in free) and base.get("patch_y0") is None:
                base["patch_y0"], base["patch_mu"] = base.get("patch_y0") or 0.3, base.get("patch_mu") or base["mu"]
            x0 = [float(base.get(f) or 0.0) for f in free]
            def build(x):
                d = {**base, **dict(zip(free, x))}
                return W(mu=max(0.05, d["mu"]), gain=max(0.05, d["gain"]), patch_y0=d.get("patch_y0"),
                         patch_mu=max(0.05, d["patch_mu"]) if d.get("patch_mu") is not None else None,
                         lens_k=d.get("lens_k") or 0.0) if d.get("patch_mu") is not None else W(mu=max(0.05, d["mu"]), gain=max(0.05, d["gain"]), lens_k=d.get("lens_k") or 0.0)
            def loss(x):
                m = build(x)
                e = sum((m.slide(c) - s) ** 2 for c, s, _ in real_cmds)
                e += sum(((m.track(c, 1)[1] - tr[1]) / DT) ** 2 for c, _, tr in real_cmds)  # same first-frame measure on both sides
                e += sum((m.perceive(r["target"]) - r["observed"]) ** 2 for r in trials)
                return e
            starts = [x0]
            if "patch_y0" in free:  # multi-start over where the region begins (loss is non-convex in it)
                i = free.index("patch_y0")
                starts = [x0[:i] + [y] + x0[i + 1:] for y in (0.15, 0.25, 0.35, 0.45, 0.55)]
            best = min((minimize(loss, st, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 4000}) for st in starts), key=lambda r: r.fun) if free else None
            fitted = build(best.x if free else [])
            res = tool("test_hypothesis", {"model": asdict(fitted)})
            return {"fitted_model": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(fitted).items()}, **res}
        if name == "probe_real":
            cmds = [float(c) for c in args.get("commands", [])][: max(0, probe_budget - used)]
            used += len(cmds)
            out = []
            for c in cmds:
                tr = real.track(c)
                real_cmds.append((c, real.slide(c), tr))
                out.append({"command": c, "stop": round(real.slide(c), 3), "launch_speed": round((tr[1] - tr[0]) / DT, 3)})
            return {"results": out, "probe_budget_left": probe_budget - used}
        return {"error": "unknown tool"}

    committed = None
    t0 = time.time()
    for step in range(max_steps):
        last = step == max_steps - 1
        if last:
            msgs.append({"role": "user", "content": "Step budget exhausted: call commit now with your best model."})
        r = cli.chat.completions.create(model=model, messages=msgs, tools=TOOLS, temperature=0.2, max_tokens=4096,
                                        **({"tool_choice": {"type": "function", "function": {"name": "commit"}}} if last else {}))
        usage[0] += r.usage.prompt_tokens; usage[1] += r.usage.completion_tokens
        msg = r.choices[0].message
        msgs.append({"role": "assistant", "content": msg.content or "", "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])]})
        if not msg.tool_calls:
            msgs.append({"role": "user", "content": "Use the tools; finish with commit."})
            log.append(("text", (msg.content or "")[:200]))
            continue
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            if tc.function.name == "commit":
                committed = args
                log.append(("commit", args))
                break
            out = tool(tc.function.name, args)
            log.append((tc.function.name, args, out if tc.function.name != "decel_profile" else f"{len(out)} bins"))
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)})
        if committed:
            break
    return committed, log, usage, used, time.time() - t0


WORLDS = {
    "patch_low": W(patch_y0=0.35, patch_mu=0.4),
    "patch_high": W(patch_y0=0.30, patch_mu=1.2),
    "lens": W(lens_k=0.25),
    "compound": W(gain=0.85, patch_y0=0.40, patch_mu=0.5, lens_k=-0.15),
    "closed_gain": W(gain=0.8),
}

if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "local"
    names = sys.argv[2:] or list(WORLDS)
    rng = random.Random(0)
    targets = [rng.uniform(0.2, 0.6) for _ in range(20)]
    holdout = [rng.uniform(0.2, 0.6) for _ in range(40)]
    nominal = W()
    for n in names:
        real = WORLDS[n]
        fit = rule_fit(real, rule_fit(real, nominal, targets), targets)
        committed, log, usage, probes, secs = run_agent(real, nominal, targets, provider)
        agent_sim = nominal
        if committed:
            m = committed["model"]
            agent_sim = W(mu=m["mu"], gain=m["gain"], patch_y0=m.get("patch_y0") if m.get("patch_mu") else None,
                          patch_mu=m.get("patch_mu"), lens_k=m.get("lens_k") or 0.0)
        print(f"\n=== {n}: truth {asdict(real)}")
        for entry in log:
            print("  ", str(entry)[:260])
        print(f"  nominal {success(real, nominal, holdout):.0%} | rule-fit x2 {success(real, fit, holdout):.0%} | agent {success(real, agent_sim, holdout):.0%} "
              f"| oracle {success(real, real, holdout):.0%} | probes {probes} | tokens {usage} | {secs:.0f}s")
