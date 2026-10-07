"""Spike: can Nemotron spot a friction patch from per-trial evidence (local Ollama, free)?"""
import json, sys, time, random
from openai import OpenAI
sys.path.insert(0, "spike")
from open_world_spike import World, inverse_cmd, slide, track, DT, G

real = World(patch_y0=0.35, patch_mu=0.4)
sim = World()
rng = random.Random(0)
targets = sorted(rng.uniform(0.2, 0.6) for _ in range(12))
rows = []
for t in targets:
    c = inverse_cmd(sim, t)
    tr = track(real, c)
    # per-trial: where it stopped, and decel measured in first 0.15 m vs last part of its slide
    rows.append({"target_m": round(t, 3), "real_stop_m": round(slide(real, c), 3), "sim_stop_m": round(slide(sim, c), 3),
                 "track_y_every_3_frames": [round(y, 3) for y in tr[:40:3]]})
evidence = {"sim_current": {"mu_eff": 0.8, "actuator_gain": 1.0}, "frame_dt_s": round(DT, 4), "trials": rows}
SYSTEM = """You diagnose the sim-to-real gap of a cube push task. Sim model: launch speed v = command*actuator_gain,
constant deceleration a = mu_eff*g on the whole table, stop distance v^2/(2a). Real and sim used identical commands.
You may propose ANY change to the sim model that the evidence supports, including changes the current model cannot
express (e.g. position-dependent friction, nonlinear actuator, nonlinear perception). Give numeric values.
Answer JSON only: {"reasoning": str (<=80 words), "model_changes": [{"kind": str, "params": {..}}]}"""
cli = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
t0 = time.time()
r = cli.chat.completions.create(model=sys.argv[1] if len(sys.argv) > 1 else "nemotron-3-nano:30b",
    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(evidence)}],
    max_tokens=4096, temperature=0.2)
print(round(time.time() - t0, 1), "s")
print(r.choices[0].message.content)
