# Devpost draft — GapCloser

Status: draft 2026-10-05. Numbers come from `eval.compare --env newton` and `eval.record_demo`; refresh before submitting. Token Factory runs still pending (needs `NEBIUS_API_KEY`).

## Tagline

An agent that fixes the simulator when a robot fails in the real world.

## Inspiration

A policy that succeeds 98% of the time in simulation can fail on a real robot because friction, motor strength or camera placement differ slightly. Engineers close this gap by hand: watch the failure, guess the cause, edit the simulator, retrain, try again. It takes days per cycle. We wanted an agent to run that cycle.

## What it does

GapCloser trains a push policy in NVIDIA Newton, runs it in a second Newton world whose physics are hidden from the agent, and measures where every push stops. When pushes miss, a Nemotron 3 model reads the evidence (end positions, the cube's tracked launch speed and deceleration, perceived vs known target positions), names the simulator parameters that are wrong with numeric estimates, and the planner rewrites the simulator config. The policy is retrained and measured again until 90% of real pushes land within ±3 cm.

The agent console shows each step: training, measurement, Nemotron's diagnosis in its own words, the config diff, Newton-rendered clips of sim vs real, the success curve, and a "reveal truth" view that grades the diagnosis against the hidden parameters.

## How we built it

- **NVIDIA Newton 1.6** (Warp) for physics and camera rendering, on a laptop CPU.
- **NVIDIA Nemotron 3** for diagnosis: Nemotron 3 Super via **Nebius Token Factory** (OpenAI-compatible API, JSON-schema output), and Nemotron 3 Nano through Ollama for local experiments. Same code path, chosen by one environment variable.
- A deterministic offline gate (24 tests) that replays a recorded real Nemotron response, so the loop is tested without network or credits.
- A self-contained dashboard (no build step) styled after NVIDIA's developer sites.

## Results

| Strategy (10 hidden worlds, NVIDIA Newton) | Real success |
|---|---|
| Domain randomization over all parameters | 18% |
| Nominal simulator | 24% |
| GapCloser, end positions only | 89% |
| GapCloser, tracked motion | 94% |

The "weak motor" world shows why the evidence matters: from end positions, a weaker motor looks exactly like a stickier table. The end-position-only agent stays at 0%; with tracked motion, Nemotron identifies `actuator_gain ≈ 0.76` and the loop reaches 100% in one fix.

## Challenges

- Confounded causes (friction vs motor gain vs camera pitch) are indistinguishable from outcomes alone. Tracking the first frames of motion separates them.
- Real physics surprises: at effective friction near 1 the cube tips over instead of sliding. Our sliding model does not cover it, and we report that world as a failure.
- Keeping LLM output safe to act on: schema-constrained JSON, validation against the parameter space, clamping, de-duplication, and a rule-based fallback so the loop never stalls.

## What's next

- Feed Newton-rendered frames to a multimodal Nemotron model so the diagnosis comes from video, not only from tracked numbers.
- A robot arm (Franka) instead of an impulse push; GR00T policies; Isaac Lab rendering.
- The same loop against real hardware.

## Built with

nvidia-newton, nvidia-warp, nemotron, nebius-token-factory, ollama, python, openai-python

## Track

Physical AI. No physical hardware; the video shows the application modules in action.
