# Decisions

Record of architectural or hard-to-undo choices (provider, infra, data model, doc policy, public workflow).

## 2026-10-05 — Zero-cost, no-hardware scope (Tier 0 only)
- **Decision**: No money spent, no robot/Jetson purchase. Submit Tier 0 (hidden-physics second simulator) only. Tier 1 (SO-101) and Tier 2 (Jetson) dropped.
- **Reason**: User constraint. Physical AI track allows "key application modules in action" instead of hardware footage; demo URL not required.
- **Impact**: Loses the real-robot before/after scene; the quantitative Tier 0 comparison becomes the core claim.

## 2026-10-05 — Run locally on Mac; Nebius Token Factory for models; AWS optional
- **Decision**: App runs on the local Mac (M4 Max, 48GB). Nemotron (and the video diagnoser) via Nebius Token Factory API with free credits ($25 hackathon form + $25 Builder Program). AWS ($165.93 credit, GPU quota currently 0) only for optional extras (self-hosted Cosmos Reason, GR00T/Isaac demo).
- **Reason**: Rules require "runs on Nebius Token Factory or Nebius AI Cloud + ≥1 NVIDIA open model"; Token Factory API calls satisfy this. No GPU credits are offered by Nebius. Azure free trial has no GPU quota.
- **Impact**: Token Factory budget ($50) is the hard limit on LLM calls → mock-first development, Nano for bulk, Ultra sparingly.

## 2026-10-05 — Physics: NVIDIA Newton (CPU on Mac), not MuJoCo/Isaac Lab
- **Decision**: Simulator is NVIDIA Newton 1.6 (Warp, Apache-2.0) on Mac CPU. Isaac Lab only as optional AWS demo.
- **Reason**: Keeps the visible stack NVIDIA (Nemotron + Cosmos/Nano Omni + Newton) at $0. Spike passed: friction gap reproduced, SensorTiledCamera renders video on CPU (60 frames × 2 worlds in 5.9 s). Newton/Warp has no Metal backend; workload (one arm, few worlds) does not need a GPU; same code runs with `--device cuda` on AWS if needed.
- **Impact**: Newton combines contact friction as the arithmetic mean (`kernels.py:191`) — param design must account for it.

## 2026-10-05 — Tier 0 headline metric: method comparison, not "guess the hidden values"
- **Decision**: Core claim = real (hidden-world) success rate under equal budget for A full domain randomization / B nominal / GapCloser. Diagnosis accuracy vs hidden truth is secondary.
- **Reason**: Several params are confounded from outcome data alone (friction vs actuator gain vs camera pitch all scale the slide). Measured success is robust; the confound is where visual diagnosis (Cosmos/Nano Omni) should add value and can be shown as heuristic-vs-LLM.
- **Impact**: Eval harness `eval/compare.py` is the reference; the analytic push task is the offline gate's stand-in for the Newton env.

## 2026-10-05 — Trajectory evidence breaks the confound; dashboard is the demo surface
- **Decision**: Add `TrajectoryDiagnoser` (launch speed/command → actuator gain, deceleration → effective friction, perceived vs known target → camera), compared relative to the sim's own tracked motion so engine biases cancel. Planner applies direct estimates first. Demo surface = self-contained dashboard (`dashboard/`, `make demo`), NVIDIA visual language from live nvidia.com CSS, **no NVIDIA logo** (brand rules), published as a private artifact: https://claude.ai/artifact/WVBNfVMAzeNg71kDSFqYAf.
- **Reason**: End-position-only diagnosis fails the weak-motor world (0% in Newton); tracking is what a video model would see and gives Nemotron/Nano Omni concrete evidence to reason over. Judges need a visual agent story without hardware.
- **Impact**: Benchmark (analytic, 10 worlds): full DR 19% / nominal 33% / outcome-only 94% / tracking 100%. Newton reveals cube size matters at high friction (~8 mm) — not modeled by the agent yet; sticky-table scenario uses light as distractor instead.

## 2026-10-05 — Local Nemotron via Ollama for experiments; Token Factory for submission
- **Decision**: One OpenAI-compatible client (`agent/llm.py`) with providers `local` (Ollama, `nemotron-3-nano:4b`/`:30b`) and `tokenfactory`. Model ids resolved from hints via the provider's model list. Tests replay a recorded real Nemotron response (`tests/fixtures/nemotron_weak_motor.json`).
- **Reason**: User asked to experiment locally; no API key yet. Same model family as Token Factory (Nemotron 3 Nano 30B), so prompts/validation carry over. Ollama build is text-only (no vision).
- **Impact**: Submission still needs a Token Factory run (rules). Local 4B diagnosed all probe worlds correctly at ~2.4k prompt tokens/call.
