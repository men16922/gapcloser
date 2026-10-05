# Progress Log

Last Updated: 2026-10-05

This file maintains only recent incremental summaries (newest 3-5 items, ≤120 lines). Older entries are archived to `docs/archive/progress-YYYY-MM.md` via `/tidy-docs`.

## 2026-10-05 — Local completion: fresh-clone check, Docker on Token Factory, video, site
- Status: Everything that can be done locally is done and verified; remaining items are account/publishing actions (`docs/submission/CHECKLIST.md`).
- Changed: dashboard deep links (`#run.sN.iN.reveal`), final KPIs hidden mid-replay; `video/make_video.py` + `make video`; `make site`; CHECKLIST.
- Verified: fresh clone → `make check` 33 passed, `make demo` OK; Docker image rebuilt and ran a live run on Token Factory (gain 0.8 + pitch 3° → 0.799 / 3.0 → 100%, 1 call, no errors); video 1920x1080, 110 s, AAC audio (mean −16 dB), frames inspected; narration numbers match the screen (fixed a rounding mismatch 18% vs 19%).
- Next: user actions in the checklist.

## 2026-10-05 — Token Factory live: demo and benchmark on Nemotron 3 Super
- Status: Rules requirement met — the app runs Nemotron 3 Super 120B on Nebius Token Factory. Dashboard v5 recorded on Token Factory.
- Changed: `.env` loader in `agent/llm.py`; compare prints LLM usage; prices read from `/v1/models?verbose=true`.
- Verified: models listed (Nano 30B, Super 120B, Ultra 550B, 3.5 Lightning; no Omni). Benchmark 10 worlds: Super 94% (P 0.94 / R 1.00), 12 calls, 14.2k in + 12.0k out ≈ $0.015. Record: 4 sliding scenarios 0→100%, tipping edge 85% (outcome-only 75%), 23 calls ≈ $0.034.
- Next: deploy, video.

## 2026-10-05 — Live demo server + Docker
- Status: `make serve` runs a live console where visitors hide physics and watch the agent (SSE). Docker image builds from a clean clone and ran a full live run against host Ollama.
- Changed: `server/app.py` (FastAPI, /api/runs start+list, SSE, cost guards), dashboard live variant + New run form, `record_scenario` streaming, `Dockerfile`, `requirements.txt` (min deps incl. trimesh/pycollada/scipy/GitPython), `deploy/hf-space/README.md`, `docs/deploy/DEPLOY.md`; `runs/demo` now tracked.
- Verified: 33 tests (5 server tests incl. budget fallback, rate limit, validation). Local live run: gain 1.2 + object_mu 0.45 → Nemotron 30B estimates 1.2 / μ_eff 0.63 → 100% in 59 s. Container run: table_mu 0.4 → μ_eff 0.6 → 100%.
- Bugs caught by tests: `max_llm_calls=0` treated as unset; missing output dir when rendering is off.
- Next: deploy (HF Space or Nebius VM) once NEBIUS_API_KEY exists.

## 2026-10-05 — Franka arm clips, Nemotron-only demo, benchmark with LLM column
- Status: All 4 sliding-regime scenarios diagnosed directly by Nemotron 3 Nano 30B (no fallback), 0→100% each; dashboard v4 with Franka FR3 clips.
- Changed: `render_trial_arm` (IK-driven kinematic Franka, cube in Newton); recorder `--only`, `--clips-only`, `--bench-only`; LLM retry + 60-word reasoning; tipping-edge scenario.
- Verified: `make check` 28 passed. Benchmark Newton 10 worlds: full DR 18 / nominal 24 / outcome-only 80 / tracking 96 / Nemotron 99 (run 1), 93 (run 2).
- Blockers: Token Factory key (user).
- Next: Token Factory re-record; video; tipping-aware planning.

## 2026-10-05 — Local Nemotron in the loop, tipping found, dashboard v2
- Status: Nemotron 3 Nano 30B (Ollama, local) drives diagnosis in all recorded scenarios; dashboard v2 published (https://claude.ai/artifact/WVBNfVMAzeNg71kDSFqYAf).
- Changed: `agent/llm.py` provider-agnostic (local/tokenfactory, max_tokens cap 2048, timeout 180 s); `agent/llm_diagnoser.py` (validation, vague-suspect filter, vision role + frames); tipping detection in Newton env + diagnosers skip tipped trials; `eval.compare --env newton --llm`, diagnosis P/R; `extract_frames`; README, LICENSE (Apache-2.0), Devpost draft, video storyboard; 2 commits.
- Verified: `make check` 28 passed. Recorded (Newton, Nemotron 30B): slippery 0→100, camera 0→100, sticky (table μ 1.05) 0→100, weak motor 0→100 (outcome-only 0), tipping edge (table μ 1.18) 5% (outcome-only 75%). Newton benchmark w/o LLM: 18/24/89/94%, P/R outcome 0.43/1.00 vs tracking 1.00/1.00.
- Blockers: Token Factory run (key). Root-caused a stall: 30B generated 31k tokens without stopping → fixed by max_tokens.
- Next: Token Factory re-record; tipping-aware planning (outcome-only beats model-based diagnosis at the tipping edge).

## 2026-10-05 — Newton env, trajectory diagnoser, agent console dashboard
- Status: Visible demo exists — dashboard with 4 recorded Newton scenarios + benchmark tab (https://claude.ai/artifact/WVBNfVMAzeNg71kDSFqYAf).
- Changed: `sim/newton_push.py` (Newton env, same rollout contract, cube tracking, `render_trial` animated WebP); `Trial.track` + `analytic_track`; `agent/loop.py` emit events, `fit_launch`, `TrajectoryDiagnoser`, planner uses direct estimates; `eval/compare.py` 4 methods; `eval/record_demo.py`; `dashboard/template.html` + `dashboard/build.py`; `docs/design/REFERENCE.md`; `make demo`/`make dashboard`; tests (17).
- Verified: `make check` green (17 incl. 2 Newton tests). Newton vs analytic < 1 cm. Scenarios (Newton): slippery 0→100%, camera 0→100%, sticky 5→95%, weak motor 0→100% (outcome-only agent stays 0%). Headless Chrome render checked.
- Blockers: NEBIUS_API_KEY (user) for the LLM diagnoser.
- Next: LLM diagnoser over the same evidence (tracks + frames), recorded fixtures; dashboard shows model reasoning text.

## 2026-10-05 — Harness install, zero-cost pivot, Newton spike, Tier 0 loop skeleton
- Status: Tier 0 offline reference loop works end to end (analytic push task + heuristic diagnoser/planner).
- Changed: git init; overnight harness + `make check` gate (venv python, py_compile + pytest); `docs/setup/NEBIUS_SETUP.md`; `spike/newton_gap_spike.py`; `sim/params.py` (10 params), `sim/push_task.py` (analytic env, policy, GridTrainer), `agent/loop.py` (RealWorld black box, Diagnoser/Planner protocols, run_loop + JSON log), `eval/compare.py`; tests.
- Verified: `make check` green (13 tests). Newton spike: friction gap 0.350 m vs 0.486 m, camera GIF/PNG on CPU. `python -m eval.compare --worlds 10`: full_dr 19% / nominal 33% / gapcloser 94% (2.1 trains). Fixed GridTrainer plateau-edge bias (picked first max c → every push short).
- Blockers: Nebius Builder signup + NEBIUS_API_KEY (user). Video-input support of Nemotron 3 Nano Omni unverified.
- Next: Newton push env implementing the same rollout contract; Token Factory client with mock; LLM diagnoser.
