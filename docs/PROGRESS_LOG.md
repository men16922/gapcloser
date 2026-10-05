# Progress Log

Last Updated: 2026-10-05

This file maintains only recent incremental summaries (newest 3-5 items, ≤120 lines). Older entries are archived to `docs/archive/progress-YYYY-MM.md` via `/tidy-docs`.

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
