# Next Plan

Last Updated: 2026-10-05

Rolling plan containing open work only. Completed history is in `docs/COMPLETED_SUMMARY.md`.

## Priority 0 — Tier 0 core (10/6–10/12)

- [ ] [manual] User: Nebius Builder signup + hackathon credit form + `NEBIUS_API_KEY` (`docs/setup/NEBIUS_SETUP.md`).
- [ ] [auto] `agent/llm.py`: Token Factory client (OpenAI-compatible, model ids from env/config) + `RecordedLLM` mock replaying fixture JSON. Done: unit test with the mock, no network in `make check`.
- [ ] [auto] `agent/llm_diagnoser.py`: Diagnoser that builds a prompt from rollout stats (+ optional frame paths) and parses a JSON suspects list; robust to malformed output (falls back to HeuristicDiagnoser). Done: tests with recorded good/malformed responses.
- [ ] [auto] `eval/compare.py`: add `--env newton` (benchmark in Newton, not only the surrogate) and a diagnosis-precision column vs hidden truth. Done: test on 2 worlds with newton marker.
- [ ] [auto] Dashboard: render LLM diagnosis text and model id per diagnose event when present (`agent`, `reasoning` fields); keep layout unchanged when absent. Done: `make dashboard` builds and a unit test checks the template contains the `reasoning` hook.
- [ ] [manual] Token Factory live check: list models, test Nano Omni with a spike GIF/frames; record responses as fixtures.

## Priority 1 — Demo quality (10/13–10/19, freeze 10/19)

- [ ] [manual] Review dashboard (artifact) and decide video storyboard around the weak-motor scenario.

- [ ] [manual] Live LLM runs: 10 hidden worlds × {heuristic, llm}; publish table.

## Priority 2 — Submission (10/20–10/29)

- [ ] [manual] Optional extras if time: Cosmos Reason (build.nvidia.com/AWS), Franka arm instead of initial-velocity push, Isaac Lab demo on AWS.
- [ ] [manual] 3-min video, README, LICENSE, Devpost text; submit 10-29.

## Rules

- Read `docs/AGENT_BRIEF.md` → `docs/STATUS.md` → this file in order before starting work.
- Design snapshots live in `docs/plans/YYYY-MM-DD-<topic>.md`.

### Automation Tags (for overnight loop)

- `[auto]` — verifiable via local, deterministic, offline gate (`make check`).
- `[manual]` — human judgment, hardware, or external service. Skipped by unattended loop.
- `[blocked]` — unmet dependencies or 2 accumulated Blockers (runner marks automatically).
- **No Tag = Not for unattended run.**
