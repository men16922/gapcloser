# Status

Last Updated: 2026-10-05

## Current Baseline

- Scope: zero cost, no hardware, Tier 0 only (see `docs/DECISIONS.md`). Local Mac + NVIDIA Newton + Nebius Token Factory (pending key); AWS optional.
- `make check` green: 17 tests (params, loop, compare, trajectory diagnoser, 2 Newton tests). Offline/deterministic, `.venv`.
- Newton push env (`sim/newton_push.py`) matches the analytic surrogate within 1 cm; renders camera clips on Mac CPU.
- Agent: `TrajectoryDiagnoser` + `HeuristicPlanner`. Benchmark 10 worlds: full DR 19% / nominal 33% / outcome-only 94% / tracking 100%.
- Demo: `make demo` → `dashboard/dist/gapcloser.standalone.html`; published private artifact https://claude.ai/artifact/WVBNfVMAzeNg71kDSFqYAf.

## Active Focus

Authority: `docs/NEXT_PLAN.md`.

0. LLM diagnoser (Nemotron / Nano Omni via Token Factory) over tracks + frames, with recorded fixtures; show its reasoning in the dashboard.

## Open Risks

- Nebius signup/API key pending (user). Token Factory budget $50; unit prices not yet seen.
- Nano Omni video input unverified; Cosmos Reason 2 not on Token Factory.
- Newton shows cube size matters at high friction (~8 mm) — unmodeled; keep or add to agent's param handling.
- Analytic benchmark vs Newton scenarios: benchmark should move to Newton before submission.
