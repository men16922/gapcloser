# Agent Brief

Last Updated: 2026-10-08

> ▶ NEXT SESSION: video v2 storyboard around the Three faults notebook + 3D viewer + Gap-Bench (`docs/submission/VIDEO_STORYBOARD.md`, `make video`), then optional HF Space deploy.

## Snapshot

GapCloser (Nebius × NVIDIA Global AI Hackathon, Physical AI track): an agent closes the Sim2Real gap — measurements in a hidden-physics "real" Newton world, including open-world faults (friction strip, lens) → Nemotron 3 Super tool agent (look, fit, probe, commit) on Token Factory → new sim config → retrain → re-measure; Cosmos Reason 2 (local) as a second opinion on tipping. Near-zero cost, no hardware: local Mac + NVIDIA Newton + Token Factory (~$2 of $25 spent). Deadline 2026-10-30 10:00 PDT (KST 10-31 02:00); submit 10-29. Original proposal: `suggestion.md` (v3 notice at top).

## Active Work

Authority: `docs/NEXT_PLAN.md`.

1. Award plan (`docs/plans/2026-10-07-award-plan.md`): open-world faults (friction strip, lens), Nemotron tool agent (fit/probe/commit), Gap-Bench vs rule + sysID, 3D viewer. Freeze 10-19.

## Read Order

1. Current State: `docs/STATUS.md`
2. Next Work: `docs/NEXT_PLAN.md`
3. Decisions: `docs/DECISIONS.md`
4. Recent Log: `docs/PROGRESS_LOG.md`

## Commands

- Verification Gate: `make check`  (offline; py_compile + pytest via .venv)
- Tier 0 table: `.venv/bin/python -m eval.compare --worlds 10`
- Gap-Bench: `.venv/bin/python -m eval.open_bench --worlds 6 [--env newton] [--llm tokenfactory]`
- Open scenarios: `.venv/bin/python -m eval.record_demo --open-only --llm tokenfactory`
- Demo + dashboard: `make demo` → `dashboard/dist/gapcloser.standalone.html` (artifact: https://claude.ai/artifact/WVBNfVMAzeNg71kDSFqYAf)

## Guardrails

- Success rates are always measured by rollout, never LLM-estimated. The agent never reads hidden params (only `RealWorld.rollout`).
- No spending: no paid cloud/GPU/hardware without explicit user approval. Token Factory calls cost credits — never inside `make check`; use mocks/recorded fixtures.
- Secrets only via env vars (`NEBIUS_API_KEY`); never commit keys.
