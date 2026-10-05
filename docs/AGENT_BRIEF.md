# Agent Brief

Last Updated: 2026-10-05

> ▶ NEXT SESSION: `docs/NEXT_PLAN.md` Priority 0 — `agent/llm.py` Token Factory client + RecordedLLM mock, then `agent/llm_diagnoser.py` over tracks/frames.

## Snapshot

GapCloser (Nebius × NVIDIA Global AI Hackathon, Physical AI track): an agent closes the Sim2Real gap — failure video/measurements in a hidden-physics "real" sim → diagnosis (Nemotron 3 / Nano Omni or Cosmos Reason via Token Factory) → sim config diff → retrain → re-measure. Zero cost, no hardware: local Mac + NVIDIA Newton + Token Factory API. Deadline 2026-10-30 10:00 PDT (KST 10-31 02:00); submit 10-29. Original proposal: `suggestion.md` (v3 notice at top).

## Active Work

Authority: `docs/NEXT_PLAN.md`.

1. Tier 0: Newton env, LLM diagnoser with mocks, compare heuristic vs LLM diagnoser on confounded worlds. Freeze 10-19.

## Read Order

1. Current State: `docs/STATUS.md`
2. Next Work: `docs/NEXT_PLAN.md`
3. Decisions: `docs/DECISIONS.md`
4. Recent Log: `docs/PROGRESS_LOG.md`

## Commands

- Verification Gate: `make check`  (offline; py_compile + pytest via .venv)
- Tier 0 table: `.venv/bin/python -m eval.compare --worlds 10`
- Demo + dashboard: `make demo` → `dashboard/dist/gapcloser.standalone.html` (artifact: https://claude.ai/artifact/WVBNfVMAzeNg71kDSFqYAf)

## Guardrails

- Success rates are always measured by rollout, never LLM-estimated. The agent never reads hidden params (only `RealWorld.rollout`).
- No spending: no paid cloud/GPU/hardware without explicit user approval. Token Factory calls cost credits — never inside `make check`; use mocks/recorded fixtures.
- Secrets only via env vars (`NEBIUS_API_KEY`); never commit keys.
