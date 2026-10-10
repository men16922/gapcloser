# Agent Brief

Last Updated: 2026-10-11

> ▶ NEXT SESSION: S-level plan executed (`docs/plans/2026-10-11-s-level-plan.md`, section 8). Real-object validation done on EV-RealPhys. Next: demo video with ElevenLabs narration (`python -m video.make_video --voice elevenlabs`, add a real-benchmark scene), then commit/push when the user approves.

## Snapshot

Tether (Nebius × NVIDIA Global AI Hackathon, Physical AI track): an agent closes the Sim2Real gap — measurements in a hidden-physics "real" Newton world, including open-world faults (friction strip, lens) → Nemotron 3 Super tool agent (look, fit, probe, commit) on Token Factory → new sim config → retrain → re-measure; Cosmos Reason 2 (local) as a second opinion on tipping. Near-zero cost, no hardware: local Mac + NVIDIA Newton + Token Factory (~$2 of $25 spent). Deadline 2026-10-30 10:00 PDT (KST 10-31 02:00); submit 10-29. Original proposal: `suggestion.md` (v3 notice at top).

## Active Work

0. Studio product (`docs/plans/2026-10-09-studio-product.md`): `studio/` core, `server/studio_api.py`, `dashboard/studio.html`; samples in `studio/samples/` (Newton-rendered, truth JSON beside each).

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
- Studio: `make serve` → http://localhost:8000/studio; CLI `python -m studio calibrate LOG.csv | video CLIP.mp4 --corners ...`; `make studio-samples`, `make studio-record`, `make studio-bench`
- Demo + dashboard: `make demo` → `dashboard/dist/gapcloser.standalone.html` (artifact: https://claude.ai/artifact/WVBNfVMAzeNg71kDSFqYAf)

## Guardrails

- Success rates are always measured by rollout, never LLM-estimated. The agent never reads hidden params (only `RealWorld.rollout`).
- No spending: no paid cloud/GPU/hardware without explicit user approval. Token Factory calls cost credits — never inside `make check`; use mocks/recorded fixtures.
- Secrets only via env vars (`NEBIUS_API_KEY`); never commit keys.
