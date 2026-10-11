# Agent Brief

Last Updated: 2026-10-11

> ▶ NEXT SESSION: Submission ready (2026-10-11). Pages are Overview / Studio / Benchmark (`/benchmark`); app code lives in the `tether/` package (map: README "Repository map"). Everything to upload is in `submission/` (`make submission`), field by field in `docs/submission/SUBMIT.md`; the owner uploads to YouTube and submits on Devpost. Live demo on Cloud Run (`make deploy`). Owner's guide: `docs/guide/00_처음_읽어주세요.md`. Deployed 2026-10-11 (Cloud Run revision with /benchmark); film re-recorded with the new nav (2:28). Open: film feedback.

## Snapshot

Tether (Nebius × NVIDIA Global AI Hackathon, Physical AI track): an agent closes the Sim2Real gap — measurements in a hidden-physics "real" Newton world, including open-world faults (friction strip, lens) → Nemotron 3 Super tool agent (look, fit, probe, commit) on Token Factory → new sim config → retrain → re-measure; Cosmos Reason 2 (local) as a second opinion on tipping. Near-zero cost, no hardware: local Mac + NVIDIA Newton + Token Factory (~$2 of $25 spent). Deadline 2026-10-30 10:00 PDT (KST 10-31 02:00); submit 10-29. 

## Active Work

0. Studio product (`docs/project/plans/2026-10-09-studio-product.md`): `tether/studio/` core, `tether/server/studio_api.py`, `tether/web/studio.html`; samples in `tether/studio/samples/` (Newton-rendered, truth JSON beside each).

Authority: `docs/project/NEXT_PLAN.md`.

1. Award plan (`docs/project/plans/2026-10-07-award-plan.md`): open-world faults (friction strip, lens), Nemotron tool agent (fit/probe/commit), Gap-Bench vs rule + sysID, 3D viewer. Freeze 10-19.

## Read Order

1. Current State: `docs/project/STATUS.md`
2. Next Work: `docs/project/NEXT_PLAN.md`
3. Decisions: `docs/project/DECISIONS.md`
4. Recent Log: `docs/project/PROGRESS_LOG.md`

## Commands

- Verification Gate: `make check`  (offline; py_compile + pytest via .venv)
- Tier 0 table: `.venv/bin/python -m tether.eval.compare --worlds 10`
- Gap-Bench: `.venv/bin/python -m tether.eval.open_bench --worlds 6 [--env newton] [--llm tokenfactory]`
- Open scenarios: `.venv/bin/python -m tether.eval.record_demo --open-only --llm tokenfactory`
- Studio: `make serve` → http://localhost:8000/studio; CLI `python -m tether.studio calibrate LOG.csv | video CLIP.mp4 --corners ...`; `make studio-samples`, `make studio-record`, `make studio-bench`
- Demo + dashboard: `make demo` → `tether/web/dist/benchmark.standalone.html` (artifact: https://claude.ai/artifact/WVBNfVMAzeNg71kDSFqYAf)

## Guardrails

- Success rates are always measured by rollout, never LLM-estimated. The agent never reads hidden params (only `RealWorld.rollout`).
- No spending: no paid cloud/GPU/hardware without explicit user approval. Token Factory calls cost credits — never inside `make check`; use mocks/recorded fixtures.
- Secrets only via env vars (`NEBIUS_API_KEY`); never commit keys.
