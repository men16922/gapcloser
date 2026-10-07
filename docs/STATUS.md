# Status

Last Updated: 2026-10-07

## Current Baseline

- Scope: zero cost, no hardware, Tier 0. Local Mac: NVIDIA Newton; Nemotron 3 Super 120B on Nebius Token Factory (key in `.env`, gitignored); Ollama Nano for offline experiments.
- Public repo: https://github.com/men16922/gapcloser
- Open world: params `patch_y0`/`patch_mu` (friction strip, seam-free Newton kernel) and `lens_k`; `InverseTrainer` (policy inverts the sim); `agent/tool_agent.py` (Nemotron tool agent: decel_profile, perception_check, fit/test_hypothesis, probe_real, commit).
- Gap-Bench Newton 6/tier (Super): closed 100 all; open rule 84 / sysID 100 / agent 100; compound rule 41 / sysID 93 / agent 99 (fewer real trials). Analytic: agent = sysID = 100.
- 3D viewer (three.js replay of Newton poses) in dashboard.
- `make check` green: 50 tests (was 33 (incl. live server, Newton env, tipping, recorded real Nemotron response replay, vision-role request shape).
- Agent: LLMDiagnoser (Nemotron) → fallback TrajectoryDiagnoser; HeuristicPlanner applies estimates. Tipped trials excluded from fits.
- Benchmark (Newton, 10 worlds): full DR 18% / nominal 24% / outcome-only 89% / tracking 94%; Nemotron 30B 99% (run 1) / 93% (run 2), P 0.89/0.94, R 1.00.
- Live server: `make serve` / Docker image (CPU) — verified with Token Factory in a container; deploy guide `docs/deploy/DEPLOY.md`.
- Submission assets: `make site` (static page), `make video` (1080p draft, ~110 s), fresh-clone reproduction verified.
- Demo: 5 scenarios recorded with Nemotron 30B; dashboard v2 https://claude.ai/artifact/WVBNfVMAzeNg71kDSFqYAf.

## Active Focus

Authority: `docs/NEXT_PLAN.md`.

0. Award plan workstreams (`docs/plans/2026-10-07-award-plan.md`); submission checklist after freeze.

## Open Risks

- Honest caveat: passive sysID with the same fitter matches the agent on clean (analytic) data; edge shows only on Newton compound worlds (n small).
- Token Factory spend so far ≈ $0.6 of $25 (Super ≈ $0.0013 per diagnosis). Nano Omni not offered on Token Factory.
- Tipping regime (μ_eff ≳ 0.95): model-based diagnosis refuses to explain it; outcome-only fitting does better there.
- Local 30B can run away without max_tokens (fixed: cap 2048, timeout 180 s).
