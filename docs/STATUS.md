# Status

Last Updated: 2026-10-05

## Current Baseline

- Scope: zero cost, no hardware, Tier 0. Local Mac: NVIDIA Newton; Nemotron 3 Super 120B on Nebius Token Factory (key in `.env`, gitignored); Ollama Nano for offline experiments.
- `make check` green: 33 tests (incl. live server, Newton env, tipping, recorded real Nemotron response replay, vision-role request shape).
- Agent: LLMDiagnoser (Nemotron) → fallback TrajectoryDiagnoser; HeuristicPlanner applies estimates. Tipped trials excluded from fits.
- Benchmark (Newton, 10 worlds): full DR 18% / nominal 24% / outcome-only 89% / tracking 94%; Nemotron 30B 99% (run 1) / 93% (run 2), P 0.89/0.94, R 1.00.
- Live server: `make serve` / Docker image (CPU), deploy guide `docs/deploy/DEPLOY.md`.
- Demo: 5 scenarios recorded with Nemotron 30B; dashboard v2 https://claude.ai/artifact/WVBNfVMAzeNg71kDSFqYAf.

## Active Focus

Authority: `docs/NEXT_PLAN.md`.

0. Deploy the live demo (HF Space free or Nebius VM ~$0.06/h); record the video.

## Open Risks

- Token Factory spend so far ≈ $0.05 of $25 (Super ≈ $0.0013 per diagnosis). Nano Omni not offered on Token Factory.
- Tipping regime (μ_eff ≳ 0.95): model-based diagnosis refuses to explain it; outcome-only fitting does better there.
- Local 30B can run away without max_tokens (fixed: cap 2048, timeout 180 s).
