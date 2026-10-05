# Status

Last Updated: 2026-10-05

## Current Baseline

- Scope: zero cost, no hardware, Tier 0. Local Mac: NVIDIA Newton + Nemotron 3 Nano (Ollama 4B/30B). Token Factory run pending key.
- `make check` green: 28 tests (incl. Newton env, tipping, recorded real Nemotron response replay, vision-role request shape).
- Agent: LLMDiagnoser (Nemotron) → fallback TrajectoryDiagnoser; HeuristicPlanner applies estimates. Tipped trials excluded from fits.
- Benchmark (Newton, 10 worlds): full DR 18% / nominal 24% / outcome-only 89% / tracking 94%; Nemotron 30B 99% (run 1) / 93% (run 2), P 0.89/0.94, R 1.00.
- Demo: 5 scenarios recorded with Nemotron 30B; dashboard v2 https://claude.ai/artifact/WVBNfVMAzeNg71kDSFqYAf.

## Active Focus

Authority: `docs/NEXT_PLAN.md`.

0. Token Factory: verify models, re-record demo/benchmark with `--llm tokenfactory` (rules require it).

## Open Risks

- NEBIUS_API_KEY pending (user). Nano Omni image input unverified.
- Tipping regime (μ_eff ≳ 0.95): model-based diagnosis refuses to explain it; outcome-only fitting does better there.
- Local 30B can run away without max_tokens (fixed: cap 2048, timeout 180 s).
