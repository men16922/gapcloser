# Award plan (2026-10-07)

Goal: move GapCloser from "guess hidden params in a closed space" to "an agent that runs experiments
in an open world and beats rule-based and black-box baselines", and make the simulation watchable.

## Why the current state is not award level

1. Nemotron Super 93–94% < rule-based tracking 96%: no answer to "why an LLM?".
2. Hidden worlds live inside the agent's known 10-param space, so rules solve them.
3. The LLM fills one JSON per iteration: no tools, no experiment design ("agent runtime" in the track text).
4. Only Nemotron of the named models (Cosmos/GR00T absent); video diagnosis is "next".
5. `tipped` comes from simulator state of the "real" world (privileged); tipping still unsolved.
6. Weak visuals (1-D cube push, kinematic arm, small WebP clips) and n=10 worlds without CIs.

## Workstreams

| ID | What | Done when |
|---|---|---|
| S1 | Open-world Gap-Bench: faults outside the rule diagnoser's map (table friction patch, actuator deadzone/saturation, compound 3–4 faults, tipping, distractors); sim config can express them; policy inverts the sim per target (SimInversePolicy) instead of one-parameter `c` | ≥50 worlds in 3 tiers; closed tier parity; open tier Nemotron > rule-based outside 95% CI |
| S2 | Tool-using agent (Token Factory Nemotron 3 supports `tools`): probe_real, simulate, compare, set_config, train, measure; hypothesis → discriminating probe → update | fewer real trials to 90% on open tier than single-shot |
| S3 | Black-box sysID baseline (CMA-ES/BO over sim params, same real-trial budget) | success-vs-real-budget curves |
| S4 | Cosmos Reason 2 8B locally (GGUF/MLX, $0) as the eyes: events from clips (tipped, bounced, stalled) replace the privileged `tipped` flag | tipping detection ≥90% vs truth |
| V1 | Watchable simulation: Newton trajectories (cube + Franka links) exported per frame and replayed in a three.js 3D viewer in the dashboard / static site (orbit, sim vs real ghost, scrub) | viewer embedded in dashboard; works offline in the standalone HTML |
| A1 | Tipping fixed at policy level (lower speed / two-stage push) | tipping-edge ≥95% reproducibly |
| A2 | Dynamic Franka (joint drives push the cube in Newton) | CPU speed spike passes, then scenarios use it |
| A3 | Nemotron tiering: Nano triage/summaries, Super diagnosis, Ultra critic on disagreement | escalations and cost visible in dashboard |
| A4 | Stats: 50–100 worlds, seeds, CIs, ablations | table with CIs in README |
| B1 | MIT Push Dataset Real2Sim validation (quasi-static, separate Newton scene) | optional |
| B2 | Video 2.5–3 min around "rules break, Nemotron experiments, Cosmos sees, success" | |
| B3 | HF Space + "stump the agent" challenge | |

Out of scope: GR00T, Isaac Lab, Cosmos Transfer (GPU cost) → roadmap only.

## Schedule

- 10/7–10/9 spikes: open faults break rules; minimal TF tool loop; Cosmos local; viewer prototype. Go/no-go 10/9.
- 10/10–10/15 S1–S3 + benchmark.
- 10/16–10/19 S4, A1, A3, TF benchmark run (budget ≤ $5 total). Freeze.
- 10/20–10/24 A2 (if spike ok), dashboard v3, B3.
- 10/25–10/29 video, README/Devpost, fresh-clone check, submit 10/29.

## Spike results (2026-10-07)

`spike/open_world_spike.py`, `spike/tool_agent_spike.py` (analytic model, Nemotron 3 Super on Token Factory, 40 held-out targets):

| World | rule-fit | Nemotron tool agent | oracle |
|---|---|---|---|
| friction patch 0.4 beyond 0.35 m | 38% | 100% (2/2) | 100% |
| friction patch 1.2 beyond 0.30 m | 60% | 100% | 100% |
| lens distortion k 0.25 | 30% | 100% | 100% |
| compound (gain, patch, lens) | 95% | 60–100% (100% only when it probed far region and fit the patch) | 100% |
| closed: gain 0.8 | 100% | 100% | 100% |

- Division of labor that works: LLM chooses structure + experiments; `fit_hypothesis` (Nelder-Mead, multi-start on patch_y0) fits numbers. Without it, Super hand-tuned numbers and never committed.
- Launch-speed evidence must be compared with the sim's own first-frame measure (biased otherwise: gain 0.95 for truth 1.0).
- Agent probes only when coverage is explicit in the evidence (`real_track_coverage_m`).
- ~17k in + 2k out tokens per world with Super ≈ $0.007.
- Dropped faults: deadzone (rule-fit + inverse policy already reaches 100%), saturation (unfixable by sim).
- Honest caveat: exhaustive model-library fitting (S3) can likely match on these; compare real trials, sim calls, explanation.

## Decision after Gap-Bench (2026-10-07 evening)

- Newton n=15/tier: compound rule 54% / sysID 97% / agent 96%; analytic one-fix budget: sysID 98% / agent 96%. Active probing gives no edge here because iteration-0 data already covers the target range.
- Do not engineer worlds to make the LLM beat sysID. Claim: the Nemotron agent reaches hand-built system-ID accuracy on its own (no hand-ordered structure library or thresholds), designs probes, explains every fix, ~$0.02/world; rule-based tuning stays at 54–83%.
- Spend remaining effort on: dashboard v3 notebook, Cosmos eyes (S4), live "stump the agent" demo (B3), video.

## Nemotron family on Gap-Bench (Newton, open + compound, 6 worlds each, agent only)

| model | open | compound | tokens in/out | cost |
|---|---|---|---|---|
| Nemotron 3 Super 120B | 100% | 99% ±2 | (from 18-world run) | ~$0.07 for these 12 |
| Nemotron 3.5 Lightning | 100% | 95% ±6 | 548k / 17k | ~$0.04 |
| Nemotron 3 Ultra 550B | 100% | 83% ±21 | 363k / 52k | ~$0.52 |
| Nemotron 3 Nano 30B | 97% ±7 | 53% ±32 | 737k / 417k | ~$0.14 |

Bigger is not better here; Super stays the default, Lightning is the cost-efficient option. n=6, treat as indicative.
