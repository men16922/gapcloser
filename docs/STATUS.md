# Status

Last Updated: 2026-10-11

## Current Baseline

- **Public live demo (2026-10-11): Google Cloud Run** https://tether-454741001655.us-central1.run.app (`make deploy`: Cloud Build from committed files, key in Secret Manager, max 1 instance, scales to zero). Verified: pages, Nemotron analysis 42 s, Ask 2 s, video tracking 30 s, retraining 169 s. Hugging Face Docker Spaces now need PRO (402); GitHub Pages rolled back (static only).
- **Three domains (2026-10-09):** robot manipulation, autonomous vehicles (car braking to a stop line, wet/icy section, speed control, front camera; Froude-scaled 1:25, full-size units on the page and exports, CARLA export), factory inspection (pusher, oily rail, inspection camera). Same physics and pipeline; `studio/domains.py`, tabs on the Studio data step, samples `stop-line` and `press-line`. Details: `reference/05_세가지_업무.md`.
- **Studio simulation mode:** no video needed. The visitor sets the hidden table (friction, a region of different friction, robot motor/camera for logs), NVIDIA Newton renders a phone video or writes a robot log (`POST /api/studio/simulate`), the Studio diagnoses it blind, "Run these pushes in NVIDIA Newton" executes the next experiment in the same world (`/simulate-more`), and the visitor's settings are revealed as truth.

- **Studio (product surface, 2026-10-09):** `/studio` on the live server and `dashboard/dist/studio.standalone.html` (recorded). Flow: phone video (A4 sheet → focal/pose, parallax-corrected tracking, push segmentation) or robot log → Nemotron agent (offline: probe_real queues next-experiment cards) + cross-check → bootstrap 90% intervals, what-if worlds for unmeasured table, ghost boxes (old vs calibrated sim) over the user's video → Newton / Isaac Lab EventTermCfg / Markdown / JSON exports. CLI `python -m studio`. Code: `studio/`, `server/studio_api.py`, `dashboard/studio.html`.
- Studio samples (Newton, truth revealed after; 19 of 21 truths inside 90% intervals across the six examples): flick take 2 μ 0.555 / region 0.376 m / μ 0.282 (truth 0.55 / 0.364 / 0.30); lab-bench μ 0.696, gain 0.872, region 0.376/0.449, pitch 1.9° (truth 0.70, 0.88, 0.38/0.45, 2°); short-reach refuses to invent the unmeasured region and asks for far pushes.
- Artifacts (private): Studio https://claude.ai/artifact/B8mi26WNiRJSiv9NTLoTjw ↔ Console https://claude.ai/artifact/WVBNfVMAzeNg71kDSFqYAf (v8, links to Studio).
- Experiment savings (`make studio-bench`): real pushes to 95%: suggested 5.8 / sweep 5.9–6.2 / random 7.7–7.9 (analytic 50, Newton 20 worlds).

- Scope: zero cost, no hardware, Tier 0. Local Mac: NVIDIA Newton; Nemotron 3 Super 120B on Nebius Token Factory (key in `.env`, gitignored); Ollama Nano for offline experiments.
- Public repo: https://github.com/men16922/tether
- Open world: params `patch_y0`/`patch_mu` (friction strip, seam-free Newton kernel) and `lens_k`; `InverseTrainer` (policy inverts the sim); `agent/tool_agent.py` (Nemotron tool agent: decel_profile, perception_check, fit/test_hypothesis, probe_real, commit).
- Gap-Bench Newton 15/tier (Super, $0.30): closed 100 all; open rule 83±9 / sysID 100 / agent 100; compound rule 54±15 / sysID 97±4 / agent 96±6. Agent ≈ sysID, both ≫ rule. (The 6/tier edge 99 vs 93 was noise.)
- Dashboard v3 (artifact https://claude.ai/artifact/WVBNfVMAzeNg71kDSFqYAf): agent lab notebook, 3D viewer with friction strips, Gap-Bench panel, Cosmos eyes strip, live "Stump the agent" (`make serve`).
- Cosmos Reason 2 8B local (llama.cpp, `agent/cosmos_eyes.py`): second opinion on tipping, 21/21 agree on demo clips; held-out tip recall ~50%.
- Nemotron family (open+compound, 6 each): Super 100/99, Lightning 100/95, Ultra 100/83, Nano 97/53.
- `make check` green: 122 tests (2026-10-11), plus `integrations/nat_tether/tests` under nvidia-nat
- Real objects (EV-RealPhys, MPI, tilt-test friction): 4/5 within ±0.05 on log and video paths, mean error 0.019 / 0.029 (paper's estimator 0.082). `make real-benchmark`. (incl. Studio core/API/video; live server, Newton env, tipping, recorded real Nemotron response replay, vision-role request shape).
- Agent: LLMDiagnoser (Nemotron) → fallback TrajectoryDiagnoser; HeuristicPlanner applies estimates. Tipped trials excluded from fits.
- Benchmark (Newton, 10 worlds): full DR 18% / nominal 24% / outcome-only 89% / tracking 94%; Nemotron 30B 99% (run 1) / 93% (run 2), P 0.89/0.94, R 1.00.
- Live server: `make serve` / Docker image (CPU) — verified with Token Factory in a container; deploy guide `docs/deploy/DEPLOY.md`.
- Submission assets (2026-10-11): `submission/` via `make submission` (film 2:28 with ~68 s of app footage from `video/footage.py`, thumbnail, gallery, YouTube description with chapters, Devpost About text); `docs/submission/SUBMIT.md` lists every field. Owner's guide: `reference/00_처음_읽어주세요.md`.
- Demo: console scenarios recorded with Nemotron 3 Super; dashboard v3 https://claude.ai/artifact/WVBNfVMAzeNg71kDSFqYAf.

## Active Focus

Authority: `docs/NEXT_PLAN.md`.

0. Award plan workstreams (`docs/plans/2026-10-07-award-plan.md`); submission checklist after freeze.

## Open Risks

- Studio has only been validated on Newton-rendered video; a real phone video has not been run yet (user to film; tracking is template NCC and may struggle with hands occluding, motion blur, glossy objects).
- Token Factory latency spiked on 2026-10-09 (95 s for 20 tokens); recording runs need long timeouts.

- Honest caveat: passive sysID with the same fitter matches the agent on clean (analytic) data; no measurable edge on Newton either at n=15. The agent's value is matching sysID without a hand-ordered structure library, choosing probes, and explaining fixes.
- Token Factory spend so far ≈ $2 of $25 (Super ≈ $0.0013 per diagnosis). Nano Omni not offered on Token Factory.
- Tipping regime (μ_eff ≳ 0.95): model-based diagnosis refuses to explain it; outcome-only fitting does better there.
- Local 30B can run away without max_tokens (fixed: cap 2048, timeout 180 s).
