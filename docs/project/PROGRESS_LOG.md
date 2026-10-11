# Progress Log

Last Updated: 2026-10-11

This file maintains only recent incremental summaries (newest 3-5 items, ≤120 lines). Older entries are archived to `docs/archive/progress-YYYY-MM.md` via `/tidy-docs`.

## 2026-10-11 — Tech article (EN/KO) published on the portfolio
- Status: done; waiting on the owner's YouTube upload.
- Changed: `docs/article/tether.{en,ko}.md` (+ `img/`, served from GitHub raw); published on https://men16922.github.io as `/articles/tether` and `/kr/articles/tether` (portfolio repo commits bc0c40c, 292223e: new "Portfolio" tab, `ArticlePost` page rendering `public/posts/<slug>/<lang>.md` with marked; `.gitattributes` now marks jpg etc. binary after JPGs were line-ending converted). Devpost replay numbers corrected (4-8 mm table scale, 18-27 cm driving).
- Verified: live `.md` files 200, live image hashes equal the originals, `make check` 122 passed (earlier this day), Prettier check passed on the portfolio.
- Blockers: none; YouTube upload and Devpost submission are the owner's.
- Next: add the YouTube link to both articles (repo + portfolio) once the owner sends it; optionally a Tether card in the portfolio Projects section.

## 2026-10-11 — One clear structure: pages Overview / Studio / Benchmark, app code in `tether/`
- Changed: "Agent console" is now **Benchmark** (KO 검증) at `/benchmark` (`/console` kept as an alias); every page has the same top bar (tagline, nav, AI status, EN/KO), the Benchmark page states its job on top (test the AI on worlds whose answer is known; Gap-Bench) and links to Studio; Overview and Studio link to it the same way. Repo: app code moved into one package `tether/` (agent, sim, studio, server, web (was dashboard), eval, `paths.py`); `film/` (was video, `film.py` -> `render.py`); `docs/guide/` (was reference); harness docs in `docs/project/`; Makefile grouped by purpose, `make pages` (was `make dashboard`); README repository map and Korean folder map (guide 00, section 10). Tech article (EN/KO) in `docs/article/`.
- Verified: `make check` 122 passed; clean copy of `tether/` + `runs/demo` builds the pages and serves /, /studio, /benchmark; Cloud Run redeployed (/, /studio, /benchmark, /console 200; Nemotron online); app footage and the Ask clip re-recorded on the live server, film 147.9 s, `make submission` rebuilt.

## 2026-10-10 — Tether: retrain in parallel Newton worlds, proof run, demo video v2
- Changed: renamed to Tether; overview page (`/`), console at `/console`, Studio deep links; Newton replay of the export; Retrain step (policy learned by trial in 16 parallel Newton worlds per condition, per-world physics kernel `push_worlds`); Newton-rendered before/after rollout and 4x4 training montage; `make prove`; demo video 2:48 (`make video`); Devpost and reference rewritten; CARLA export checked against API shape; vehicle-log driving sample.
- Verified: `make check` 97 passed; `make prove` 312 s: coverage 92% (118 values), replay 0.6-0.8 cm vs 3-25 cm, retrain Tether 100% on 4/6 examples (46% short reach, 58% roadside); studio-bench 5.76 vs 7.6.

## 2026-10-09 — Three domains: robot manipulation, autonomous vehicles, factory inspection
- Changed: `tether/studio/domains.py` (names after the FlywheelFit workspace tabs); domain tabs on the Studio data step; Newton-rendered road (car, lanes, stop line) and rail (guides, inspection window, camera post) scenes; driving is Froude-scaled 1:25 (friction unchanged), shown and exported at full size; CARLA 0.9.16 export (tire_friction ratio, friction trigger, braking check); samples `stop-line` (roadside video, 2 takes) and `press-line` (pusher log), recorded with Nemotron; chat summary in domain units; robot-log bootstrap adds a 1% length-scale term (`LOG_ERROR`).
- Verified: `make check` 93 passed; interval coverage over 24 random worlds 82% -> 90.5%; studio-bench unchanged (suggested 5.76, random 7.60, sweep 5.88); stop-line truths inside intervals; press-line region start misses by 0.9 mm (0.300 vs 0.276-0.299); headless Chrome EN/KO for all three domains; Studio artifact v5.

## 2026-10-09 — Simulation mode, EN/KO, Ask GapCloser
- Changed: Studio simulation mode (visitor sets hidden physics, Newton renders video / writes log, next experiment runs in the same world); EN default + KO switch on console and Studio (whole-sentence templates + DOM translation + patterns for server strings); `POST /api/studio/chat` + Ask panel (Nemotron, grounded in the session, truth only after reveal, 30 questions/session); fitter pre-fit before region search; Nebius VM deployed, verified, then deleted at user request (`deploy/nebius_vm.sh`).
- Verified: `make check` 91 passed; simulated world recovered inside intervals; chat answers EN/KO in 2-5 s; KO/EN switch restores originals; 390 px no overflow.

## 2026-10-09 — GapCloser Studio: calibrate from your own data
- Status: Product surface built and verified end to end (live API in headless Chrome, recorded standalone, mobile 390 px no overflow).
- Changed: `tether/studio/` (session ingest, fit + bootstrap intervals + what-if worlds, query-by-committee next experiment, Newton/Isaac Lab/Markdown/JSON export, A4-sheet camera recovery + parallax-corrected tracking, CLI, Newton-rendered sample videos/logs, recorder); `tether/server/studio_api.py`; `tether/web/studio.html` (ghost boxes over the user's video, friction painted on the frame, forest plot with truth reveal); tool agent: offline probe queue, Savitzky-Golay decel profile, `unexplained` residual checks, per-field simplex + patch-mu multistart (pitch was stuck at 0 before); Studio cross-check + identifiability guard; `tether/eval/studio_bench.py`.
- Verified: `make check` 88 passed; Docker image (CPU) runs Studio end to end (track, analyse, export); sample truths inside 90% intervals (see STATUS); bench suggested 5.8 vs random 7.7–7.9 vs sweep 5.9–6.2 real pushes to 95%; Gap-Bench analytic sysID 100/100/100, rule 100/80/59 (no regression).
- Next: real phone video; Studio artifact; video v2.

## 2026-10-08 — Dashboard v3, live Stump the agent, Cosmos eyes, README/Devpost rewrite
- Status: Award-plan core done except video. Artifact republished (v3).
- Changed: lab notebook + strip rendering + Gap-Bench panel (subagent); live server open-world path with streaming agent steps, presets, Surprise me (subagent); `tether/agent/cosmos_eyes.py` + recorder `--eyes`, eyes strip (subagent); tipped flag uses peak tilt; README and DEVPOST rewritten with honest System-ID comparison; Nemotron family bench.
- Verified: `make check` 67 passed; demo clips Cosmos vs physics 21/21; mobile 390 px no page overflow (CDP emulation).
- Next: video v2, optional deploy.

## 2026-10-07 — Open world, Nemotron tool agent, Gap-Bench, 3D viewer
- Status: Repo public (github.com/men16922/gapcloser). Award plan written; open-world faults break the rule-based agent and the Nemotron tool agent closes them in NVIDIA Newton.
- Changed: open params (friction strip via seam-free Warp kernel, lens distortion), full tracks, `InverseTrainer`, `RealWorld.push` probes; `tether/agent/tool_agent.py` + `LLM.chat` tool calling with record/replay; `tether/eval/open_bench.py` (tiers, sysID baseline, CIs); recorder `--open-only`/`--attach-bench`; 4 open scenarios recorded on Token Factory; three.js 3D replay viewer (subagent, merged).
- Verified: `make check` 50 passed; closed benchmark JSON identical to before. Gap-Bench Newton 6/tier: compound rule 41% / sysID 93% / agent 99% (50 vs 57 real trials); analytic agent = sysID = 100%. Spend ≈ $0.6 total.
- Correction (n=15/tier): compound rule 54% / sysID 97% / agent 96% — no agent edge over sysID; the 6/tier gap was noise.
- Next: dashboard v3 merge, Cosmos eyes.

## 2026-10-05 — Local completion: fresh-clone check, Docker on Token Factory, video, site
- Status: Everything that can be done locally is done and verified; remaining items are account/publishing actions (`docs/submission/CHECKLIST.md`).
- Changed: dashboard deep links (`#run.sN.iN.reveal`), final KPIs hidden mid-replay; `video/make_video.py` + `make video`; `make site`; CHECKLIST.
- Verified: fresh clone → `make check` 33 passed, `make demo` OK; Docker image rebuilt and ran a live run on Token Factory (gain 0.8 + pitch 3° → 0.799 / 3.0 → 100%, 1 call, no errors); video 1920x1080, 110 s, AAC audio (mean −16 dB), frames inspected; narration numbers match the screen (fixed a rounding mismatch 18% vs 19%).
- Next: user actions in the checklist.

## 2026-10-05 — Token Factory live: demo and benchmark on Nemotron 3 Super
- Status: Rules requirement met — the app runs Nemotron 3 Super 120B on Nebius Token Factory. Dashboard v5 recorded on Token Factory.
- Changed: `.env` loader in `tether/agent/llm.py`; compare prints LLM usage; prices read from `/v1/models?verbose=true`.
- Verified: models listed (Nano 30B, Super 120B, Ultra 550B, 3.5 Lightning; no Omni). Benchmark 10 worlds: Super 94% (P 0.94 / R 1.00), 12 calls, 14.2k in + 12.0k out ≈ $0.015. Record: 4 sliding scenarios 0→100%, tipping edge 85% (outcome-only 75%), 23 calls ≈ $0.034.
- Next: deploy, video.

## 2026-10-05 — Live demo server + Docker
- Status: `make serve` runs a live console where visitors hide physics and watch the agent (SSE). Docker image builds from a clean clone and ran a full live run against host Ollama.
- Changed: `tether/server/app.py` (FastAPI, /api/runs start+list, SSE, cost guards), dashboard live variant + New run form, `record_scenario` streaming, `Dockerfile`, `requirements.txt` (min deps incl. trimesh/pycollada/scipy/GitPython), `deploy/hf-space/README.md`, `docs/deploy/DEPLOY.md`; `runs/demo` now tracked.
- Verified: 33 tests (5 server tests incl. budget fallback, rate limit, validation). Local live run: gain 1.2 + object_mu 0.45 → Nemotron 30B estimates 1.2 / μ_eff 0.63 → 100% in 59 s. Container run: table_mu 0.4 → μ_eff 0.6 → 100%.
- Bugs caught by tests: `max_llm_calls=0` treated as unset; missing output dir when rendering is off.
- Next: deploy (HF Space or Nebius VM) once NEBIUS_API_KEY exists.

## 2026-10-05 — Franka arm clips, Nemotron-only demo, benchmark with LLM column
- Status: All 4 sliding-regime scenarios diagnosed directly by Nemotron 3 Nano 30B (no fallback), 0→100% each; dashboard v4 with Franka FR3 clips.
- Changed: `render_trial_arm` (IK-driven kinematic Franka, cube in Newton); recorder `--only`, `--clips-only`, `--bench-only`; LLM retry + 60-word reasoning; tipping-edge scenario.
- Verified: `make check` 28 passed. Benchmark Newton 10 worlds: full DR 18 / nominal 24 / outcome-only 80 / tracking 96 / Nemotron 99 (run 1), 93 (run 2).
- Blockers: Token Factory key (user).
- Next: Token Factory re-record; video; tipping-aware planning.

## 2026-10-05 — Local Nemotron in the loop, tipping found, dashboard v2
- Status: Nemotron 3 Nano 30B (Ollama, local) drives diagnosis in all recorded scenarios; dashboard v2 published (https://claude.ai/artifact/WVBNfVMAzeNg71kDSFqYAf).
- Changed: `tether/agent/llm.py` provider-agnostic (local/tokenfactory, max_tokens cap 2048, timeout 180 s); `tether/agent/llm_diagnoser.py` (validation, vague-suspect filter, vision role + frames); tipping detection in Newton env + diagnosers skip tipped trials; `eval.compare --env newton --llm`, diagnosis P/R; `extract_frames`; README, LICENSE (Apache-2.0), Devpost draft, video storyboard; 2 commits.
- Verified: `make check` 28 passed. Recorded (Newton, Nemotron 30B): slippery 0→100, camera 0→100, sticky (table μ 1.05) 0→100, weak motor 0→100 (outcome-only 0), tipping edge (table μ 1.18) 5% (outcome-only 75%). Newton benchmark w/o LLM: 18/24/89/94%, P/R outcome 0.43/1.00 vs tracking 1.00/1.00.
- Blockers: Token Factory run (key). Root-caused a stall: 30B generated 31k tokens without stopping → fixed by max_tokens.
- Next: Token Factory re-record; tipping-aware planning (outcome-only beats model-based diagnosis at the tipping edge).

## 2026-10-05 — Newton env, trajectory diagnoser, agent console dashboard
- Status: Visible demo exists — dashboard with 4 recorded Newton scenarios + benchmark tab (https://claude.ai/artifact/WVBNfVMAzeNg71kDSFqYAf).
- Changed: `tether/sim/newton_push.py` (Newton env, same rollout contract, cube tracking, `render_trial` animated WebP); `Trial.track` + `analytic_track`; `tether/agent/loop.py` emit events, `fit_launch`, `TrajectoryDiagnoser`, planner uses direct estimates; `tether/eval/compare.py` 4 methods; `tether/eval/record_demo.py`; `tether/web/benchmark.html` + `tether/web/build.py`; `docs/design/REFERENCE.md`; `make demo`/`make pages`; tests (17).
- Verified: `make check` green (17 incl. 2 Newton tests). Newton vs analytic < 1 cm. Scenarios (Newton): slippery 0→100%, camera 0→100%, sticky 5→95%, weak motor 0→100% (outcome-only agent stays 0%). Headless Chrome render checked.
- Blockers: NEBIUS_API_KEY (user) for the LLM diagnoser.
- Next: LLM diagnoser over the same evidence (tracks + frames), recorded fixtures; dashboard shows model reasoning text.

## 2026-10-05 — Harness install, zero-cost pivot, Newton spike, Tier 0 loop skeleton
- Status: Tier 0 offline reference loop works end to end (analytic push task + heuristic diagnoser/planner).
- Changed: git init; overnight harness + `make check` gate (venv python, py_compile + pytest); `docs/setup/NEBIUS_SETUP.md`; `spike/newton_gap_spike.py`; `tether/sim/params.py` (10 params), `tether/sim/push_task.py` (analytic env, policy, GridTrainer), `tether/agent/loop.py` (RealWorld black box, Diagnoser/Planner protocols, run_loop + JSON log), `tether/eval/compare.py`; tests.
- Verified: `make check` green (13 tests). Newton spike: friction gap 0.350 m vs 0.486 m, camera GIF/PNG on CPU. `python -m tether.eval.compare --worlds 10`: full_dr 19% / nominal 33% / gapcloser 94% (2.1 trains). Fixed GridTrainer plateau-edge bias (picked first max c → every push short).
- Blockers: Nebius Builder signup + NEBIUS_API_KEY (user). Video-input support of Nemotron 3 Nano Omni unverified.
- Next: Newton push env implementing the same rollout contract; Token Factory client with mock; LLM diagnoser.
