# Next Plan

Last Updated: 2026-10-11

Rolling plan containing open work only. Completed history is in `docs/project/COMPLETED_SUMMARY.md`; the S-level plan and its
progress table are in `docs/project/plans/2026-10-11-s-level-plan.md`.

Current round: make the claims match the evidence. Source: external review `FEEDBACK.md`, each claim checked against the code
in `docs/project/plans/2026-10-11-feedback-hardening.md` (findings table, designs, schedule, risks). No new features until it is done.

## Priority 0 — Submission (deadline 2026-10-30 10:00 PDT, target 10-29)

- [x] [auto] Upload folder `submission/` (`make submission`): film 2:28 with ~68 s of the app in action, YouTube thumbnail and description, Devpost gallery and About text; steps in `docs/submission/SUBMIT.md`.
- [ ] [manual] Owner: **hold the YouTube upload until P1-A is done** (the film's real-objects scene quotes the paper's 0.082; a published video cannot be replaced under the same link). Then upload (public) and submit on Devpost following `docs/submission/SUBMIT.md`.
- [ ] [auto] After the YouTube upload: put the link into `docs/article/tether.{en,ko}.md` and the portfolio copies (`men16922.github.io/public/posts/tether/`), and README; optional Tether card in the portfolio Projects section. Pushing the portfolio is outward-facing: confirm first.
- [ ] [manual] Film feedback from the owner, if any: re-render with `make film`, then `make submission`.

## Priority 1 — Make the claims match the evidence (FEEDBACK, 2026-10-11)

Order: A → B → C → D/E → F → G. Each block ends with `make check` green and `tests/test_docs_numbers.py` matching the recorded proof files.

### A. Comparison protocol (real benchmark) · ~5 h · 10-11 ~ 10-12

- [ ] [auto] `tether/eval/real_benchmark.py`: baselines on the identical pushes, inits and exclusions (per-push constant-deceleration μ median via `constant_decel_mu`; per-object mean) and leave-one-push-out stop prediction (median and p90 in cm; "not evaluable" for single-push objects). JSON gains `baselines`, `held_out`, `excluded` (20 of 45 video tracks, by reason) and `inputs` (dataset camera calibration, tracking start projected from the mocap rest position, mocap object height). Done: `make real-benchmark` rerun, one table of Tether vs baselines.
- [ ] [auto] Move the paper's 0.082 to a separate reference line "different protocol (RGB-D online tracking and filtering)" in README, `docs/submission/DEVPOST.md`, `submission/about-the-project.md`, `docs/article/tether.{en,ko}.md`, `docs/guide/00_*`, `docs/guide/04_*`, STATUS, `film/stage.html:223`, `film/render.py:111`. Put the inputs paragraph under the README table. Done: docs-number test fails when 0.082 appears without the protocol note; `make film` and `make submission` regenerated (narration unchanged, TTS cache).

### B. Reliability status and predictive interval · ~9 h · 10-12 ~ 10-15

- [ ] [auto] Diagnose first (skill `overnight-harness:diagnose`): why the cracker box is off on every path (tall-box tilt, tracking, the tilt-test value itself); check whether the per-push constant-deceleration μ scatter is a usable signal. Done: cause written in the plan's progress table with evidence.
- [ ] [auto] `tether/studio/reliability.py` `assess(session, cal)` → `level` (ok / limited / unreliable) + `reasons` as code + params; hide the interval when bootstrap is meaningless (one push); push-count thresholds are product policy chosen from a coverage-vs-n sweep (EV-RealPhys subsamples, 24 cross-engine worlds), never described as a statistical rule.
- [ ] [auto] `predict_stop(cal, launch_speed)`: median and 90% range of the next push's stop (ensemble + residual noise). The "Where it stops" chart shows the parameter band and the predictive band separately.
- [ ] [auto] UI: one status chip on the result, reasons folded; same status in report.md, JSON and snippet headers. EN and KO copy (natural 합니다체), `tests/test_i18n.py` covers it; screenshots EN/KO, desktop and 390 px.
- [ ] [auto] Evaluation table in README: on the 10 EV-RealPhys cases (5 objects × 2 paths), are the cases with error > 0.05 flagged, and what is the interval coverage among unflagged cases. If a case is not flagged, report it as a limit; do not tune signals to the benchmark.

### C. Export checked as a file, one path end to end · ~7 h · 10-12 ~ 10-18

- [ ] [auto] Isaac Lab snippet: table material fixed at `MU_EFF`, object range `(2·lo − MU_EFF, 2·hi − MU_EFF)` (clamped at 0) so the averaged pair spans `[lo, hi]`; confirm the combine default and `randomize_rigid_body_material` arguments in the Isaac Lab docs first. Update the `EventTerm(` count assertion in `tests/test_studio_api.py`. Done: stand-in-module test (like the CARLA one) shows the effective friction range equals the calibrated interval.
- [ ] [auto] `calibration.json` carries the bootstrap ensemble (≤ 30 draws); Isaac snippet header says "per-variable ranges, not joint draws".
- [ ] [auto] `tether/studio/roundtrip.py` + `python -m tether.studio replay EXPORT_DIR ...`: read `calibration.json` (base units, domain scale handled), run new pushes in `NewtonPushEnv` in a **separate process**, compare stops; also parse the constants of `newton_calibration.py` with `ast` and check they match the JSON. Done: `tests/test_roundtrip.py` (Newton-marked) passes for the robot and the driving domain.
- [ ] [auto] `verify`: held-out mode (calibrate without some pushes, report new-push error) when there are enough pushes; otherwise label the number "replay of the pushes used for the fit". Export tabs state the verification level: Newton "executed", Isaac Lab and CARLA "format-checked against a stand-in, not executed here".

### D. Nemotron's contribution, separated · ~5 h · 10-16 ~ 10-18

- [ ] [auto] Switch in `pipeline.analyze` to skip the cross-check; harness over 24 cross-engine worlds + 6 Newton samples + 5 EV-RealPhys log objects with three conditions: offline structure search / Nemotron alone / Nemotron + cross-check. Metrics: held-out stop error, structure accuracy (worlds with truth), failure rate, latency, cost, times the cross-check overruled the agent. Responses stored as `RecordedLLM` fixtures so tests replay offline. Replaces the earlier "tool-agent ablations" item.
- [ ] [manual] Approve one live Token Factory run (~$0.1 of existing credit; never inside `make check`). Decision rule fixed beforehand: if (c) ≈ (a) within the interval, position the agent as guidance for model review and next-experiment choice, with no accuracy claim.
- [ ] [manual] Optional: 3–5 first-time users read a result and pick the next experiment (15 min each). If skipped, the docs say "not tested with users".

### E. One independent phone case · tooling ~3 h + filming 1–2 h · filming 10-17 ~ 10-18

- [ ] [auto] `tether/eval/prove_real.py`: calibrate on the first k flicks, predict the rest from their measured launch speeds, report held-out stop error and predicted-range coverage, list failed flicks with reasons, and add the file round trip from C. `docs/guide/06_*`: 15 flicks per pair (10 calibrate + 5 evaluate; a small design, not a statistically sufficient sample).
- [ ] [manual] Owner films at least one object–surface pair (phone + A4, tilt angle with a phone level; ideally three pairs), `make prove-real`. Fallback if no filming: the leave-one-push-out result from A.

### F. Messaging, article, film · ~5 h · 10-26 ~ 10-28 (after A–E numbers)

- [ ] [auto] One-line pitch everywhere (README, Overview hero, DEVPOST, article, YouTube description): Tether estimates the physics gap of a pushing task from video or logs, shows where it does not know yet, and provides the next experiment and a verifiable setup. Robot/factory pushing is the lead case; driving is "the same model, scaled".
- [ ] [auto] Article opens with input video → calibration → prediction on pushes not used for the fit. Replace "숨겨진 실제 환경" with "숨겨진 시뮬레이션 환경" (ko 20/162/191, en 20) and add a vocabulary test; tag every table and figure with its data origin (Newton synthetic / MuJoCo cross-engine / public real / own recording) and mark recomputed vs quoted numbers. Sync the portfolio copy after confirmation.

### G. Submission and judging period · ~2 h + owner checks · 10-27 ~ 10-29

- [ ] [auto] `make presubmit`: `make check`, docs-number test, placeholder search (`YOUTUBE_LINK` etc.), link list.
- [ ] [manual] Owner re-reads the official rules (public YouTube under 3 min, ≥ 1 min of the core module running, Token Factory inference counts as Nebius use) and checks that the Token Factory key and credit last through the judging window (12-01 ~ 12-15). DEVPOST gets a note on how to reproduce if the live demo is down (`make serve`, standalone pages).
- [ ] [auto] Final `make deploy`, then verify pages, analysis and export on the live URL. Code freeze 10-27.

## Calendar

| Date | Work |
|---|---|
| 10-11 ~ 10-12 | A, B diagnosis, C Isaac fix → re-render film → YouTube upload |
| 10-13 ~ 10-15 | B |
| 10-16 ~ 10-18 | C rest, D, E tooling; filming on the weekend; interim `make deploy` |
| 10-19 ~ 10-25 | DEA first; only `make check`, filming results, fixes (1–2 h/day) |
| 10-26 ~ 10-28 | F, G, final deploy, film re-render only if visible UI changed |
| 10-29 ~ 10-30 | Owner submits; critical fixes only |

## Frozen until after submission

W4 engineering (rest: server-side unit conversion, single domain source, studio.html modules), Studio 3D before/after, tipping-aware planning, new domains/engines/agents, running Isaac Lab or CARLA for real.

## Rules

- Read `docs/project/AGENT_BRIEF.md` → `docs/project/STATUS.md` → this file in order before starting work.
- Design snapshots live in `docs/project/plans/YYYY-MM-DD-<topic>.md`.
- Tags: `[auto]` verifiable by the offline gate (`make check`); `[manual]` needs the user, hardware or an external service.
