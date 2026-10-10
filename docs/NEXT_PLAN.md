# Next Plan

Last Updated: 2026-10-11

Rolling plan containing open work only. Completed history is in `docs/COMPLETED_SUMMARY.md`.

## Priority 0 — B → S plan (authoritative: `docs/plans/2026-10-11-s-level-plan.md`)

- [x] [auto] Phase 0: overstated claims fixed in README/DEVPOST/reference; per-take Newton replay and retraining recorded (no dead buttons on the shared page); Korean coverage test for server sentences; session store safety (busy sessions never evicted, atomic analyse/train start, readable Newton errors, temp dirs removed); dead code removed.
- [x] [auto] W1 evidence: MuJoCo hidden worlds with contact launch and 3 off-menu effects (Tether 83-89% vs best DR 9-13%), DR width sweep, pattern check, friction-law BIC on IDPP (Coulomb 36/45). Agent necessity repositioned honestly (ties System-ID).
- [x] [auto] W2 NVIDIA stack: NeMo Agent Toolkit workflow (integrations/nat_tether), Isaac Lab export with real region/gain config. Cosmos gate deferred (no tipped real clips to score).
- [x] [auto] W3 product: shared design system, Studio in 3 phases, one headline. Open: 3D before/after in Studio.
- [ ] [auto] W4 engineering (rest): server-side unit conversion, single domain source, studio_api router split, shared i18n, studio.html modules.
- [ ] [manual] W1f: 15-minute real recording (3 object–surface pairs, A4 sheet, tilt-test ground truth).

## Priority 0 — Studio product (plan: `docs/plans/2026-10-09-studio-product.md`)

- [ ] [manual] Film a real phone video (A4 sheet beside the path, 8–10 flicks, phone still) and run it through `/studio`; fix what breaks in tracking (hand occlusion, blur). This is the strongest possible demo evidence.
- [ ] [manual] Video v2 around Studio: phone video → ghost boxes → next experiment → take 2 → export.

## Priority 0b — Award plan (10/7–10/19), see `docs/plans/2026-10-07-award-plan.md`

- [ ] [auto] Tool-agent ablations on analytic env with RecordedLLM fixtures (no probes / no fit tool); Done: offline tests pass.
- [ ] [manual] Video v2 (2.5–3 min): rules break → Nemotron notebook (Three faults) → 3D viewer strip → Cosmos eyes → Gap-Bench with honest System-ID line. Update `docs/submission/VIDEO_STORYBOARD.md`, `video/make_video.py`.
- [ ] [manual] Optional: HF Space deploy of the live "Stump the agent" server (Token Factory budget cap).
- [ ] [manual] Devpost submission 10-29 from `docs/submission/DEVPOST.md`.


- [ ] [manual] Vision: Nano Omni is not on Token Factory (2026-10-05 model list). Candidate: `openbmb/MiniCPM-V-4_5` (not NVIDIA) or skip; if used, check image support then if yes pass `frames=lambda: extract_frames(real_clip, ...)` to `LLMDiagnoser` in `eval/record_demo.py` (extractor and vision role already implemented and tested).

- [ ] [manual] Deploy the live demo: Hugging Face Space (free) with `NEBIUS_API_KEY` secret, or a Nebius CPU VM (~$0.06/h, user decision). Guide: `docs/deploy/DEPLOY.md`. Image builds and runs locally (verified).

## Priority 1 — Demo quality (10/13–10/19, freeze 10/19)

- [ ] [manual] Review dashboard (artifact) and decide video storyboard around the weak-motor scenario.


## Priority 2 — Submission (10/20–10/29)

- [ ] [manual] Optional extras if time: Cosmos Reason (build.nvidia.com/AWS), dynamic Franka push (arm currently kinematic in clips), Isaac Lab demo on AWS (needs GPU quota request + approval).
- [ ] [auto] Tipping-aware planning: when ≥30% of real trials tip, propose lowering push speed (policy ceiling) instead of leaving the gap; Done: Newton-marked test on table μ 1.18 improves over the current result.
- [ ] [manual] Review the video draft (`make video`), optionally re-voice it; push a public GitHub repo; Devpost submission 10-29. See `docs/submission/CHECKLIST.md`.

## Rules

- Read `docs/AGENT_BRIEF.md` → `docs/STATUS.md` → this file in order before starting work.
- Design snapshots live in `docs/plans/YYYY-MM-DD-<topic>.md`.

### Automation Tags (for overnight loop)

- `[auto]` — verifiable via local, deterministic, offline gate (`make check`).
- `[manual]` — human judgment, hardware, or external service. Skipped by unattended loop.
- `[blocked]` — unmet dependencies or 2 accumulated Blockers (runner marks automatically).
- **No Tag = Not for unattended run.**
