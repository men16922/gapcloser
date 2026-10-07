# Next Plan

Last Updated: 2026-10-08

Rolling plan containing open work only. Completed history is in `docs/COMPLETED_SUMMARY.md`.

## Priority 0 — Award plan (10/7–10/19), see `docs/plans/2026-10-07-award-plan.md`

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
