# Next Plan

Last Updated: 2026-10-05

Rolling plan containing open work only. Completed history is in `docs/COMPLETED_SUMMARY.md`.

## Priority 0 — Tier 0 core (10/6–10/12)

- [ ] [manual] User: Nebius Builder signup + hackathon credit form + `NEBIUS_API_KEY` (`docs/setup/NEBIUS_SETUP.md`).
- [ ] [manual] Token Factory live check: `python -m agent.llm --provider tokenfactory --models`, then `--ping`; re-record demo with `--llm tokenfactory` (Devpost requires Token Factory in the running app).

- [ ] [manual] With Token Factory: check whether Nemotron 3 Nano Omni accepts images; if yes pass `frames=lambda: extract_frames(real_clip, ...)` to `LLMDiagnoser` in `eval/record_demo.py` (extractor and vision role already implemented and tested).

- [ ] [manual] Deploy the live demo: Hugging Face Space (free) with `NEBIUS_API_KEY` secret, or a Nebius CPU VM (~$0.06/h, user decision). Guide: `docs/deploy/DEPLOY.md`. Image builds and runs locally (verified).

## Priority 1 — Demo quality (10/13–10/19, freeze 10/19)

- [ ] [manual] Review dashboard (artifact) and decide video storyboard around the weak-motor scenario.


## Priority 2 — Submission (10/20–10/29)

- [ ] [manual] Optional extras if time: Cosmos Reason (build.nvidia.com/AWS), dynamic Franka push (arm currently kinematic in clips), Isaac Lab demo on AWS (needs GPU quota request + approval).
- [ ] [auto] Tipping-aware planning: when ≥30% of real trials tip, propose lowering push speed (policy ceiling) instead of leaving the gap; Done: Newton-marked test on table μ 1.18 improves over the current result.
- [ ] [manual] Record the 3-min video from the dashboard (`docs/submission/VIDEO_STORYBOARD.md`), finalize `docs/submission/DEVPOST.md`, push to a public GitHub repo, submit 10-29. README and LICENSE are done.

## Rules

- Read `docs/AGENT_BRIEF.md` → `docs/STATUS.md` → this file in order before starting work.
- Design snapshots live in `docs/plans/YYYY-MM-DD-<topic>.md`.

### Automation Tags (for overnight loop)

- `[auto]` — verifiable via local, deterministic, offline gate (`make check`).
- `[manual]` — human judgment, hardware, or external service. Skipped by unattended loop.
- `[blocked]` — unmet dependencies or 2 accumulated Blockers (runner marks automatically).
- **No Tag = Not for unattended run.**
