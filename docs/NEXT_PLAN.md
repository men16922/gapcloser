# Next Plan

Last Updated: 2026-10-05

Rolling plan containing open work only. Completed history is in `docs/COMPLETED_SUMMARY.md`.

## Priority 0 — Tier 0 core (10/6–10/12)

- [ ] [manual] User: Nebius Builder signup + hackathon credit form + `NEBIUS_API_KEY` (`docs/setup/NEBIUS_SETUP.md`).
- [ ] [manual] Token Factory live check: `python -m agent.llm --provider tokenfactory --models`, then `--ping`; re-record demo with `--llm tokenfactory` (Devpost requires Token Factory in the running app).

- [ ] [manual] With Token Factory: check whether Nemotron 3 Nano Omni accepts images; if yes pass `frames=lambda: extract_frames(real_clip, ...)` to `LLMDiagnoser` in `eval/record_demo.py` (extractor and vision role already implemented and tested).

## Priority 1 — Demo quality (10/13–10/19, freeze 10/19)

- [ ] [manual] Review dashboard (artifact) and decide video storyboard around the weak-motor scenario.

- [ ] [manual] Live LLM runs: 10 hidden worlds × {heuristic, llm}; publish table.

## Priority 2 — Submission (10/20–10/29)

- [ ] [manual] Optional extras if time: Cosmos Reason (build.nvidia.com/AWS), Franka arm instead of initial-velocity push, Isaac Lab demo on AWS.
- [ ] [manual] 3-min video, README, LICENSE, Devpost text; submit 10-29.

## Rules

- Read `docs/AGENT_BRIEF.md` → `docs/STATUS.md` → this file in order before starting work.
- Design snapshots live in `docs/plans/YYYY-MM-DD-<topic>.md`.

### Automation Tags (for overnight loop)

- `[auto]` — verifiable via local, deterministic, offline gate (`make check`).
- `[manual]` — human judgment, hardware, or external service. Skipped by unattended loop.
- `[blocked]` — unmet dependencies or 2 accumulated Blockers (runner marks automatically).
- **No Tag = Not for unattended run.**
