# Next Plan

Last Updated: 2026-10-11

Rolling plan containing open work only. Completed history is in `docs/COMPLETED_SUMMARY.md`; the S-level plan and its
progress table are in `docs/plans/2026-10-11-s-level-plan.md`.

## Priority 0 — Submission (deadline 2026-10-30 10:00 PDT, target 10-29)

- [ ] [manual] Film review: the user watches `video/out/tether_film.mp4`; re-render with `make film` after changes.
- [ ] [manual] Upload the film to YouTube or Vimeo (public or unlisted) for the Devpost video URL.
- [ ] [manual] Devpost: paste `docs/submission/DEVPOST.md`, add the video URL, https://github.com/men16922/tether and the live demo https://tether-454741001655.us-central1.run.app, choose the Physical AI track.

## Priority 1 — Engineering

- [ ] [auto] W4 engineering (rest): server-side unit conversion, single domain source, studio.html modules.
- [ ] [auto] Studio: 3D before/after on the results step (W3 open item).
- [ ] [auto] Tool-agent ablations on analytic env with RecordedLLM fixtures (no probes / no fit tool); Done: offline tests pass.
- [ ] [auto] Tipping-aware planning: when ≥30% of real trials tip, propose lowering push speed (policy ceiling) instead of leaving the gap; Done: Newton-marked test on table μ 1.18 improves over the current result.
- [ ] [manual] Optional: own real recording (3 object–surface pairs, A4 sheet, tilt-test ground truth; `make prove-real`).

## Rules

- Read `docs/AGENT_BRIEF.md` → `docs/STATUS.md` → this file in order before starting work.
- Design snapshots live in `docs/plans/YYYY-MM-DD-<topic>.md`.
- Tags: `[auto]` verifiable by the offline gate (`make check`); `[manual]` needs the user, hardware or an external service.
