# Next Plan

Last Updated: 2026-10-11

Rolling plan containing open work only. Completed history is in `docs/COMPLETED_SUMMARY.md`; the S-level plan and its
progress table are in `docs/plans/2026-10-11-s-level-plan.md`.

## Priority 0 — Submission (deadline 2026-10-30 10:00 PDT, target 10-29)

- [x] [auto] Upload folder `submission/` (`make submission`): film 2:28 with ~68 s of the app in action, YouTube thumbnail and description, Devpost gallery and About text; steps in `docs/submission/SUBMIT.md`.
- [ ] [manual] Owner: upload the film to YouTube (public), then submit on Devpost following `docs/submission/SUBMIT.md`.
- [ ] [manual] Film feedback from the owner, if any: re-render with `make film`, then `make submission`.

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
