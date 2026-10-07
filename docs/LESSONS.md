# Lessons

Last Updated: 2026-10-08

Repo-specific facts an agent learned the hard way, one per line, newest first. Read before
starting work; append when a blocker, gate failure, or repair taught you something the docs did
not say. Keep it under 40 lines — `/tidy-docs` compresses older entries into `docs/DECISIONS.md`
or `docs/COMPLETED_SUMMARY.md` when they stop being surprises.

Format (one line each, no prose paragraphs):

```text
- YYYY-MM-DD <area>: <what is true and how you found out>. Applies to: <path/command>.
```

Not for: task status (that is `docs/PROGRESS_LOG.md`), open work (`docs/NEXT_PLAN.md`), or
decisions with rationale (`docs/DECISIONS.md`).

## Entries

- 2026-10-08 benchmarks: a 6-worlds/tier run showed the LLM agent beating sysID (99% vs 93%); at 15/tier it vanished (96% vs 97%). Never claim a gap from n<=6. Applies to: eval/open_bench.py, README/DEVPOST claims.
- 2026-10-08 tool-agent: Nemotron hand-tuning numbers never committed; splitting "LLM picks structure, least-squares fitter fits numbers" (`fit_hypothesis`) fixed it. Applies to: agent/tool_agent.py.
- 2026-10-08 evidence: launch speed must be compared with the sim's own first-frame measure, not command*gain (gain read 0.95 for truth 1.0). Applies to: agent/tool_agent.py Workbench.loss.
- 2026-10-08 newton: a seam between two table bodies makes cubes tip; friction strips are done by switching the cube's own mu with ground mu 0 (XPBD averages the two). Applies to: sim/newton_push.py `_patch_setup`.
- 2026-10-08 newton: `tipped` from the final pose misses a double roll (ends upright); use peak tilt over frames. Applies to: sim/newton_push.py, sim/replay.py.
- 2026-10-08 params: new fixed params must not draw from the RNG (`Randomization.sample`) or the closed-world benchmark changes; verified by identical `eval.compare --json`. Applies to: sim/params.py.
- 2026-10-08 cosmos: raw 480x270 clips (cube ~10 px) give 17-33% tip recall; tracked 448 px crops + per-frame "tilted?" + no `<think>` give 88% overall, ~50% held-out recall; video input and thinking were worse. Applies to: agent/cosmos_eyes.py, spike/cosmos/README.md.
- 2026-10-08 headless Chrome: window cannot go below ~485 px, so check phone layout with CDP device emulation (scratchpad `cdp_probe.py` pattern), not --window-size. Applies to: dashboard visual checks.
