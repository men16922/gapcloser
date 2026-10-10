# Lessons

Last Updated: 2026-10-11

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

- 2026-10-11 deploy: Hugging Face Docker Spaces need PRO since 2026 (create_repo → 402); Cloud Run works but needs --no-cpu-throttling (retraining runs between polls) and max 1 instance (sessions in memory); retraining takes ~170 s there vs ~36 s on the Mac (single-threaded Warp CPU, more vCPU won't help).
- 2026-10-09 docker: the VM's legacy builder makes `COPY --chown` parent dirs root-owned (runs/), the app then crashes on mkdir; BuildKit on the Mac hid it. Applies to: Dockerfile.

- 2026-10-09 intervals: a bootstrap over pushes alone gave video intervals that excluded the truth (systematic tracking/scale error is shared by all pushes); each replicate now redraws scale/speed error too. Applies to: studio/fit.py bootstrap.

- 2026-10-09 fitter: scipy Nelder-Mead's default simplex (5% of x0, 0.00025 at 0) left camera_pitch_deg/lens_k stuck at 0; pass initial_simplex with per-field steps. Applies to: agent/tool_agent.py fit_hypothesis.
- 2026-10-09 fitter: a friction region started with patch_mu == mu_eff is flat in patch_y0, so the fit never moves; multistart patch_mu at 0.6x/1.4x. Applies to: agent/tool_agent.py.
- 2026-10-09 real tracks: 1.5 mm camera jitter makes 2-frame finite-difference deceleration useless (±0.3 g) and misled Nemotron; use a 7-frame Savitzky-Golay second derivative and skip windows that reach the stop. Applies to: Workbench.decel_profile.
- 2026-10-09 agent: Nemotron fitted stops by trading friction for actuator gain (stops depend on gain^2/mu) and ignored the launch residual; tool results now carry `unexplained` warnings and Studio cross-checks against a structure library. Applies to: agent/tool_agent.py, studio/pipeline.py.
- 2026-10-09 video: release frame must be refined on raw forward differences; the smoothed-speed peak lands one frame late (+3-6 cm start error). Applies to: studio/video.py segment.
- 2026-10-09 shell: `cat > file` with no heredoc waits on stdin forever and stalls the tool call. Applies to: any Bash command.

- 2026-10-08 benchmarks: a 6-worlds/tier run showed the LLM agent beating sysID (99% vs 93%); at 15/tier it vanished (96% vs 97%). Never claim a gap from n<=6. Applies to: eval/open_bench.py, README/DEVPOST claims.
- 2026-10-08 tool-agent: Nemotron hand-tuning numbers never committed; splitting "LLM picks structure, least-squares fitter fits numbers" (`fit_hypothesis`) fixed it. Applies to: agent/tool_agent.py.
- 2026-10-08 evidence: launch speed must be compared with the sim's own first-frame measure, not command*gain (gain read 0.95 for truth 1.0). Applies to: agent/tool_agent.py Workbench.loss.
- 2026-10-08 newton: a seam between two table bodies makes cubes tip; friction strips are done by switching the cube's own mu with ground mu 0 (XPBD averages the two). Applies to: sim/newton_push.py `_patch_setup`.
- 2026-10-08 newton: `tipped` from the final pose misses a double roll (ends upright); use peak tilt over frames. Applies to: sim/newton_push.py, sim/replay.py.
- 2026-10-08 params: new fixed params must not draw from the RNG (`Randomization.sample`) or the closed-world benchmark changes; verified by identical `eval.compare --json`. Applies to: sim/params.py.
- 2026-10-08 cosmos: raw 480x270 clips (cube ~10 px) give 17-33% tip recall; tracked 448 px crops + per-frame "tilted?" + no `<think>` give 88% overall, ~50% held-out recall; video input and thinking were worse. Applies to: agent/cosmos_eyes.py, spike/cosmos/README.md.
- 2026-10-08 headless Chrome: window cannot go below ~485 px, so check phone layout with CDP device emulation (scratchpad `cdp_probe.py` pattern), not --window-size. Applies to: dashboard visual checks.
