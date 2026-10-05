# 3-minute video storyboard

Source: the dashboard (`make demo`, open `dashboard/dist/gapcloser.standalone.html`), screen-recorded at 1440p. Track rule: with no hardware, show "the key application modules in action".

| Time | Screen | Voice-over (draft) |
|---|---|---|
| 0:00–0:15 | Weak motor tab, it 0 selected: sim clip lands on the line, real clip stops short. KPI "Before 0%". | "This push policy is perfect in simulation. In the real world, every push falls short." |
| 0:15–0:35 | Stop-error chart: green band, red real dots below, grey sim circles on zero. | "Same commands, different physics. Something in the simulator is wrong, but what?" |
| 0:35–1:05 | Press Replay run. Timeline fills: Train → Measure → Diagnose. Zoom on Nemotron's reasoning quote and the suspect bar `actuator_gain ↓ 0.76×`. | "GapCloser hands the evidence to Nemotron: the cube launched 24% slower than commanded while it decelerated normally. That is the motor, not the friction." |
| 1:05–1:25 | Plan card: config diff `actuator_gain 0.76× ± 0.03×`. Train it 1. | "The agent rewrites the simulator and retrains." |
| 1:25–1:45 | Measure it 1: real 20/20, success chart jumps to 100%, clips both on the line. | "Measured in the real world again: 100%." |
| 1:45–2:05 | Reveal truth: triangle lands inside the belief band. KPI "Outcome-only agent 0%". | "The hidden truth was 0.76. An agent that only looks at where the cube stopped blames friction and never recovers." |
| 2:05–2:35 | Benchmark tab: tiles 18% / 24% / 89% / 94%, precision 0.43 vs 1.00. | "Across ten random worlds, GapCloser beats domain randomization and the nominal simulator, with every diagnosis correct." |
| 2:35–3:00 | Architecture (README mermaid) + stack chips: NVIDIA Newton, Nemotron via Nebius Token Factory. | "Newton for physics, Nemotron on Nebius Token Factory for reasoning, all on a laptop. Days of manual sim tuning become one agent loop." |

Checklist before recording:
- [ ] Re-record with `--llm tokenfactory` so the chips and timeline show Token Factory.
- [ ] Larger font zoom (browser 125%) for legibility.
- [ ] Mention the tipping limitation only if asked; it is documented in the README.
