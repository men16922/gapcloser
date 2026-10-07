# Devpost draft — GapCloser

Status: draft 2026-10-08. Numbers come from `runs/bench/open_newton_super_n15.json` (Gap-Bench, Newton, Nemotron 3 Super on Nebius Token Factory), `runs/bench/open_newton_{nano,lightning,ultra}.json` and `runs/demo/bundle.json`.

## Tagline

An agent that fixes the simulator when a robot fails in the real world, and shows its work.

## Inspiration

A policy that works 98% of the time in simulation fails on the real robot, because the simulator is wrong in ways nobody wrote down. Typical examples are a slippery strip on the table, a weaker motor, or a camera lens that bends distances. Engineers close this gap by hand. They watch the failures, guess a cause, edit the simulator, retrain and try again, and each cycle takes days. Tuning scripts only adjust the parameters someone thought of in advance. We wanted an agent that runs the cycle like a scientist: it forms hypotheses, runs experiments, fixes the simulator and explains why.

## What it does

GapCloser trains a push policy in **NVIDIA Newton**, runs it in a second Newton world whose physics are hidden from the agent, and measures where all 20 pushes stop. When pushes miss, the **Nemotron 3 Super** agent on **Nebius Token Factory** investigates with tools:

- **Look:** it reads a deceleration profile from camera tracks and a perception check of perceived vs true target positions.
- **Hypothesize and fit:** it proposes simulator *structures*, such as "uniform friction + weaker motor", "+ a friction strip from 0.40 m" or "+ lens distortion". A least-squares fitter returns the numbers and residuals for each. The LLM chooses the structure and the optimizer does the arithmetic.
- **Experiment:** when the real data does not cover the target range, it spends a few extra real pushes where the data is missing.
- **Commit:** it hands over a new simulator config with a plain-language explanation. The policy is retrained and measured again until 90% of pushes land within ±3 cm.

**NVIDIA Cosmos Reason 2**, running locally, watches the real camera clips and reports events such as "slid 0.17 s → tipped 0.69 s" as a second opinion next to the physics.

The agent console shows the whole investigation:

- a lab notebook of every tool call, with charts and accepted or rejected hypotheses
- a 3D replay of each push (a Franka arm, the real cube, and a ghost of the agent's simulated cube), including the friction strip the agent believes in vs the real one
- the success curve
- a "reveal truth" view that grades the diagnosis

On the live server, visitors can **hide physics themselves** ("Stump the agent") and watch Nemotron work it out.

## How we built it

- **NVIDIA Newton 1.6** (Warp) for physics, camera rendering and per-frame poses, on a laptop CPU. The friction strip is implemented without geometry seams: a small Warp kernel switches the cube's contact friction when it crosses the strip, because a seam between two table bodies made cubes tip.
- **NVIDIA Nemotron 3** via **Nebius Token Factory**, using OpenAI-compatible tool calling. Super is the default agent. We benchmarked Nano, Super, Ultra and 3.5 Lightning.
- **NVIDIA Cosmos Reason 2 8B**, run locally through llama.cpp at zero cost. It looks at eight cropped frames per clip and is asked "is it tilted?" for each, and the events are built in code.
- **Policy:** it inverts the simulator for each target, so the policy is exactly as good as the simulator. That makes the measured success a direct grade of the simulator fix.
- **Gap-Bench:** an open-world benchmark with closed, open and compound tiers, a rule-based baseline and a system-identification baseline, 95% CIs, and a count of real trials.
- **Offline gate:** 67 tests that replay recorded Nemotron tool-call sessions, with no network and no credits. The dashboard needs no build step and the 3D viewer uses three.js.

## Results

Gap-Bench (NVIDIA Newton, 15 hidden worlds per tier, mean real success ± 95% CI):

| Tier | Nominal sim | Domain randomization | Rule-based agent | System-ID baseline | Nemotron tool agent |
|---|---|---|---|---|---|
| Closed (known params) | 46% | 0% | 100% | 100% | 100% |
| Open (friction strip or lens) | 39% | 6% | 83% ±9 | 100% | 100% |
| Compound (strip + lens + 1) | 22% | 14% | 54% ±15 | 97% ±4 | 96% ±6 |

- Rule-based tuning breaks once the world contains an effect it has no parameter for. The Nemotron agent closes those gaps in about two iterations, at about **$0.007 per world** on Token Factory.
- We report an honest strong baseline: a system-identification pipeline with a hand-ordered library of model structures and the same fitter matches the agent. What the agent adds is that nobody has to write that library, its search order or its thresholds. Nemotron decides which structures and experiments to try, and explains each fix.
- A first run with 6 worlds per tier showed the agent ahead on compound worlds (99% vs 93%). That gap disappeared at 15 worlds per tier, so we do not claim it.
- Nemotron family on the open and compound tiers (6 worlds each): Super 100/99%, 3.5 Lightning 100/95% (cheapest), Ultra 100/83%, Nano 97/53%. Bigger was not better here.

**Example (Three faults).** The hidden world had a weak motor, a slick strip from 0.40 m and lens distortion.

1. The agent's first model fit perfectly, but the real pushes only reached 0.40 m.
2. It spent 3 probe pushes beyond that point.
3. The model then missed by 5.75 cm, so it added a friction strip at 0.40 m with μ 0.50, and the residual fell to 0.06 cm.
4. Retrained in that simulator, the policy went from 0% to 100%.

## Challenges

- **Making the LLM useful rather than decorative.** When it tuned numbers by hand it never committed. Splitting the work so the LLM picks the structure and an optimizer fits the numbers fixed that.
- **Honest evaluation.** Our first small benchmark suggested the LLM beat system identification. More worlds showed it does not, and we kept the stronger baseline in the results.
- **Real physics surprises.** At effective friction near 1 the cube tips over instead of sliding. A seam between two table bodies also tipped cubes, which we solved with a friction-switching kernel. We also found a double roll that ended upright and so slipped past the tipped flag; we fixed it by using peak tilt.
- **Cosmos on tiny objects.** The cube is about 10 px wide in the clips. Tracking crops and per-frame yes/no questions raised slid-vs-tipped accuracy to 88%. Recall on unseen tips is still about 50%, so Cosmos is a second opinion and never replaces the physics.

## What's next

- Open-ended structures beyond our tool schema, such as several strips or non-linear actuators, where a fixed library no longer scales.
- Dynamic Franka contact instead of an impulse push, and GR00T policies.
- Run the same loop against real hardware logs.

## Built with

nvidia-newton, nvidia-warp, nemotron, nebius-token-factory, cosmos-reason, llama.cpp, three.js, fastapi, python, openai-python, scipy

## Track

Physical AI. There is no physical hardware; the video shows the application modules in action.
