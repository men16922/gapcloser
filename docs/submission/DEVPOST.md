# Devpost draft — Tether

Status: draft 2026-10-10 (renamed from GapCloser; three domains; Newton replay). Numbers come from
`runs/bench/open_newton_super_n15.json` (Gap-Bench), `runs/bench/studio_bench_analytic.json`, `runs/studio-demo/*.json`
(recorded Studio runs with Nemotron) and the coverage study noted in `studio/fit.py`. Demo video: `video/out/tether_demo.mp4`.

## Tagline

Tie your simulator to the real world: measured physics, honest intervals, and configs for NVIDIA Newton, Isaac Lab and CARLA.

## Inspiration

A policy that works 98% of the time in simulation misses in the real world because the simulator is wrong in ways
nobody wrote down: a wet section of road, an oily stretch of rail, a weaker actuator, a camera that misjudges
distance. Engineers close this gap by hand: watch failures, guess a cause, edit the simulator, retrain, repeat. Each
cycle takes days, and tuning scripts only adjust the parameters someone thought of in advance.

The same gap shows up in three industries with the same physics underneath: an object launched toward a target
slides to a stop. A robot pushing a box onto a line, a car braking to a stop line and a pusher sliding a part to an
inspection camera differ in scale and vocabulary, not in the equations. We built one tool for all three.

## What it does

Tether has three connected surfaces: an **overview**, an **agent console** that shows how the AI works, and
**Studio**, where you bring your own data.

**Studio (your data → a calibrated simulator)**

1. **Pick a domain:** robot manipulation, autonomous vehicles or factory inspection.
2. **Bring data:** a phone, roadside or line-camera video (any rectangle of known size gives scale and camera pose),
   or the log your robot, car or pusher already writes. No data? Set the hidden physics yourself and **NVIDIA
   Newton** renders the video or writes the log.
3. **Measure:** the reference rectangle is found automatically (about 1 px on clean video, 3 px on a shaky, blurred,
   compressed one); homography from it gives focal length, height and angle; tracking is parallax-corrected;
   launch speeds come from a local deceleration fit. On Newton-rendered video this is within 1 mm and 0.6%.
4. **Diagnose:** **NVIDIA Nemotron 3 Super** on **Nebius Token Factory** works as a tool-using agent. It profiles
   deceleration along the surface, checks perception, fits candidate *structures* ("uniform friction + a wet section
   from 8.5 m") and asks for the runs that would settle the rest. The language model chooses structures and
   experiments; least squares computes every number. A library search cross-checks the agent and overrules it if its
   model leaves evidence unexplained.
5. **Calibrated sim:** every value with a 90% interval (bootstrap that also redraws measurement error), friction
   painted on your own video, ghost objects replaying each run in the old and the calibrated simulator, predicted
   success before/after, and an explicit "unknown" where nothing was measured.
6. **Retrain and test:** the same learner trains a policy three ways in parallel NVIDIA Newton worlds (each world
   with its own friction, region and actuator through a per-body Warp kernel): on your current simulator, on wide
   domain randomization, and on worlds drawn from Tether's bootstrap ensemble. When the hidden world is known, each
   policy is scored there, and Newton renders the three policies acting side by side.
7. **Verify in NVIDIA Newton:** before export, every measured run is replayed in the full engine, from its own start
   and launch, with the exported physics and with your current simulator.
8. **Export:** NVIDIA Newton materials and friction region, an Isaac Lab `EventTermCfg` whose randomization ranges
   are the measured intervals, and for driving a **CARLA 0.9.16** script (tire friction scaled by measured/simulated
   μ, a `static.trigger.friction` box on the wet section, and a braking-distance check), plus a report and JSON.

**Agent console (how the AI works)** runs the full loop on hidden worlds: train a policy in Newton, measure it in a
world whose physics the agent cannot see, let Nemotron investigate, fix the simulator, retrain. Each example has a
"Run this case yourself in Studio" button that opens Studio's virtual table with the same hidden physics.
**NVIDIA Cosmos Reason 2**, running locally, watches the clips as a second opinion on whether the object slid or tipped.

## How we built it

- **NVIDIA Newton** (Warp, CPU) for physics, camera rendering (SensorTiledCamera) and the replay check. A small Warp
  kernel switches the body's friction when it crosses a region, because a seam between two surface bodies tipped
  sliding objects.
- **Three domains, one engine:** driving uses Froude similarity. Lengths ×25, speeds ×5, friction unchanged, so
  stop = v²/(2μg) holds at both scales. The fit runs at the base scale; pages and exports are full size.
- **NVIDIA Nemotron 3** on **Nebius Token Factory** with OpenAI-compatible tool calling. Super is the default; we
  benchmarked Nano, Super, Ultra and 3.5 Lightning. A chat panel ("Ask Tether") answers questions grounded in the
  visitor's own session, in English or Korean.
- **NVIDIA Cosmos Reason 2 8B** locally through llama.cpp at zero cost.
- FastAPI with Server-Sent Events, a no-build front end (three.js for the 3D replay), 97 offline tests that replay
  recorded Nemotron sessions without network or credits.

## Results

**Studio examples** (hidden physics in NVIDIA Newton, revealed only after the diagnosis; recorded with Nemotron):

| Example | Found (90% interval) | Truth |
|---|---|---|
| Roadside video, braking (11 runs) | road μ 0.738 (0.698–0.769), wet from 8.78 m (8.44–9.12), wet μ 0.274 (0.253–0.300) | 0.75, 8.82 m, 0.30 |
| Vehicle log, braking (18 runs) | road μ 0.783, speed control 0.926×, wet from 9.93 m, wet μ 0.380 | 0.78, 0.93×, 10.0 m, 0.38 |
| Robot log (20 pushes) | μ 0.696, arm 0.872×, region from 0.376 m at μ 0.449, camera 1.9° | 0.70, 0.88×, 0.38 m / 0.45, 2.0° |
| Pusher log (18 strokes) | μ 0.558, pusher 0.911×, oily from 0.281 m (0.276–0.299), oil μ 0.316, camera −1.6° | 0.55, 0.92×, **0.300 m**, 0.32, −1.5° |

- 19 of the 21 hidden values in the six examples fall inside their 90% intervals; the two misses are by 0.9 mm (oily
  section start) and 0.001 (robot arm gain). Across 24 random hidden worlds (118 values) the 90% intervals contain the
  truth 92% of the time; before we added a 1% length-scale term for logs it was 82%. `make prove` recomputes all of
  this, plus the replay and retraining tables, in about five minutes with no network (`runs/proof/PROOF.md`).
- **Replay in NVIDIA Newton:** with the exported physics, rms stop error over the six examples is 6–9 mm at robot
  scale; the uncalibrated simulators are off by 3–25 cm. On the roadside example at full scale: 22 cm vs 2.7 m.
- **Retrain and test** (policy learned by trial in 16 parallel Newton worlds × 11 table points, 10 iterations; success
  in the hidden world, 24 targets):

  | Example | Current sim | Wide randomization | Tether's ranges |
  |---|---|---|---|
  | Vehicle log | 42% | 0% | **100%** |
  | Robot phone video | 0% | 13% | **100%** |
  | Robot log | 63% | 0% | **100%** |
  | Pusher log | 0% | 38% | **100%** |
  | Roadside video | 29% | 8% | **58%** |
  | Short pushes only | 0% | 25% | **46%** |

  The two below 100% are honest: the short log never measured the far table (Studio says so before you train), and
  on the roadside video the true wet-section friction sits at the edge of its interval. Wide randomization trains a
  policy that is mediocre everywhere; the current simulator is confidently wrong.
- **Phone robustness** (same hidden table filmed clean vs hand-held with motion blur, exposure flicker and heavy
  compression; automatic corners, tracking and fit end to end): 11 of 11 pushes found in both; launch speed reads
  0.8% low clean and 6.5% low on the bad video; on the bad video the intervals widen to cover the truth instead of
  narrowing around a wrong value (3/3 inside), on the clean one the region friction misses its interval by 0.004.
- **Next experiment:** reaching 95% real success takes 5.8 real runs with Tether's suggestions vs 7.6 random, about
  the same as a well-designed manual sweep (5.9).

**Gap-Bench** (agent console; NVIDIA Newton, 15 hidden worlds per tier, mean real success ± 95% CI):

| Tier | Nominal sim | Domain randomization | Rule-based | System-ID baseline | Nemotron tool agent |
|---|---|---|---|---|---|
| Closed (known params) | 46% | 0% | 100% | 100% | 100% |
| Open (friction region or lens) | 39% | 6% | 83% ±9 | 100% | 100% |
| Compound (region + lens + 1) | 22% | 14% | 54% ±15 | 97% ±4 | 96% ±6 |

- Rules break once the world has an effect they have no parameter for. The agent closes those gaps for about
  $0.007 per world on Token Factory.
- We keep an honest strong baseline: system identification with a hand-ordered structure library matches the agent.
  What the agent adds is that nobody writes that library, its order or its thresholds; it chooses experiments and
  explains each fix. An early 6-world run suggested the agent was ahead on compound worlds; at 15 worlds that gap
  disappeared, so we do not claim it.
- Nemotron family (open + compound, 6 worlds each): Super 100/99%, 3.5 Lightning 100/95% (cheapest), Ultra 100/83%,
  Nano 97/53%. Bigger was not better.

## Challenges

- **Making the LLM useful rather than decorative:** when it tuned numbers by hand it never committed. Structure from
  the LLM, numbers from the optimizer fixed that.
- **Intervals that do not lie:** our first video intervals excluded the truth because they ignored systematic camera
  error; log intervals covered only 82%. We measured coverage and fixed both.
- **One engine, three domains:** a car is not a box on a table, but braking with locked or ABS-limited wheels is
  Coulomb sliding. Froude scaling let the same Newton scene and fitter serve driving at full-size units.
- **Physics surprises:** cubes tipping at μ≈1, a seam tipping cubes (solved with a friction-switching kernel), a
  double roll that ended upright and hid a tip (solved with peak tilt).
- **Cosmos on tiny objects:** about 10 px wide; crops and per-frame questions got slid-vs-tipped to 88%, so it stays
  a second opinion.

## Limits we state plainly

- All data so far is synthetic (Newton-rendered video and Newton logs) so the truth can be revealed. A real phone video
  goes through the same path but has not been validated yet.
- The physics is one family: launched, sliding to a stop. No grasping, steering or lane changes.
- The CARLA export was checked against CARLA 0.9.16's documented API and run against a stand-in module, not inside
  CARLA itself. GR00T is not used.

## What's next

- Real phone and vehicle data; more structures (several regions, non-linear actuators); dynamic Franka contact and
  GR00T policies trained on the calibrated, randomized simulator.

## Built with

nvidia-newton, nvidia-warp, nemotron, nebius-token-factory, cosmos-reason, isaac-lab, carla, llama.cpp, three.js,
fastapi, python, openai-python, scipy, opencv

## Track

Physical AI. There is no physical hardware; the video shows the application modules in action (overview, Studio
on the driving domain, Newton replay, retraining and the three policies braking side by side, exports, factory
domain, agent console).

## How it maps to the judging criteria

| Criterion | Where to look |
|---|---|
| Technical implementation | Nemotron tool agent + cross-check, bootstrap intervals with measured coverage, Newton replay of the export, retraining in parallel Newton worlds scored in the hidden world, Froude-scaled driving, 97 offline tests |
| Design | One route: overview → console → Studio → export; friction painted on the visitor's own video; ghost replays; EN/KO |
| Potential impact | Same tool for robots, vehicles and production lines; exports into Newton, Isaac Lab and CARLA |
| Quality of the idea | An agent that fixes the simulator instead of the policy, and says how sure it is |
