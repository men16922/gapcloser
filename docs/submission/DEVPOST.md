# Devpost draft — Tether

Status: final 2026-10-11. Numbers come from `runs/proof/*.json` (`make prove`), `runs/bench/open_newton_super_n15.json`
(Gap-Bench) and `runs/studio-demo/*.json` (recorded Studio runs with Nemotron). Demo film: `film/out/tether_film.mp4`
(2:28). Live demo: https://tether-454741001655.us-central1.run.app · Code: https://github.com/men16922/tether.
How to fill in each Devpost field: `docs/submission/SUBMIT.md`.

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

Tether has three connected pages: an **Overview**, a **Benchmark** that tests the AI on worlds whose answer is known, and
**Studio**, where you bring your own data.

**Studio (your data → a calibrated simulator)**

1. **Pick a domain:** robot manipulation, autonomous vehicles or factory inspection.
2. **Bring data:** a phone, roadside or line-camera video (any rectangle of known size gives scale and camera pose),
   or the log your robot, car or pusher already writes. No data? Set the hidden physics yourself and **NVIDIA
   Newton** renders the video or writes the log.
3. **Measure:** the reference rectangle is found automatically (on Newton-rendered video: about 1 px clean, 3 px shaky,
   blurred and compressed); homography from it gives focal length, height and angle; tracking is parallax-corrected;
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

**Benchmark (how the AI is tested)** runs the full loop on hidden worlds: train a policy in Newton, measure it in a
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
- **NVIDIA NeMo Agent Toolkit:** the calibration agent also runs as a NAT workflow (`nat run`, `nat serve`): NAT
  holds the LLM config and every tool call of the agent lands in NAT's event stream for profiling and tracing.
- **MuJoCo** as a second, independent engine for hidden worlds (contact-driven launch, soft contacts), so Tether is
  not only scored on data made by the model family it fits.
- **NVIDIA Cosmos Reason 2 8B** locally through llama.cpp at zero cost.
- FastAPI with Server-Sent Events, a no-build front end with one shared design system (three.js for the 3D replay),
  offline tests that replay recorded Nemotron sessions without network or credits.

## Results

**Studio examples** (hidden physics in NVIDIA Newton, revealed only after the diagnosis; recorded with Nemotron):

| Example | Found (90% interval) | Truth |
|---|---|---|
| Roadside video, braking (11 runs) | road μ 0.740 (0.720–0.750), wet from 8.60 m (8.36–8.94), wet μ 0.294 (0.283–0.303) | 0.75, 8.82 m, 0.30 |
| Vehicle log, braking (18 runs) | road μ 0.786, speed control 0.927×, wet from 9.93 m, wet μ 0.381, camera 1.3° | 0.78, 0.93×, 10.0 m, 0.38, 1.2° |
| Robot log (20 pushes) | μ 0.701, arm 0.875×, region from 0.377 m at μ 0.453, camera 2.3° | 0.70, 0.88×, 0.38 m / 0.45, 2.0° |
| Pusher log (18 strokes) | μ 0.559, pusher 0.917×, oily from 0.290 m (0.281–0.300), oil μ 0.319, camera −1.2° | 0.55, 0.92×, 0.300 m, 0.32, −1.5° |
| Short pushes only (8) | μ 0.609, arm 1.00×. Nemotron proposed a region at 0.25 m; the cross-check left it out (0.8 mm better, within noise) | 0.60; region from 0.33 m, never reached |

- 21 of the 22 hidden values in the six examples fall inside their 90% intervals; the miss is the pusher log's oily
  section, estimated to start at 0.291 m (truth 0.300 m), whose interval stops 0.1 mm short. Across 24 random hidden worlds (118 values)
  the 90% intervals contain the truth 97% of the time; before we added a 1% length-scale term for logs it was 82%. That term was tuned on these
  worlds, so we checked it on fresh ones made by a different engine (MuJoCo, error model frozen): 88% of 120 values.
  `make prove` recomputes all of this, plus the replay and retraining tables, with no network (`runs/proof/PROOF.md`).
- **Replay in NVIDIA Newton:** with the exported physics, rms stop error is 4–8 mm on the four table-scale examples,
  where the uncalibrated simulators are off by 3–25 cm; on the two driving examples at full size, 18–27 cm vs 1.1–2.7 m
  (`runs/proof/PROOF.md`). This replays in the engine that generated the data, so it is a self-consistency check of the
  export, not an independent validation.
- **Retrain and test** (policy learned by trial in 16 parallel Newton worlds × 11 table points, 10 iterations; success
  in the hidden world, 24 targets):

  | Example | Current sim | Wide randomization | Tether's ranges |
  |---|---|---|---|
  | Vehicle log | 42% | 0% | **100%** |
  | Robot phone video | 0% | 13% | **100%** |
  | Robot log | 63% | 0% | **100%** |
  | Pusher log | 0% | 38% | **100%** |
  | Roadside video | 29% | 8% | **96%** |
  | Short pushes only | 0% | 25% | **46%** |

  The short log stays below 100% for an honest reason: it never measured the far table, and Studio says so before you
  train. Wide randomization trains a
  policy that is mediocre everywhere; the current simulator is confidently wrong.
- **Phone robustness** (same hidden table filmed clean vs hand-held with motion blur, exposure flicker and heavy
  compression; automatic corners, tracking and fit end to end): 11 of 11 pushes found in both; launch speed reads
  0.8% low clean and 6.5% low on the bad video; the hidden values fall inside their intervals 3 of 3 on the clean video
  and 2 of 3 on the bad one.
- **Real footage** (52 public iPhone slow-motion clips of objects sliding to a stop on six real surfaces, with
  dynamometer friction; IDPP real split, Apache-2.0; `python -m tether.eval.real_friction --download`): Tether's tracker
  follows 45 of them, and constant deceleration (the Coulomb sliding Tether fits) explains each slide with median
  R² 0.9985. The clips have no size reference and no stated slow-motion factor, so friction itself is only known up
  to one constant per object and camera: calibrated on the other surfaces, the predictions correlate with the
  dynamometer (r 0.70) but are not more accurate than guessing the mean (MAE 0.068 vs 0.058). That is the reason
  Studio asks for a sheet of known size and the frame rate.
- **A different engine makes the data** (`make cross-engine`; MuJoCo, 24 random hidden worlds per condition, the
  robot's first-day log, policies run in the hidden world, 24 targets ±3 cm):

  | Hidden world (MuJoCo) | Current sim | Best-width DR | DR over the true distribution* | Exact parameters | **Tether** (95% CI) |
  |---|---|---|---|---|---|
  | In-menu effects | 9% | 12% | 25% | 82% | **89%** (85–92) |
  | + speed-dependent friction | 10% | 12% | 26% | 58% | **83%** (75–92) |
  | + a second friction region | 10% | 13% | 25% | 85% | **83%** (76–90) |
  | + a 1.5–3° tilt | 8% | 9% | 26% | 33% | **90%** (87–93) |

  \* a DR baseline no user could build. Tether is at least as good as the best DR in 94 of 96 worlds. Held-out stops:
  1.6–2.0 cm vs 18–30 cm. The new pattern check names what the model misses instead of thresholding its size: it
  reports speed dependence in 24 of 24 worlds that have it, a change along the surface in 15 of 24 two-region
  worlds, and raises 2 false alarms on 24 in-menu worlds (the old 1 cm rule flagged 23).
- **Real objects, real video, independently measured friction** (EV-RealPhys, a public MPI benchmark: YCB objects
  pushed across a real table at 30 Hz; each object's friction measured separately with a tilted table, never shown to
  Tether):

  | Object | Tilt test | Tether, motion-capture log | Tether, RGB video |
  |---|---|---|---|
  | mug | 0.110 | 0.115 | 0.120 |
  | mustard bottle | 0.159 | 0.156 | 0.164 |
  | bleach cleanser | 0.169 | 0.159 | 0.181 |
  | pitcher | 0.220 | 0.216 | 0.229 |
  | cracker box | 0.280 | 0.208 | 0.173 |

  4 of 5 objects within ±0.05 on both paths; mean error 0.019 (log) and 0.029 (video), against 0.082 for the paper's
  own estimator on the same sequences. The cracker box misses on every path, and on real data the 90% intervals are
  too narrow (3 of 5 and 2 of 5 contain the tilt-test value). The benchmark changed Tether: the fit now reads how long
  each slide takes to stop (a jittery clock made launch speeds 20–30% off), and tracks that jump or run backwards are
  dropped with a note.
- **Real footage, friction law:** on the 45 tracked real slides, Coulomb sliding (Tether's model) beats a viscous
  law decisively on 35 and is the best of three laws on 36 (scale-free BIC).
- **Next experiment:** reaching 95% hidden-world success takes 5.76 runs with Tether's suggestions vs 6.84 random, the
  same as a well-designed manual sweep (5.76): the suggestions match an expert's sweep without knowing the table.

**Gap-Bench** (Benchmark page; NVIDIA Newton, 15 hidden worlds per tier, mean hidden-world success ± 95% CI):

| Tier | Nominal sim | Domain randomization | Rule-based | System-ID baseline | Nemotron tool agent |
|---|---|---|---|---|---|
| Closed (known params) | 46% | 0% | 100% | 100% | 100% |
| Open (friction region or lens) | 39% | 6% | 83% ±9 | 100% | 100% |
| Compound (region + lens + 1) | 22% | 14% | 54% ±15 | 97% ±4 | 96% ±6 |

- Rules break once the world has an effect they have no parameter for. The agent closes those gaps for about
  $0.007 per world on Token Factory.
- We keep an honest strong baseline: system identification with a hand-ordered structure library matches the agent.
  The agent chooses from the same structure menu without a hand-set order or thresholds, ties System-ID, and uses
  5–9% more hidden-world pushes (it probes); what it adds today is the experiments it asks for and the explanation of each fix. An early 6-world run suggested the agent was ahead on compound worlds; at 15 worlds that gap
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

- Most results use simulated data (Newton, and MuJoCo as a second engine) so the truth can be revealed. On real
  objects (EV-RealPhys) friction lands within ±0.05 for 4 of 5, with the dataset's camera calibration standing in for
  the A4 sheet; our own sheet-referenced phone recording is not in the results yet. Intervals are too narrow on real data.
- The physics is one family: launched, sliding to a stop. No grasping, steering or lane changes.
- The CARLA export was checked against CARLA 0.9.16's documented API and run against a stand-in module, not inside
  CARLA itself. GR00T is not used.

## What's next

- Real phone and vehicle data; more structures (several regions, non-linear actuators); dynamic Franka contact and
  GR00T policies trained on the calibrated, randomized simulator.

## Built with

nvidia-newton, nvidia-warp, nemotron, nebius-token-factory, nemo-agent-toolkit, cosmos-reason, mujoco, llama.cpp, three.js,
fastapi, python, openai-python, scipy, opencv. Export targets (generated, not executed here): Isaac Lab, CARLA.

## Track

Physical AI. There is no physical hardware; the video shows the application modules in action for over a minute
(Studio on the driving domain: tracking, the Nemotron agent's notebook, the calibrated simulator, the Newton replay
check and the export; Ask Tether on the live server), plus retraining in parallel NVIDIA Newton worlds and the
validation on real objects (EV-RealPhys) and on worlds from a different engine (MuJoCo).

## How it maps to the judging criteria

| Criterion | Where to look |
|---|---|
| Technical implementation | Nemotron tool agent + cross-check (also as a NeMo Agent Toolkit workflow), bootstrap intervals with measured coverage, a pattern check for what the model misses, Newton replay of the export, retraining in parallel Newton worlds scored in the hidden world, a second engine (MuJoCo) for hidden worlds, Froude-scaled driving, offline tests |
| Design | One design system on three pages; Studio in three steps (Data → Analysis → Results) with one headline number; friction painted on the visitor's own video; ghost replays; EN/KO |
| Potential impact | Same tool for robots, vehicles and production lines; exports into Newton, Isaac Lab and CARLA |
| Quality of the idea | An agent that fixes the simulator instead of the policy, and says how sure it is |
