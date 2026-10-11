# Fix the simulator, not the policy

*How Tether uses NVIDIA Nemotron on Nebius Token Factory and NVIDIA Newton to find the physics a simulator gets wrong, and to say how sure it is.*

![Tether](https://raw.githubusercontent.com/men16922/tether/main/docs/article/img/cover.jpg)

**Live demo:** https://tether-454741001655.us-central1.run.app (Studio at `/studio`) · **Code (Apache-2.0):** https://github.com/men16922/tether · **Demo video (2:28):** YOUTUBE_LINK

Built for the Nebius × NVIDIA Global AI Hackathon, Physical AI track.

---

## A policy that is 98% right in the simulator

A braking controller is trained in simulation until it stops at the line nearly every time. On the real road it overshoots. Nothing is wrong with the controller. The road is wet from about 8.6 m onward, and nobody told the simulator.

This is the Sim2Real gap, and the usual fixes are slow or blunt:

- **By hand:** watch the failures, guess a cause, edit the simulator, retrain, and repeat. Each cycle takes days, and a tuning script only adjusts the parameters someone thought of in advance.
- **Domain randomization (DR):** randomize the physics widely and hope the policy becomes robust. In our tests, wide DR trained policies that were mediocre everywhere, scoring 0–38% in the hidden real world.

Tether takes a third route. It **measures** what is different, gives each value an interval it has checked, and only then retrains, randomizing over what the data still allows.

![Same policy, different physics](https://raw.githubusercontent.com/men16922/tether/main/docs/article/img/gap.jpg)

## One physics family, three industries

We picked one problem and found it in three industries: something is launched toward a target and slides to a stop.

| Domain | Scene | What Tether finds |
|---|---|---|
| Robot manipulation | an arm pushes a box onto a line | table friction, a wet patch, arm strength, camera tilt |
| Autonomous vehicles | a car brakes to a stop line | tire-road friction, a wet or icy section, speed control, camera pitch |
| Factory inspection | a pusher slides a part to a camera | part-rail friction, an oily section, pusher strength, camera tilt |

The equation underneath is the same: stop distance = v² / (2μg). A braking car with locked or ABS-limited wheels is Coulomb sliding, just like a box on a table.

To run all three on one engine, driving uses Froude similarity. Lengths are scaled by 25, speeds by 5, and friction is unchanged, so v²/(2μg) holds at both scales. The fit runs at table scale; the pages and exports show full-size metres.

## Architecture at a glance

![Tether pipeline](https://raw.githubusercontent.com/men16922/tether/main/docs/article/img/pipeline-en.png)

- **Physics, rendering and checks:** NVIDIA Newton (Warp, on CPU). It renders the synthetic videos, replays every measured run, and hosts the parallel training worlds.
- **Reasoning:** NVIDIA Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`) on Nebius Token Factory, via the OpenAI-compatible API with tool calling. The same agent also runs as an NVIDIA NeMo Agent Toolkit workflow (`nat run`, `nat serve`).
- **Server and front end:** FastAPI with Server-Sent Events, and a no-build front end (three.js for the 3D replay). It is deployed on Google Cloud Run.
- **Optional eyes:** NVIDIA Cosmos Reason 2 8B, run locally through llama.cpp, as a second opinion on whether an object slid or tipped.

The rule that shaped everything else: **the language model chooses structures and experiments; least squares computes every number.** We arrived at it the hard way. More on that below.

## Step 1. Measure: a camera from a sheet of paper

Studio accepts a phone or roadside video, or the log a robot, car or pusher already writes. For video, the only calibration is a rectangle of known size in view: an A4 sheet on a table, or a painted box on the road.

- **Camera pose:** the rectangle's corners are found automatically. On Newton-rendered video they land within about 1 px on clean footage and 3 px on shaky, blurred, compressed footage. A homography from the corners gives focal length, camera height and angle. On the sample video it recovers 0.481 m and 579 px, against a truth of 0.48 m and 579 px.
- **Tracking:** object positions are corrected for the parallax of the object's own height, then split into separate runs.
- **Release frame:** the smoothed-speed peak lands one frame late, which adds 3–6 cm of start error. We refine the release on the raw forward differences.
- **Deceleration:** with 1.5 mm of camera jitter, a 2-frame finite difference swings by ±0.3 g. That noise misled Nemotron into seeing friction changes that were not there. A 7-frame Savitzky–Golay second derivative fixed it.

On Newton-rendered video, slides come out within 1 mm and launch speeds within 0.6% rms.

![Measuring the runs in Studio](https://raw.githubusercontent.com/men16922/tether/main/docs/article/img/measure.jpg)

## Step 2. Diagnose: Nemotron as a tool-using scientist

Token Factory exposes an OpenAI-compatible endpoint, so the client is ordinary:

```python
from openai import OpenAI

client = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/",
                api_key=os.environ["NEBIUS_API_KEY"])
turn = client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b",
    messages=messages, tools=TOOLS, temperature=0.2)
```

The agent gets six tools:

```python
TOOLS = [
  decel_profile,    # measured deceleration, binned by position along the path
  perception_check, # perceived vs known target positions (camera calibration)
  fit_hypothesis,   # least-squares fit of the chosen free fields of a model structure
  test_hypothesis,  # replay every real command in a candidate model
  probe_real,       # costs real trials: ask for runs at chosen commands
  commit,           # final model + a short explanation
]
```

A model structure is a choice among a few effects: global friction, actuator gain, a region of different friction (where it starts and its friction), camera offset and pitch, and lens distortion.

### Why the LLM does not pick numbers

Our first agent asked Nemotron to propose parameter values and refine them. It never converged. It kept adjusting numbers and never committed. Splitting the job fixed that:

- Nemotron proposes a **structure**, for example "uniform friction plus a wet section", and the fields to free.
- `fit_hypothesis` fits those fields by least squares against every measurement: stop positions, launch motion and perception.
- Nemotron reads the residuals and decides what to try next.

### Telling the agent what is still unexplained

The second failure was subtler. Stop distance depends on gain² / μ, so a model can match every stop by trading friction against actuator strength while getting the launch speeds wrong. Nemotron did exactly that.

Every tool result now carries an `unexplained` list built from generic checks, the same for every world:

```python
if stop_rms > 0.01:
    warn.append("stops not explained (> 1 cm rms): the structure is missing an effect")
if launch_rms > 0.08:
    warn.append("launch speeds disagree with the model: the stops may fit only by trading "
                "friction against actuator_gain (stops depend on gain^2/mu); free actuator_gain")
if perception_rms > 0.003:
    warn.append("perceived target distances not explained: try camera_pitch_deg or lens_k")
```

### What a session looks like

Here are the roadside braking example's agent steps, in the full-size units the notebook shows:

```text
fit_hypothesis  free = [friction]
  -> friction 0.586, stops off by 1.35 m rms
  -> unexplained: stops not explained, the structure is missing an effect
fit_hypothesis  free = [friction, region start, region friction]
  -> friction 0.739, wet from 8.62 m at 0.293, stops off by 9.3 cm rms
probe_real      longer braking runs
  -> recorded data has no robot attached: becomes a "next experiment" card
commit          two-region friction model
```

A session takes 5–32 Nemotron calls depending on the data (22 for this one), and about 40 s on the live server.

### A cross-check that can overrule the agent

After the agent commits, a plain library search fits every structure in a fixed library and keeps the one with the lowest misfit, preferring fewer parameters on ties. If the agent's model leaves evidence unexplained and a library structure explains the data clearly better, the library structure wins, and the page says so.

A friction region the agent adds must also beat the plain model by more than the noise. On a log with only short pushes, Nemotron proposed a slick region at 0.25 m. The cross-check left it out, because it improved the stops by 0.8 mm, which is within noise.

![The agent's notebook in Studio](https://raw.githubusercontent.com/men16922/tether/main/docs/article/img/notebook.jpg)

## Step 3. Calibrate: intervals we checked

Every value comes with a 90% interval from a bootstrap. Our first version resampled only the runs, and its video intervals excluded the truth. Tracking and scale errors are systematic: they are shared by every run, so resampling runs cannot see them.

Each bootstrap replicate now also redraws a systematic error: sheet scale ±1%, tracked speed ±1.5%, and stop ±3 mm.

We measured whether the intervals hold:

- **24 random hidden worlds, 118 values:** the 90% interval contains the truth 97% of the time. Before we added a 1% length-scale term for logs, it was 82%.
- **That term was tuned on those worlds,** so we checked it on fresh worlds from a different engine (MuJoCo, error model frozen): 88% of 120 values.
- **The six Studio examples:** 21 of 22 hidden values fall inside their intervals. The miss is an oily section estimated to start at 0.291 m (truth 0.300 m); its interval stops 0.1 mm short.

A pattern check then looks for what the model still misses after subtracting it: deceleration that changes with speed, or along the surface. It replaced a size threshold that flagged 23 of 24 healthy worlds. On MuJoCo worlds with speed-dependent friction, it reports "depends on speed" in 24 of 24.

![The calibrated simulator](https://raw.githubusercontent.com/men16922/tether/main/docs/article/img/calibrated.jpg)

### The next experiment

Where the plausible models still disagree, Studio says so and names the runs that would settle it. The roadside example shows why that matters:

- **After the first 7 runs:** the wet section starts somewhere between 7.9 and 10.8 m, and its friction is anywhere from 0.12 to 0.31. Studio marks the road past 13.4 m as unmeasured and asks for longer runs. A policy retrained on these ranges succeeds 63% of the time in the hidden world.
- **After the 4 runs it asked for:** wet from 8.60 m (8.36–8.94), friction 0.294 (0.283–0.303). The truth is 8.82 m and 0.30. The retrained policy succeeds 96% of the time.

On a benchmark of 50 hidden worlds, reaching 95% success takes 5.76 runs with the suggestions, against 6.84 with random runs. That ties a well-designed manual sweep (5.76). The difference is that the suggestions adapt to the surface, so you do not need to know in advance where to push.

![Next experiment cards](https://raw.githubusercontent.com/men16922/tether/main/docs/article/img/next-experiment.jpg)

## Step 4. Retrain in parallel NVIDIA Newton worlds

The same learner trains a policy three ways:

1. on your current simulator;
2. on wide domain randomization;
3. on worlds drawn from Tether's bootstrap ensemble.

Each set uses 16 parallel Newton worlds. The policy is deliberately simple: a command table over 11 points, learned by trial in 10 iterations, taking the median over worlds. Because the learner is the same, the only difference between the three policies is the physics they were trained on.

Each world needs its own friction region, and that took a detour. Our first friction strip was a second surface body, and the seam between the two bodies tipped sliding cubes. Newton's XPBD solver averages the two shapes' friction, so we set the ground's friction to 0. A small Warp kernel then switches each body's own friction, as twice the region's effective value, when it crosses the region start:

```python
@wp.kernel
def k(body_q: wp.array(dtype=wp.transform), bodies: wp.array(dtype=int), shapes: wp.array(dtype=int),
      y0: wp.array(dtype=float), mu_near: wp.array(dtype=float), mu_far: wp.array(dtype=float),
      shape_mu: wp.array(dtype=float)):
    i = wp.tid()
    y = wp.transform_get_translation(body_q[bodies[i]])[1]
    shape_mu[shapes[i]] = wp.where(y >= y0[i], mu_far[i], mu_near[i])
```

Hidden-world success, 24 targets:

| Example | Current sim | Wide DR | Tether's ranges |
|---|---|---|---|
| Vehicle braking log | 42% | 0% | **100%** |
| Robot phone video | 0% | 13% | **100%** |
| Robot log | 63% | 0% | **100%** |
| Pusher log | 0% | 38% | **100%** |
| Roadside video | 29% | 8% | **96%** |
| Short pushes only | 0% | 25% | **46%** |

The last row stays low for an honest reason: that log never measured the far table, and Studio says so before you train. The current simulator is confidently wrong; wide DR is mediocre everywhere.

![Policies retrained in parallel Newton worlds](https://raw.githubusercontent.com/men16922/tether/main/docs/article/img/retrain.jpg)

## Step 5. Verify in Newton, then export

Before export, every measured run is replayed in the full Newton engine, from its own start and with its own launch, once with the exported physics and once with your current simulator.

With the exported physics, the rms stop error is 4–8 mm on the four table-scale examples, where the uncalibrated simulators are off by 3–25 cm. On the two driving examples at full size, it is 18–27 cm, against 1.1–2.7 m. This is a self-consistency check of the export, since the examples were generated by Newton; the independent checks come in the next section.

![Verified in NVIDIA Newton](https://raw.githubusercontent.com/men16922/tether/main/docs/article/img/verify.jpg)

The export is written for the tools people already use:

- **NVIDIA Newton:** `ShapeConfig` friction and the friction region.
- **Isaac Lab:** an `EventTermCfg` whose randomization ranges are the measured 90% intervals. Instead of randomizing over a guess, you randomize over what the data still allows.
- **CARLA 0.9.16 (driving):** tire friction scaled by measured/simulated μ, a `static.trigger.friction` box over the wet section, and a braking-distance check. We checked this script against CARLA's documented API with a stand-in module, not inside CARLA itself.

![Export to CARLA](https://raw.githubusercontent.com/men16922/tether/main/docs/article/img/export-carla.jpg)

## Does it hold up outside its own simulator?

Fitting data that Newton generated, with a Newton-family model, is only a consistency check. So we tested three ways the data could come from somewhere else.

**A different engine.** In these hidden worlds, every "real" run comes from MuJoCo, which has soft contacts and a paddle that pushes the object up to speed. In three of the four conditions, the world also contains an effect Tether's model has no field for. There are 24 worlds per condition, and policies run in the hidden MuJoCo world.

![Hidden worlds from a different engine](https://raw.githubusercontent.com/men16922/tether/main/docs/article/img/cross-engine.jpg)

| Hidden world (MuJoCo) | Current sim | Best-width DR | DR over the true distribution* | **Tether** |
|---|---|---|---|---|
| In-menu effects | 9% | 12% | 25% | **89%** |
| + speed-dependent friction | 10% | 12% | 26% | **83%** |
| + a second friction region | 10% | 13% | 25% | **83%** |
| + a 1.5–3° tilt | 8% | 9% | 26% | **90%** |

\* An oracle no user could build. Tether is at least as good as the best DR in 94 of 96 worlds. On held-out runs, stop error is 1.6–2.0 cm against 18–30 cm for the current simulator.

**Real objects, with independently measured friction.** EV-RealPhys (Kandukuri, Strecke, Stueckler 2023, MPI) films household objects pushed by hand across a real table, and measures each object's friction separately with a tilt test. Tether never sees those values. On the video path, Tether's tracker places pixels on the table using the dataset's camera calibration, in place of the A4 sheet.

![Real objects](https://raw.githubusercontent.com/men16922/tether/main/docs/article/img/real-objects.jpg)

Four of five objects land within ±0.05 of the tilt test on both the log path and the video path. Mean error is 0.019 for the log and 0.029 for the video, against 0.082 for the paper's own estimator on the same sequences. The cracker box misses on every path. On real data the intervals are too narrow: they contain the tilt-test value for only 3 of 5 objects (log) and 2 of 5 (video).

This benchmark also changed Tether. The dataset's clock pairs 240 Hz poses with 60 Hz images, so launch speeds read off a few frames can be 20–30% off. The fit now reads the whole shape of each slide, meaning how long it takes to stop, and that pins friction anyway.

**Which friction law?** On 52 public slow-motion clips of real slides (IDPP), the tracker follows 45. Constant deceleration, the Coulomb sliding Tether fits, explains each slide with a median R² of 0.9985, and it is the best of three laws on 36 of 45. Without a size reference or a known frame rate, though, the friction value itself is not identifiable. That is why Studio asks for a reference rectangle and the frame rate.

## Is the LLM actually needed?

We benchmarked the agent against simpler methods on hidden Newton worlds (Gap-Bench, on the live demo's [Benchmark](https://tether-454741001655.us-central1.run.app/benchmark) page), with 15 worlds per tier, reporting mean hidden-world success:

| Tier | Nominal sim | DR | Rule-based | System ID | **Nemotron agent** |
|---|---|---|---|---|---|
| Closed (known parameters) | 46% | 0% | 100% | 100% | 100% |
| Open (friction region or lens) | 39% | 6% | 83% | 100% | 100% |
| Compound (region + lens + one more) | 22% | 14% | 54% | 97% | 96% |

The honest reading:

- Rules break once the world has an effect they have no parameter for.
- A strong system-identification baseline with a hand-ordered structure library ties the agent. The agent picks from the same menu without a hand-set order or thresholds, and spends 5–9% more runs because it probes.
- What the agent adds today is the experiments it asks for and an explanation of each fix, at about $0.007 per world on Token Factory.
- An early 6-world run suggested the agent was ahead on compound worlds. At 15 worlds the gap disappeared, so we do not claim it.

We also ran the Nemotron family on the open and compound tiers, with only 6 worlds each, so read this as a rough ranking. Super scored 100% and 99%, 3.5 Lightning 100% and 95% (and was the cheapest), Ultra 100% and 83%, and Nano 97% and 53%. Bigger was not better.

## Ask Tether

Studio has a chat panel that answers questions about your own session in English or Korean. It also runs on Nemotron via Token Factory, grounded in the session's numbers. Asked "Which number should I trust least, and why?" about a robot log, it named the camera-tilt estimate, whose interval spans about 8°, the widest of all the parameters, and suggested what extra runs would narrow it.

![Ask Tether](https://raw.githubusercontent.com/men16922/tether/main/docs/article/img/ask.jpg)

## Lessons that cost us a day each

- **Never trust n ≤ 6.** At 6 worlds per tier the LLM agent beat system ID 99% to 93%; at 15 worlds it was 96% to 97%.
- **SciPy's default Nelder–Mead simplex** steps 5% of the starting value, and 0.00025 when the start is 0, so camera pitch and lens terms starting at 0 never moved. Pass `initial_simplex` with per-field steps.
- **A friction region that starts with the same friction as the rest is flat in its start position,** so the fit never moves it. Use multiple starts at 0.6× and 1.4×.
- **Judge tipping by peak tilt.** A cube that rolls twice and ends upright looks untipped if you only check its final pose.
- **On small objects, help Cosmos.** At about 10 px wide, raw clips gave 17–33% tip recall. Tracked 448 px crops with a per-frame "tilted?" question reached 88% on slid vs tipped, so Cosmos stays a second opinion.

## Shipping it on a near-zero budget

The live demo runs on Google Cloud Run, built from the committed files (`git archive HEAD`), so `.env` and local runs never leave the machine. The API key lives in Secret Manager. Three settings matter:

- **At most one instance,** because sessions live in memory.
- **Scale to zero** when nobody is visiting. The first visit after an idle spell takes about 15 s.
- **`--no-cpu-throttling`,** because retraining keeps running between the page's polls. It takes about 170 s there, against about 36 s on a laptop, since Warp on CPU is single-threaded.

The 122 tests run offline: they replay recorded Nemotron sessions, so CI needs no network and spends no credits. `make prove` recomputes the headline numbers offline.

## What it cannot do yet

- Most results use simulated data, from Newton and MuJoCo, so the truth can be revealed. Real-object results come from a public benchmark; our own phone recording with a reference sheet is not in the results yet.
- On real data, the intervals are too narrow.
- The physics is one family: launched, sliding to a stop. There is no grasping, steering or lane changing.
- The CARLA export has not run inside CARLA. GR00T is not used.

## Try it

Open the [live Studio](https://tether-454741001655.us-central1.run.app/studio):

1. Pick a field, such as **Autonomous vehicles**, press **Open an example**, and choose **Roadside camera: braking to a stop line**.
2. Press **Track pushes**, then read Nemotron's **Agent notebook**.
3. Under **Simulate your own**, hide the physics yourself, then press **Reveal the hidden physics** to see whether Tether found it.

Code, tests and the proof script are at https://github.com/men16922/tether.

---

*Real-object footage: EV-RealPhys (Kandukuri, Strecke, Stueckler 2023, Max Planck Institute), CC BY-SA 4.0. Real slide clips: IDPP, Apache-2.0. Tether is an independent hackathon project, not affiliated with or endorsed by NVIDIA or Nebius.*
