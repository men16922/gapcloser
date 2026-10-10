# Tether: proof run

2026-10-10 23:46 · git c9bd732 · arm64 Darwin · 776 s · full

Everything below except section 5 was recomputed by `python -m eval.prove` on this machine, with no network.

## 1. Interval coverage

Over 24 random hidden worlds (118 fitted values), the 90% interval contained the truth **97%** of the time (target 90%). By field: mu_eff 96%, actuator_gain 100%, camera_pitch_deg 92%, patch_y0 96%, patch_mu 100%.

## 2. Newton replay and 3. retrain and test (six Studio examples)

| Example | Domain | Runs | Truth inside 90% | Replay rms: exported / current | Retrain, hidden-world success: current / wide / Tether |
|---|---|---|---|---|---|
| brake-log | driving | 18 | 5/5 | 0.18 m / 1.11 m | 42% / 0% / **100%** |
| flick | robot | 11 | 3/3 | 0.8 cm / 19.7 cm | 0% / 13% / **100%** |
| lab-bench | robot | 20 | 5/5 | 0.6 cm / 3.1 cm | 63% / 0% / **100%** |
| press-line | factory | 18 | 4/5 | 0.8 cm / 24.7 cm | 0% / 38% / **100%** |
| short-reach | robot | 8 | 1/1 | 0.4 cm / 5.1 cm | 0% / 25% / **46%** |
| stop-line | driving | 11 | 3/3 | 0.27 m / 2.74 m | 29% / 8% / **96%** |

## 4. Next experiment

Real runs to reach 95% success: suggested **5.76**, random 6.84, hand-made sweep 5.76 (50 worlds).

## 5. Gap-Bench (recorded Nemotron runs, quoted)

| Tier | Nominal | Domain rand. | Rule-based | System ID | Nemotron agent |
|---|---|---|---|---|---|
| closed | 46% | 0% | 100% | 100% | 100% |
| open | 39% | 6% | 83% | 100% | 100% |
| compound | 22% | 14% | 54% | 97% | 96% |

## 6. Phone robustness (automatic sheet corners, tracking, fit)

| Condition | Corner error | Pushes found | Launch-speed bias | Truth inside 90% | Intervals (mu, region start, region mu) |
|---|---|---|---|---|---|
| clean phone | 1.0 px | 11/11 | -0.8% | 3/3 | [0.535, 0.56], [0.337, 0.364], [0.296, 0.311] |
| hand-held, blurred, compressed | 3.32 px | 11/11 | -6.5% | 2/3 | [0.486, 0.559], [0.321, 0.365], [0.269, 0.295] |

## 7. Second engine: hidden worlds made by MuJoCo 3.12.0 (quoted from `make cross-engine`)

The "real" pushes come from MuJoCo instead of Newton: soft contacts, a paddle that pushes the object up to speed instead of an assigned velocity, and in three conditions an effect the fitter has no field for. The Studio analyses each first-day log offline; policies are trained in each simulator and run in the hidden MuJoCo world (24 targets, ±3 cm). DR: domain randomization around the current sim at 25-100% of each parameter's range (best width shown); DR oracle: randomized over the hidden worlds' own distribution, which no user knows.

| Condition | Worlds | Current sim | Best DR | DR oracle | Exact parameters | **Tether** | Tether ≥ best DR | Left unexplained flagged | Hold-out stop error: Tether / current |
|---|---|---|---|---|---|---|---|---|---|
| in-menu | 24 | 9% | 12% (100%) | 25% | 82% | **89%** | 24/24 | 2/24 | 1.7 / 27.5 cm |
| speed | 24 | 10% | 12% (50%) | 26% | 58% | **83%** | 23/24 | 24/24 | 2.0 / 30.3 cm |
| two regions | 24 | 10% | 13% (100%) | 25% | 85% | **83%** | 23/24 | 15/24 | 1.6 / 18.3 cm |
| slope | 24 | 8% | 9% (100%) | 26% | 33% | **90%** | 24/24 | 4/24 | 1.8 / 24.1 cm |

"Exact parameters" trains on the hidden world's own MuJoCo parameter values: it misses where MuJoCo's contact behaves differently from its parameters (a few % in friction) or where an off-menu effect acts; Tether fits the behaviour, not the parameter. Interval coverage of MuJoCo's parameter values: in-menu 88%, speed 47%, two regions 57%, slope 65% (the parameter is not the behaviour here).

## 8. Real footage (IDPP, quoted from `make real-check`)

45 of 52 real slow-motion slides tracked. Which friction law explains each slide (scale-free, BIC): Coulomb (constant deceleration, Tether's model) best on 36 of 45; viscous decisively worse on 35; a mixed law decisively better on 5. The friction *value* is not identifiable from these clips (no size reference, unknown slow-motion factor): leave-one-surface-out error 0.0675 vs 0.0581 for guessing the mean.

## 9. Real objects with independently measured friction (EV-RealPhys, quoted from `make real-benchmark`)

YCB objects pushed by hand across a real table, filmed at 30 Hz (RealSense D455, RGB only here) with motion capture; each object's friction was measured separately by tilting the table (Kandukuri et al. 2023, Table 5). Tether never sees those values. Log: the motion-capture centre of mass as a robot log. Video: Tether's tracker on the RGB frames, pixels put on the table with the dataset's camera calibration, one click per clip at the rest position.

| Object | Tilt test | Tether, log (90%) | Tether, video (90%) | Pushes log / video |
|---|---|---|---|---|
| pitcher | 0.220 | 0.216 (0.210–0.225) | 0.229 (0.201–0.260) | 10 / 4 |
| mustard bottle | 0.159 | 0.156 (0.137–0.163) | 0.164 (0.160–0.168) | 10 / 7 |
| bleach cleanser | 0.169 | 0.159 (0.155–0.165) | 0.181 (0.153–0.198) | 10 / 5 |
| cracker box | 0.280 | 0.208 (0.201–0.213) | 0.173 (0.170–0.178) | 5 / 1 |
| mug | 0.110 | 0.115 (0.110–0.123) | 0.120 (0.114–0.127) | 10 / 8 |

Mean absolute error: log 0.019 (median 0.005), video 0.029 (median 0.011); within ±0.05: 4/5 and 4/5. The paper's own estimator on the same real sequences: mean 0.082, median 0.0246. Intervals are too narrow on real data: they contain the tilt-test value for 3 of 5 (log) and 2 of 5 (video).

Limits: sections 1-6 use synthetic data (NVIDIA Newton) so the truth is known; section 7 uses a second engine; section 8 is real footage without a scale reference; section 9 is real objects and real video with an independent friction measurement, calibrated by the dataset's camera calibration instead of a sheet of paper.
