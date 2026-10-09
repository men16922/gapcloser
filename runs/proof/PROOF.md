# Tether: proof run

2026-10-10 02:13 · git a662519 · arm64 Darwin · 415 s · full

Everything below except section 5 was recomputed by `python -m eval.prove` on this machine, with no network.

## 1. Interval coverage

Over 24 random hidden worlds (118 fitted values), the 90% interval contained the truth **92%** of the time (target 90%). By field: mu_eff 88%, actuator_gain 100%, camera_pitch_deg 92%, patch_y0 83%, patch_mu 100%.

## 2. Newton replay and 3. retrain and test (six Studio examples)

| Example | Domain | Runs | Truth inside 90% | Replay rms: exported / current | Retrain, hidden-world success: current / wide / Tether |
|---|---|---|---|---|---|
| brake-log | driving | 18 | 4/4 | 0.18 m / 1.11 m | 42% / 0% / **100%** |
| flick | robot | 11 | 3/3 | 0.6 cm / 19.7 cm | 0% / 12% / **100%** |
| lab-bench | robot | 20 | 4/5 | 0.6 cm / 3.1 cm | 62% / 0% / **100%** |
| press-line | factory | 18 | 4/5 | 0.8 cm / 24.7 cm | 0% / 38% / **100%** |
| short-reach | robot | 8 | 1/1 | 0.4 cm / 5.1 cm | 0% / 25% / **46%** |
| stop-line | driving | 11 | 3/3 | 0.22 m / 2.74 m | 29% / 8% / **58%** |

## 4. Next experiment

Real runs to reach 95% success: suggested **5.76**, random 7.6, hand-made sweep 5.88 (50 worlds).

## 5. Gap-Bench (recorded Nemotron runs, quoted)

| Tier | Nominal | Domain rand. | Rule-based | System ID | Nemotron agent |
|---|---|---|---|---|---|
| closed | 46% | 0% | 100% | 100% | 100% |
| open | 39% | 6% | 83% | 100% | 100% |
| compound | 22% | 14% | 54% | 97% | 96% |

## 6. Phone robustness (automatic sheet corners, tracking, fit)

| Condition | Corner error | Pushes found | Launch-speed bias | Truth inside 90% | Intervals (mu, region start, region mu) |
|---|---|---|---|---|---|
| clean phone | 1.0 px | 11/11 | -0.8% | 2/3 | [0.511, 0.573], [0.342, 0.379], [0.258, 0.296] |
| hand-held, blurred, compressed | 3.32 px | 11/11 | -6.5% | 3/3 | [0.425, 0.631], [0.15, 0.589], [0.204, 0.391] |

Limits: all data is synthetic (NVIDIA Newton) so the truth is known; a real phone video has not been validated yet.
