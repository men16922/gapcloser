# GapCloser Studio — product plan (2026-10-09)

Goal: turn GapCloser from "a demo where the real world is a hidden sim" into a tool a robotics engineer
can point at **their own data** and walk away with a calibrated simulator they can paste into NVIDIA
Newton or Isaac Lab, an honest uncertainty, and the next experiment worth running.

## User story

> My policy works in sim and misses on the real table. I have a phone (or my robot's logs). In five
> minutes I want: what is physically different, how sure we are, which pushes would settle the rest,
> and a config I can drop into my sim.

## Product surface

| Step | What the user does | What GapCloser returns |
|---|---|---|
| 1 Data | Upload a phone video of pushes (with an A4 sheet on the table) **or** a robot log (CSV/JSON: command, stop, optional track/target/perceived) **or** pick a sample | Parsed trials, validation errors in plain words |
| 2 Calibrate (video) | Click the 4 corners of the A4 sheet on the first frame | Metric top-down rectification, tracked pushes, overlay preview |
| 3 Diagnose | Watch the Nemotron tool agent (or offline fitter) work | Lab notebook: evidence → hypotheses → fits → committed model |
| 4 Results | Read | Params with 90% intervals (bootstrap), measured vs model curves, friction map along the push path, predicted success over the target range before/after |
| 5 Next experiment | Optional: run the suggested pushes, append, re-run | Commands ranked by ensemble disagreement where it matters (query by committee) |
| 6 Export | Copy / download | NVIDIA Newton snippet, Isaac Lab event-term config (DR ranges = intervals), JSON, Markdown report |

Same core is a CLI (`python -m studio calibrate log.csv --out out/`) and a REST API (`/api/studio/*`, OpenAPI at `/docs`).

## Agent change for real data

Offline data has no robot to probe. `probe_real` becomes a **request** to the human: the agent's
proposed pushes are returned as "next experiment" cards; a session can be resumed with the new data.

## Honesty rules

- Sample video is rendered by NVIDIA Newton (SensorTiledCamera) from a known world and labelled synthetic;
  ground truth is shown only after the diagnosis. A real phone video path exists and is documented.
- Intervals are bootstrap over trials (resample, refit), not LLM confidence.
- Success predictions are rollouts of the fitted model, labelled as predictions.

## Milestones

- P1 core: `studio/session.py` (ingest + validation), `studio/fit.py` (fit + bootstrap + predictions),
  `studio/design.py` (next experiment), `studio/export.py`; CLI; tests.
- P2 video: `studio/video.py` (A4 homography, tracking, push segmentation) + Newton-rendered sample video
  with ground truth; tests on the sample.
- P3 API + Studio UI page (`/studio`) + static recorded session inside the standalone dashboard.
- P4 agent in the loop (Token Factory) on the samples; experiment-savings bench (real pushes to reach a
  calibrated model: suggested vs random).
- P5 docs/README/Devpost, checkpoint.
