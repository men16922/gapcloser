# Retrain and visuals (2026-10-10)

Goal: prove that Tether's measured physics trains a better policy, and show it.

| # | Item | Status | Where |
|---|---|---|---|
| 1 | Retrain and test: one learner, three training-world sets (current sim / wide DR / Tether ensemble) in parallel Newton worlds, scored in the hidden world | done | `studio/train.py`, `sim/newton_push.py::push_worlds`, Studio step 5, `POST/GET /api/studio/sessions/{sid}/train` |
| 2 | Parallel training worlds, rendered (4x4 tiled Newton camera, iterations 0/1/3/10) | done | `studio/train_montage.py`, demo video |
| 3 | Before/after: the three policies acting in the hidden world side by side | done | `studio/rollout_video.py`, Studio button, demo video |
| 4 | 3D viewer in Studio | dropped: the Newton-rendered rollout and montage cover the need with less risk | |
| 5 | Proof and docs | done | `make prove` -> `runs/proof/PROOF.md`; Devpost, README, reference/05 |

Results (hidden-world success, current / wide / Tether): vehicle log 42/0/100, phone video 0/12/100, robot log 62/0/100,
pusher log 0/38/100, roadside video 29/8/58, short pushes 0/25/46.
