# Spike: NVIDIA Cosmos Reason 2 (8B) as local "eyes"

**Question.** Can `nvidia/Cosmos-Reason2-8B` run locally on the M4 Max (48 GB) with image/video
input, fast enough to be the agent's eyes, and can it tell from a short Newton clip whether the
pushed cube **slid** or **tipped**?

**Answer: conditional GO.** It runs locally at zero cost (llama.cpp + Metal, about 6 s per clip). With
the right setup, which is a tracked close-up crop, a per-frame "is it tilted?" question, no `<think>`, and
greedy decoding, it scores **25/26 on the tuning set, 20/20 on the existing demo clips, and 24/32 on a
held-out set**. Slides are reliable (about 96% recall). Tips are missed about 1 time in 4. Overall
it got 69/78 (88%).

Use it as **corroborating evidence and for the demo story ("Built on NVIDIA Cosmos")**, not as the only source
of the `tipped` flag. The Newton state already knows the exact tilt, so keep that as ground truth.
The things that did not work are as useful to know as the result: video input, `<think>` reasoning,
full-frame input and contact sheets were all much worse (see the table).

## Setup (reproducible, about 6.2 GB download, no account or token)

```bash
brew install llama.cpp ffmpeg        # llama.cpp 0.6.0 (Metal); ffmpeg is only needed for --mode video
mkdir -p ~/models/cosmos-reason2-8b && cd ~/models/cosmos-reason2-8b
B=https://huggingface.co/mradermacher/Cosmos-Reason2-8B-GGUF/resolve/main
curl -LO $B/Cosmos-Reason2-8B.Q4_K_M.gguf        # 5.0 GB
curl -LO $B/Cosmos-Reason2-8B.mmproj-f16.gguf    # 1.2 GB vision projector (required for images)
llama-server -m Cosmos-Reason2-8B.Q4_K_M.gguf --mmproj Cosmos-Reason2-8B.mmproj-f16.gguf \
  -ngl 99 -c 8192 --cache-ram 0 -np 1 --port 8080          # add --video-fps 15 for --mode video

# from the repo root (the repo .venv already has Pillow/numpy/scipy/requests; nothing is installed into it)
.venv/bin/python spike/cosmos/classify.py spike/cosmos/clips/tip-c30-side.webp
# {"events": [{"t": 0.07, "type": "slid"}, {"t": 0.13, "type": "tipped"}], "outcome": "tipped", "latency_s": 6.3}
.venv/bin/python spike/cosmos/classify.py --labels spike/cosmos/clips/labels.json            # tuning set
.venv/bin/python spike/cosmos/classify.py --labels spike/cosmos/clips_holdout/labels.json    # held-out
.venv/bin/python spike/cosmos/classify.py --labels spike/cosmos/clips/labels_demo.json       # runs/demo clips
.venv/bin/python spike/cosmos/render_set.py [--holdout]   # re-render the labeled clips (Newton, about 1 min)
```

Notes on runtimes:
- **llama.cpp: works.** The image path is solid. Video works through a `video_url` content part, with ffmpeg
  decoding at `--video-fps`, but it was useless for this task (see below).
- **MLX / mlx-vlm: not viable for 8B.** The official repo is gated (HTTP 401 without accepting the
  license), and the only community MLX conversions are of the **2B** model.
- **Ollama: not tested end to end.** The HF registry manifest for
  `hf.co/mradermacher/Cosmos-Reason2-8B-GGUF:Q4_K_M` *does* include an
  `application/vnd.ollama.image.projector` layer (the Q8_0 mmproj), so the projector is not dropped.
  Whether Ollama 0.33.2 runs the qwen3vl architecture from a split projector is still unverified.
  llama.cpp is the known-good path.

## Labeled data

| set | clips | labels | source |
|---|---|---|---|
| `clips/` (tuning) | 26 = 13 cases x {side cam `render_trial`, Franka cam `render_trial_arm`}: 14 slid / 12 tipped | **physics**: per-frame `tilt_deg`, tipped if peak > 30 deg | nominal at commands 1.0 to 3.4, slippery, restitution 0.4, `table_mu=1.18` at commands 2.4 to 4.0 |
| `clips_holdout/` | 32 = 16 cases x 2 views: 20 slid / 12 tipped | physics | **not used while choosing the prompt**: table_mu 1.05/1.1/1.2, light 0.6/1.4, half-size 0.02/0.033, camera_dx +/-0.02 to 0.028, density, actuator gain |
| `labels_demo.json` | the 20 `runs/demo/clips/*-real.webp`: 12 slid / 8 tipped | **visual** (by eye from tracked crops) | `bundle.json` clips are rendered at target 0.45, which matches no trial, so there is no per-clip physics flag. Tipped: tipping-edge it0 to it4, sticky-table it2 to it4 |

Side finding: `tip-c24` rolls twice (180 deg). Its final tilt is 0 deg, so the repo's final-only
`tipped` flag (`sim/newton_push.py` `tilt_deg` > `TIP_DEG` on the final pose) **misses it**.
`render_set.py` labels from the peak tilt instead.

## Results (tuning set, 26 clips)

The cube is about 10 px wide in the 480x270 render. Every row except the first two uses `--crop track`:
a small-blob tracker seeded by projecting the cube's start pose through the known Newton camera
gives a 448x448 close-up per frame.

| # | prompt | input | think | temp | acc | tipped recall | slid recall | JSON ok | median latency |
|---|---|---|---|---|---|---|---|---|---|
| 1 | events | 8 frames, workspace strip | yes | 0 | 0.69 | 0.33 | 1.00 | 0.88 | 13.9 s |
| 2 | events | same + `--image-min-tokens 1024` | yes | 0 | 0.62 | 0.17 | 1.00 | 1.00 | 33.1 s |
| 3 | events | 8 tracked crops @256 | yes | 0 | 0.73 | 0.50 | 0.93 | 0.50 | 30.4 s |
| 4 | events | 8 tracked crops @448 | yes | 0.6 | 0.73 | 0.67 | 0.79 | 0.77 | 19.0 s |
| 5 | events | contact sheet (1 image, 8 crops) | yes | 0.6 | 0.69 | 0.42 | 0.93 | 0.69 | 58.0 s |
| 6 | events | **mp4 video** of crops (15 fps) | yes | 0.6 | 0.54 | **0.00** | 1.00 | 1.00 | 14.2 s |
| 7 | events | 8 tracked crops | no | 0.6 | 0.81 | 0.83 | 0.79 | 1.00 | 6.9 s |
| 8 | events | 8 tracked crops | no | 0 | 0.73 | 0.42 | 1.00 | 1.00 | 2.3 s |
| 9 | tilt | 8 tracked crops | yes | 0.6 | 0.69 | 0.75 | 0.64 | 1.00 | 16.8 s |
| 10 | tilt | 12 tracked crops | no | 0 | 0.88 | 0.75 | 1.00 | 1.00 | 11.6 s |
| 11 | tilt | 8 tracked crops, motion window only | no | 0 | 0.85 | 0.67 | 1.00 | 1.00 | 5.9 s |
| **12** | **tilt** | **8 tracked crops** | **no** | **0** | **0.96** | **0.92** | **1.00** | **1.00** | **3.5 s** (6.1 s rerun) |

Best config (row 12, now the default) on the other sets:

| set | acc | tipped recall | slid recall | median / max latency |
|---|---|---|---|---|
| tuning (rerun) | 25/26 = 0.96 | 11/12 | 14/14 | 6.1 / 7.5 s |
| **held-out** | 24/32 = 0.75 | **6/12** | 18/20 | 7.7 / 8.1 s |
| demo clips | 20/20 = 1.00 | 8/8 | 12/12 | 6.7 / 7.8 s |
| pooled | 69/78 = 0.88 | 25/32 = 0.78 | 44/46 = 0.96 | |

Held-out misses: tips where the tilt shows in only one of the 8 sampled frames (dark, bright, small
and big cubes, `mu=1.2`). The two false positives are the `camera_dx=-0.028` slide, where the cube near the frame
edge looks skewed in perspective. For comparison, a naive pixel baseline (min blob fill ratio of the
tracked cube) scored 0.54 to 0.62, close to chance. So the VLM is doing real perception, but with only
8 frames at 30 fps, a fast tip that shows in a single frame can be missed.

What we learned:
- **Ask Cosmos to perceive, let code reason.** A per-frame `tilted: true/false` list, with the
  event logic in Python ("first tilted frame means tipped"), beat asking the model for the event list
  directly (0.96 vs 0.73 to 0.81).
- **`<think>` hurt** on every variant. The reasoning confidently describes the cube as "upright" and
  often loops to `max_tokens` (2048), which takes 60 to 200 s. Cosmos's recommended `<think>` format is
  for its post-trained reasoning tasks. For small-object perception, a short JSON answer was better.
- **Crop is essential.** On the raw 480x270 frames the cube is about 10 px and tipped recall was 0.17 to 0.33.
  Upscaling the full frame (`--image-min-tokens 1024`) did not help and made it 3x slower.
- **Video input failed** (0 of 12 tips detected). llama.cpp's Qwen3-VL video path merges frame pairs,
  and a 0.1 s tip disappears.

### Best prompt (`--prompt tilt`, no think, temperature 0)

System: `You are a helpful assistant.` User: 8 images, each preceded by `Frame k (t=0.13s):`, then:

```
These are 8 frames in time order from a robot push experiment rendered in a physics simulator.
Each frame is a close-up crop centred on a small light-grey cube on a dark checkered table. For EACH
frame decide whether the cube sits flat and upright (a face on the table, its outline an axis-aligned
square/box) or is TILTED (rotated, balanced on an edge or corner, its outline a diamond or skewed box).
Output strict JSON on one line: {"frames":[{"i":<frame number>,"tilted":true|false}]}
Reply with the JSON only.
```

The model sometimes returns a bare list or wraps it in a code fence twice, and `parse_tilt` accepts both.
Events are built in code: `[{"t":<first moving frame>,"type":"slid"}, {"t":<first tilted frame>,"type":"tipped"}]`.

## Latency and memory (M4 Max, Metal, Q4_K_M)

- Per clip: **3.5 to 8 s** median end to end (8 crops at 448 px is about 1.9k prompt tokens, plus about 60 output tokens).
  Prompt eval ran at 300 to 540 tok/s and generation at 77 tok/s cold, dropping to 25 to 45 tok/s during
  long back-to-back sweeps (thermal or shared load). An agent iteration that looks at 1 to 2 clips adds about 10 s.
- Memory: `footprint` of `llama-server` is **2.4 to 2.8 GB** with `-c 8192 --cache-ram 0`, plus the
  mmapped 5.0 GB weights and 1.2 GB projector, so about **9 GB of unified memory** in total. With the defaults (`-c 32768`,
  8 GB prompt cache) the footprint grew to 16 GB, so use the lean flags next to Ollama/Nemotron.

## License

`nvidia/Cosmos-Reason2-8B` is under the **NVIDIA Open Model License** (HF card: `license: other`,
`nvidia-open-model-license`; the official repo is gated). The license allows commercial use and
derivative models, and NVIDIA claims no ownership of outputs. Redistributing the model requires a copy of the
agreement and the notice *"Licensed by NVIDIA Corporation under the NVIDIA Open Model License"*. For
Cosmos models, it also requires **"Built on NVIDIA Cosmos"** on related websites, UIs or docs.
Bypassing safety guardrails terminates the license. Our use is local inference with no redistribution of
weights, which is fine. **Add "Built on NVIDIA Cosmos" to the dashboard/README if CosmosEyes ships.** The GGUF
is a community quantization (mradermacher), a derivative under the same license.

## Recommendation and integration

Go, as an optional perception layer:

```python
from spike.cosmos.classify import CosmosEyes          # move to agent/cosmos_eyes.py when adopted
eyes = CosmosEyes(server="http://127.0.0.1:8080")     # defaults = best config (tilt, track, n=8, no think, temp 0)
events = eyes.events(Path("runs/demo/clips/tipping-edge-it0-real.webp"))
# -> [{"t": 0.17, "type": "slid"}, {"t": 0.69, "type": "tipped"}]
```

1. Add `agent/cosmos_eyes.py` (lift `CosmosEyes` with `track_cube` and `parse_tilt`). Give it a
   `--eyes cosmos|none` switch in `eval/record_demo.py`. The default stays `none`, so CI and offline
   gates do not need the 6 GB model.
2. Per iteration, run it on the real clip (and the sim clip). Put `{"vision_events": [...], "vision_model":
   "Cosmos-Reason2-8B (local)"}` into the measure event and into the Nemotron diagnoser's input
   next to `real_trials_cube_tipped_over`. Make the diagnoser treat vision as a *second opinion*:
   physics-flag tipped plus vision tipped means a confident tipping diagnosis, and disagreement goes to a note.
3. Show the per-frame tilt flags on the dashboard clip, with "Built on NVIDIA Cosmos".
4. To improve tip recall: render clips at 60 fps or sample 12 to 16 frames only during the strike,
   OR-combine the two camera views, or try the Q8_0 quant. The arm-view clip and a dense sample around
   the strike are the cheapest next steps. "bounced" and "stalled" were not evaluated: no
   clip in this task bounces. Stall is better computed from the tracker displacement than from the VLM.

## Files

- `classify.py`: `CosmosEyes` (tracker crop, prompts, llama-server client, parsers), eval and `--rescore`
- `render_set.py`: renders the labeled clips with Newton physics labels (`--holdout` for the held-out set)
- `clips/`, `clips_holdout/`: webp clips and `labels.json`. `clips/labels_demo.json` covers `runs/demo/clips`
- `results/*.jsonl`: every run, with raw model output, usage and timings, one row per clip
