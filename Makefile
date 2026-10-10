# Tether — Sim2Real self-closing agent (Nebius x NVIDIA hackathon)
# Gate must stay OFFLINE + DETERMINISTIC: no GPU, no network, no Nebius/Token Factory calls.
.PHONY: check test lint smoke-local demo dashboard replays franka-mesh serve docker deploy studio-samples studio-record studio-bench prove cross-engine real-check prove-real real-benchmark film submission

PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

PY_SRC := $(shell find agent sim eval dashboard server video studio -name '*.py' 2>/dev/null)

check: lint test

lint:
	@for f in docs/AGENT_BRIEF.md docs/STATUS.md docs/NEXT_PLAN.md; do test -f $$f || { echo "missing $$f"; exit 1; }; done
	@if [ -n "$(PY_SRC)" ]; then $(PY) -m py_compile $(PY_SRC) && echo "py_compile OK"; else echo "no python sources yet"; fi

test:
	@if [ -d tests ] && ls tests/test_*.py >/dev/null 2>&1; then $(PY) -m pytest -q tests; else echo "no tests yet"; fi

smoke-local: check

# record Newton demo runs (local CPU, no network) and build the self-contained dashboard
demo:
	$(PY) -m eval.record_demo
	$(PY) -m dashboard.build

dashboard:
	$(PY) -m dashboard.build

# 3D viewer data for the recorded bundle (Newton on CPU, no LLM, no network), then rebuild the dashboard
replays:
	$(PY) -m eval.record_demo --replays-only
	$(PY) -m dashboard.build

# regenerate dashboard/assets/franka_fr3.json from the Newton Franka asset (decimated visual meshes)
franka-mesh:
	$(PY) -m dashboard.franka_mesh

# Studio sample data: Newton robot logs + Newton-rendered phone videos with ground truth (studio/samples)
studio-samples:
	$(PY) -m studio.samples
	$(PY) -m studio.video_sample

# record the samples through the Studio API (Nemotron agent when TETHER_LLM / --llm is set), rebuild pages
studio-record:
	$(PY) -m studio.record --llm $${TETHER_LLM:-tokenfactory}
	$(PY) -m dashboard.build

real-check:  # public real footage (IDPP, ~310 MB download): tracker + sliding model on 52 real clips
	$(PY) -m eval.real_friction --download

cross-engine:  # hidden worlds made by a second engine (MuJoCo, contact launch, off-menu effects): pip install -r requirements-eval.txt
	$(PY) -m eval.cross_engine --worlds 24

real-benchmark:  # EV-RealPhys real objects (1.7 GB download, CC BY-SA 4.0): Tether vs tilt-test friction -> runs/proof/real_benchmark.json
	$(PY) -m eval.real_benchmark --download

prove-real:  # your own real clips + tilt-test angles (video/real/truth.json, see reference/06) -> runs/proof/real_calibration.json
	$(PY) -m eval.prove_real

prove:  # recompute every headline number (no network, ~5 min) -> runs/proof/PROOF.md
	$(PY) -m eval.prove

studio-bench:
	$(PY) -m eval.studio_bench --worlds 50 --json runs/bench/studio_bench_analytic.json

# live server on http://localhost:8000 (Studio at /studio) (TETHER_LLM=local|tokenfactory|none)
serve: dashboard
	$(PY) -m uvicorn server.app:app --host 0.0.0.0 --port $${PORT:-8000}

docker:
	docker build -t tether .

# public deployment on Google Cloud Run (gcloud logged in; key from .env into Secret Manager), see docs/deploy/DEPLOY.md
deploy:
	$(PY) -m deploy.cloud_run

# demo film: motion-graphics scenes rendered in headless Chrome, ElevenLabs narration (.env), generated music bed
film:  # -> video/out/tether_film.mp4 (needs make real-benchmark, and make serve with TETHER_LLM=tokenfactory for the Ask clip)
	$(PY) -m video.footage
	$(PY) -m studio.rollout_video brake-log
	$(PY) -m studio.train_montage brake-log
	$(PY) -m video.film --voice elevenlabs

# everything to upload (film, YouTube thumbnail and description, Devpost gallery and text) -> submission/, see docs/submission/SUBMIT.md
submission:
	$(PY) -m video.submission
