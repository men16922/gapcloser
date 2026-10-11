# Tether: tie your simulator to the real world (Nebius x NVIDIA Global AI Hackathon, Physical AI track)
# The gate (make check) stays OFFLINE + DETERMINISTIC: no GPU, no network, no Nebius/Token Factory calls.
.PHONY: check test lint smoke-local serve pages docker deploy demo replays franka-mesh studio-samples studio-record \
        prove studio-bench cross-engine real-check real-benchmark prove-real film submission

PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
PY_SRC := $(shell find tether film deploy -name '*.py' 2>/dev/null)

# ---- check ---------------------------------------------------------------------------------------------------
check: lint test

lint:
	@for f in docs/project/AGENT_BRIEF.md docs/project/STATUS.md docs/project/NEXT_PLAN.md; do test -f $$f || { echo "missing $$f"; exit 1; }; done
	@$(PY) -m py_compile $(PY_SRC) && echo "py_compile OK"

test:
	$(PY) -m pytest -q tests

smoke-local: check

# ---- run the app ---------------------------------------------------------------------------------------------
# live server on http://localhost:8000: Overview /, Studio /studio, Benchmark /benchmark (TETHER_LLM=local|tokenfactory|none)
serve: pages
	$(PY) -m uvicorn tether.server.app:app --host 0.0.0.0 --port $${PORT:-8000}

# build the pages into tether/web/dist (live variants for the server, standalone copies that work offline)
pages:
	$(PY) -m tether.web.build

docker:
	docker build -t tether .

# public deployment on Google Cloud Run (gcloud logged in; key from .env into Secret Manager), see docs/deploy/DEPLOY.md
deploy:
	$(PY) -m deploy.cloud_run

# ---- recorded data the pages show (runs/) --------------------------------------------------------------------
# Benchmark page: Newton runs of the agent on hidden worlds (local CPU, no network) -> runs/demo
demo:
	$(PY) -m tether.eval.record_demo
	$(PY) -m tether.web.build

# 3D viewer data for the recorded bundle (Newton on CPU, no LLM, no network), then rebuild the pages
replays:
	$(PY) -m tether.eval.record_demo --replays-only
	$(PY) -m tether.web.build

# regenerate tether/web/assets/franka_fr3.json from the Newton Franka asset (decimated visual meshes)
franka-mesh:
	$(PY) -m tether.web.franka_mesh

# Studio examples: Newton robot logs + Newton-rendered phone videos with ground truth -> tether/studio/samples
studio-samples:
	$(PY) -m tether.studio.samples
	$(PY) -m tether.studio.video_sample

# record the examples through the Studio API (Nemotron agent when TETHER_LLM / --llm is set) -> runs/studio-demo
studio-record:
	$(PY) -m tether.studio.record --llm $${TETHER_LLM:-tokenfactory}
	$(PY) -m tether.web.build

# ---- evidence: every headline number (runs/proof, runs/bench) ------------------------------------------------
prove:  # recompute every headline number (no network) -> runs/proof/PROOF.md
	$(PY) -m tether.eval.prove

studio-bench:  # does the next-experiment card save runs? -> runs/bench/studio_bench_analytic.json
	$(PY) -m tether.eval.studio_bench --worlds 50 --json runs/bench/studio_bench_analytic.json

cross-engine:  # hidden worlds made by a second engine (MuJoCo, contact launch, off-menu effects): pip install -r requirements-eval.txt
	$(PY) -m tether.eval.cross_engine --worlds 24

real-check:  # public real footage (IDPP, ~310 MB download): tracker + sliding model on 52 real clips
	$(PY) -m tether.eval.real_friction --download

real-benchmark:  # EV-RealPhys real objects (1.7 GB download, CC BY-SA 4.0): Tether vs tilt-test friction -> runs/proof/real_benchmark.json
	$(PY) -m tether.eval.real_benchmark --download

prove-real:  # your own real clips + tilt-test angles (film/real/truth.json, see docs/guide/06) -> runs/proof/real_calibration.json
	$(PY) -m tether.eval.prove_real

# ---- demo film and submission (film/) ------------------------------------------------------------------------
# app footage, Newton renders, motion-graphics scenes in headless Chrome, ElevenLabs narration (.env), generated music
film:  # -> film/out/tether_film.mp4 (needs make real-benchmark, and make serve with TETHER_LLM=tokenfactory for the Ask clip)
	$(PY) -m film.footage
	$(PY) -m tether.studio.rollout_video brake-log
	$(PY) -m tether.studio.train_montage brake-log
	$(PY) -m film.render --voice elevenlabs

# everything to upload (film, YouTube thumbnail and description, Devpost gallery and text) -> submission/, see docs/submission/SUBMIT.md
submission:
	$(PY) -m film.submission
