# GapCloser live demo: NVIDIA Newton on CPU + FastAPI. Nemotron is called over HTTPS
# (Nebius Token Factory), so the container needs no GPU.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# non-root user (Hugging Face Spaces runs containers as uid 1000)
RUN useradd -m -u 1000 app
USER app
ENV HOME=/home/app PATH=/home/app/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/app/gapcloser

COPY --chown=app requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=app agent agent
COPY --chown=app sim sim
COPY --chown=app eval eval
COPY --chown=app server server
COPY --chown=app dashboard dashboard
COPY --chown=app runs/demo runs/demo

# fetch the Franka asset and compile Warp CPU kernels at build time, so the first visitor
# doesn't wait for them; also builds the live dashboard page
RUN python -m dashboard.build --out dashboard/dist/gapcloser.html \
 && python -c "from pathlib import Path; from sim.params import ParamSet; from sim.newton_push import NewtonPushEnv, render_trial_arm; from sim.push_task import Policy, eval_targets; NewtonPushEnv().rollout(ParamSet.nominal(), Policy(15.7), eval_targets(3, 1)); render_trial_arm(ParamSet.nominal(), 2.0, 0.4, Path('/tmp/warm.webp'))"

ENV GAPCLOSER_LLM=tokenfactory \
    GAPCLOSER_MAX_LLM_CALLS=200 \
    GAPCLOSER_RUNS_PER_HOUR=6 \
    PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "python -m uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
