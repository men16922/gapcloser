# Deploying GapCloser

The same Docker image runs everywhere. It needs no GPU: NVIDIA Newton runs on CPU and Nemotron is called over HTTPS on Nebius Token Factory.

| Target | What visitors get | Cost |
|---|---|---|
| GitHub Pages | Recorded runs only (static page) | free |
| Hugging Face Spaces (Docker, CPU basic) | Live demo: visitors hide physics and watch the agent | free (sleeps when idle) |
| Nebius AI Cloud CPU VM | Live demo on Nebius | about $0.06/hour (2 vCPU / 8 GB, price from 2026-10-01) |

All live targets need the secret `NEBIUS_API_KEY`. Never commit it; set it as a platform secret or environment variable.

## Build and check locally

```bash
make demo            # record scenarios (runs/demo), build the dashboard
make serve           # http://localhost:8000  (GAPCLOSER_LLM=local uses Ollama)
make docker          # docker build -t gapcloser .
docker run --rm -p 7860:7860 -e NEBIUS_API_KEY=$NEBIUS_API_KEY gapcloser   # http://localhost:7860
# without a key, point the container at the host's Ollama:
docker run --rm -p 7860:7860 -e GAPCLOSER_LLM=local -e OLLAMA_BASE_URL=http://host.docker.internal:11434/v1/ gapcloser
```

The image bundles `runs/demo` (recorded scenarios and clips, committed to git). Re-run `make demo` only to refresh them.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `GAPCLOSER_LLM` | `tokenfactory` in the image | `tokenfactory`, `local` (Ollama) or `none` (rule-based diagnoser) |
| `NEBIUS_API_KEY` | — | Token Factory key |
| `GAPCLOSER_DIAG_MODEL` | `nemotron super` | model hint for diagnosis (words matched against the model list) |
| `GAPCLOSER_MAX_LLM_CALLS` | 200 | total LLM calls for the server's lifetime; then the rule-based diagnoser takes over |
| `GAPCLOSER_RUNS_PER_HOUR` | 6 | runs per visitor IP per hour |
| `GAPCLOSER_MAX_TURNS_PER_RUN` | 24 | Nemotron tool-agent chat turns per live run (max 8 per diagnosis); past it, the rule-based diagnoser takes over |
| `PORT` | 7860 | listen port |

"Stump the agent": with an LLM configured, every live run uses the open-world path. Visitors pick a preset (wet strip, rough strip, lens distortion, three faults, surprise me) or build a world, including a friction strip (`patch_y0` + `patch_mu`, counted as one fault) and lens distortion; the Nemotron tool agent's lab notebook streams step by step (`agent_step` events) while it experiments.

Cost guards built in: one run at a time, max 4 iterations, max 3 hidden faults (a strip counts as one), values inside the parameter bounds and `patch_mu` ≤ 0.95 (stickier strips tip the cube), per-IP rate limit, a global LLM-call cap, a per-run cap of 24 agent turns, and a 2048-token output cap. Every chat turn counts as one call. Tool-agent turns resend the growing conversation (about 2–6k prompt tokens each; the spike measured ~17k in + 2k out per world with Super, ≈ $0.007), so 200 calls stay around 1M tokens in the worst case.

## Hugging Face Spaces (free)

1. Create a Space at huggingface.co/new-space: SDK **Docker**, hardware **CPU basic**.
2. In the Space settings, add the secret `NEBIUS_API_KEY`.
3. Push the project with the Space README:
   ```bash
   make demo
   git clone https://huggingface.co/spaces/<user>/gapcloser hf-space && cd hf-space
   rsync -a --exclude .git --exclude .venv --exclude runs/live ../ ./
   cp deploy/hf-space/README.md README.md
   git lfs track "*.webp" && git add -A && git commit -m "GapCloser demo" && git push
   ```
4. The Space builds the Dockerfile (about 3 minutes) and serves on port 7860.

## Nebius AI Cloud CPU VM

1. In the Nebius console, create a Compute VM: smallest CPU preset (2 vCPU / 8 GB), Ubuntu 22.04, public IP, your SSH key.
2. On the VM:
   ```bash
   sudo apt-get update && sudo apt-get install -y docker.io git
   git clone <your GitHub repo> gapcloser && cd gapcloser
   sudo docker build -t gapcloser .
   sudo docker run -d --restart unless-stopped -p 80:7860 -e NEBIUS_API_KEY=... gapcloser
   ```
3. Open `http://<public-ip>/`. Stop the VM after judging to stop billing.

## GitHub Pages (static, free)

```bash
make demo
mkdir -p site && cp dashboard/dist/gapcloser.standalone.html site/index.html
```

Publish `site/` with GitHub Pages (Settings → Pages → deploy from a branch or an Actions upload). The page is self-contained (clips are embedded), about 4 MB.
