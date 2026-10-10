# Deploying Tether

One Docker image serves the pages and the API. It needs no GPU: NVIDIA Newton runs on CPU and Nemotron is called over
HTTPS on Nebius Token Factory. The public deployment runs on Google Cloud Run.

## Google Cloud Run

```bash
make deploy        # = .venv/bin/python -m deploy.cloud_run [--project ID] [--region us-central1]
```

`deploy/cloud_run.py` does the whole thing with gcloud (logged in, a project with billing on):

1. Puts `NEBIUS_API_KEY` from `.env` into Secret Manager (`tether-nebius-key`) and lets the service read it.
2. Builds the Dockerfile on Cloud Build from the committed files only (`git archive HEAD`), so `.env`, uploads and
   local runs never leave the machine. The build takes about 10 minutes.
3. Deploys the service `tether`: 2 vCPU, 4 GiB, **one instance at most** (Studio sessions live in memory), scales to
   zero when nobody visits, CPU kept on while the instance is up (analyses and retraining run between the page's polls).

The first visit after an idle spell starts the container (about 10–20 s). Cost: Cloud Run bills only while the
instance is up (it stops about 15 minutes after the last request) and the monthly free tier covers dozens of hours;
the image in Artifact Registry costs cents per month. To take it down:
`gcloud run services delete tether --region us-central1`.

## Build and check locally

```bash
make serve           # http://localhost:8000  (TETHER_LLM=tokenfactory|local|none)
make docker          # docker build -t tether .
docker run --rm -p 7860:7860 -e NEBIUS_API_KEY tether   # http://localhost:7860
# without a key, point the container at the host's Ollama:
docker run --rm -p 7860:7860 -e TETHER_LLM=local -e OLLAMA_BASE_URL=http://host.docker.internal:11434/v1/ tether
```

The image bundles `runs/demo` (recorded scenarios and clips, committed to git). Re-run `make demo` only to refresh them.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `TETHER_LLM` | `tokenfactory` in the image | `tokenfactory`, `local` (Ollama) or `none` (rule-based diagnoser) |
| `NEBIUS_API_KEY` | — | Token Factory key |
| `TETHER_DIAG_MODEL` | `nemotron super` | model hint for diagnosis (words matched against the model list) |
| `TETHER_MAX_LLM_CALLS` | 200 | total LLM calls for the server's lifetime; then the rule-based diagnoser takes over |
| `TETHER_RUNS_PER_HOUR` | 6 | runs per visitor IP per hour |
| `TETHER_MAX_TURNS_PER_RUN` | 24 | Nemotron tool-agent chat turns per live run (max 8 per diagnosis); past it, the rule-based diagnoser takes over |
| `TETHER_MAX_SESSIONS` | 60 (20 on Cloud Run) | Studio sessions kept; the oldest idle one is dropped with its uploads |
| `PORT` | 7860 | listen port (Cloud Run sets it) |

"Stump the agent": with an LLM configured, every live run uses the open-world path. Visitors pick a preset (wet strip, rough strip, lens distortion, three faults, surprise me) or build a world, including a friction strip (`patch_y0` + `patch_mu`, counted as one fault) and lens distortion; the Nemotron tool agent's lab notebook streams step by step (`agent_step` events) while it experiments.

Cost guards built in: one run at a time, max 4 iterations, max 3 hidden faults (a strip counts as one), values inside the parameter bounds and `patch_mu` ≤ 0.95 (stickier strips tip the cube), per-IP rate limit, a global LLM-call cap, a per-run cap of 24 agent turns, and a 2048-token output cap. Every chat turn counts as one call. Tool-agent turns resend the growing conversation (about 2–6k prompt tokens each; about 17k in + 2k out per world with Super, ≈ $0.007), so 200 calls stay around 1M tokens in the worst case.

**Studio** (`/studio`) is served by the same container. Visitors upload a phone video (up to 120 MB, 90 s analysed) or a robot log and get a calibrated simulator. Uploads live under `runs/studio/<session>` and the newest `TETHER_MAX_SESSIONS` sessions are kept. Studio analyses share the global LLM-call cap and are capped at 8 agent turns each. One agent analysis runs at a time; others wait up to 2 minutes, then fall back to the offline fit.
