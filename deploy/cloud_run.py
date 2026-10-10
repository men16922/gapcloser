"""Deploy the live server (pages + API) to Google Cloud Run with gcloud.

Builds the Dockerfile on Cloud Build from the committed files it copies (git archive HEAD, so .env and local runs never
leave the machine), keeps NEBIUS_API_KEY in Secret Manager, and runs one instance at most (sessions live in memory)
that scales to zero when nobody visits. CPU stays allocated while the instance is up, so analyses and retraining keep
running between the page's polls.

Needs gcloud logged in with a project that has billing on. Run:
  .venv/bin/python -m deploy.cloud_run [--project ID] [--region us-central1] [--no-secret]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVICE, SECRET = "tether", "tether-nebius-key"
# what the Dockerfile copies (keep in step with it)
PATHS = ["Dockerfile", ".dockerignore", "requirements.txt", "agent", "sim", "eval", "server", "studio", "dashboard", "runs/demo"]


def gcloud(*args: str, stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["gcloud", *args], input=stdin, text=True, check=check, capture_output=stdin is not None or not check)


def stage(out: Path) -> None:
    tar = subprocess.run(["git", "archive", "--format=tar", "HEAD", *PATHS], cwd=ROOT, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=BytesIO(tar)) as t:
        t.extractall(out, filter="data")


def put_secret(project: str) -> None:
    from agent.llm import load_dotenv

    load_dotenv()
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        raise SystemExit("NEBIUS_API_KEY is not set (env or .env): the service would run without Nemotron")
    if gcloud("secrets", "describe", SECRET, "--project", project, check=False).returncode:
        gcloud("secrets", "create", SECRET, "--project", project, "--replication-policy", "automatic", "--data-file=-", stdin=key)
    else:
        gcloud("secrets", "versions", "add", SECRET, "--project", project, "--data-file=-", stdin=key)
    number = gcloud("projects", "describe", project, "--format=value(projectNumber)", check=False).stdout.strip()
    gcloud("secrets", "add-iam-policy-binding", SECRET, "--project", project, "--role", "roles/secretmanager.secretAccessor",
           "--member", f"serviceAccount:{number}-compute@developer.gserviceaccount.com", "--condition=None", stdin="")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True,
                                                        text=True).stdout.strip())
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--no-secret", action="store_true", help="keep the secret's current version")
    a = ap.parse_args()
    if not a.no_secret:
        put_secret(a.project)
    with tempfile.TemporaryDirectory() as td:
        stage(Path(td))
        subprocess.run(["gcloud", "run", "deploy", SERVICE, "--source", td, "--project", a.project, "--region", a.region,
                        "--allow-unauthenticated", "--cpu", "2", "--memory", "4Gi", "--min-instances", "0", "--max-instances", "1",
                        "--concurrency", "40", "--timeout", "3600", "--no-cpu-throttling", "--cpu-boost",
                        "--execution-environment", "gen2", "--set-secrets", f"NEBIUS_API_KEY={SECRET}:latest",
                        "--set-env-vars", "TETHER_MAX_SESSIONS=20", "--quiet"],
                       check=True, env={**os.environ, "CLOUDSDK_BUILDS_TIMEOUT": "1800"})
    url = gcloud("run", "services", "describe", SERVICE, "--project", a.project, "--region", a.region,
                 "--format=value(status.url)", check=False).stdout.strip()
    print(f"live: {url}  (studio: {url}/studio, console: {url}/console)")


if __name__ == "__main__":
    main()
