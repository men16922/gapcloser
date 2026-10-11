"""Where things live in the repository, in one place, so no module counts parent directories."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"  # recorded runs the pages and tests read (runs/demo, runs/studio-demo, runs/proof, runs/bench)
WEB = REPO / "tether" / "web"  # page sources; built pages go to web/dist
SAMPLES = REPO / "tether" / "studio" / "samples"  # Studio's example data with ground truth
FILM_OUT = REPO / "film" / "out"  # demo film renders and app footage (git-ignored)
