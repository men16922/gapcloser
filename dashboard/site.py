"""The public site: Overview (index.html), Studio and Agent console as static pages that replay the recorded runs,
linked to each other with relative URLs. Serves from any static host (GitHub Pages: `make pages`).

Run: .venv/bin/python -m dashboard.site [--out site]
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

HERE = Path(__file__).parent


def build_site(out: Path) -> list[Path]:
    os.environ.update(TETHER_HOME_URL="./", TETHER_STUDIO_URL="studio.html", TETHER_CONSOLE_URL="console.html")
    from dashboard.build import HEAD, build, build_home, build_studio

    dist = HERE / "dist"
    build_studio()
    build_home()
    build(Path("runs/demo/bundle.json"), dist / "console.html")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dist / "studio.standalone.html", out / "studio.html")
    shutil.copyfile(dist / "console.standalone.html", out / "console.html")
    (out / "index.html").write_text(HEAD + (dist / "home.html").read_text() + "</body></html>")  # home is built as a fragment
    (out / ".nojekyll").write_text("")  # serve files as they are
    return [out / n for n in ("index.html", "studio.html", "console.html")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("site"))
    for p in build_site(ap.parse_args().out):
        print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB)")
