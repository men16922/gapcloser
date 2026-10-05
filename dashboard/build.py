"""Build a self-contained dashboard: inline runs/demo/bundle.json and its clips as data URIs.

Run: .venv/bin/python -m dashboard.build  ->  dashboard/dist/gapcloser.html (open directly in a browser)
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

HERE = Path(__file__).parent


def build(bundle_path: Path, out: Path) -> Path:
    bundle = json.loads(bundle_path.read_text())
    base = bundle_path.parent
    for run in bundle["runs"]:
        for e in run["events"]:
            clip = e.get("clip")
            if not clip:
                continue
            for k in ("real", "sim"):
                data = (base / clip[k]).read_bytes()
                clip[k] = "data:image/webp;base64," + base64.b64encode(data).decode()
    tpl = (HERE / "template.html").read_text()
    payload = json.dumps(bundle, separators=(",", ":")).replace("</", "<\\/")
    html = tpl.replace('"__BUNDLE__"', payload, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)  # fragment: the Artifact publisher adds the document skeleton
    standalone = out.with_name(out.stem + ".standalone.html")
    standalone.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                          '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
                          '</head><body style="margin:0">' + html + "</body></html>")
    return out


def build_live(out: Path) -> Path:
    """Server variant: no embedded data; the page loads /api/bundle and /api/status at runtime."""
    tpl = (HERE / "template.html").read_text().replace('"__BUNDLE__"', "null", 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
                   '</head><body style="margin:0">' + tpl + "</body></html>")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=Path("runs/demo/bundle.json"))
    ap.add_argument("--out", type=Path, default=HERE / "dist" / "gapcloser.html")
    a = ap.parse_args()
    live = build_live(a.out.with_name("gapcloser.live.html"))
    print(f"wrote {live}")
    if a.bundle.exists():
        out = build(a.bundle, a.out)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
