"""Build a self-contained dashboard: inline runs/demo/bundle.json, its clips (data URIs) and 3D replays.

The 3D viewer's three.js build (vendored, MIT) and the decimated Franka FR3 meshes are inlined too, so the
standalone page works offline. Run: .venv/bin/python -m dashboard.build  ->  dashboard/dist/gapcloser.html
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
THREE = ASSETS / "three.min.js"
FRANKA = ASSETS / "franka_fr3.json"
HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
        '</head><body style="margin:0">')


def _script_safe(s: str) -> str:
    return s.replace("</", "<\\/")


def _template(bundle_json: str) -> str:
    """template.html with the bundle, three.js and the Franka meshes substituted."""
    tpl = (HERE / "template.html").read_text()
    three = THREE.read_text() if THREE.exists() else ""
    if three.startswith("console.warn("):  # silence the UMD-deprecation banner, keep the expression valid
        three = "void(" + three[len("console.warn("):]
    franka = FRANKA.read_text() if FRANKA.exists() else "null"
    tpl = tpl.replace("/*__THREE__*/", _script_safe(three), 1)
    tpl = tpl.replace('"__FRANKA__"', _script_safe(franka), 1)
    return tpl.replace('"__BUNDLE__"', bundle_json, 1)


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
            rp = clip.get("replay")
            if isinstance(rp, str):
                path = base / rp
                clip["replay"] = json.loads(path.read_text()) if path.exists() else None
    payload = _script_safe(json.dumps(bundle, separators=(",", ":")))
    html = _template(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)  # fragment: the Artifact publisher adds the document skeleton
    standalone = out.with_name(out.stem + ".standalone.html")
    standalone.write_text(HEAD + html + "</body></html>")
    return out


def build_live(out: Path) -> Path:
    """Server variant: no embedded data; the page loads /api/bundle and /api/status at runtime."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HEAD + _template("null") + "</body></html>")
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
