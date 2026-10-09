"""Build a self-contained dashboard: inline runs/demo/bundle.json, its clips (data URIs) and 3D replays.

The 3D viewer's three.js build (vendored, MIT) and the decimated Franka FR3 meshes are inlined too, so the
standalone page works offline. Run: .venv/bin/python -m dashboard.build  ->  dashboard/dist/gapcloser.html
"""

from __future__ import annotations

import argparse
import base64
import json
import os
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
    studio = "/studio" if bundle_json == "null" else os.environ.get("GAPCLOSER_STUDIO_URL", "")
    tpl = tpl.replace("__STUDIO_URL__", studio, 1)
    tpl = tpl.replace("__HOME_URL__", "/" if bundle_json == "null" else os.environ.get("GAPCLOSER_HOME_URL", ""))
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


STUDIO_REC = Path("runs/studio-demo")
MIME = {".mp4": "video/mp4", ".jpg": "image/jpeg", ".webm": "video/webm"}


def build_studio(rec_dir: Path = STUDIO_REC, out_dir: Path = HERE / "dist", console_url: str | None = None) -> list[Path]:
    """Studio pages: studio.live.html (served at /studio, talks to the API) and, when recorded sessions exist,
    studio.standalone.html (sessions + media inlined; replays the recorded Nemotron runs offline)."""
    from studio.domains import page_data

    tpl = (HERE / "studio.html").read_text()
    tpl = tpl.replace("/*__STUDIO_DOMAINS__*/null", _script_safe(json.dumps(page_data(), separators=(",", ":"))), 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    live = out_dir / "studio.live.html"
    live.write_text(tpl.replace("__HOME_URL__", "/"))
    tpl = tpl.replace("__HOME_URL__", os.environ.get("GAPCLOSER_HOME_URL", ""))  # the published overview, if any
    outs = [live]
    order = ["flick", "lab-bench", "short-reach", "stop-line", "brake-log", "press-line"]
    recs = [json.loads((rec_dir / f"{k}.json").read_text()) for k in order if (rec_dir / f"{k}.json").exists()]
    if not recs:
        return outs
    media = {}
    for r in recs:
        for v in r["videos"]:
            for key in ("file", "frame"):
                path = rec_dir / v[key]
                media[v[key]] = f"data:{MIME[path.suffix]};base64," + base64.b64encode(path.read_bytes()).decode()
        for st in r["stages"]:  # the page re-derives these; keep the payload small
            if st.get("result"):
                st["result"].pop("session", None)
    model = next((r["stages"][-1]["result"]["agent"]["model_name"] for r in recs
                  if (r["stages"][-1]["result"] or {}).get("agent")), None)
    console_url = console_url or os.environ.get("GAPCLOSER_CONSOLE_URL")
    data = {"samples": recs, "media": media, "model": model, "provider": "tokenfactory", "console_url": console_url}
    html = tpl.replace("/*__STUDIO_DATA__*/null", _script_safe(json.dumps(data, separators=(",", ":"))), 1)
    sa = out_dir / "studio.standalone.html"
    sa.write_text(html)
    outs.append(sa)
    # Artifact fragment: the publisher adds the document skeleton (doctype, html/head/body, charset, viewport)
    frag = html
    for tag in ("<!doctype html>", '<html lang="en">', "<head>", "</head>", "<body>", "</body>", "</html>",
                '<meta charset="utf-8">', '<meta name="viewport" content="width=device-width, initial-scale=1">'):
        frag = frag.replace(tag, "", 1)
    art = out_dir / "studio.html"
    art.write_text(frag.strip() + "\n")
    outs.append(art)
    return outs


def build_home(out_dir: Path = HERE / "dist") -> list[Path]:
    """Overview as a shared page: Newton stills inlined, links to the published Console and Studio (env
    GAPCLOSER_CONSOLE_URL / GAPCLOSER_STUDIO_URL). Without both links there is nothing to point at; skipped."""
    console, studio = os.environ.get("GAPCLOSER_CONSOLE_URL"), os.environ.get("GAPCLOSER_STUDIO_URL")
    if not (console and studio):
        return []
    from studio.video import first_frame_jpeg

    samples = HERE.parent / "studio" / "samples"
    img = {}
    for key, src in (("flick", "flick-video.mp4"), ("stop-line", "stop-line-video.mp4"), ("press-line", "press-line-frame.jpg")):
        data = (samples / src).read_bytes() if src.endswith(".jpg") else first_frame_jpeg(samples / src)[0]
        img[key] = "data:image/jpeg;base64," + base64.b64encode(data).decode()
    html = (HERE / "home.html").read_text().replace(
        "/*__HOME_STATIC__*/null", _script_safe(json.dumps({"console": console, "studio": studio, "img": img})), 1)
    for tag in ("<!doctype html>", '<html lang="en">', "<head>", "</head>", "<body>", "</body>", "</html>",
                '<meta charset="utf-8">', '<meta name="viewport" content="width=device-width, initial-scale=1">'):
        html = html.replace(tag, "", 1)
    out = out_dir / "home.html"
    out.write_text(html.strip() + "\n")
    return [out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=Path("runs/demo/bundle.json"))
    ap.add_argument("--out", type=Path, default=HERE / "dist" / "gapcloser.html")
    a = ap.parse_args()
    live = build_live(a.out.with_name("gapcloser.live.html"))
    print(f"wrote {live}")
    for p in build_studio() + build_home():
        print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB)")
    if a.bundle.exists():
        out = build(a.bundle, a.out)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
