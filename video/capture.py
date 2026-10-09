"""Screen captures for the demo video, driven through Chrome DevTools (headless Chrome, 1920x1080).

Walks the real pages: the overview and console on the live server, and the Studio standalone page, which replays
the recorded Nemotron + Newton runs through the same UI a visitor uses. Each step is either a navigation, a short
wait, a JavaScript action (a click, a scroll), or a screenshot of the viewport or of one element.

Run: .venv/bin/python -m video.capture [--out video/out/caps]   (needs `make serve` on localhost:8000)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
LIVE = "http://localhost:8000"
STUDIO = (ROOT / "dashboard/dist/studio.standalone.html").as_uri()


def click(sel: str) -> str:
    return f"document.querySelector({json.dumps(sel)}).click(); 'ok'"


def show(sel: str, block: str = "start") -> str:
    return f"document.querySelector({json.dumps(sel)}).scrollIntoView({{block: {json.dumps(block)}}}); 'ok'"


def plan() -> list[dict]:
    hide_fab = "document.querySelector('#askfab') && (document.querySelector('#askfab').style.display='none'); 'ok'"
    return [
        {"go": f"{LIVE}/?lang=en"}, {"wait": 3500}, {"shot": "home_hero"},
        {"js": show(".doms", "center")}, {"wait": 800}, {"shot": "home_domains"},
        {"js": show(".ev", "center")}, {"wait": 500}, {"shot": "home_evidence"},
        {"go": f"{STUDIO}?lang=en#driving"}, {"wait": 2000}, {"js": hide_fab}, {"shot": "studio_tabs"},
        {"js": click("[data-id=stop-line]")}, {"wait": 1500}, {"js": hide_fab}, {"shot": "studio_measure"},
        {"js": click("#track")}, {"wait": 2500}, {"shot": "studio_tracked"},
        {"js": click("#to-diag")}, {"wait": 600}, {"js": click("#run")}, {"wait": 9000},
        {"js": show("#nb li:nth-child(3)")}, {"wait": 400}, {"shot": "studio_diag"},
        {"js": click("#to-res")}, {"wait": 2500}, {"js": "window.scrollTo(0,0); 'ok'"}, {"wait": 2600}, {"shot": "studio_results"},
        {"js": click("#reveal")}, {"wait": 1200}, {"js": show("#r-forest", "center")}, {"wait": 400}, {"shot": "studio_truth"},
        # take 2: the runs the Studio asked for, then the final calibration
        {"js": click("#more")}, {"wait": 1500}, {"js": click("#track")}, {"wait": 2500},
        {"js": click("#to-diag")}, {"wait": 600}, {"js": click("#run")}, {"wait": 9000},
        {"js": click("#to-res")}, {"wait": 2500}, {"js": click("#to-x")}, {"wait": 800},
        {"js": click("#v-run")}, {"wait": 1800}, {"js": show("#verify", "center")}, {"wait": 400}, {"shot": "studio_verify"},
        {"js": click("[data-id=carla]")}, {"wait": 600}, {"js": show("#x-code", "center")}, {"wait": 400}, {"shot": "studio_carla"},
        {"go": f"{STUDIO}?lang=en#factory"}, {"wait": 1800}, {"js": hide_fab}, {"js": click("[data-id=press-line]")}, {"wait": 1200},
        {"js": click("#to-diag")}, {"wait": 600}, {"js": click("#run")}, {"wait": 8000}, {"js": click("#to-res")}, {"wait": 2000},
        {"js": click("#reveal")}, {"wait": 1000}, {"js": "window.scrollTo(0,0); 'ok'"}, {"wait": 500}, {"shot": "factory_results"},
        {"go": f"{LIVE}/console?lang=en#gap-bench"}, {"wait": 4000}, {"shot": "console_bench"},
    ]


async def run(steps: list[dict], out: Path) -> None:
    import websockets

    port = 9334
    prof = tempfile.mkdtemp()
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
                             "--hide-scrollbars", "--autoplay-policy=no-user-gesture-required", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
                url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:  # noqa: BLE001 - Chrome still starting
                time.sleep(0.2)
        async with websockets.connect(url, max_size=1 << 28) as ws:
            n = 0

            async def send(method, params=None):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params or {}}))
                while True:
                    m = json.loads(await ws.recv())
                    if m.get("method") == "Runtime.exceptionThrown":
                        print("page error:", json.dumps(m["params"]["exceptionDetails"])[:300])
                    if m.get("id") == n:
                        return m

            await send("Runtime.enable")
            await send("Page.enable")
            await send("Emulation.setDeviceMetricsOverride", {"width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False})
            for st in steps:
                if "go" in st:
                    await send("Page.navigate", {"url": st["go"]})
                    await asyncio.sleep(1.2)
                if "wait" in st:
                    end = time.time() + st["wait"] / 1000
                    while time.time() < end:
                        await send("Runtime.evaluate", {"expression": "1"})
                        await asyncio.sleep(0.2)
                if "js" in st:
                    r = await send("Runtime.evaluate", {"expression": st["js"], "awaitPromise": True, "returnByValue": True})
                    if r.get("result", {}).get("exceptionDetails"):
                        print("step failed:", st["js"][:80], json.dumps(r["result"]["exceptionDetails"])[:200])
                if "shot" in st:
                    r = await send("Page.captureScreenshot", {"format": "png"})
                    (out / f"{st['shot']}.png").write_bytes(base64.b64decode(r["result"]["data"]))
                    print("shot", st["shot"])
    finally:
        proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "video" / "out" / "caps")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(plan(), a.out))


if __name__ == "__main__":
    main()
