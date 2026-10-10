"""App footage for the demo film: the Studio in action, recorded as real screen video (Chrome DevTools screencast).

Two clips:
  studio  the Studio page replaying a recorded Nemotron session (driving: roadside video -> tracking -> the agent's
          notebook -> results -> Newton replay check -> CARLA export); deterministic, no network, no credits
  ask     "Ask Tether" on the live server: a question about the visitor's own session, answered by Nemotron on
          Nebius Token Factory (a few LLM calls)
A drawn cursor moves to each control before it is clicked, so the viewer can follow the action.

Run: .venv/bin/python -m video.footage [--live URL] [--only studio|ask]   -> video/out/footage/<clip>/ + index.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "video" / "out" / "footage"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
STUDIO = (ROOT / "dashboard/dist/studio.standalone.html").as_uri()
W, H = 1440, 810

CURSOR = """(() => {
  if (document.getElementById('__cur')) return 'ok';
  const st = document.createElement('style');
  st.textContent = `#__cur{position:fixed;z-index:2147483647;pointer-events:none;transition:left .75s cubic-bezier(.45,0,.2,1),top .75s cubic-bezier(.45,0,.2,1)}
    .__ring{position:fixed;z-index:2147483646;pointer-events:none;width:44px;height:44px;margin:-22px 0 0 -22px;border-radius:50%;
      border:3px solid #76b900;animation:__r .55s ease-out forwards}@keyframes __r{from{transform:scale(.3);opacity:1}to{transform:scale(1.4);opacity:0}}
    #askfab{display:none}`;
  document.head.appendChild(st);
  const c = document.createElement('div'); c.id = '__cur';
  c.innerHTML = '<svg width="28" height="28" viewBox="0 0 28 28"><path d="M4 2 L4 22 L9.5 17 L13.2 25.5 L16.8 24 L13.2 15.8 L21 15.8 Z" fill="#fff" stroke="#111" stroke-width="1.6" stroke-linejoin="round"/></svg>';
  c.style.left = '760px'; c.style.top = '560px'; document.body.appendChild(c); return 'ok';
})()"""


def act(sel: str, scroll: bool = True) -> str:
    """Move the drawn cursor to the element, then click it with a ring."""
    return f"""(async () => {{
  const e = document.querySelector({json.dumps(sel)}); if (!e) return 'missing {sel}';
  {"e.scrollIntoView({block: 'nearest', behavior: 'smooth'}); await new Promise(f => setTimeout(f, 350));" if scroll else ""}
  const r = e.getBoundingClientRect(), x = r.left + Math.min(r.width / 2, 60), y = r.top + r.height / 2, c = document.getElementById('__cur');
  c.style.left = x + 'px'; c.style.top = y + 'px'; await new Promise(f => setTimeout(f, 850));
  const g = document.createElement('div'); g.className = '__ring'; g.style.left = x + 'px'; g.style.top = y + 'px';
  document.body.appendChild(g); setTimeout(() => g.remove(), 700); e.click(); return 'ok';
}})()"""


def glide(target: str, ms: int) -> str:
    """Scroll the page smoothly so the element (or a pixel offset) sits near the top."""
    to = f"(document.querySelector({json.dumps(target)}).getBoundingClientRect().top + scrollY - 70)" if not target.isdigit() else target
    return f"""(async () => {{
  const y0 = scrollY, y1 = Math.max(0, {to}), t0 = performance.now();
  await new Promise(res => {{ (function f(now) {{ const p = Math.min(1, (now - t0) / {ms}), e = p < .5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
    scrollTo(0, y0 + (y1 - y0) * e); p < 1 ? requestAnimationFrame(f) : res(); }})(performance.now()); }}); return 'ok';
}})()"""


def typing(sel: str, text: str) -> str:
    return f"""(async () => {{
  const e = document.querySelector({json.dumps(sel)}); e.focus(); e.value = '';
  for (const ch of {json.dumps(text)}) {{ e.value += ch; e.dispatchEvent(new Event('input', {{bubbles: true}})); await new Promise(f => setTimeout(f, 55)); }}
  return 'ok';
}})()"""


def studio_plan() -> list[dict]:
    return [
        {"go": f"{STUDIO}?lang=en#driving"}, {"wait": 2500}, {"js": CURSOR}, {"js": "window.scrollTo(0,0); 'ok'"},
        {"rec": "studio"}, {"wait": 900},
        {"js": act("[data-id=stop-line]")}, {"wait": 2600},
        {"mark": "track"}, {"js": act("#track")}, {"wait": 1600},
        {"js": act("#vtabs [data-v=measure]", scroll=False)}, {"wait": 3600},  # the tracked runs on the video
        {"mark": "agent"}, {"js": act("#vtabs [data-v=diag]", scroll=False)}, {"js": glide("#nb", 1000)}, {"wait": 8500},
        {"mark": "results"}, {"js": act("#to-res")}, {"wait": 1800}, {"js": glide("0", 600)}, {"wait": 2600},
        {"js": glide("#r-map", 1800)}, {"wait": 3200},
        {"js": glide("#r-forest", 1800)}, {"wait": 2600},
        {"mark": "retrain"}, {"js": act("#to-x")}, {"wait": 1200}, {"js": glide("0", 500)},
        {"js": act("#t-run")}, {"wait": 7500}, {"js": glide("#t-body", 1400)}, {"wait": 3000},
        {"js": glide("#t-vid-box", 1400)}, {"wait": 4500},
        {"mark": "export"}, {"js": act("#to-x2")}, {"wait": 1400}, {"js": glide("0", 500)},
        {"js": act("#v-run")}, {"wait": 2800}, {"js": glide("#verify", 1200)}, {"wait": 2200},
        {"js": act("[data-id=carla]")}, {"wait": 900}, {"js": glide("#x-code", 1200)}, {"wait": 3000},
        {"stop": True},
    ]


def ask_plan(live: str) -> list[dict]:
    q = "Which number should I trust least, and why?"
    return [
        {"go": f"{live}/studio?lang=en#robot"}, {"wait": 4000}, {"js": CURSOR},
        {"js": "document.querySelector('[data-id=lab-bench]').click(); 'ok'"},
        {"until": "typeof state !== 'undefined' && !!state.result", "max": 150000},
        {"wait": 3000}, {"js": "document.querySelector('#to-res').click(); 'ok'"}, {"wait": 2500}, {"js": "window.scrollTo(0,0); 'ok'"},
        {"js": "document.getElementById('askfab').style.display='block'; 'ok'"}, {"wait": 800},
        {"rec": "ask"}, {"wait": 700},
        {"js": act("#askfab", scroll=False)}, {"wait": 1100},
        {"js": act("#chat-in", scroll=False)}, {"js": typing("#chat-in", q)}, {"wait": 400},
        {"js": act("#chat-send", scroll=False)},
        {"until": "document.querySelectorAll('#chat .msg.bot').length > 0", "max": 30000},
        {"wait": 6500}, {"stop": True},
    ]


async def run(steps: list[dict], port: int = 9341) -> None:
    import websockets

    prof = tempfile.mkdtemp()
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
                             "--hide-scrollbars", "--autoplay-policy=no-user-gesture-required", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    clip, frames, marks, n, ack = None, [], {}, 0, 10 ** 6
    try:
        for _ in range(50):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
                url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:  # noqa: BLE001 - Chrome still starting
                time.sleep(0.2)
        async with websockets.connect(url, max_size=1 << 28) as ws:

            async def send(method, params=None):
                nonlocal n, ack
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params or {}}))
                while True:
                    m = json.loads(await ws.recv())
                    if m.get("method") == "Page.screencastFrame":
                        p = m["params"]
                        ack += 1
                        await ws.send(json.dumps({"id": ack, "method": "Page.screencastFrameAck", "params": {"sessionId": p["sessionId"]}}))
                        if clip is not None:
                            k = len(frames) + 1
                            (OUT / clip / f"f{k:05d}.jpg").write_bytes(base64.b64decode(p["data"]))
                            frames.append(p["metadata"].get("timestamp") or time.time())
                    elif m.get("method") == "Runtime.exceptionThrown":
                        print("page error:", json.dumps(m["params"]["exceptionDetails"])[:300])
                    if m.get("id") == n:
                        return m

            async def evaluate(js):
                r = await send("Runtime.evaluate", {"expression": js, "awaitPromise": True, "returnByValue": True})
                return r.get("result", {}).get("result", {}).get("value")

            await send("Runtime.enable")
            await send("Page.enable")
            await send("Emulation.setDeviceMetricsOverride", {"width": W, "height": H, "deviceScaleFactor": 1, "mobile": False})
            t_start = None
            for st in steps:
                if "go" in st:
                    await send("Page.navigate", {"url": st["go"]})
                    await asyncio.sleep(1.2)
                if "rec" in st:
                    clip, frames, marks = st["rec"], [], {}
                    shutil.rmtree(OUT / clip, ignore_errors=True)
                    (OUT / clip).mkdir(parents=True)
                    t_start = time.time()
                    await send("Page.startScreencast", {"format": "jpeg", "quality": 88, "maxWidth": W, "maxHeight": H, "everyNthFrame": 1})
                if "mark" in st:
                    marks[st["mark"]] = time.time()
                if "wait" in st:
                    end = time.time() + st["wait"] / 1000
                    while time.time() < end:
                        await send("Runtime.evaluate", {"expression": "1"})
                        await asyncio.sleep(0.03)
                if "until" in st:
                    end = time.time() + st.get("max", 30000) / 1000
                    while time.time() < end and not await evaluate(st["until"]):
                        await asyncio.sleep(0.25)
                if "js" in st:
                    v = await evaluate(st["js"])
                    if v != "ok":
                        print("step:", str(v)[:120])
                if "stop" in st:
                    await send("Page.stopScreencast")
                    t_end = time.time()
                    idx = json.loads((OUT / "index.json").read_text()) if (OUT / "index.json").exists() else {}
                    idx[clip] = {"w": W, "h": H, "times": [round(t - t_start, 3) for t in frames],
                                 "marks": {k: round(v - t_start, 3) for k, v in marks.items()}, "length": round(t_end - t_start, 3)}
                    (OUT / "index.json").write_text(json.dumps(idx, indent=1))
                    print(f"{clip}: {len(frames)} frames over {t_end - t_start:.1f} s, marks {idx[clip]['marks']}")
                    clip = None
    finally:
        proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="http://localhost:8000", help="server for the Ask clip (needs TETHER_LLM=tokenfactory)")
    ap.add_argument("--only", choices=["studio", "ask"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "studio"):
        asyncio.run(run(studio_plan()))
    if a.only in (None, "ask"):
        asyncio.run(run(ask_plan(a.live.rstrip("/")), port=9342))


if __name__ == "__main__":
    main()
