"""Tether demo film: motion graphics drawn in a browser stage (video/stage.html) around about a minute of the app in
action (screen recordings from video/footage.py), captured frame by frame with headless Chrome, narrated (ElevenLabs
from .env, or macOS say), over a soft generated music bed that ducks under the voice. Every number comes from
runs/proof/*.json and the recorded Studio sessions.

Needs: Google Chrome, ffmpeg, the EV-RealPhys clips (make real-benchmark), the app footage (python -m video.footage)
and the Newton renders (python -m studio.rollout_video brake-log; python -m studio.train_montage brake-log).
Run: .venv/bin/python -m video.film [--voice elevenlabs|say] [--only scene]   -> video/out/tether_film.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import shutil
import subprocess
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "video" / "out"
WORK = OUT / "film"
FOOTAGE = OUT / "footage"
TTS_CACHE = OUT / "tts-cache"  # ElevenLabs audio per sentence, so a rebuild does not spend credits twice
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FPS = 30
SR = 44100

# scene, narration, minimum seconds
SCRIPT = [
    ("cold", "Every robot learns in a simulator first. Then it meets the real world.", 8.5),
    ("gap", "And the real world is a little different. The table is slipperier, the motor is weaker, and a policy that "
            "lands every push in simulation misses on the real one.", 11),
    ("title", "Tether ties your simulator to the real world.", 6),
    # the app in action (screen recordings, video/footage.py)
    ("appInput", "Open Tether Studio and pick a field. Give it a phone video, or the log a robot already writes. A box of "
                 "known size gives the scale, and every run is tracked to a stop.", 12),
    ("appAgent", "Then NVIDIA Nemotron 3 Super, running on Nebius Token Factory, works through the evidence like an "
                 "engineer: where things slow down, which effects explain it, which runs would settle the rest. Least "
                 "squares computes every number, each with an honest interval.", 17),
    ("appResult", "What comes back is your simulator, corrected: friction mapped along the road, what differs from your "
                  "current simulator, and the next experiment worth running.", 13),
    ("retrain", "Then Tether retrains the policy in sixteen parallel NVIDIA Newton worlds drawn from what it measured. "
                "On the vehicle log, hidden-world success goes from forty-two percent to one hundred.", 15),
    ("appExport", "Before export, every run is replayed in NVIDIA Newton, and the calibrated physics goes straight into "
                  "Newton, Isaac Lab or CARLA.", 11),
    ("appAsk", "And you can ask Tether about your own result. Nemotron answers from your session's numbers.", 10),
    ("proofReal", "It holds outside our own simulator. On real objects from a public benchmark, friction lands within "
                  "five hundredths of an independent measurement, for four of five.", 11),
    ("proofEngine", "On worlds built by a different physics engine, policies trained with Tether reach eighty-nine "
                    "percent. The best domain randomization reaches twelve.", 9),
    ("honest", "And when the AI proposes physics the data cannot support, the cross-check takes it out.", 8),
    ("close", "Tether. Built with NVIDIA Newton, and Nemotron on Nebius Token Factory.", 7),
]


# ------------------------------------------------------------------ assets and numbers
def frames_of(video: Path, name: str, scale: str | None = None) -> dict:
    d = WORK / "assets" / name
    if not d.exists() or not any(d.glob("f*.jpg")):
        d.mkdir(parents=True, exist_ok=True)
        vf = ["-vf", f"scale={scale}"] if scale else []
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), *vf, "-q:v", "3", str(d / "f%04d.jpg")], check=True)
    return {"dir": f"assets/{name}", "n": len(list(d.glob("f*.jpg")))}


def build_data() -> dict:
    from eval.real_benchmark import DATA as EV

    data = {}
    mug = EV / "ev-realphys" / "val_sliding" / "000015" / "rgb"
    if not mug.exists():
        raise SystemExit("EV-RealPhys not found: run `make real-benchmark` first")
    link = WORK / "assets" / "mug"
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(mug)
    n_gt = len(json.loads((mug.parent / "scene_gt.json").read_text()))
    data["mug"] = {"dir": "assets/mug", "n": n_gt}

    # screen recordings of the app (video/footage.py)
    idx = json.loads((FOOTAGE / "index.json").read_text()) if (FOOTAGE / "index.json").exists() else {}
    if not {"studio", "ask"} <= set(idx):
        raise SystemExit("app footage missing: run `python -m video.footage` (needs make serve for the Ask clip)")
    data["footage"] = {}
    for clip, meta in idx.items():
        link = WORK / "assets" / f"footage-{clip}"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(FOOTAGE / clip)
        data["footage"][clip] = {**meta, "dir": f"assets/footage-{clip}"}
    data["montage"] = frames_of(OUT / "montage-brake-log.mp4", "montage")
    bl = json.loads((ROOT / "runs/studio-demo/brake-log.json").read_text())
    tr = bl["training"]["conditions"]
    data["rollout"] = {**frames_of(OUT / "rollout-brake-log.mp4", "rollout"), "cols": [
        {"x": 20, "y": 296, "w": 620, "label": "Your current simulator", "success": tr["current"]["final_real"]},
        {"x": 654, "y": 296, "w": 620, "label": "Wide domain randomization", "success": tr["wide"]["final_real"]},
        {"x": 1286, "y": 296, "w": 620, "label": "Tether's measured ranges", "success": tr["tether"]["final_real"]}]}

    rb = json.loads((ROOT / "runs/proof/real_benchmark.json").read_text())
    rows = sorted(rb["objects"], key=lambda o: o["mu_tilt_test"])
    data["real"] = {"within": rb["summary"]["video"]["within_0_05"], "objects": rb["summary"]["video"]["objects"],
                    "mae": rb["summary"]["video"]["mean_abs_error"], "paper": rb["paper_estimator_on_real"]["mean_abs_error"],
                    "rows": [{"object": o["object"], "truth": o["mu_tilt_test"], "tether": o["video"]["mu"],
                              "ok": abs(o["video"]["mu"] - o["mu_tilt_test"]) <= 0.05} for o in rows]}
    ce = json.loads((ROOT / "runs/proof/cross_engine.json").read_text())["summary"]["in-menu"]
    m = ce["success_mean"]
    data["engine"] = [{"label": "Tether's calibration", "v": m["tether"], "hi": True},
                      {"label": "best domain randomization", "v": m[ce["best_dr_width"]]},
                      {"label": "randomization over the true distribution (an oracle)", "v": m["dr oracle"]}]
    rng = np.random.default_rng(3)
    data["gap"] = [{"sim": round(0.45 + float(rng.normal(0, 0.008)), 4), "real": round(0.45 * 0.78 + float(rng.normal(0, 0.018)), 4)} for _ in range(12)]
    return data


# ------------------------------------------------------------------ narration and music
def _elevenlabs(text: str, out: Path) -> None:
    import hashlib
    import os

    from agent.llm import load_dotenv

    load_dotenv()
    key = os.environ.get("ELEVENLAB_API_KEY") or os.environ.get("ELEVENLABS_API_KEY")
    actor = os.environ.get("ELEVENLAB_ACTOR") or os.environ.get("ELVENLAB_ACTOR") or os.environ.get("ELEVENLABS_VOICE")
    if not key or not actor:
        raise SystemExit("ElevenLabs narration needs ELEVENLAB_API_KEY and ELEVENLAB_ACTOR (voice id or name) in .env")
    TTS_CACHE.mkdir(parents=True, exist_ok=True)
    cached = TTS_CACHE / (hashlib.sha256(f"{actor}|{text}".encode()).hexdigest()[:20] + ".mp3")
    if not cached.exists():
        voice = actor
        if not (len(actor) >= 16 and actor.isalnum()):  # a name: look its id up
            req = urllib.request.Request("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": key})
            voices = json.loads(urllib.request.urlopen(req, timeout=60).read())["voices"]
            voice = next((v["voice_id"] for v in voices if v["name"].lower() == actor.lower()), None)
            if voice is None:
                raise SystemExit(f"no ElevenLabs voice named {actor!r} on this account")
        body = json.dumps({"text": text, "model_id": "eleven_multilingual_v2",
                           "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}).encode()
        req = urllib.request.Request(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}", data=body, method="POST",
                                     headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"})
        cached.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(cached), str(out)], check=True)


def narrate(text: str, out: Path, voice: str) -> float:
    """One narration line to audio (ElevenLabs, cached; or macOS say) -> its length in seconds."""
    if voice == "elevenlabs":
        _elevenlabs(text, out)
    else:
        subprocess.run(["say", "-v", "Samantha", "-r", "178", "-o", str(out), text], check=True)
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                       check=True, capture_output=True, text=True)
    return float(r.stdout.strip())


def to_wav(src: Path, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ar", str(SR), "-ac", "2", str(dst)], check=True)


def music(seconds: float, out: Path) -> None:
    """A slow ambient bed: four chords, soft partials with a long attack, a little detune between ears, and a simple
    feedback-delay room. Generated here, so there is nothing to license."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    chords = [[146.83, 220.0, 261.63, 329.63, 349.23],   # Dm9-ish
              [116.54, 174.61, 233.08, 293.66, 349.23],  # Bbmaj7
              [87.31, 174.61, 220.0, 261.63, 392.0],     # F add9
              [130.81, 196.0, 246.94, 293.66, 329.63]]   # C add9
    span = 9.0
    L = np.zeros(n)
    R = np.zeros(n)
    k = 0
    while k * span < seconds + span:
        c = chords[k % 4]
        t0 = k * span - 1.5
        env = np.clip((t - t0) / 3.0, 0, 1) * np.clip((t0 + span + 3.0 - t) / 3.0, 0, 1)
        env = env * env * (3 - 2 * env)
        for j, f in enumerate(c):
            a = 0.16 / (1 + j * 0.35)
            L += a * env * (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t + 0.3))
            R += a * env * (np.sin(2 * np.pi * f * 1.0015 * t + 0.7) + 0.25 * np.sin(2 * np.pi * 2 * f * 1.0015 * t))
        k += 1
    trem = 1 + 0.06 * np.sin(2 * np.pi * 0.11 * t)
    L, R = L * trem, R * trem
    for d_s, g in ((0.173, 0.35), (0.291, 0.25), (0.437, 0.18)):  # small room
        d = int(d_s * SR)
        L[d:] += g * R[:-d]
        R[d:] += g * L[:-d]
    fade = np.clip(t / 2.5, 0, 1) * np.clip((seconds - t) / 3.5, 0, 1)
    st = np.stack([L * fade, R * fade], axis=1)
    st = st / (np.abs(st).max() + 1e-9) * 0.5
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


# ------------------------------------------------------------------ capture
async def capture(page: Path, plan: list[tuple[str, float]], only: str | None) -> None:
    import websockets

    port = 9339
    prof = tempfile.mkdtemp()
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
                             "--allow-file-access-from-files", "--hide-scrollbars", "--force-device-scale-factor=1", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
                ws_url = next(x["webSocketDebuggerUrl"] for x in tabs if x["type"] == "page")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.25)
        async with websockets.connect(ws_url, max_size=1 << 28) as ws:
            mid = 0

            async def send(method, params=None):
                nonlocal mid
                mid += 1
                me = mid
                await ws.send(json.dumps({"id": me, "method": method, "params": params or {}}))
                while True:
                    m = json.loads(await ws.recv())
                    if m.get("method") == "Runtime.exceptionThrown":
                        print("JS error:", json.dumps(m["params"]["exceptionDetails"])[:400])
                    if m.get("id") == me:
                        return m

            await send("Runtime.enable")
            await send("Page.enable")
            await send("Emulation.setDeviceMetricsOverride", {"width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False})
            await send("Page.navigate", {"url": page.as_uri()})
            await asyncio.sleep(1.5)
            await send("Runtime.evaluate", {"expression": "window.ready", "awaitPromise": True})
            for name, d in plan:
                if only and name != only:
                    continue
                fd = WORK / "frames" / name
                shutil.rmtree(fd, ignore_errors=True)
                fd.mkdir(parents=True)
                n = int(round(d * FPS))
                t0 = time.time()
                for i in range(n):
                    r = await send("Runtime.evaluate", {"expression": f"render({json.dumps(name)}, {i / FPS:.4f}, {d:.4f})", "awaitPromise": True})
                    if "exceptionDetails" in r.get("result", {}):
                        print("render error", name, i, json.dumps(r["result"]["exceptionDetails"])[:300])
                    shot = await send("Page.captureScreenshot", {"format": "jpeg", "quality": 92})
                    (fd / f"f{i:05d}.jpg").write_bytes(base64.b64decode(shot["result"]["data"]))
                print(f"{name:12s} {n} frames in {time.time() - t0:.0f} s", flush=True)
    finally:
        proc.kill()


# ------------------------------------------------------------------ assembly
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", choices=["elevenlabs", "say"], default="elevenlabs")
    ap.add_argument("--only", default=None, help="re-capture one scene")
    ap.add_argument("--out", type=Path, default=OUT / "tether_film.mp4")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    data = build_data()
    page = WORK / "stage.html"
    page.write_text((Path(__file__).parent / "stage.html").read_text().replace("/*__DATA__*/null", json.dumps(data), 1))

    lead = 0.45  # voice starts a beat after the cut
    plan, voices = [], []
    for name, text, min_s in SCRIPT:
        raw = WORK / "voice" / f"{name}.aiff"
        raw.parent.mkdir(parents=True, exist_ok=True)
        dur = narrate(text, raw, a.voice)
        d = max(min_s, lead + dur + 0.9)
        plan.append((name, round(d, 3)))
        voices.append(raw)
    total = sum(d for _, d in plan)
    (WORK / "plan.json").write_text(json.dumps(plan))  # scene lengths, read by video/submission.py for chapters and stills
    print(f"plan: {total:.1f} s  " + "  ".join(f"{n} {d:.1f}" for n, d in plan))

    asyncio.run(capture(page, plan, a.only))

    # picture
    parts = []
    for name, d in plan:
        out = WORK / "frames" / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", str(WORK / "frames" / name / "f%05d.jpg"),
                        "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-tune", "film", "-pix_fmt", "yuv420p", str(out)], check=True)
        parts.append(out)
    lst = WORK / "parts.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    picture = WORK / "picture.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(picture)], check=True)

    # voice track: each line placed at its scene's start + lead
    voice = WORK / "voice.wav"
    inputs, filt, offset = [], [], 0.0
    for i, ((name, d), raw) in enumerate(zip(plan, voices)):
        wav = WORK / "voice" / f"{name}.wav"
        to_wav(raw, wav)
        inputs += ["-i", str(wav)]
        ms = int((offset + lead) * 1000)
        filt.append(f"[{i}:a]adelay={ms}|{ms}[v{i}]")
        offset += d
    mix = "".join(f"[v{i}]" for i in range(len(plan)))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex",
                    ";".join(filt) + f";{mix}amix=inputs={len(plan)}:normalize=0,apad=whole_dur={total:.3f}[out]", "-map", "[out]",
                    "-ar", str(SR), "-ac", "2", str(voice)], check=True)
    bed = WORK / "music.wav"
    music(total, bed)
    final_audio = WORK / "audio.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(voice), "-i", str(bed), "-filter_complex",
                    "[1:a]volume=0.22[m];[m][0:a]sidechaincompress=threshold=0.015:ratio=8:attack=40:release=900[md];"
                    "[0:a][md]amix=inputs=2:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11[a]", "-map", "[a]", "-ar", str(SR), str(final_audio)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(picture), "-i", str(final_audio), "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(a.out)], check=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(a.out)],
                               capture_output=True, text=True).stdout)
    print(f"wrote {a.out} ({dur:.1f} s, {a.out.stat().st_size / 1e6:.1f} MB)")
    _ = math


if __name__ == "__main__":
    main()
