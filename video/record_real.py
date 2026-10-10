"""Guided recording of a real table with the Mac camera: waits until the reference sheet is in view (studio.video
detect_sheet on a preview frame, three times in a row), announces the start, records, announces the end and posts a
macOS notification. Spoken prompts in Korean.

Run: .venv/bin/python -m video.record_real [--seconds 90] [--device 0] [--wait 600]
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import cv2

from studio.video import detect_sheet

OUT = Path(__file__).resolve().parent / "real"
VOICE = "Yuna"


def say(text: str) -> None:
    subprocess.run(["say", "-v", VOICE, text], check=False)


def notify(text: str) -> None:
    subprocess.run(["osascript", "-e", f'display notification "{text}" with title "Tether" sound name "Glass"'], check=False)


def grab(device: str, path: Path) -> bool:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "avfoundation", "-framerate", "30",
                        "-pixel_format", "nv12", "-video_size", "1280x720", "-i", f"{device}:none", "-frames:v", "1", str(path)],
                       capture_output=True, timeout=30)
    return r.returncode == 0 and path.exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=90)
    ap.add_argument("--device", default="0")
    ap.add_argument("--wait", type=int, default=600)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    say("Tether 녹화를 준비합니다. 책상 위 상자가 지나갈 길 옆에 A4 용지를 놓고, 카메라가 길과 종이를 함께 비추게 해 주세요.")
    preview = OUT / "preview.jpg"
    t0, hits, last_hint = time.time(), 0, 0.0
    while time.time() - t0 < a.wait:
        ok = grab(a.device, preview)
        found = ok and detect_sheet(cv2.imread(str(preview))) is not None
        hits = hits + 1 if found else 0
        print(f"{time.time() - t0:5.0f}s sheet {'found' if found else '-'} ({hits}/3)", flush=True)
        if hits >= 3:
            break
        if time.time() - last_hint > 60:
            say("아직 A4 용지가 보이지 않습니다. 종이가 화면에 다 들어오게 해 주세요.")
            last_hint = time.time()
        time.sleep(2)
    else:
        say("종이를 찾지 못해 녹화를 취소합니다.")
        notify("종이를 찾지 못해 녹화를 취소했습니다")
        raise SystemExit(2)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = OUT / f"real-{stamp}.mp4"
    say(f"종이를 찾았습니다. 3초 뒤 녹화를 시작합니다. {a.seconds}초 동안 상자를 여덟 번에서 열 번 튕겨 주세요. 매번 멈출 때까지 기다려 주세요.")
    time.sleep(3)
    say("녹화 시작")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "avfoundation", "-framerate", "30", "-pixel_format", "nv12",
                    "-video_size", "1280x720", "-i", f"{a.device}:none", "-t", str(a.seconds), "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", str(out)], check=True, timeout=a.seconds + 60)
    say("녹화 끝. 수고하셨습니다. 분석을 시작합니다.")
    notify(f"녹화 완료: {out.name}")
    print("RECORDED", out, flush=True)


if __name__ == "__main__":
    main()
