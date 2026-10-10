"""Korean coverage of text the server writes: the Studio page translates server sentences (notes, reasons, how a
model was chosen) with exact keys and regex patterns (dashboard/studio.html KO, KO_RX, KO_SERVER). If a server
sentence changes and no pattern matches any more, the page silently falls back to English. This test runs the real
pipeline on the samples, collects every sentence the page passes through tx(), and checks each one translates."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGE = (ROOT / "dashboard" / "studio.html").read_text()


def _block(start: str, end: str = "\n};") -> str:
    i = PAGE.index(start)
    return PAGE[i:PAGE.index(end, i)]


def ko_keys() -> set[str]:
    body = _block("const KO = {")
    return {json.loads(f'"{m}"') for m in re.findall(r'"((?:[^"\\]|\\.)*)"\s*:', body)}


def ko_patterns() -> list[re.Pattern]:
    out = []
    for name in ("const KO_RX = [", "const KO_SERVER = ["):
        body = _block(name, "\n];")
        for src, flags in re.findall(r"\[/((?:\\/|[^/\n])+)/([a-z]*),", body):
            out.append(re.compile(src.replace("\\/", "/"), re.I if "i" in flags else 0))
    return out


KEYS, PATTERNS = ko_keys(), ko_patterns()


def translatable(text: str) -> bool:
    k = text.strip()
    if not k or k in KEYS:
        return True
    core = k[:-1] if k.endswith(".") and not k.endswith("…") else k
    return core in KEYS or any(p.search(core) for p in PATTERNS)


def server_sentences() -> set[str]:
    from studio.pipeline import analyze
    from studio.session import load

    out: set[str] = set()
    samples = ROOT / "studio" / "samples"
    for name in ("lab-bench", "short-reach", "press-line", "brake-log"):
        s = load((samples / f"{name}.csv").read_text(), f"{name}.csv", name)
        res = analyze(s, None)
        out |= {n[0].upper() + n[1:] for n in s.notes}
        out.add(res.calibration.chosen_by)
        out |= set(res.calibration.residuals.get("unexplained") or [])
        out |= {w["why"][0].upper() + w["why"][1:] for w in res.next_experiment["suggestions"]}
    from server.studio_api import truth_view
    from studio.simulate import truth as sim_truth

    for t in samples.glob("*.truth.json"):
        out.add(truth_view(json.loads(t.read_text()))["source"])
    out.add(sim_truth({"mu_eff": 0.5})["source"])
    # sentences the server builds outside the samples' path (agent fallbacks, the cross-check verdict)
    out |= {"agent busy with another session: offline search",
            "the agent's model leaves launch speeds disagree with the model — the library structure mu_eff+actuator_gain explains the data"}
    # the pattern check's findings and the region it adds (studio/structure.py, studio/fit.py)
    out |= {"offline structure search (simplest model within noise of the best); friction region added at 0.31 m (deceleration steps there)",
            "friction rises as the object slows (-0.05 per m/s): it depends on sliding speed, which the model has no field for",
            "friction falls as the object slows (+0.04 per m/s): it depends on sliding speed, which the model has no field for",
            "2 push(es) left out: the tracker lost the object while it moved",
            "cross-check (agent's friction region within noise)",
            "the friction region the agent proposed at 0.25 m improves the stops by 0.8 mm rms only, within noise: left out",
            "deceleration still changes along the table (strongest near 0.52 m, +0.06): the friction map needs a change the model does not have"}
    return out


def test_patterns_parse():
    assert len(KEYS) > 150 and len(PATTERNS) > 10


def test_every_server_sentence_on_the_page_has_korean():
    missing = sorted(t for t in server_sentences() if not translatable(t))
    assert not missing, "no Korean for:\n" + "\n".join(missing)


@pytest.mark.parametrize("text", [
    "Models disagree by ±2.55 m here; your pushes stop by 13.3 m, so the table beyond is unmeasured",
    "Camera recovered from the sheet: 12.0 m above the table, focal 579 px",
    "offline structure search (simplest model within noise of the best); friction region past 0.29 m dropped (no push measured it)",
])
def test_driving_scaled_sentences_still_translate(text):
    assert translatable(text)


def test_server_errors_have_korean():
    """Every fixed HTTP 4xx message the Studio API raises is translated on the page (call() passes it through tx())."""
    src = (ROOT / "server" / "studio_api.py").read_text()
    msgs = set(re.findall(r'HTTPException\(4(?:09|15|22|29), "([^"]+)"\)', src)) | {"Newton is busy with other visitors. Try again in a minute."}
    missing = sorted(m for m in msgs if not translatable(m) and not m.startswith(("Give the", "Only ", "The last message", "The new rows")))
    assert not missing, missing
