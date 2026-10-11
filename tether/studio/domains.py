"""Three domains, one physics: an object launched toward a line slides to a stop under Coulomb friction, a stretch
of the surface may differ, the actuator may under- or over-deliver, and a camera judges the distance to the line.

  robot     a robot arm pushes a box to a line on a table                   (the base scale)
  driving   a car brakes to stop at a stop line; wet or icy road section  (Froude-scaled: lengths x25)
  factory   a pneumatic pusher slides a part to the inspection station      (base scale, metal rail)

Driving uses Froude similarity: lengths scale by L, speeds and times by sqrt(L), friction is unchanged, so
stop = v^2 / (2 mu g) holds at both scales. Sessions store base units (the fitter, the agent and the benchmark
are unchanged); the page and the exports convert to the domain's units (`model_to_domain`, `scale_text`).

Names follow the FlywheelFit workspace tabs (autonomous vehicles / robot manipulation / factory inspection).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Domain:
    id: str
    scale: float  # length scale vs the base (robot) task
    name: dict
    blurb: dict
    words: dict  # UI vocabulary per language: object, surface, actuator, target, region, camera
    reference: str  # the known rectangle used to recover the camera from video
    object_height_m: float
    tolerance_label: dict
    world_default: dict = field(default_factory=dict)

    @property
    def speed_scale(self) -> float:
        return math.sqrt(self.scale)

    def public(self) -> dict:
        return {"id": self.id, "scale": self.scale, "speed_scale": round(self.speed_scale, 6), "name": self.name,
                "blurb": self.blurb, "words": self.words, "reference": self.reference, "object_height_m": self.object_height_m,
                "tolerance_label": self.tolerance_label, "world_default": self.world_default}


DOMAINS = {
    "robot": Domain(
        "robot", 1.0,
        {"en": "Robot manipulation", "ko": "로봇 조작"},
        {"en": "A robot arm pushes a box so it stops on a line. The table may be slicker, wet in places, or the arm weaker than in simulation.",
         "ko": "로봇 팔이 상자를 밀어 선에 세웁니다. 실제 탁자는 더 미끄럽거나 일부가 젖어 있거나, 팔의 힘이 시뮬레이션과 다를 수 있습니다."},
        {"en": {"object": "box", "surface": "table", "actuator": "arm strength", "target": "line", "region": "friction region (spill, mat, worn spot)", "camera": "robot camera"},
         "ko": {"object": "상자", "surface": "탁자", "actuator": "팔 출력", "target": "선", "region": "마찰이 다른 구간", "camera": "로봇 카메라"}},
        "a4", 0.06, {"en": "within 3 cm of the line", "ko": "선에서 3 cm 이내"},
        {"mu_eff": 0.45, "patch_y0": 0.32, "patch_mu": 0.25}),
    "driving": Domain(
        "driving", 25.0,
        {"en": "Autonomous vehicles", "ko": "자율주행 차량"},
        {"en": "A car brakes to stop at a stop line. The real road may be wet or icy from some point on, the brakes weaker, or the camera misjudge distance.",
         "ko": "차량이 정지선에 맞춰 제동합니다. 실제 도로는 어느 지점부터 젖었거나 얼어 있을 수 있고, 제동력이나 카메라의 거리 판단이 시뮬레이션과 다를 수 있습니다."},
        {"en": {"object": "car", "surface": "road", "actuator": "brake/drive response", "target": "stop line", "region": "wet or icy section", "camera": "front camera"},
         "ko": {"object": "차량", "surface": "도로", "actuator": "속도 추종", "target": "정지선", "region": "젖거나 언 구간", "camera": "전방 카메라"}},
        "crosswalk", 1.5, {"en": "within 0.75 m of the stop line", "ko": "정지선에서 0.75 m 이내"},
        {"mu_eff": 0.75, "patch_y0": 0.36, "patch_mu": 0.30}),
    "factory": Domain(
        "factory", 1.0,
        {"en": "Factory inspection", "ko": "제조 검사"},
        {"en": "A pneumatic pusher slides a part along a rail to the inspection camera. Oil on the rail, a weaker pusher or a shifted camera puts the part off-position for inspection.",
         "ko": "공압 푸셔가 부품을 레일 위로 밀어 검사 카메라 아래에 세웁니다. 레일의 기름, 약해진 푸셔, 어긋난 카메라 때문에 부품이 검사 위치를 벗어납니다."},
        {"en": {"object": "part", "surface": "rail", "actuator": "pusher pressure", "target": "inspection position", "region": "oily section", "camera": "inspection camera"},
         "ko": {"object": "부품", "surface": "레일", "actuator": "푸셔 출력", "target": "검사 위치", "region": "기름 묻은 구간", "camera": "검사 카메라"}},
        "a4", 0.06, {"en": "within 3 cm of the inspection position", "ko": "검사 위치에서 3 cm 이내"},
        {"mu_eff": 0.35, "patch_y0": 0.30, "patch_mu": 0.20}),
}


def get(domain_id: str | None) -> Domain:
    return DOMAINS.get(domain_id or "robot", DOMAINS["robot"])


# conversions between base units (robot scale) and a domain's real units
def length(d: Domain, base_m: float | None) -> float | None:
    return None if base_m is None else base_m * d.scale


def speed(d: Domain, base_mps: float | None) -> float | None:
    return None if base_mps is None else base_mps * d.speed_scale


def to_base_length(d: Domain, real_m: float | None) -> float | None:
    return None if real_m is None else real_m / d.scale


def to_base_speed(d: Domain, real_mps: float | None) -> float | None:
    return None if real_mps is None else real_mps / d.speed_scale


def model_to_domain(d: Domain, model: dict) -> dict:
    """A fitted model in the domain's units (friction and gain are scale-free; pitch too)."""
    out = dict(model)
    for k in ("patch_y0", "camera_dx"):
        if out.get(k) is not None:
            out[k] = round(out[k] * d.scale, 4)
    if out.get("lens_k") is not None:
        out["lens_k"] = round(out["lens_k"] / d.scale, 6)  # perceived += k * d^2: k carries 1/length
    return out


def interval_to_domain(d: Domain, field_name: str, iv):
    if iv is None:
        return None
    k = d.scale if field_name in ("patch_y0", "camera_dx") else (1 / d.scale if field_name == "lens_k" else 1.0)
    return [round(x * k, 6) for x in iv]


_SPEED = re.compile(r"(-?\d+(?:\.\d+)?)\s?m/s")
_LEN = re.compile(r"(±?)(-?\d+(?:\.\d+)?)\s?(cm|mm|m)\b(?!/)")


def scale_text(d: Domain, text: str) -> str:
    """Rewrite lengths and speeds in base units inside free text (agent notes, reasons) into the domain's units."""
    if d.scale == 1.0 or not text:
        return text

    def sp(m):
        v = float(m.group(1)) * d.speed_scale
        return f"{v * 3.6:.0f} km/h"

    def ln(m):
        v = float(m.group(2)) * {"cm": 0.01, "mm": 0.001, "m": 1.0}[m.group(3)] * d.scale
        return f"{m.group(1)}{v:.2f} m" if abs(v) < 10 else f"{m.group(1)}{v:.1f} m"

    return _LEN.sub(ln, _SPEED.sub(sp, text))


FIELD_LABELS = {  # what each fitted field means in each domain (English; the page translates)
    "robot": {"mu_eff": "Box-table friction", "patch_y0": "Friction region starts at", "patch_mu": "Friction in that region",
              "actuator_gain": "Arm strength (actual / commanded speed)", "camera_dx": "Camera offset",
              "camera_pitch_deg": "Camera tilt error", "lens_k": "Lens distortion"},
    "driving": {"mu_eff": "Tire-road friction", "patch_y0": "Wet or icy section starts at", "patch_mu": "Friction in that section",
                "actuator_gain": "Speed control (actual / commanded speed)", "camera_dx": "Front camera range offset",
                "camera_pitch_deg": "Front camera pitch error", "lens_k": "Lens distortion"},
    "factory": {"mu_eff": "Part-rail friction", "patch_y0": "Oily section starts at", "patch_mu": "Friction in the oily section",
                "actuator_gain": "Pusher strength (actual / commanded speed)", "camera_dx": "Inspection camera offset",
                "camera_pitch_deg": "Inspection camera tilt error", "lens_k": "Lens distortion"},
}

FIELD_LABELS_KO = {
    "robot": {"mu_eff": "상자-탁자 마찰계수", "patch_y0": "마찰이 다른 구간의 시작 위치", "patch_mu": "해당 구간 마찰계수",
              "actuator_gain": "팔 출력 (실제/명령 속도 비)", "camera_dx": "카메라 위치 오차", "camera_pitch_deg": "카메라 기울기 오차",
              "lens_k": "렌즈 왜곡"},
    "driving": {"mu_eff": "타이어-노면 마찰계수", "patch_y0": "젖거나 언 구간 시작 위치", "patch_mu": "해당 구간 마찰계수",
                "actuator_gain": "속도 추종 (실제/명령 속도 비)", "camera_dx": "전방 카메라 거리 오차", "camera_pitch_deg": "전방 카메라 기울기 오차",
                "lens_k": "렌즈 왜곡"},
    "factory": {"mu_eff": "부품-레일 마찰계수", "patch_y0": "기름 묻은 구간 시작 위치", "patch_mu": "기름 구간 마찰계수",
                "actuator_gain": "푸셔 출력 (실제/명령 속도 비)", "camera_dx": "검사 카메라 위치 오차", "camera_pitch_deg": "검사 카메라 기울기 오차",
                "lens_k": "렌즈 왜곡"},
}


def page_data() -> dict:
    """What the Studio page needs to name and scale each domain (injected at build time, also at /api/studio/domains)."""
    return {d.id: d.public() | {"labels": {"en": FIELD_LABELS[d.id], "ko": FIELD_LABELS_KO[d.id]}} for d in DOMAINS.values()}


def of_session(session) -> Domain:
    return get((getattr(session, "meta", None) or {}).get("domain"))
