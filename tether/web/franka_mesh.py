"""Export the Franka FR3 visual meshes (as Newton loads them) to a compact per-link file for the 3D viewer.

Meshes come from the same URDF and builder Newton uses (`newton.utils.download_asset("franka_emika_panda")`),
so link frames match the per-frame body transforms recorded by tether/sim/replay.py. Each link's visual shapes are
grouped by color, decimated with trimesh (quadric), and quantized to uint16 positions + uint16 indices.

Run: .venv/bin/python -m tether.web.franka_mesh  ->  tether/web/assets/franka_fr3.json (~0.5 MB)
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "assets" / "franka_fr3.json"


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _quantize_color(c) -> tuple[int, int, int]:
    # the DAE materials carry many near-identical whites/greys; snap them so parts merge
    return tuple(min(255, int(round(float(x) * 15)) * 17) for x in c)


def export(out: Path = OUT, tri_budget: int = 36000) -> dict:
    import trimesh
    import warp as wp

    import newton

    wp.config.quiet = True
    b = newton.ModelBuilder()
    b.add_urdf(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf", floating=False)
    labels = list(getattr(b, "body_label", None) or b.body_key)
    vis = int(newton.ShapeFlags.VISIBLE)
    groups: dict[tuple[int, tuple], list] = {}
    for i in range(b.shape_count):
        src = b.shape_source[i]
        if src is None or not (int(b.shape_flags[i]) & vis) or (int(b.shape_flags[i]) & int(newton.ShapeFlags.COLLIDE_SHAPES)):
            continue
        v = np.asarray(src.vertices, dtype=np.float64) * np.asarray(b.shape_scale[i])
        tf = b.shape_transform[i]
        p, q = np.asarray(tf[:3]), np.asarray(tf[3:7])  # xyzw
        m = trimesh.transformations.quaternion_matrix([q[3], q[0], q[1], q[2]])[:3, :3]
        v = v @ m.T + p
        f = np.asarray(src.indices, dtype=np.int64).reshape(-1, 3)
        groups.setdefault((int(b.shape_body[i]), _quantize_color(b.shape_color[i])), []).append(trimesh.Trimesh(v, f, process=False))

    meshes = {k: trimesh.util.concatenate(ms) for k, ms in groups.items()}
    for mesh in meshes.values():
        mesh.merge_vertices()
    total = sum(len(m.faces) for m in meshes.values())
    ratio = min(1.0, tri_budget / total)
    links: dict[int, list] = {}
    kept = 0
    for (body, color), mesh in sorted(meshes.items()):
        target = max(12, int(len(mesh.faces) * ratio))
        if target < len(mesh.faces):
            mesh = mesh.simplify_quadric_decimation(face_count=target)
        mesh.remove_unreferenced_vertices()
        if len(mesh.faces) == 0:
            continue
        lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
        span = np.maximum(hi - lo, 1e-6)
        qv = np.round((mesh.vertices - lo) / span * 65535).astype(np.uint16)
        idx = mesh.faces.astype(np.uint16 if len(mesh.vertices) < 65536 else np.uint32)
        kept += len(mesh.faces)
        links.setdefault(body, []).append({
            "color": "#%02x%02x%02x" % color,
            "lo": [round(float(x), 6) for x in lo], "span": [round(float(x), 6) for x in span],
            "v": _b64(qv), "i": _b64(idx), "i32": idx.dtype == np.uint32,
        })
    data = {
        "schema": "gapcloser.franka_mesh/1",
        "source": "newton.utils.download_asset('franka_emika_panda') urdf/fr3_franka_hand.urdf (visual meshes, decimated)",
        "triangles": kept,
        "bodies": labels,
        "links": [{"body": body, "name": labels[body], "parts": parts} for body, parts in sorted(links.items())],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tris", type=int, default=36000, help="total triangle budget after decimation")
    a = ap.parse_args()
    d = export(a.out, a.tris)
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.2f} MB, {d['triangles']} triangles, {len(d['links'])} links)")


if __name__ == "__main__":
    main()
