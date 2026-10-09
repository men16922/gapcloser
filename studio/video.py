"""Phone video -> metric push tracks.

1. Scale and pose from a sheet of paper: the user clicks the 4 corners of an A4 (or Letter) sheet lying on
   the table. A plane homography gives the table plane; with the principal point at the image centre it also
   pins the focal length (two orthogonality constraints) and so the full camera pose.
2. The object is tracked by multi-scale normalized cross-correlation against its first-frame appearance.
   Its image centre is back-projected onto the plane z = height/2 (not the table plane), which removes the
   parallax a box's own height would otherwise add (~8% scale error for a 6 cm box seen from 50 cm).
3. Pushes are segmented by speed. Each push starts at release (peak speed) and ends at rest; positions are
   measured along the dominant push direction, from a common table origin, so a friction region found in
   one push lines up with the others.

Output: a Session of video pushes (launch speed + track + start position) and a debug dict for overlays.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from sim.push_task import FRAME_DT
from studio.session import Push, Session, SessionError, resample

SHEETS = {"a4": (0.210, 0.297), "letter": (0.2159, 0.2794)}
MAX_SECONDS = 90
MAX_WIDTH = 960
MOVE_SPEED = 0.06  # m/s: faster than this counts as moving
MIN_PUSH_FRAMES = 5


def _cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise SessionError("video support needs opencv-python-headless (pip install opencv-python-headless)") from e
    return cv2


# ---------------------------------------------------------------------------------------------------- geometry
def order_corners(pts: np.ndarray) -> np.ndarray:
    """Counter-clockwise (in image coordinates, y down) order starting anywhere."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """DLT, src (world plane) -> dst (pixels), 4+ points."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(A, float))
    Hm = vt[-1].reshape(3, 3)
    return Hm / Hm[2, 2]


def _focal(Hc: np.ndarray) -> list[float]:
    """Focal estimates from a centred homography (principal point at origin, square pixels)."""
    h = Hc
    out = []
    d1 = h[2, 0] * h[2, 1]
    if abs(d1) > 1e-12:
        f2 = -(h[0, 0] * h[0, 1] + h[1, 0] * h[1, 1]) / d1
        if f2 > 0:
            out.append(math.sqrt(f2))
    d2 = h[2, 0] ** 2 - h[2, 1] ** 2
    if abs(d2) > 1e-12:
        f2 = -((h[0, 0] ** 2 + h[1, 0] ** 2) - (h[0, 1] ** 2 + h[1, 1] ** 2)) / d2
        if f2 > 0:
            out.append(math.sqrt(f2))
    return out


class PlaneCamera:
    """Camera pose relative to the sheet's plane (world: sheet corner origin, z up out of the table)."""

    def __init__(self, corners_px, size_wh: tuple[float, float], image_wh: tuple[int, int], focal_px: float | None = None):
        pts = order_corners(np.asarray(corners_px, float).reshape(4, 2))
        w, l = size_wh
        cx, cy = image_wh[0] / 2, image_wh[1] / 2
        best = None
        for rot in range(4):  # which clicked corner is the origin and whether the long side comes first
            p = np.roll(pts, rot, axis=0)
            for world in ([(0, 0), (w, 0), (w, l), (0, l)], [(0, 0), (l, 0), (l, w), (0, w)],
                          [(0, 0), (0, l), (w, l), (w, 0)], [(0, 0), (0, w), (l, w), (l, 0)]):  # both turning senses
                Hm = _homography(np.asarray(world, float), p - [cx, cy])
                fs = _focal(Hm) if focal_px is None else [focal_px]
                if not fs:
                    continue
                f = float(np.exp(np.mean(np.log(fs))))
                spread = (max(fs) / min(fs)) if len(fs) > 1 else 1.5
                pose = self._pose(Hm, f)
                if pose is None:
                    continue
                score = spread + (0 if 0.35 < f / image_wh[0] < 3.0 else 10)
                if best is None or score < best[0]:
                    best = (score, f, Hm, pose, world)
        if best is None:
            raise SessionError("could not recover the camera from those 4 corners: click the corners of the flat sheet, "
                               "in any order, with the whole sheet in view")
        _, self.f, self.H, (self.R, self.t), world = best
        self.cx, self.cy = cx, cy
        self.sheet_world = np.asarray(world, float)
        self.center = -self.R.T @ self.t  # camera centre in world
        self.height_m = float(self.center[2])

    @staticmethod
    def _pose(Hm: np.ndarray, f: float):
        Kinv = np.diag([1 / f, 1 / f, 1.0])
        B = Kinv @ Hm
        lam = 1.0 / np.linalg.norm(B[:, 0])
        if B[2, 2] * lam < 0:
            lam = -lam
        r1, r2, t = B[:, 0] * lam, B[:, 1] * lam, B[:, 2] * lam
        R = np.stack([r1, r2, np.cross(r1, r2)], axis=1)
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            return None
        C = -R.T @ t
        if C[2] <= 0:  # camera must be above the table: mirrored assignment
            return None
        return R, t

    def ray_to_plane(self, uv: np.ndarray, z: float = 0.0) -> np.ndarray:
        """World points where pixel rays hit the horizontal plane at height z (m)."""
        uv = np.atleast_2d(uv).astype(float)
        d_cam = np.stack([(uv[:, 0] - self.cx) / self.f, (uv[:, 1] - self.cy) / self.f, np.ones(len(uv))], axis=1)
        d = d_cam @ self.R  # R^T d for each row
        s = (z - self.center[2]) / d[:, 2]
        return self.center[None, :] + s[:, None] * d

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X).astype(float)
        pc = X @ self.R.T + self.t
        return np.stack([self.f * pc[:, 0] / pc[:, 2] + self.cx, self.f * pc[:, 1] / pc[:, 2] + self.cy], axis=1)


# ---------------------------------------------------------------------------------------------------- tracking
def read_frames(path: Path, max_seconds: float = MAX_SECONDS):
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SessionError("could not open the video (mp4/mov/webm from a phone are fine)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not 5 <= fps <= 480:
        fps = 30.0
    frames, scale = [], 1.0
    while len(frames) < max_seconds * fps:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[1] > MAX_WIDTH:
            scale = MAX_WIDTH / fr.shape[1]
            fr = cv2.resize(fr, (MAX_WIDTH, round(fr.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()
    if len(frames) < 10:
        raise SessionError("the video is too short (need at least a second)")
    return frames, float(fps), scale


def first_frame_jpeg(path: Path) -> tuple[bytes, dict]:
    cv2 = _cv2()
    frames, fps, scale = read_frames(path, max_seconds=0.5)
    ok, buf = cv2.imencode(".jpg", frames[0], [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes(), {"width": frames[0].shape[1], "height": frames[0].shape[0], "scale": scale, "fps": fps}


def auto_object(frames: list[np.ndarray], cam: PlaneCamera | None = None) -> tuple[float, float, int]:
    """Where the object sits in the first frame: the most colourful large blob that differs from the per-pixel
    median background (the object moves around, so most of the time any given pixel shows the table).
    Returns (x, y, size_px)."""
    cv2 = _cv2()
    idx = np.linspace(0, len(frames) - 1, min(len(frames), 41)).astype(int)
    bg = np.median(np.stack([frames[k] for k in idx]).astype(np.int16), axis=0)
    diff = np.abs(frames[0].astype(np.int16) - bg).max(axis=2).astype(np.uint8)
    mask = (cv2.GaussianBlur(diff, (5, 5), 0) > 30).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    hsv = cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV)
    best, best_score = None, -1.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 30:
            continue
        sat = float(hsv[..., 1][lab == i].mean())
        score = area * (0.5 + sat / 255.0)
        if score > best_score:
            size = int(max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]))
            best, best_score = (float(cents[i][0]), float(cents[i][1]), size), score
    if best is None:
        raise SessionError("could not find the object in the first frame; click it instead")
    return best


def track(frames: list[np.ndarray], obj_px: tuple[float, float], box_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame object centre (pixels) and match score: normalized cross-correlation against the first-frame
    appearance at a few scales (the object shrinks as it slides away). Search near the last position at the
    last scale and its neighbours; the whole frame only when that fails (the object was picked up and put back)."""
    cv2 = _cv2()
    x, y = obj_px
    r = max(8, box_px // 2)
    h_img, w_img = frames[0].shape[:2]
    x0, y0 = int(max(0, x - r)), int(max(0, y - r))
    x1, y1 = int(min(w_img, x + r)), int(min(h_img, y + r))
    tmpl0 = frames[0][y0:y1, x0:x1]
    off0 = (x - x0, y - y0)
    scales = tuple(round(0.55 * 1.06 ** i, 3) for i in range(15))  # 0.55 .. 1.24, fine steps
    templates = [cv2.resize(tmpl0, (max(6, round(tmpl0.shape[1] * s)), max(6, round(tmpl0.shape[0] * s)))) for s in scales]
    out = np.zeros((len(frames), 2))
    score = np.zeros(len(frames))
    prev, si = np.array([x, y], float), min(range(len(scales)), key=lambda i: abs(scales[i] - 1.0))
    win = 4 * r

    def search(gray, box, idxs):
        sx0, sy0, sx1, sy1 = box
        roi = gray[sy0:sy1, sx0:sx1]
        best = (-2.0, None, si)
        for i in idxs:
            t = templates[i]
            if roi.shape[0] <= t.shape[0] or roi.shape[1] <= t.shape[1]:
                continue
            res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv > best[0]:
                best = (mv, np.array([sx0 + ml[0] + off0[0] * scales[i], sy0 + ml[1] + off0[1] * scales[i]]), i)
        return best

    for k, fr in enumerate(frames):
        gray = fr
        box = (int(max(0, prev[0] - win)), int(max(0, prev[1] - win)), int(min(w_img, prev[0] + win)), int(min(h_img, prev[1] + win)))
        mv, c, i = search(gray, box, range(max(0, si - 2), min(len(scales), si + 3)))
        if mv < 0.6:
            g = search(gray, (0, 0, w_img, h_img), range(len(scales)))
            if g[0] >= 0.75 and g[0] > mv:  # a weak global hit is noise: keep the local one
                mv, c, i = g
        if c is None:
            c = prev
        out[k], score[k] = c, mv
        prev, si = c, i
    return out, score


def _smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    if len(x) < k:
        return x
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def segment(s: np.ndarray, fps: float) -> list[tuple[int, int]]:
    """(release, rest) frame index pairs of each push from position along the push axis."""
    v = np.gradient(_smooth(s, 3)) * fps
    moving = v > MOVE_SPEED
    segs, k = [], 0
    while k < len(s):
        if moving[k]:
            j = k
            while j < len(s) and (moving[j] or (j + 1 < len(s) and moving[j + 1])):
                j += 1
            if j - k >= MIN_PUSH_FRAMES:
                rel = k + int(np.argmax(v[k:j]))  # release = peak speed (the hand stops accelerating it)
                fwd = np.diff(s) * fps  # refine on raw positions: the frame whose forward step is fastest
                lo = max(0, rel - 3)
                rel = lo + int(np.argmax(fwd[lo:min(len(fwd), rel + 2)]))
                segs.append((rel, min(j + 2, len(s) - 1)))
            k = j + 1
        else:
            k += 1
    return segs


def track_video(path: Path, corners_px, sheet: str = "a4", name: str | None = None, object_px=None,
                object_height_m: float | None = None, max_seconds: float = MAX_SECONDS,
                prior: dict | None = None) -> tuple[Session, dict]:
    """`prior` = meta of an earlier video session of the same table: positions use its origin so the pushes of
    both videos share one table frame (the sheet must not move between them)."""
    if sheet not in SHEETS:
        raise SessionError(f"unknown sheet {sheet!r} (a4 or letter)")
    corners = np.asarray(corners_px, float).reshape(4, 2)
    frames, fps, scale = read_frames(Path(path), max_seconds)
    corners = corners * scale
    h_img, w_img = frames[0].shape[:2]
    cam = PlaneCamera(corners, SHEETS[sheet], (w_img, h_img))
    if object_px is not None:
        obj, blob = tuple(np.asarray(object_px, float) * scale), None
    else:
        *obj, blob = auto_object(frames, cam)
        obj = tuple(obj)
    # box size in pixels from the object's footprint on the table: assume ~6 cm unless told
    hgt = object_height_m if object_height_m is not None else 0.05
    ground = cam.ray_to_plane(np.asarray([obj]), hgt / 2)[0]
    px_per_m = np.linalg.norm(cam.project(ground + [0.02, 0, 0]) - cam.project(ground - [0.02, 0, 0])) / 0.04
    box_px = int(np.clip(blob * 1.15 if blob else px_per_m * max(hgt, 0.03) * 1.6, 14, 200))
    uv, score = track(frames, obj, box_px)
    P = cam.ray_to_plane(uv, hgt / 2)[:, :2]
    good = score > 0.45
    # push axis: dominant direction of frame-to-frame motion while moving
    dP = np.diff(P, axis=0)
    sp = np.linalg.norm(dP, axis=1) * fps
    mv = (sp > MOVE_SPEED) & good[1:] & good[:-1]
    if mv.sum() < MIN_PUSH_FRAMES:
        raise SessionError("no push found: is the object visible and does it slide? (click it if detection failed)")
    D = dP[mv]
    axis = D.sum(axis=0)
    axis /= np.linalg.norm(axis)
    sheet_c = cam.sheet_world.mean(axis=0)
    s_abs = (P - sheet_c) @ axis  # along the push, from the sheet centre: the same table frame in every video
    segs = segment(s_abs, fps)
    if not segs:
        raise SessionError("no push found: the object moves too little (needs > 6 cm/s)")
    prior = prior or {}
    origin = float(prior["origin_from_sheet_m"]) if "origin_from_sheet_m" in prior else float(np.median([s_abs[r] for r, _ in segs]))
    pushes, debug_pushes = [], []
    from agent.loop import fit_launch

    for i, (r, e) in enumerate(segs):
        if not good[r:e + 1].mean() > 0.8:
            continue
        rest = float(np.median(s_abs[e:min(len(s_abs), e + 5)]))
        tr_native = list(s_abs[r:e + 1] - s_abs[r]) + [rest - s_abs[r]] * 3
        tr = resample([float(x) for x in tr_native], fps)
        fit = fit_launch(tr)
        v0 = fit[0] if fit and fit[0] > 0 else (tr[1] - tr[0]) / FRAME_DT
        start = float(s_abs[r] - origin)
        stop = rest - float(s_abs[r])
        pushes.append(Push(round(stop, 4), None, round(float(v0), 4), None, None, False, [round(x, 4) for x in tr], "video",
                           round(start, 4)))
        debug_pushes.append({"push": i, "release_frame": r, "rest_frame": e, "start_m": round(start, 4), "stop_m": round(stop, 4),
                             "launch_speed_mps": round(float(v0), 3),
                             "path_px": [[round(float(a) / scale, 1), round(float(b) / scale, 1)] for a, b in uv[r:e + 1:2]]})
    if not pushes:
        raise SessionError("pushes were found but the object was lost during them (occluded?)")
    notes = [f"camera recovered from the sheet: {cam.height_m:.2f} m above the table, focal {cam.f / scale:.0f} px"]
    spread = max(abs(p.start) for p in pushes)
    if spread > 0.03:
        notes.append(f"release points spread over ±{spread * 100:.0f} cm; positions are measured from their median")
    session = Session(name or Path(path).stem, "video", pushes, notes=notes,
                      meta={"fps": fps, "frames": len(frames), "camera_height_m": round(cam.height_m, 3),
                            "focal_px": round(cam.f / scale, 1), "origin_from_sheet_m": round(origin, 4)})
    # geometry for overlays: map an along-axis distance from the origin to image pixels (table plane)
    r0 = segs[0][0]
    o_world = P[r0] + (origin - s_abs[r0]) * axis

    def to_px(d: float, lateral: float = 0.0) -> list[float]:
        n = np.array([-axis[1], axis[0]])
        w = np.append(o_world + d * axis + lateral * n, 0.0)
        u = cam.project(w)[0] / scale
        return [round(float(u[0]), 1), round(float(u[1]), 1)]

    debug = {"fps": fps, "frames": len(frames), "scale": scale, "camera": {"height_m": round(cam.height_m, 3), "focal_px": round(cam.f / scale, 1)},
             "object_px": [round(obj[0] / scale, 1), round(obj[1] / scale, 1)], "pushes": debug_pushes,
             "axis_px": {f"{d:.2f}": to_px(d) for d in np.arange(0.0, 0.91, 0.05)},
             "to_px": to_px, "track_score_min": round(float(score[good].min()) if good.any() else 0.0, 3)}
    return session, debug
