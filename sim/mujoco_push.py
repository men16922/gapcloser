"""A second, independent engine for the hidden "real" world: MuJoCo (soft contacts, elliptic friction cones).

Everything Tether fits is NVIDIA Newton or its analytic surrogate. Scoring Tether on data made by the same family
of models only checks self-consistency, so this env makes the "real" pushes differently:

  - a different contact model (MuJoCo's soft, convex contact solver instead of Newton's XPBD),
  - a contact-driven launch: a velocity-controlled paddle pushes the cube from behind, accelerating over a 12 cm
    stroke, and brakes at the launch point, so the launch speed comes out of contact, not an assigned velocity,
  - optional effects the Studio's fitter has no field for (off-menu), set per world in `effects`:
      speed_slope   friction changes with sliding speed: mu(v) = mu * (1 + speed_slope * (v - 1 m/s)), clipped
                    (negative: velocity-weakening, like a wet surface; positive: velocity-strengthening, like grease)
      region2       (y0, mu): a second friction region further out, after the first one
      slope_deg     the table tilts along the push axis (downhill > 0)

Same contract as AnalyticPushEnv / NewtonPushEnv: rollout(params, policy, targets) and push(params, commands).
Perception (where the robot's camera thinks the target is) uses the same camera model as the other envs.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from sim.params import GRAVITY, ParamSet, effective_friction
from sim.push_task import FRAME_DT, FULL_TRACK_FRAMES, PATCH_OFF, Rollout, Trial, observe

DT = 0.0005
PADDLE_GAP = 0.001  # paddle starts just behind the cube and accelerates with it (a push, not a hit)
STROKE = 0.2  # m the paddle pushes the cube before the launch point (smoothly reaches speed over half of it, then holds)
REST_SPEED = 2e-3  # m/s
MAX_T = 8.0  # s of sliding before giving up

XML = """
<mujoco>
  <option timestep="{dt}" gravity="0 {gy} {gz}" cone="elliptic" impratio="10" integrator="implicitfast"/>
  <worldbody>
    <geom name="table" type="plane" size="6 6 0.1" friction="0.0001 0 0"/>
    <body name="paddle" pos="0 {py} {pz}">
      <joint name="stroke" type="slide" axis="0 1 0"/>
      <geom name="paddle" type="capsule" fromto="-0.04 0 0 0.04 0 0" size="0.006" mass="2.0" friction="0.0001 0 0" solref="0.004 1"/>
    </body>
    <body name="cube" pos="0 {cy} {h}">
      <freejoint/>
      <geom name="cube" type="box" size="{h} {h} {h}" density="{rho}" friction="{mu} 0.001 0.0001" priority="1"/>
    </body>
  </worldbody>
  <actuator><velocity name="drive" joint="stroke" kv="800" ctrlrange="-10 10"/></actuator>
</mujoco>
"""


def _mu_at(p: ParamSet, y: float, effects: dict) -> float:
    mu = effective_friction(p["object_mu"], p["table_mu"])
    if p["patch_y0"] < PATCH_OFF and y >= p["patch_y0"]:
        mu = p["patch_mu"]
    r2 = effects.get("region2")
    if r2 and y >= r2[0]:
        mu = r2[1]
    return mu


class MujocoPushEnv:
    def __init__(self, effects: dict | None = None):
        self.effects = dict(effects or {})

    def _model(self, p: ParamSet) -> tuple[mujoco.MjModel, mujoco.MjData]:
        h = p["object_half_size"]
        th = math.radians(self.effects.get("slope_deg", 0.0))
        # the paddle is a horizontal rod at the cube's centre height: the push goes through the centre of mass,
        # so even a fast stroke does not pitch the cube
        xml = XML.format(dt=DT, gy=GRAVITY * math.sin(th), gz=-GRAVITY * math.cos(th), h=h, pz=h,
                         py=-STROKE - h - PADDLE_GAP - 0.006, cy=-STROKE, rho=p["object_density"], mu=_mu_at(p, 0.0, self.effects))
        m = mujoco.MjModel.from_xml_string(xml)
        return m, mujoco.MjData(m)

    def _slide(self, p: ParamSet, command: float) -> tuple[float, list[float], bool]:
        m, d = self._model(p)
        cube_geom = m.geom("cube").id
        qy, vy = m.jnt_qposadr[m.joint(1).id] + 1, m.jnt_dofadr[m.joint(1).id] + 1  # free joint: x y z, then quat
        speed = command * p["actuator_gain"]
        sv = self.effects.get("speed_slope", 0.0)
        # strike: the paddle drives at the commanded speed until the cube reaches the launch point (y = 0), then brakes
        ramp = STROKE / max(speed, 1e-3)  # smooth-step to full speed over the first half of the stroke
        t = 0.0
        while d.qpos[qy] < 0.0 and t < 2.0:
            u = min(1.0, t / ramp)
            d.ctrl[0] = speed * u * u * (3.0 - 2.0 * u)
            m.geom_friction[cube_geom, 0] = _mu_at(p, d.qpos[qy], self.effects)
            mujoco.mj_step(m, d)
            t += DT
        d.ctrl[0] = -1.0  # pull back out of the way
        y0 = float(d.qpos[qy])
        cube_body = m.body("cube").id
        track, t, next_frame, still, max_tilt = [0.0], 0.0, FRAME_DT, 0.0, 0.0
        while t < MAX_T:
            y, v = float(d.qpos[qy]), float(d.qvel[vy])
            mu = _mu_at(p, y, self.effects)  # regions are fixed on the table; the launch point is y = 0
            if sv:
                mu = min(1.5, max(0.05, mu * (1.0 + sv * (abs(v) - 1.0))))
            m.geom_friction[cube_geom, 0] = mu
            mujoco.mj_step(m, d)
            t += DT
            if t >= next_frame - 1e-9:
                track.append(float(d.qpos[qy]) - y0)
                next_frame += FRAME_DT
            max_tilt = max(max_tilt, math.acos(max(-1.0, min(1.0, float(d.xmat[cube_body][8])))))
            still = still + DT if math.hypot(*d.qvel[vy - 1:vy + 1]) < REST_SPEED else 0.0
            if still > 0.1 and len(track) > 3:
                break
        stop = float(d.qpos[qy]) - y0
        while len(track) < FULL_TRACK_FRAMES + 1:
            track.append(stop)
        tipped = max_tilt > math.radians(20)  # the cube pitched or rolled over instead of sliding flat
        return stop, track[:FULL_TRACK_FRAMES + 1], tipped

    def rollout(self, params: ParamSet, policy, targets: list[float]) -> Rollout:
        trials = []
        for target in targets:
            obs = observe(target, params)
            cmd = policy.command(obs)
            stop, track, tipped = self._slide(params, cmd)
            trials.append(Trial(target, obs, cmd, stop, track, tipped))
        return Rollout(trials)

    def push(self, params: ParamSet, commands: list[float]) -> Rollout:
        nan = float("nan")
        out = []
        for c in commands:
            stop, track, tipped = self._slide(params, c)
            out.append(Trial(nan, nan, c, stop, track, tipped))
        return Rollout(out)


def launch_speed_error(params: ParamSet, commands: list[float]) -> list[float]:
    """How far the collision-made launch speed is from command x gain (fraction), from the first tracked frame."""
    env = MujocoPushEnv()
    out = []
    for c in commands:
        _, track, _ = env._slide(params, c)
        v_meas = (track[1] - track[0]) / FRAME_DT
        out.append(v_meas / (c * params["actuator_gain"]) - 1.0)
    return out


if __name__ == "__main__":
    from sim.push_task import slide_distance

    p = ParamSet.nominal().with_(object_mu=0.6, table_mu=0.6, actuator_gain=0.9, patch_y0=0.35, patch_mu=0.4)
    env = MujocoPushEnv()
    for c in (1.5, 2.0, 2.5, 3.0):
        stop, track, tipped = env._slide(p, c)
        print(f"cmd {c}: mujoco {stop:.4f} m, analytic {slide_distance(c, p):.4f} m, tipped {tipped}, v0 {(track[1]) / FRAME_DT:.3f}")
    print("launch speed error", np.round(launch_speed_error(p, [1.5, 2.5, 3.5]), 4))
