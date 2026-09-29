"""Run a trained policy in MuJoCo without ROS (quick check, evaluation, videos).

    python3 -m h1_rl.play                          # interactive viewer, pretrained policy
    python3 -m h1_rl.play --policy logs/h1_walk/<run>/policy_latest.npz
    python3 -m h1_rl.play --headless               # scripted test, prints tracking errors
    python3 -m h1_rl.play --headless --video walk.mp4     # (no display: MUJOCO_GL=egl)

    python3 -m h1_rl.play --config config/h1_walk_terrain.yaml --terrain-level 5   # rough ground

Viewer keys:  Up/Down = forward speed,  Left/Right = turn rate,
              PageUp/PageDown = sideways speed,  Home = stop,  End = reset robot
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import terrain as terrain_mod
from .config import load_config, resolve_path
from .controller import FALLEN, PolicyController
from .robot import foot_geom_ids, place_on_ground
from .sim import H1Sim

DEFAULT_POLICY = "policies/h1_walk.npz"

# (duration [s], vx [m/s], vy [m/s], yaw rate [rad/s])
DEMO_SCRIPT = [
    (3.0, 0.0, 0.0, 0.0),     # stand still
    (4.0, 0.5, 0.0, 0.0),
    (4.0, 1.0, 0.0, 0.0),
    (3.0, 0.0, 0.0, 0.0),     # stop: finish the step, then stand
    (4.0, 0.0, 0.0, 0.8),
    (4.0, 0.6, 0.0, -0.6),
    (3.0, 0.0, 0.0, 0.0),
    (4.0, 0.0, 0.4, 0.0),
    (4.0, 0.0, -0.4, 0.0),
    (4.0, -0.5, 0.0, 0.0),
    (3.0, 0.0, 0.0, 0.0),
]


def run_script(sim: H1Sim, ctrl: PolicyController, script, latency_steps: int = 0,
               renderer=None, frames=None, fps: int = 30, verbose: bool = True) -> dict:
    """Drive the robot through a command script; returns tracking statistics."""
    dec = sim.robot.decimation
    render_every = max(1, int(round(1.0 / (fps * sim.dt))))
    step_count = 0
    results, falls = [], 0
    for duration, vx, vy, wz in script:
        cmd = np.array([vx, vy, wz])
        n_ctrl = int(round(duration / ctrl.dt))
        vels = []
        for k in range(n_ctrl):
            q, dq, _ = sim.joint_state()
            quat, gyro, _ = sim.imu()
            out = ctrl.step(q, dq, quat, gyro, cmd)
            if latency_steps:
                sim.step(latency_steps)
            sim.set_command(out.position, out.velocity, out.kp, out.kd, out.effort)
            for _ in range(dec - latency_steps):
                sim.step()
                step_count += 1
                if renderer is not None and step_count % render_every == 0:
                    renderer.update_scene(sim.data, camera="track")
                    frames.append(renderer.render())
            if k > 0.4 * n_ctrl:
                lin, ang = sim.base_twist_body()
                vels.append([lin[0], lin[1], ang[2]])
            if ctrl.state == FALLEN or sim.fallen():
                falls += 1
                if verbose:
                    print(f"  fell during command {cmd.tolist()} at t={sim.time:.1f}s -> reset")
                sim.reset(support=False)
                ctrl.reset()
        v = np.mean(vels, axis=0) if vels else np.full(3, np.nan)
        results.append({"cmd": cmd.tolist(), "achieved": v.tolist(), "error": np.abs(v - ctrl.clip_command(cmd)).tolist()})
        if verbose:
            print(f"  cmd vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f} -> achieved vx={v[0]:+.2f} vy={v[1]:+.2f} wz={v[2]:+.2f}")
    err = np.array([r["error"] for r in results])
    return {"segments": results, "falls": falls,
            "mean_abs_error": np.nanmean(err, axis=0).tolist() if len(err) else None}


def make_terrain(sim: H1Sim, cfg: dict, level: int, kind: str, seed: int):
    """Fill the scene's height field and return a respawn function that lands on it."""
    import mujoco

    m = sim.model
    if m.nhfield == 0:
        raise SystemExit("this scene has no height field: use --config config/h1_walk_terrain.yaml")
    tcfg = cfg["terrain"]
    n, patch = int(m.hfield_nrow[0]), float(tcfg["patch"])
    fraction = level / max(int(tcfg["levels"]), 1)
    height = terrain_mod.generate(np.random.default_rng(seed), n, fraction, kind,
                                  float(tcfg["amplitude"]))
    terrain_mod.write_to_model(m, height, float(m.hfield_size[0][2]))
    feet = foot_geom_ids(m)
    print(f"terrain: {kind}, level {level}/{tcfg['levels']} -> relief 0..{height.max():.3f} m")

    def respawn(support: bool = False) -> None:
        sim.reset(support=support)
        d = sim.data
        place_on_ground(m, d, feet)
        d.qpos[2] += terrain_mod.sample(height, patch, d.qpos[0], d.qpos[1]) + 0.01
        mujoco.mj_forward(m, d)

    return respawn


def interactive(sim: H1Sim, ctrl: PolicyController, respawn=None) -> None:
    import mujoco.viewer

    cmd = np.zeros(3)
    reset_flag = [False]

    def on_key(key: int) -> None:
        if key == 265:     # Up
            cmd[0] += 0.1
        elif key == 264:   # Down
            cmd[0] -= 0.1
        elif key == 263:   # Left
            cmd[2] += 0.2
        elif key == 262:   # Right
            cmd[2] -= 0.2
        elif key == 266:   # PageUp
            cmd[1] += 0.1
        elif key == 267:   # PageDown
            cmd[1] -= 0.1
        elif key == 268:   # Home
            cmd[:] = 0.0
        elif key == 269:   # End
            reset_flag[0] = True
        cmd[:] = ctrl.clip_command(cmd)
        print(f"command: vx={cmd[0]:+.1f} m/s  vy={cmd[1]:+.1f} m/s  yaw={cmd[2]:+.1f} rad/s")

    dec = sim.robot.decimation
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as viewer:
        viewer.cam.type = 1  # tracking camera
        viewer.cam.trackbodyid = sim.pelvis
        viewer.cam.distance, viewer.cam.elevation = 3.5, -15
        next_t = time.perf_counter()
        while viewer.is_running():
            if reset_flag[0] or sim.fallen(0.3):
                (respawn or sim.reset)(support=False)
                ctrl.reset()
                reset_flag[0] = False
            with viewer.lock():
                q, dq, _ = sim.joint_state()
                quat, gyro, _ = sim.imu()
                out = ctrl.step(q, dq, quat, gyro, cmd)
                sim.set_command(out.position, out.velocity, out.kp, out.kd, out.effort)
                sim.step(dec)
            viewer.sync()
            next_t += ctrl.dt
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            elif sleep < -0.2:
                next_t = time.perf_counter()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--config", default=None)
    ap.add_argument("--headless", action="store_true", help="run the scripted test without a window")
    ap.add_argument("--video", default=None, help="save an mp4/gif of the scripted test")
    ap.add_argument("--latency-steps", type=int, default=0, help="physics steps of actuation delay (0-3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--terrain-level", type=int, default=0,
                    help="0 = as configured; 1-10 fills the height field at that level")
    ap.add_argument("--terrain-kind", default="rough", help="rough | waves | steps | flat")
    args = ap.parse_args()

    cfg = load_config(args.config)
    policy_path = resolve_path(args.policy)
    ctrl = PolicyController(str(policy_path))
    sim = H1Sim(cfg, visual=True)
    respawn = None
    if args.terrain_level > 0:
        respawn = make_terrain(sim, cfg, args.terrain_level, args.terrain_kind, args.seed)
        respawn(support=False)
    else:
        sim.reset(support=False)
    print(f"policy: {policy_path} (trained {ctrl.meta.get('iteration')} iterations)")

    if not args.headless and not args.video:
        interactive(sim, ctrl, respawn)
        return

    renderer, frames = None, None
    if args.video:
        import mujoco
        renderer, frames = mujoco.Renderer(sim.model, height=480, width=640), []
    stats = run_script(sim, ctrl, DEMO_SCRIPT, latency_steps=args.latency_steps, renderer=renderer, frames=frames)
    e = stats["mean_abs_error"]
    print(f"falls: {stats['falls']} | mean |error| vx {e[0]:.3f} m/s, vy {e[1]:.3f} m/s, yaw {e[2]:.3f} rad/s")
    if renderer is not None:
        renderer.close()
    if args.video and frames:
        import imageio.v2 as imageio
        imageio.mimsave(args.video, frames, fps=30)
        print(f"video saved to {args.video}")


if __name__ == "__main__":
    main()
