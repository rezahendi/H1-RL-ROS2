"""Export a training checkpoint to a NumPy policy file for the ROS 2 controller.

    python3 -m h1_rl.export --checkpoint logs/h1_walk/<run>/model_3000.pt \
                            --output policies/h1_walk.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .obs import FRAME_DIM, ObsScales
from .robot import RobotSpec


def policy_meta(cfg: dict, iteration: int | None = None) -> dict:
    """Everything the controller needs to reproduce the training setup."""
    robot = RobotSpec.from_config(cfg)
    return {
        "format": "h1_rl_policy_v1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "iteration": iteration,
        "activation": cfg["ppo"].get("activation", "elu"),
        "joint_names": robot.joint_names,
        "policy_joints": robot.policy_joints,
        "default_q": robot.default_q.tolist(),
        "kp": robot.kp.tolist(),
        "kd": robot.kd.tolist(),
        "torque_limit": robot.torque_limit.tolist(),
        "sim_dt": robot.timestep,
        "decimation": robot.decimation,
        "control_dt": robot.control_dt,
        "action_scale": float(cfg["control"]["action_scale"]),
        "action_clip": float(cfg["control"]["action_clip"]),
        "obs_scales": ObsScales(cfg).to_dict(),
        "frame_dim": FRAME_DIM,
        "history_length": int(cfg["observations"]["history_length"]),
        "gait": {k: float(cfg["gait"][k]) for k in ("period", "offset", "stance_ratio", "swing_height")},
        "ref_amplitude": float(cfg["gait"].get("ref_amplitude", 0.0)),
        "residual_reference": bool(cfg["control"].get("residual_reference", False)),
        "stand_when_idle": bool(cfg["gait"].get("stand_when_idle", False)),
        "commands": {**{k: list(map(float, cfg["commands"][k])) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")},
                     "small_cmd_threshold": float(cfg["commands"].get("small_cmd_threshold", 0.0)),
                     "small_yaw_threshold": float(cfg["commands"].get("small_yaw_threshold", 0.0)),
                     "accel_limit": list(map(float, cfg["commands"].get("accel_limit", [0.0, 0.0, 0.0])))},
        "termination": dict(cfg["env"]["termination"]),
    }


def export_checkpoint(checkpoint: str | Path, output: str | Path) -> Path:
    import torch  # only needed here

    ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    state = ck["policy"]
    layer_ids = sorted({int(k.split(".")[1]) for k in state if k.startswith("actor.") and k.endswith(".weight")})
    arrays = {}
    for n, i in enumerate(layer_ids):
        arrays[f"w{n}"] = state[f"actor.{i}.weight"].numpy().astype(np.float32)
        arrays[f"b{n}"] = state[f"actor.{i}.bias"].numpy().astype(np.float32)
    norm = ck.get("obs_norm")
    num_obs = arrays["w0"].shape[1]
    if norm is not None:
        mean = norm["mean"].numpy().reshape(-1)
        std = np.sqrt(norm["var"].numpy().reshape(-1)) + float(ck.get("norm_eps", 1e-2))
    else:
        mean, std = np.zeros(num_obs), np.ones(num_obs)
    meta = policy_meta(cfg, ck.get("iteration"))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, meta=np.array(json.dumps(meta)), num_layers=np.array(len(layer_ids)),
             obs_mean=mean.astype(np.float32), obs_std=std.astype(np.float32), **arrays)
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="h1_walk.npz")
    args = ap.parse_args()
    out = export_checkpoint(args.checkpoint, args.output)
    print(f"Exported policy to {out}")


if __name__ == "__main__":
    main()
