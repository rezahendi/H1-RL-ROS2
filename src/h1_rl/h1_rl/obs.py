"""Observation construction shared by the training env and the ROS 2 controller.

Keeping this in one place guarantees the controller feeds the policy exactly
what it saw during training. One observation frame (41 values):

    [ 0: 3]  base angular velocity (IMU gyro, body frame)   * ang_vel scale
    [ 3: 6]  gravity direction in the body frame (from the IMU orientation)
    [ 6: 9]  command (vx, vy, yaw rate)                        * command scale
    [ 9:19]  leg joint angles - default angles                 * dof_pos scale
    [19:29]  leg joint velocities                              * dof_vel scale
    [29:39]  previous action
    [39:41]  gait clock: sin(2*pi*phase), cos(2*pi*phase)

The actor sees the last `history_length` frames, oldest first.
"""

from __future__ import annotations

import numpy as np

GRAVITY = np.array([0.0, 0.0, -1.0])
FRAME_DIM = 41


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world-frame vectors into the frame described by quaternion q (w, x, y, z)."""
    q_wxyz = np.asarray(q_wxyz, dtype=np.float64)
    w = q_wxyz[..., 0:1]
    qv = q_wxyz[..., 1:4]
    v = np.broadcast_to(v, qv.shape)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qv, v) * w * 2.0
    c = qv * np.sum(qv * v, axis=-1, keepdims=True) * 2.0
    return a - b + c


def projected_gravity(q_wxyz: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse(q_wxyz, GRAVITY)


def leg_phases(phase: np.ndarray, offset: float) -> np.ndarray:
    """Phase of (left, right) leg in [0, 1)."""
    phase = np.asarray(phase)
    return np.stack([phase % 1.0, (phase + offset) % 1.0], axis=-1)


class GaitClock:
    """Gait phase as an exact integer tick counter (one tick per control step).

    With `stand_when_idle`, the clock keeps running while the robot walks. When the command
    becomes zero it runs on to the next double-support instant (phase 0.0 or 0.5, both feet
    on the ground) and stops there, so the robot finishes its step and then stands still.
    Requires a right-leg offset of 0.5 and a stance ratio above 0.5.
    """

    def __init__(self, period: float, dt: float, num: int = 1, stand_when_idle: bool = True):
        self.ticks_per_cycle = int(round(period / dt))
        if self.ticks_per_cycle % 2:
            raise ValueError("gait period / control dt must be an even number of ticks")
        self.half = self.ticks_per_cycle // 2
        self.stand_when_idle = stand_when_idle
        self.ticks = np.zeros(num, dtype=np.int64)

    @property
    def phase(self) -> np.ndarray:
        return self.ticks / self.ticks_per_cycle

    def standing(self, walking) -> np.ndarray:
        """True where the clock is stopped (idle command and both feet in stance)."""
        if not self.stand_when_idle:
            return np.zeros(self.ticks.shape, dtype=bool)
        return ~np.asarray(walking, dtype=bool) & (self.ticks % self.half == 0)

    def reset(self, ids=slice(None)) -> None:
        self.ticks[ids] = 0

    def advance(self, walking) -> None:
        moving = ~self.standing(walking)
        self.ticks = np.where(moving, (self.ticks + 1) % self.ticks_per_cycle, self.ticks)


def ramp_command(current: np.ndarray, target: np.ndarray, max_delta: np.ndarray) -> np.ndarray:
    """Move the command towards the target by at most `max_delta` (= acceleration limit * dt).

    Without this, a command that drops to zero the moment both feet are on the ground would
    freeze the gait clock instantly and the robot would have no step left to brake with.
    """
    return current + np.clip(np.asarray(target) - np.asarray(current), -max_delta, max_delta)


def command_deadband(cmd: np.ndarray, xy_threshold: float, yaw_threshold: float) -> np.ndarray:
    """Zero out tiny commands (|v_xy| or |yaw rate| below the thresholds), row-wise."""
    cmd = np.array(cmd, dtype=np.float64, copy=True)
    xy = cmd[..., :2]
    xy[np.linalg.norm(xy, axis=-1) < xy_threshold] = 0.0
    yaw = cmd[..., 2]
    yaw[np.abs(yaw) < yaw_threshold] = 0.0
    return cmd


def is_walking(cmd: np.ndarray) -> np.ndarray:
    return np.any(np.asarray(cmd) != 0.0, axis=-1)


def reference_pattern(policy_joints: list[str]) -> np.ndarray:
    """(2, n_actions) joint pattern of the stepping reference for the (left, right) swing leg:
    hip pitch -1, knee +2, ankle -1 (bends the leg and keeps the foot level)."""
    pattern = np.zeros((2, len(policy_joints)))
    for side, prefix in enumerate(("left", "right")):
        for joint, gain in (("hip_pitch", -1.0), ("knee", 2.0), ("ankle", -1.0)):
            pattern[side, policy_joints.index(f"{prefix}_{joint}_joint")] = gain
    return pattern


def swing_shape(phase: np.ndarray, offset: float, stance_ratio: float) -> np.ndarray:
    """(N, 2): 0 while a foot should be on the ground, half-sine bump (0..1) during its swing."""
    lp = leg_phases(phase, offset)
    progress = np.clip((lp - stance_ratio) / (1.0 - stance_ratio), 0.0, 1.0)
    return np.where(lp < stance_ratio, 0.0, np.sin(np.pi * progress))


def gait_reference(phase: np.ndarray, offset: float, stance_ratio: float, amplitude: float,
                   pattern: np.ndarray) -> np.ndarray:
    """(N, n_actions) joint offsets of the stepping-in-place reference at the given gait phase."""
    return amplitude * (swing_shape(phase, offset, stance_ratio) @ pattern)


class ObsScales:
    def __init__(self, cfg: dict):
        s = cfg["observations"]["scales"]
        self.ang_vel = float(s["ang_vel"])
        self.dof_pos = float(s["dof_pos"])
        self.dof_vel = float(s["dof_vel"])
        self.lin_vel = float(s.get("lin_vel", 2.0))
        self.command = np.asarray(s["command"], dtype=np.float64)
        self.clip = float(cfg["observations"].get("clip", 100.0))

    def to_dict(self) -> dict:
        return {"ang_vel": self.ang_vel, "dof_pos": self.dof_pos, "dof_vel": self.dof_vel,
                "lin_vel": self.lin_vel, "command": self.command.tolist(), "clip": self.clip}

    @classmethod
    def from_dict(cls, d: dict) -> "ObsScales":
        obj = cls.__new__(cls)
        obj.ang_vel, obj.dof_pos, obj.dof_vel = d["ang_vel"], d["dof_pos"], d["dof_vel"]
        obj.lin_vel = d.get("lin_vel", 2.0)
        obj.command = np.asarray(d["command"], dtype=np.float64)
        obj.clip = d.get("clip", 100.0)
        return obj


def obs_frame(scales: ObsScales, gyro, gravity, command, q_leg_rel, dq_leg, last_action, phase) -> np.ndarray:
    """Build one observation frame. All inputs are batched (N, ...) arrays; phase is (N,)."""
    phase = np.asarray(phase, dtype=np.float64)
    two_pi_phase = 2.0 * np.pi * phase
    frame = np.concatenate(
        [
            np.asarray(gyro) * scales.ang_vel,
            np.asarray(gravity),
            np.asarray(command) * scales.command,
            np.asarray(q_leg_rel) * scales.dof_pos,
            np.asarray(dq_leg) * scales.dof_vel,
            np.asarray(last_action),
            np.sin(two_pi_phase)[:, None],
            np.cos(two_pi_phase)[:, None],
        ],
        axis=1,
    )
    return np.clip(frame, -scales.clip, scales.clip)


class ObsHistory:
    """Rolling buffer of observation frames, flattened oldest-first for the actor."""

    def __init__(self, num_envs: int, frame_dim: int, length: int):
        self.length = int(length)
        self.buf = np.zeros((num_envs, self.length, frame_dim), dtype=np.float64)

    def reset(self, ids, frames: np.ndarray) -> None:
        self.buf[ids] = np.asarray(frames)[:, None, :]

    def push(self, frames: np.ndarray) -> None:
        self.buf[:, :-1] = self.buf[:, 1:].copy()
        self.buf[:, -1] = frames

    def flat(self) -> np.ndarray:
        return self.buf.reshape(self.buf.shape[0], -1)
