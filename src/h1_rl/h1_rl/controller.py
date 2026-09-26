"""Policy controller: IMU + joint encoder readings in, joint PD commands out (no ROS).

The ROS 2 node `policy_controller` is a thin wrapper around this class, and
`h1_rl.play` uses it directly, so the exact same code path is tested with and
without ROS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .obs import (DisturbanceDetector, GaitClock, ObsHistory, ObsScales, command_deadband,
                  frame_dim, gait_reference, is_walking, obs_frame, projected_gravity,
                  ramp_command, reference_pattern)
from .policy import Policy

RUNNING, FALLEN = "running", "fallen"


@dataclass
class JointCommand:
    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray
    kp: np.ndarray
    kd: np.ndarray


class PolicyController:
    def __init__(self, policy_path: str, fall_tilt: float | None = None,
                 recover_tilt: float = 0.2, recover_time: float = 1.0, damping_kd: float = 5.0):
        self.policy = Policy(policy_path)
        meta = self.policy.meta
        self.meta = meta
        self.joint_names: list[str] = list(meta["joint_names"])
        self.policy_joints: list[str] = list(meta["policy_joints"])
        self.leg = np.array([self.joint_names.index(j) for j in self.policy_joints])
        self.default_q = np.asarray(meta["default_q"], dtype=np.float64)
        self.kp = np.asarray(meta["kp"], dtype=np.float64)
        self.kd = np.asarray(meta["kd"], dtype=np.float64)
        self.action_scale = float(meta["action_scale"])
        self.action_clip = float(meta["action_clip"])
        self.dt = float(meta["control_dt"])
        self.scales = ObsScales.from_dict(meta["obs_scales"])
        self.gait_period = float(meta["gait"]["period"])
        self.gait_offset = float(meta["gait"]["offset"])
        self.stance_ratio = float(meta["gait"]["stance_ratio"])
        self.residual_reference = bool(meta.get("residual_reference", False))
        self.ref_amplitude = float(meta.get("ref_amplitude", 0.0))
        self.ref_pattern = reference_pattern(self.policy_joints,
                                             float(meta.get("arm_swing", 0.0)))
        # older policy files: clock never stops, no command deadband, no recovery step
        rec = dict(meta.get("recovery") or {})
        self.disturbance = DisturbanceDetector(1, rec.get("tilt_change", 0.0),
                                               rec.get("ang_vel", 0.0), rec.get("filter", 0.02))
        self.clock = GaitClock(self.gait_period, float(meta["control_dt"]), 1,
                               bool(meta.get("stand_when_idle", False)),
                               int(round(float(rec.get("time", 0.0)) / float(meta["control_dt"]))))
        cmds = meta["commands"]
        self.xy_deadband = float(cmds.get("small_cmd_threshold", 0.0))
        self.yaw_deadband = float(cmds.get("small_yaw_threshold", 0.0))
        accel = np.asarray(cmds.get("accel_limit", [0.0, 0.0, 0.0]), dtype=np.float64) * self.dt
        self.cmd_delta = accel if np.any(accel) else np.full(3, np.inf)
        self.cmd_low = np.array([cmds["lin_vel_x"][0], cmds["lin_vel_y"][0], cmds["ang_vel_yaw"][0]])
        self.cmd_high = np.array([cmds["lin_vel_x"][1], cmds["lin_vel_y"][1], cmds["ang_vel_yaw"][1]])
        self.fall_tilt = float(fall_tilt if fall_tilt is not None else meta["termination"]["max_tilt_gravity_xy"])
        self.recover_tilt = recover_tilt
        self.recover_steps = int(round(recover_time / self.dt))
        self.damping_kd = damping_kd
        fd = int(meta.get("frame_dim") or frame_dim(len(self.policy_joints)))
        self.history = ObsHistory(1, fd, int(meta["history_length"]))
        if self.history.length * fd != self.policy.obs_dim:
            raise ValueError("policy input size does not match its metadata")
        self.state = RUNNING
        self.reset()

    # ------------------------------------------------------------------ API
    @property
    def phase(self) -> float:
        return float(self.clock.phase[0])

    @property
    def standing(self) -> bool:
        """True while the gait clock is stopped (idle command, both feet down)."""
        return bool(self.clock.standing([is_walking(self._last_cmd)])[0])

    def reset(self) -> None:
        self.last_action = np.zeros(len(self.leg))
        self.clock.reset()
        self.disturbance.reset()
        self._last_cmd = np.zeros(3)
        self.command = np.zeros(3)       # effective command (acceleration limited)
        self._history_ready = False
        self._upright_steps = 0
        self.state = RUNNING

    def clip_command(self, command) -> np.ndarray:
        """Clip to the trained ranges and zero out tiny commands (same deadband as in training)."""
        cmd = np.clip(np.asarray(command, dtype=np.float64), self.cmd_low, self.cmd_high)
        return command_deadband(cmd, self.xy_deadband, self.yaw_deadband)

    def hold_command(self, q_des=None) -> JointCommand:
        n = len(self.joint_names)
        return JointCommand(self.default_q.copy() if q_des is None else np.asarray(q_des, float),
                            np.zeros(n), np.zeros(n), self.kp.copy(), self.kd.copy())

    def damping_command(self, q) -> JointCommand:
        n = len(self.joint_names)
        return JointCommand(np.asarray(q, float).copy(), np.zeros(n), np.zeros(n), np.zeros(n),
                            np.full(n, self.damping_kd))

    def step(self, q, dq, quat_wxyz, gyro, command) -> JointCommand:
        """One control tick (call every `self.dt` seconds). Arrays are in joint order."""
        q = np.asarray(q, dtype=np.float64)
        dq = np.asarray(dq, dtype=np.float64)
        gravity = projected_gravity(np.asarray(quat_wxyz, dtype=np.float64)[None])[0]
        tilt = float(np.linalg.norm(gravity[:2]))

        if self.state == FALLEN:
            self._upright_steps = self._upright_steps + 1 if tilt < self.recover_tilt else 0
            if self._upright_steps >= self.recover_steps:
                self.reset()  # robot was put back on its feet: start again
            else:
                return self.damping_command(q)
        elif tilt > self.fall_tilt:
            self.state = FALLEN
            self._upright_steps = 0
            return self.damping_command(q)

        # ramp towards the requested command so the robot can accelerate and brake smoothly
        self.command = ramp_command(self.command, self.clip_command(command), self.cmd_delta)
        cmd = self.command
        frame = obs_frame(self.scales, np.asarray(gyro, dtype=np.float64)[None], gravity[None], cmd[None],
                          (q[self.leg] - self.default_q[self.leg])[None], dq[self.leg][None],
                          self.last_action[None], np.array([self.phase]))
        if self._history_ready:
            self.history.push(frame)
        else:
            self.history.reset([0], frame)
            self._history_ready = True
        action = np.clip(self.policy(self.history.flat()[0]), -self.action_clip, self.action_clip)
        self.last_action = action

        target = self.default_q.copy()
        target[self.leg] += self.action_scale * action
        if self.residual_reference:  # policy output is a correction on top of the stepping reference
            target[self.leg] += gait_reference(self.clock.phase, self.gait_offset, self.stance_ratio,
                                               self.ref_amplitude, self.ref_pattern)[0]
        self._last_cmd = cmd
        # a shove restarts the clock so the robot can step to catch itself
        self.clock.trigger_recovery(self.disturbance(gravity[None],
                                                     np.asarray(gyro, dtype=np.float64)[None]))
        self.clock.advance([is_walking(cmd)])  # stops at double support when the command is zero
        return self.hold_command(target)
