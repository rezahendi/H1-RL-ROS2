"""MuJoCo simulation of the H1 with a motor-level command interface (no ROS).

Used by the ROS 2 simulator node and by `h1_rl.play`. Physics settings, PD
actuators and sensors are identical to the training environment, so a policy
sees the same dynamics here as during training.

Each joint follows the Unitree-style impedance command
    tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff      (clipped to motor limits)
The kp/kd part runs inside MuJoCo's position servos (implicit damping, same as
training); dq_des and tau_ff are added as generalized forces.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .obs import projected_gravity
from .robot import (RobotSpec, build_model, joint_qpos_qvel_index, sensor_slices, set_pd_gains,
                    standing_qpos)


class H1Sim:
    # virtual support ("elastic band") that holds the pelvis until a controller takes over
    SUPPORT_KP_LIN, SUPPORT_KD_LIN = 3000.0, 300.0     # N/m, N s/m
    SUPPORT_KP_ANG, SUPPORT_KD_ANG = 800.0, 80.0       # Nm/rad, Nm s/rad

    def __init__(self, cfg: dict, visual: bool = True):
        self.cfg = cfg
        self.robot = RobotSpec.from_config(cfg)
        self.model = build_model(cfg, visual=visual)
        self.data = mujoco.MjData(self.model)
        self.qadr, self.vadr = joint_qpos_qvel_index(self.model, self.robot.joint_names)
        self.sens = sensor_slices(self.model)
        self.pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.dt = float(self.model.opt.timestep)
        n = self.robot.num_joints
        self.q_des = self.robot.default_q.copy()
        self.dq_des = np.zeros(n)
        self.tau_ff = np.zeros(n)
        self.kp = self.robot.kp.copy()
        self.kd = self.robot.kd.copy()
        self.support = False
        self._support_force_on = False
        # simulation clock as an exact integer (ns): never jumps back, also not across resets
        self._dt_ns = int(round(self.dt * 1e9))
        self._time_ns = 0
        self.reset(support=True)

    # --------------------------------------------------------------- control
    def reset(self, support: bool = True, yaw: float = 0.0) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = standing_qpos(m, self.robot, yaw=yaw)
        self.hold_default_pose()
        d.ctrl[:] = self.q_des
        mujoco.mj_forward(m, d)
        self.support = support
        self.support_pos = d.qpos[0:3].copy()
        self.support_quat = d.qpos[3:7].copy()

    def hold_default_pose(self) -> None:
        self.set_command(self.robot.default_q, kp=self.robot.kp, kd=self.robot.kd)

    def set_command(self, q_des, dq_des=None, kp=None, kd=None, tau_ff=None) -> None:
        """All arrays are in joint order (RobotSpec.joint_names)."""
        self.q_des[:] = q_des
        self.dq_des[:] = 0.0 if dq_des is None else dq_des
        self.tau_ff[:] = 0.0 if tau_ff is None else tau_ff
        if kp is not None:
            self.kp[:] = kp
        if kd is not None:
            self.kd[:] = kd
        set_pd_gains(self.model, self.kp, self.kd)

    def step(self, nstep: int = 1) -> None:
        d = self.data
        d.ctrl[:] = self.q_des
        extra = self.tau_ff + self.kd * self.dq_des
        d.qfrc_applied[:] = 0.0
        if np.any(extra):
            lim = self.robot.torque_limit
            d.qfrc_applied[self.vadr] = np.clip(extra, -lim, lim)
        for _ in range(nstep):
            if self.support:
                self._apply_support()
                self._support_force_on = True
            elif self._support_force_on:  # band just released: remove its force once
                d.xfrc_applied[self.pelvis] = 0.0
                self._support_force_on = False
            mujoco.mj_step(self.model, d)
            self._time_ns += self._dt_ns

    def _apply_support(self) -> None:
        d = self.data
        pos, vel = d.xpos[self.pelvis], d.qvel[0:3]
        force = self.SUPPORT_KP_LIN * (self.support_pos - pos) - self.SUPPORT_KD_LIN * vel
        err = np.zeros(3)
        q_err = np.zeros(4)
        q_inv = np.zeros(4)
        mujoco.mju_negQuat(q_inv, d.qpos[3:7])
        mujoco.mju_mulQuat(q_err, self.support_quat, q_inv)
        mujoco.mju_quat2Vel(err, q_err, 1.0)          # world-frame rotation vector
        ang_vel_world = d.xmat[self.pelvis].reshape(3, 3) @ d.qvel[3:6]
        torque = self.SUPPORT_KP_ANG * err - self.SUPPORT_KD_ANG * ang_vel_world
        d.xfrc_applied[self.pelvis, 0:3] = force
        d.xfrc_applied[self.pelvis, 3:6] = torque

    # ----------------------------------------------------------------- state
    @property
    def time_ns(self) -> int:
        """Simulation time in integer nanoseconds (monotonic, exact)."""
        return self._time_ns

    @property
    def time(self) -> float:
        return self._time_ns * 1e-9

    def joint_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = self.data
        tau = d.actuator_force.copy() + d.qfrc_applied[self.vadr]
        return d.qpos[self.qadr].copy(), d.qvel[self.vadr].copy(), tau

    def imu(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(orientation quaternion w,x,y,z ; gyro [rad/s] ; accelerometer [m/s^2]), body frame."""
        d, s = self.data, self.sens
        return d.qpos[3:7].copy(), d.sensordata[s["imu_gyro"]].copy(), d.sensordata[s["imu_acc"]].copy()

    def base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.qpos[0:3].copy(), self.data.qpos[3:7].copy()

    def base_twist_body(self) -> tuple[np.ndarray, np.ndarray]:
        s = self.sens
        return self.data.sensordata[s["imu_linvel"]].copy(), self.data.sensordata[s["imu_gyro"]].copy()

    def foot_forces(self) -> np.ndarray:
        s = self.sens
        return np.array([self.data.sensordata[s["left_foot_touch"]][0], self.data.sensordata[s["right_foot_touch"]][0]])

    def fallen(self, min_height: float = 0.5, max_tilt: float = 0.85) -> bool:
        """Pelvis below `min_height` or tilted more than asin(max_tilt) (~58 deg)."""
        g = projected_gravity(self.data.qpos[3:7][None])[0]
        return bool(self.data.qpos[2] < min_height or np.hypot(g[0], g[1]) > max_tilt)
