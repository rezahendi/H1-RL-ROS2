"""Batched MuJoCo environment: Unitree H1 velocity-tracking locomotion.

Design
------
* N independent MjModel/MjData pairs (per-env copies so each env can have its
  own randomized friction, mass and PD gains). Physics runs in a thread pool;
  MuJoCo releases the GIL inside mj_step, so this scales with CPU cores.
* The policy runs at 50 Hz (4 x 5 ms physics steps) and outputs position
  offsets for the 10 leg joints. PD control happens inside MuJoCo (position
  servo actuators), like the H1 motor drivers do on the real robot.
* Rewards follow Unitree's legged_gym H1 recipe (velocity tracking + gait clock
  + regularization), observations are built with h1_rl.obs so the ROS 2
  controller reproduces them exactly.
"""

from __future__ import annotations

import copy
import os
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

from ..obs import (FRAME_DIM, GaitClock, ObsHistory, ObsScales, command_deadband, gait_reference,
                   is_walking, leg_phases, obs_frame, projected_gravity, ramp_command, reference_pattern,
                   swing_shape)
from ..robot import (RobotSpec, build_model, foot_geom_ids, joint_qpos_qvel_index,
                     place_on_ground, sensor_slices, set_pd_gains)

DEFAULT_THREADS = 4          # see h1_rl/threads.py
CONTACT_FORCE_THRESHOLD = 1.0  # [N]


class H1WalkEnv:
    def __init__(self, cfg: dict, num_envs: int, num_threads: int | None = None,
                 seed: int = 0, randomize: bool = True, obs_noise: bool = True,
                 pushes: bool = True):
        self.cfg = cfg
        self.num_envs = n = int(num_envs)
        self.robot = RobotSpec.from_config(cfg)
        self.dt = self.robot.control_dt
        self.decimation = self.robot.decimation
        self.randomize = randomize
        self.obs_noise = obs_noise
        self.pushes = pushes
        self.rng = np.random.default_rng(seed)

        # ------------------------------------------------------------ physics
        self.base_model = build_model(cfg, visual=False)
        m = self.base_model
        self.models = [copy.copy(m) for _ in range(n)]
        self.datas = [mujoco.MjData(mm) for mm in self.models]
        self.qadr, self.vadr = joint_qpos_qvel_index(m, self.robot.joint_names)
        self.sens = sensor_slices(m)
        self.feet = foot_geom_ids(m)
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.nominal_pelvis_mass = float(m.body_mass[self.pelvis])
        self.nominal_pelvis_ipos = m.body_ipos[self.pelvis].copy()
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.robot.joint_names]
        self.joint_range = m.jnt_range[jids].copy()

        # --------------------------------------------------------------- tables
        self.leg = self.robot.policy_idx
        self.num_actions = len(self.leg)
        self.default_q = self.robot.default_q.copy()
        ctrl = cfg["control"]
        self.action_scale = float(ctrl["action_scale"])
        self.action_clip = float(ctrl["action_clip"])
        rew = cfg["rewards"]
        mid = self.joint_range.mean(axis=1)
        half = 0.5 * (self.joint_range[:, 1] - self.joint_range[:, 0]) * float(rew["soft_dof_pos_limit"])
        self.soft_lower, self.soft_upper = (mid - half)[self.leg], (mid + half)[self.leg]
        names = self.robot.joint_names
        self.hip_yaw_roll = np.array([i for i in self.leg if "hip_yaw" in names[i] or "hip_roll" in names[i]])
        g = cfg["gait"]
        self.gait_period, self.gait_offset = float(g["period"]), float(g["offset"])
        self.stance_ratio, self.swing_height = float(g["stance_ratio"]), float(g["swing_height"])
        self.ref_amplitude = float(g.get("ref_amplitude", 0.3))
        # stepping reference: bend hip/knee/ankle of the swing leg (foot stays level)
        self.ref_pattern = reference_pattern(self.robot.policy_joints)
        # the policy outputs residuals on top of the reference (feedforward gait + learned feedback)
        self.residual_reference = bool(ctrl.get("residual_reference", False))
        self.cmd_cfg = cfg["commands"]
        self.clock = GaitClock(self.gait_period, self.dt, n, bool(g.get("stand_when_idle", False)))
        self.dr = cfg["domain_rand"]
        term = cfg["env"]["termination"]
        self.min_base_height = float(term["min_base_height"])
        self.max_tilt = float(term["max_tilt_gravity_xy"])
        self.max_episode_steps = int(round(float(cfg["env"]["episode_length_s"]) / self.dt))

        # -------------------------------------------------------------- buffers
        self.qpos = np.zeros((n, m.nq))
        self.qvel = np.zeros((n, m.nv))
        self.sensordata = np.zeros((n, m.nsensordata))
        self.tau = np.zeros((n, m.nu))
        self.targets = np.tile(self.default_q, (n, 1))
        self.prev_targets = self.targets.copy()
        self.actions = np.zeros((n, self.num_actions))
        self.last_actions = np.zeros((n, self.num_actions))
        self.last_dq_leg = np.zeros((n, self.num_actions))
        self.commands = np.zeros((n, 3))       # effective command (acceleration limited)
        self.cmd_target = np.zeros((n, 3))     # what was sampled / what the user asked for
        self.cmd_delta = np.asarray(self.cmd_cfg.get("accel_limit", [0.0, 0.0, 0.0]), dtype=float) * self.dt
        if not np.any(self.cmd_delta):
            self.cmd_delta = np.full(3, np.inf)
        self.cmd_timer = np.zeros(n)
        self.push_timer = np.zeros(n)
        self.delay = np.zeros(n, dtype=int)
        self.episode_step = np.zeros(n, dtype=int)
        self.friction = np.ones(n)
        self.added_mass = np.zeros(n)

        # --------------------------------------------------------- observations
        self.scales = ObsScales(cfg)
        self.noise = cfg["observations"]["noise"]
        self.history = ObsHistory(n, FRAME_DIM, int(cfg["observations"]["history_length"]))
        self.num_obs = FRAME_DIM * self.history.length
        self.num_critic_obs = FRAME_DIM + 3 + 1 + 2 + 2 + 2

        # -------------------------------------------------------------- rewards
        self.tracking_sigma = float(rew["tracking_sigma"])
        self.base_height_target = float(rew["base_height_target"])
        self.only_positive = bool(rew.get("only_positive_rewards", True))
        self.reward_scales = {k: float(v) * self.dt for k, v in rew["scales"].items() if float(v) != 0.0}
        self.reward_fns = {k: getattr(self, f"_reward_{k}") for k in self.reward_scales}
        self.episode_sums = {k: np.zeros(n) for k in self.reward_scales}

        # -------------------------------------------------------------- threads
        # NOT os.cpu_count(): throughput peaks at a few threads and then collapses
        # (GIL around mj_step + main-thread numpy). h1_rl.threads.choose_threads measures it.
        nt = int(num_threads or min(os.cpu_count() or 1, DEFAULT_THREADS))
        nt = max(1, min(nt, n))
        self.num_threads = nt
        self.chunks = [c for c in np.array_split(np.arange(n), nt) if len(c)]
        self.pool = ThreadPoolExecutor(nt) if nt > 1 else None

        self.reset_all()

    # ================================================================== API
    def reset_all(self):
        ids = np.arange(self.num_envs)
        self._reset_envs(ids)
        return self._compute_obs(ids)

    def step(self, actions: np.ndarray):
        actions = np.clip(np.asarray(actions, dtype=np.float64), -self.action_clip, self.action_clip)
        self.actions[:] = actions
        self.prev_targets[:] = self.targets
        self.targets[:, self.leg] = self.default_q[self.leg] + self.action_scale * actions
        if self.residual_reference:  # reference at the gait phase the policy just observed
            self.targets[:, self.leg] += self._reference(self.phase)

        if self.pushes:
            self._apply_pushes()
        self._simulate()
        self.episode_step += 1
        self.clock.advance(is_walking(self.commands))
        self._update_state()

        terminated = (self.base_height < self.min_base_height) | \
                     (np.linalg.norm(self.gravity_b[:, :2], axis=1) > self.max_tilt)
        time_out = (self.episode_step >= self.max_episode_steps) & ~terminated
        rewards = self._compute_rewards()

        self.last_actions[:] = actions
        self.last_dq_leg[:] = self.dq[:, self.leg]

        self.cmd_timer -= self.dt
        expired = np.nonzero(self.cmd_timer <= 0.0)[0]
        if len(expired):
            self._resample_commands(expired)
        self.commands = ramp_command(self.commands, self.cmd_target, self.cmd_delta)

        dones = terminated | time_out
        infos = {"time_outs": time_out.copy(), "terminated": terminated.copy()}
        reset_ids = np.nonzero(dones)[0]
        if len(reset_ids):
            infos["episode"] = self._episode_stats(reset_ids)
            self._reset_envs(reset_ids)
        obs, critic_obs = self._compute_obs(reset_ids)
        return obs, critic_obs, rewards, dones, infos

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None

    # ============================================================== physics
    def _simulate_chunk(self, ids):
        dec = self.decimation
        for i in ids:
            m, d = self.models[i], self.datas[i]
            k = int(self.delay[i])
            if k > 0:  # actuation latency: old target is still active for k steps
                d.ctrl[:] = self.prev_targets[i]
                mujoco.mj_step(m, d, k)
            d.ctrl[:] = self.targets[i]
            mujoco.mj_step(m, d, dec - k)
            self.qpos[i] = d.qpos
            self.qvel[i] = d.qvel
            self.sensordata[i] = d.sensordata
            self.tau[i] = d.actuator_force

    def _simulate(self):
        if self.pool is not None:
            list(self.pool.map(self._simulate_chunk, self.chunks))
        else:
            self._simulate_chunk(range(self.num_envs))

    def _apply_pushes(self):
        self.push_timer -= self.dt
        ids = np.nonzero(self.push_timer <= 0.0)[0]
        if len(ids) == 0:
            return
        v = float(self.dr["push_vel_xy"])
        for i in ids:
            self.datas[i].qvel[0:2] += self.rng.uniform(-v, v, size=2)
        lo, hi = self.dr["push_interval_s"]
        self.push_timer[ids] = self.rng.uniform(lo, hi, size=len(ids))

    def _update_state(self):
        s = self.sensordata
        sens = self.sens
        self.base_height = self.qpos[:, 2]
        self.base_quat = self.qpos[:, 3:7]
        self.q = self.qpos[:, self.qadr]
        self.dq = self.qvel[:, self.vadr]
        self.gyro = s[:, sens["imu_gyro"]]
        self.base_lin_vel = s[:, sens["imu_linvel"]]  # body frame
        self.gravity_b = projected_gravity(self.base_quat)
        self.foot_force = np.stack([s[:, sens["left_foot_touch"]][:, 0], s[:, sens["right_foot_touch"]][:, 0]], axis=1)
        self.contact = self.foot_force > CONTACT_FORCE_THRESHOLD
        self.foot_pos = np.stack([s[:, sens["left_foot_pos"]], s[:, sens["right_foot_pos"]]], axis=1)
        self.foot_vel = np.stack([s[:, sens["left_foot_vel"]], s[:, sens["right_foot_vel"]]], axis=1)
        self.phase = self.clock.phase
        self.leg_phase = leg_phases(self.phase, self.gait_offset)
        self.is_stance = self.leg_phase < self.stance_ratio
        self.swing_shape = swing_shape(self.phase, self.gait_offset, self.stance_ratio)  # (N, 2), 0..1

    def _reference(self, phase: np.ndarray) -> np.ndarray:
        return gait_reference(phase, self.gait_offset, self.stance_ratio, self.ref_amplitude, self.ref_pattern)

    # ================================================================ resets
    def _randomize_model(self, i: int):
        m, d, dr, rng = self.models[i], self.datas[i], self.dr, self.rng
        f = rng.uniform(*dr["friction"])
        m.geom_friction[self.feet, 0] = f
        dm = rng.uniform(*dr["added_base_mass"])
        m.body_mass[self.pelvis] = self.nominal_pelvis_mass + dm
        shift = float(dr.get("com_shift", 0.0))
        m.body_ipos[self.pelvis] = self.nominal_pelvis_ipos + rng.uniform(-shift, shift, size=3)
        nj = self.robot.num_joints
        kp = self.robot.kp * rng.uniform(*dr["kp_scale"], size=nj)
        kd = self.robot.kd * rng.uniform(*dr["kd_scale"], size=nj)
        set_pd_gains(m, kp, kd)
        mujoco.mj_setConst(m, d)
        self.friction[i], self.added_mass[i] = f, dm

    def _reset_envs(self, ids: np.ndarray):
        rng, dr = self.rng, self.dr
        noise = float(dr["init_joint_noise"]) if self.randomize else 0.0
        vel = float(dr["init_base_vel"]) if self.randomize else 0.0
        for i in ids:
            m, d = self.models[i], self.datas[i]
            if self.randomize:
                self._randomize_model(i)
            mujoco.mj_resetData(m, d)
            yaw = rng.uniform(-np.pi, np.pi)
            q = self.default_q + rng.uniform(-noise, noise, size=self.robot.num_joints)
            q = np.clip(q, self.joint_range[:, 0], self.joint_range[:, 1])
            d.qpos[0:3] = [0.0, 0.0, 1.0]
            d.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            d.qpos[self.qadr] = q
            place_on_ground(m, d, self.feet)
            d.qvel[0:6] = rng.uniform(-vel, vel, size=6)
            d.ctrl[:] = q
            mujoco.mj_forward(m, d)
            self.qpos[i] = d.qpos
            self.qvel[i] = d.qvel
            self.sensordata[i] = d.sensordata
            self.tau[i] = d.actuator_force
            self.targets[i] = q
            self.prev_targets[i] = q
        self.actions[ids] = 0.0
        self.last_actions[ids] = 0.0
        self.last_dq_leg[ids] = self.qvel[np.ix_(ids, self.vadr[self.leg])]
        self.episode_step[ids] = 0
        self.clock.reset(ids)
        lo, hi = dr["action_delay_steps"] if self.randomize else (0, 0)
        self.delay[ids] = rng.integers(int(lo), int(hi) + 1, size=len(ids))
        self.delay[ids] = np.minimum(self.delay[ids], self.decimation - 1)
        plo, phi = dr["push_interval_s"]
        self.push_timer[ids] = rng.uniform(plo, phi, size=len(ids))
        self._resample_commands(ids)
        self.commands[ids] = self.cmd_target[ids]  # no ramp from a stale command after a reset
        for k in self.episode_sums:
            self.episode_sums[k][ids] = 0.0
        self._update_state()

    def _resample_commands(self, ids: np.ndarray):
        c, rng, n = self.cmd_cfg, self.rng, len(ids)
        cmd = np.stack([rng.uniform(*c["lin_vel_x"], size=n),
                        rng.uniform(*c["lin_vel_y"], size=n),
                        rng.uniform(*c["ang_vel_yaw"], size=n)], axis=1)
        cmd = command_deadband(cmd, float(c["small_cmd_threshold"]), float(c.get("small_yaw_threshold", 0.0)))
        cmd[rng.random(n) < float(c["zero_prob"])] = 0.0
        self.cmd_target[ids] = cmd
        self.cmd_timer[ids] = rng.uniform(*c["resample_interval_s"], size=n)

    # ========================================================== observations
    def _compute_obs(self, reset_ids):
        leg = self.leg
        q_rel = self.q[:, leg] - self.default_q[leg]
        dq = self.dq[:, leg]
        clean = obs_frame(self.scales, self.gyro, self.gravity_b, self.commands, q_rel, dq,
                          self.last_actions, self.phase)
        if self.obs_noise:
            nz, rng, n = self.noise, self.rng, self.num_envs
            u = lambda dim, amp: rng.uniform(-1.0, 1.0, size=(n, dim)) * float(amp)  # noqa: E731
            noisy = obs_frame(self.scales,
                              self.gyro + u(3, nz["ang_vel"]),
                              self.gravity_b + u(3, nz["gravity"]),
                              self.commands,
                              q_rel + u(self.num_actions, nz["dof_pos"]),
                              dq + u(self.num_actions, nz["dof_vel"]),
                              self.last_actions, self.phase)
        else:
            noisy = clean
        self.history.push(noisy)
        if len(reset_ids):
            self.history.reset(reset_ids, noisy[reset_ids])
        obs = self.history.flat()
        critic = np.concatenate(
            [clean,
             self.base_lin_vel * self.scales.lin_vel,
             (self.base_height - self.base_height_target)[:, None],
             self.contact.astype(np.float64),
             self.foot_pos[:, :, 2],
             (self.friction - 0.8)[:, None],
             (self.added_mass * 0.1)[:, None]],
            axis=1)
        return obs.astype(np.float32), critic.astype(np.float32)

    # =============================================================== rewards
    def _compute_rewards(self) -> np.ndarray:
        total = np.zeros(self.num_envs)
        for name, scale in self.reward_scales.items():
            r = self.reward_fns[name]() * scale
            self.episode_sums[name] += r
            total += r
        if self.only_positive:
            total = np.clip(total, 0.0, None)
        return total

    def _episode_stats(self, ids) -> dict:
        ep_s = np.maximum(self.episode_step[ids], 1) * self.dt
        stats = {f"rew_{k}": float(np.mean(v[ids] / ep_s)) for k, v in self.episode_sums.items()}
        stats["episode_length_s"] = float(np.mean(self.episode_step[ids] * self.dt))
        return stats

    def _reward_tracking_lin_vel(self):
        err = np.sum((self.commands[:, :2] - self.base_lin_vel[:, :2]) ** 2, axis=1)
        return np.exp(-err / self.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        err = (self.commands[:, 2] - self.gyro[:, 2]) ** 2
        return np.exp(-err / self.tracking_sigma)

    def _reward_lin_vel_z(self):
        return self.base_lin_vel[:, 2] ** 2

    def _reward_ang_vel_xy(self):
        return np.sum(self.gyro[:, :2] ** 2, axis=1)

    def _reward_orientation(self):
        return np.sum(self.gravity_b[:, :2] ** 2, axis=1)

    def _reward_base_height(self):
        return (self.base_height - self.base_height_target) ** 2

    def _reward_dof_acc(self):
        return np.sum(((self.dq[:, self.leg] - self.last_dq_leg) / self.dt) ** 2, axis=1)

    def _reward_dof_vel(self):
        return np.sum(self.dq[:, self.leg] ** 2, axis=1)

    def _reward_torques(self):
        return np.sum(self.tau[:, self.leg] ** 2, axis=1)

    def _reward_action_rate(self):
        return np.sum((self.actions - self.last_actions) ** 2, axis=1)

    def _reward_dof_pos_limits(self):
        q = self.q[:, self.leg]
        out = -np.clip(q - self.soft_lower, None, 0.0) + np.clip(q - self.soft_upper, 0.0, None)
        return np.sum(out, axis=1)

    def _reward_alive(self):
        return np.ones(self.num_envs)

    def _reward_hip_pos(self):
        return np.sum(self.q[:, self.hip_yaw_roll] ** 2, axis=1)

    def _reward_contact_no_vel(self):
        v2 = np.sum(self.foot_vel ** 2, axis=2)
        return np.sum(v2 * self.contact, axis=1)

    def _reward_feet_swing_height(self):
        target = self.swing_height * self.swing_shape
        err = (self.foot_pos[:, :, 2] - target) ** 2
        return np.sum(err * ~self.is_stance, axis=1)

    def _reward_stand_still(self):
        # idle command and gait clock stopped: keep the legs still. Mostly a velocity penalty, because
        # the H1 cannot hold the default pose passively (weak ankles); a small pull towards it keeps
        # the posture tidy while leaving room for balance corrections.
        standing = self.clock.standing(is_walking(self.commands))
        still = np.sum(self.dq[:, self.leg] ** 2, axis=1)
        pose = np.sum(np.abs(self.q[:, self.leg] - self.default_q[self.leg]), axis=1)
        return (still + 0.2 * pose) * standing

    def _reward_contact(self):
        # +1 per foot whose contact matches the gait clock, -0.3 per mismatch
        return np.sum(np.where(self.contact == self.is_stance, 1.0, -0.3), axis=1)

    def _reward_ref_joint_pos(self):
        # follow a stepping-in-place leg motion timed by the gait clock (Humanoid-Gym style)
        ref = self.default_q[self.leg] + self._reference(self.phase)
        dist = np.linalg.norm(self.q[:, self.leg] - ref, axis=1)
        return np.exp(-2.0 * dist) - 0.2 * np.clip(dist, 0.0, 0.5)
