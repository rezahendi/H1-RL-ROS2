"""Tests that run without ROS:  python3 -m pytest src/h1_rl/test -q

The most important one checks that the deployment controller (used by the ROS
node) feeds the policy exactly the observations the training env produced.
"""

import json

import mujoco
import numpy as np
import pytest

from h1_rl.config import load_config, per_joint
from h1_rl.controller import PolicyController
from h1_rl.envs import H1WalkEnv
from h1_rl.export import policy_meta
from h1_rl.obs import projected_gravity
from h1_rl.robot import RobotSpec, build_model, standing_qpos
from h1_rl.sim import H1Sim


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def random_policy_file(tmp_path, cfg, num_obs, num_actions, seed=0):
    """A random (untrained) policy file with the real metadata layout."""
    rng = np.random.default_rng(seed)
    dims = [num_obs, 64, 32, num_actions]
    arrays = {}
    for i in range(3):
        arrays[f"w{i}"] = (rng.standard_normal((dims[i + 1], dims[i])) * 0.3 / np.sqrt(dims[i])).astype(np.float32)
        arrays[f"b{i}"] = np.zeros(dims[i + 1], dtype=np.float32)
    path = tmp_path / "policy.npz"
    np.savez(path, meta=np.array(json.dumps(policy_meta(cfg, 0))), num_layers=np.array(3),
             obs_mean=np.zeros(num_obs, np.float32), obs_std=np.ones(num_obs, np.float32), **arrays)
    return path


def test_config_tables(cfg):
    robot = RobotSpec.from_config(cfg)
    assert robot.num_joints == 19 and robot.num_actions == 10
    assert robot.kp[robot.joint_names.index("left_knee_joint")] == 200.0
    assert robot.torque_limit[robot.joint_names.index("left_shoulder_yaw_joint")] == 18.0
    assert robot.default_q[robot.joint_names.index("right_hip_pitch_joint")] == pytest.approx(-0.1)
    assert per_joint({"a": 1, "": 2}, ["xa", "y"]).tolist() == [1, 2]


def test_model_matches_config(cfg):
    m = build_model(cfg, visual=False)
    robot = RobotSpec.from_config(cfg)
    assert m.nu == 19 and m.opt.timestep == pytest.approx(robot.timestep)
    np.testing.assert_allclose(m.actuator_gainprm[:, 0], robot.kp)
    np.testing.assert_allclose(-m.actuator_biasprm[:, 2], robot.kd)
    np.testing.assert_allclose(m.actuator_forcerange[:, 1], robot.torque_limit)
    # visual and training models have identical dynamics
    mv = build_model(cfg, visual=True)
    np.testing.assert_allclose(mv.body_mass, m.body_mass)
    np.testing.assert_allclose(mv.body_inertia, m.body_inertia)


def test_standing_pose_touches_ground(cfg):
    m = build_model(cfg, visual=False)
    d = mujoco.MjData(m)
    robot = RobotSpec.from_config(cfg)
    d.qpos[:] = standing_qpos(m, robot)
    d.ctrl[:] = robot.default_q
    assert 0.95 < d.qpos[2] < 1.1
    mujoco.mj_step(m, d, 20)  # settle for 0.1 s
    assert d.ncon >= 6  # both feet on the floor
    assert abs(d.qpos[2] - 1.03) < 0.03


def test_projected_gravity():
    upright = projected_gravity(np.array([[1.0, 0, 0, 0]]))[0]
    np.testing.assert_allclose(upright, [0, 0, -1])
    th = 0.3  # pitch forward: gravity gets a +x component in the body frame
    g = projected_gravity(np.array([[np.cos(th / 2), 0, np.sin(th / 2), 0]]))[0]
    np.testing.assert_allclose(g, [np.sin(th), 0, -np.cos(th)], atol=1e-12)


def test_controller_reproduces_training_observations(cfg, tmp_path):
    env = H1WalkEnv(cfg, num_envs=1, num_threads=1, seed=3, randomize=False, obs_noise=False, pushes=False)
    ctrl = PolicyController(str(random_policy_file(tmp_path, cfg, env.num_obs, env.num_actions)))
    obs, _ = env.reset_all()
    env.cmd_target[0] = env.commands[0] = [0.5, -0.2, 0.3]
    obs = env._compute_obs(np.array([0]))[0]
    ctrl.command = env.commands[0].copy()   # both start from the same effective command
    for k in range(150):
        target = env.cmd_target[0].copy()   # the env ramps towards a new target one step later
        if k == 60:  # stop: both must ramp down and freeze the gait clock at the same tick
            env.cmd_target[0] = 0.0
        if k == 110:
            env.cmd_target[0] = [0.0, 0.3, 0.0]
        assert ctrl.phase == pytest.approx(float(env.phase[0]))  # same gait clock before the tick
        out = ctrl.step(env.q[0], env.dq[0], env.base_quat[0], env.gyro[0], target)
        np.testing.assert_allclose(ctrl.history.flat()[0], obs[0], atol=1e-5,
                                   err_msg=f"observation mismatch at control step {k}")
        obs, _, _, done, _ = env.step(ctrl.last_action[None])
        if done[0]:  # env was reset (the random policy fell): stop comparing
            break
        np.testing.assert_allclose(ctrl.command, env.commands[0], atol=1e-9)  # same command ramp
        np.testing.assert_allclose(out.position, env.targets[0], atol=1e-9)
    assert k > 20, "episode ended too early to be a useful comparison"
    env.close()


def test_gait_clock_stops_in_double_support():
    from h1_rl.obs import GaitClock, swing_shape
    clock = GaitClock(period=0.8, dt=0.02, num=1, stand_when_idle=True)
    for _ in range(27):              # walk into the middle of a step
        clock.advance([True])
    for _ in range(40):              # command goes to zero
        clock.advance([False])
    assert clock.ticks[0] in (0, 20)  # stopped at phase 0.0 or 0.5
    assert np.all(swing_shape(clock.phase, 0.5, 0.55) == 0.0)  # both feet on the ground
    assert clock.standing([False])[0]
    clock.advance([True])
    assert not clock.standing([True])[0]


def test_sim_matches_training_physics(cfg):
    """H1Sim (ROS simulator) and the training env produce the same trajectory."""
    env = H1WalkEnv(cfg, num_envs=1, num_threads=1, seed=0, randomize=False, obs_noise=False, pushes=False)
    sim = H1Sim(cfg, visual=True)
    sim.reset(support=False)
    env.datas[0].qpos[:] = sim.data.qpos
    env.datas[0].qvel[:] = 0.0
    mujoco.mj_forward(env.models[0], env.datas[0])
    rng = np.random.default_rng(0)
    for _ in range(25):
        a = rng.uniform(-0.5, 0.5, size=(1, 10))
        env.step(a)
        sim.set_command(env.targets[0], kp=sim.robot.kp, kd=sim.robot.kd)
        sim.step(sim.robot.decimation)
    np.testing.assert_allclose(sim.data.qpos, env.datas[0].qpos, atol=1e-8)
    env.close()


def test_export_roundtrip(cfg, tmp_path):
    torch = pytest.importorskip("torch")
    from h1_rl.export import export_checkpoint
    from h1_rl.policy import Policy
    from h1_rl.ppo import ActorCritic, EmpiricalNormalization

    ac = ActorCritic(205, 51, 10, [64, 32], [64, 32], "elu", 0.8)
    torch.nn.init.normal_(ac.actor[-1].weight, std=0.3)
    norm = EmpiricalNormalization(205)
    norm.update(torch.randn(500, 205) * 3 + 1)
    ck = tmp_path / "model.pt"
    torch.save({"iteration": 1, "policy": ac.state_dict(), "obs_norm": norm.state_dict(),
                "critic_norm": norm.state_dict(), "cfg": cfg, "norm_eps": norm.eps}, ck)
    out = export_checkpoint(ck, tmp_path / "p.npz")
    pol = Policy(out)
    x = np.random.default_rng(1).standard_normal((7, 205)) * 3
    norm.eval()
    with torch.no_grad():
        ref = ac.actor(norm(torch.as_tensor(x, dtype=torch.float32))).numpy()
    np.testing.assert_allclose(pol(x), ref, atol=1e-4)


def test_shipped_policy_walks(cfg):
    """The pretrained policy keeps the robot up and follows a forward command."""
    from h1_rl.config import resolve_path
    try:
        path = resolve_path("policies/h1_walk.npz")
    except FileNotFoundError:
        pytest.skip("no pretrained policy in policies/")
    sim = H1Sim(cfg, visual=False)
    sim.reset(support=False)
    ctrl = PolicyController(str(path))
    vx = []
    for k in range(int(8.0 / ctrl.dt)):
        cmd = [0.0, 0.0, 0.0] if k < 100 else [0.6, 0.0, 0.0]
        q, dq, _ = sim.joint_state()
        quat, gyro, _ = sim.imu()
        out = ctrl.step(q, dq, quat, gyro, cmd)
        sim.set_command(out.position, out.velocity, out.kp, out.kd, out.effort)
        sim.step(sim.robot.decimation)
        assert not sim.fallen(), f"robot fell at t={k * ctrl.dt:.2f}s"
        if k > 250:
            vx.append(sim.base_twist_body()[0][0])
    assert np.mean(vx) > 0.3, f"forward speed {np.mean(vx):.2f} m/s for a 0.6 m/s command"


def test_sim_clock_is_exact_and_monotonic(cfg):
    """The published /clock must never jump back (a float rounding bug once made it jump -1 s)."""
    sim = H1Sim(cfg, visual=False)
    sim.reset(support=True)
    prev = sim.time_ns
    for k in range(1, 1001):
        sim.step()
        assert sim.time_ns == prev + 5_000_000
        prev = sim.time_ns
        if k == 500:
            sim.reset(support=True)  # resets the robot, not the clock
    assert sim.time_ns == 1000 * 5_000_000
