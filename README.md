<h1 align="center">Unitree H1 — reinforcement-learning walking controller</h1>

<p align="center">
  PPO in <b>MuJoCo</b> → a <b>ROS 2</b> node that walks the H1 humanoid on <code>/cmd_vel</code>.<br/>
  Trained from scratch on 2 CPU cores. No Isaac Gym, no GPU required.
</p>

<p align="center">
  <img alt="ROS 2 Jazzy" src="https://img.shields.io/badge/ROS%202-Jazzy-22314E?logo=ros&logoColor=white">
  <img alt="MuJoCo 3.13" src="https://img.shields.io/badge/MuJoCo-3.13-ff6f00">
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="License BSD-3-Clause" src="https://img.shields.io/badge/license-BSD--3--Clause-blue">
  <a href="../../actions/workflows/ci.yml"><img alt="CI" src="../../actions/workflows/ci.yml/badge.svg"></a>
</p>

<p align="center">
  <img src="docs/demo.gif" width="620" alt="H1 walking, stopping and turning on velocity commands in MuJoCo">
</p>

<p align="center"><sub>
  Real time, no cuts. <code>cmd</code> is the velocity command, <code>meas</code> the measured base velocity.
  <a href="docs/demo.mp4">Full clip (18 s, includes walk + turn and sideways stepping)</a>.
</sub></p>

---

The policy takes only signals a real H1 actually has — IMU and joint encoders — and outputs
joint targets for the 10 leg joints at 50 Hz. At zero command it **finishes the step it is
taking, puts both feet down and stands still**, balancing actively; the next command starts
it walking again.

```
       TRAINING  (no ROS, fast)                                 RUNTIME  (ROS 2 Jazzy)                  

+------------------------------------+        +----------------------------------------------------------+
| h1_rl.train                        |        | teleop_twist_keyboard  /  your nav stack                 |
|   128 MuJoCo H1s in parallel (CPU) | policy |             | /cmd_vel                                   |
|   PPO, asymmetric actor-critic     |------->|             v                                            |
|   velocity tracking + gait rewards |  .npz  | policy_controller           NumPy MLP, 50 Hz             |
|   domain randomization, pushes,    |        |   ^ /joint_states  ^ /imu/data   | /joint_commands       |
|   actuation delay, sensor noise    |        |   |                |             v  (q, dq, tau, kp, kd) |
+------------------------------------+        | mujoco_sim    H1 physics 200 Hz, viewer, /clock,         |
                                              |               /odom, /tf, motor watchdog, auto-reset     |
                                              +----------------------------------------------------------+
```

## Highlights

- **Walks and stands still.** Velocity tracking in all three axes plus a true standstill at zero
  command — the part most H1 demos skip, because the H1 topples if you just freeze its joints.
- **Deployment-shaped interface.** `h1_msgs/JointCommand` carries exactly what Unitree's `LowCmd`
  carries (position, velocity, torque, kp, kd), so the controller talks to the simulator the same
  way it would talk to the robot.
- **One config, no drift.** `config/h1_walk.yaml` is the single source of truth for training, the
  simulator and the controller; the exported `.npz` policy embeds the settings it was trained
  with, and a test asserts the deployed controller sees *exactly* the observations training
  produced (down to 1e-5) through stand ↔ walk transitions.
- **Trains on a laptop.** ~40 M steps ≈ 2 hours on 2 CPU cores. Physics runs multi-threaded in
  MuJoCo, the GPU (if any) only does the network updates.
- **Tested.** 10 pytest tests, run in CI: controller/env observation equality, sim/training
  physics equality, exact monotonic `/clock`, policy export round-trip, and a walk test with the
  shipped policy.

## Quickstart

Requires **ROS 2 Jazzy on Ubuntu 24.04** (WSL2 works). Full install, including WSL and the
venv details: **[docs/SETUP.md](docs/SETUP.md)**.

```bash
git clone https://github.com/rezahendi/h1-rl-ros2.git ~/h1_rl_ws && cd ~/h1_rl_ws
python3 -m venv --system-site-packages ~/h1_venv && source ~/h1_venv/bin/activate
pip install -r requirements.txt                  # numpy<2, mujoco, pyyaml
source /opt/ros/jazzy/setup.bash
python -m colcon build --symlink-install         # use "python -m colcon", venv active
```

Run the pretrained policy (a MuJoCo window opens):

```bash
source ~/h1_rl_ws/env.sh                         # venv + ROS 2 + this workspace
ros2 launch h1_rl walk.launch.py
```

Drive it from a second terminal:

```bash
source ~/h1_rl_ws/env.sh
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

`i` forward · `,` backward · `j`/`l` turn · `J`/`L` sideways · `k` stop (→ stands still).
Trained range: vx ∈ [−0.6, 1.0] m/s, vy ∈ [±0.5] m/s, yaw ∈ [±1.0] rad/s; larger commands are
clipped, and all commands are acceleration limited (2 m/s², 4 rad/s²) so the robot has a step to
speed up or brake.

<details>
<summary>Other ways to run it</summary>

```bash
ros2 launch h1_rl walk.launch.py demo:=true            # scripted walk, no keyboard needed
ros2 launch h1_rl walk.launch.py rviz:=true            # RViz (robot model + odometry)
ros2 launch h1_rl walk.launch.py realtime_factor:=0.5  # slow motion for slow machines
ros2 launch h1_rl walk.launch.py policy:=/path/to/policy_latest.npz

ros2 topic hz /joint_commands                             # 50 Hz from the policy
ros2 topic echo /odom --field twist.twist.linear          # actual body velocity
ros2 service call /mujoco_sim/reset std_srvs/srv/Trigger  # put the robot back on its feet
```

Without ROS at all (MuJoCo viewer, arrow keys drive the robot):

```bash
cd src/h1_rl && python -m h1_rl.play
```

At start-up the simulator holds the pelvis in a virtual **support band** (like Unitree's elastic
band) and releases it on the first `/joint_commands` message. If the robot falls, the controller
switches to damping, the simulator auto-resets it after 2 s and the policy restarts once it is
upright.
</details>

## Results

Scripted test with the shipped policy (`python -m h1_rl.play --headless`):

| Command | Achieved | | Robustness test | Result |
|---|---|---|---|---|
| forward 0.5 m/s | 0.47 m/s | | stop from 0.3–1.0 m/s × 12 gait phases (48 runs) | **0 falls** |
| forward 1.0 m/s | 0.87 m/s | | stand 12 s, 192 robots, sensor noise | **1 fall** |
| backward 0.5 m/s | 0.39 m/s | | stand 12 s, 192 robots, randomized dynamics | **0 falls** |
| sideways ±0.4 m/s | ±0.29 m/s | | 0.8 m/s shove while walking (12 runs) | **12 survived** |
| turn 0.8 rad/s | 0.71 rad/s | | 0.4–0.8 m/s shove while standing | ~1 in 3 survives |
| zero | motionless, both feet down | | +5 ms actuation delay | no falls |

Mean tracking error over the 11 segments of the script (each averaged after the command ramp):
**0.041 m/s** (vx), **0.035 m/s** (vy), **0.040 rad/s** (yaw), no falls. The residual gap at the
edges of the trained range (1.0 m/s forward) is the usual under-tracking of a velocity-tracking
reward; segments inside the range track within a few cm/s.

<p align="center"><img src="docs/training.png" width="860" alt="PPO training curves"></p>

The shipped policy is 9800 PPO iterations (~40 M environment steps, ~2 h on 2 CPU cores). The
drops are deliberate: each stage adds a reward term or resets exploration noise and the policy
re-converges. It walks after ~500 iterations and survives full 20 s episodes after ~1500.

## How it works

**Robot.** `models/h1/h1.xml` is Unitree's H1 (51 kg, 19 actuated joints) with simple collision
shapes (4 spheres per foot). Every joint is a **PD servo inside MuJoCo**
(`τ = kp·(q* − q) − kd·q̇`, clipped to the motor torque limits) — the same impedance interface
the real motor drivers expose. Gains are Unitree's RL gains (hip 150, knee 200, ankle 40, …).

**Policy.** 50 Hz, 3-layer MLP (256-128-64, ELU), 10 leg joints; torso and arms are held at their
default pose. The output is a *correction on top of a stepping reference*:

```
target = default pose + reference(gait phase) + 0.25 × action
```

The reference is a plain stepping-in-place motion (the swing leg bends hip/knee/ankle in a
half-sine, timed by a 0.8 s gait clock). The network learns balance, foot placement and velocity
tracking on top of it. Without the reference the robot first discovers standing still and needs
far more samples to find stepping.

**Standing still.** The gait clock is an exact integer tick counter. At zero command it keeps
running to the next double-support instant (both feet down) and stops there; the first non-zero
command restarts it. Combined with acceleration-limited commands, the robot always has a step
left to brake with. Penalizing motion while standing was tried and made it *worse* — the policy
needs those small ankle/hip corrections to stay upright.

**Observations** (41 values × last 5 frames): IMU angular velocity, gravity direction, command,
leg joint angles and velocities, previous action, gait clock (sin/cos). No ground-truth velocity
— the critic gets that as privileged information, the actor never does.

**Training.** PPO with an asymmetric actor-critic. Rewards follow Unitree's `unitree_rl_gym` H1
recipe (velocity tracking, contact-vs-gait-clock matching, swing-foot clearance, penalties on
tilt, action rate, torques, joint limits) plus a small reference-tracking term as in
Humanoid-Gym. Randomization: friction 0.4–1.25, pelvis mass −1…+3 kg, CoM ±2 cm, ±10 % PD gains,
0–10 ms actuation delay, random pushes, sensor noise.

### ROS 2 interface

| Node | Topic / service | Type | Notes |
|---|---|---|---|
| `mujoco_sim` | pub `/joint_states` | `sensor_msgs/JointState` | 200 Hz, all 19 joints |
| | pub `/imu/data` | `sensor_msgs/Imu` | pelvis IMU |
| | pub `/odom`, `/tf` | `nav_msgs/Odometry` | ground truth, `odom → pelvis` |
| | pub `/clock` | `rosgraph_msgs/Clock` | run other nodes with `use_sim_time:=true` |
| | sub `/joint_commands` | `h1_msgs/JointCommand` | per-joint q, dq, τ, kp, kd |
| | srv `~/reset`, `~/set_support` | `Trigger`, `SetBool` | |
| `policy_controller` | sub `/joint_states`, `/imu/data`, `/cmd_vel` | | holds the last command (`cmd_vel_timeout` > 0 to stop after N s) |
| | pub `/joint_commands` | `h1_msgs/JointCommand` | 50 Hz |
| | srv `~/enable` | `SetBool` | |

## Repository layout

| Path | What it is |
|---|---|
| `src/h1_rl/config/h1_walk.yaml` | **Single source of truth**: joints, PD gains, rewards, randomization, PPO |
| `src/h1_rl/h1_rl/envs/h1_walk.py` | Batched, multi-threaded MuJoCo training environment |
| `src/h1_rl/h1_rl/ppo.py`, `train.py` | PPO, asymmetric critic, TensorBoard logging, checkpoints |
| `src/h1_rl/h1_rl/obs.py` | Observation + gait-clock code shared by training and deployment |
| `src/h1_rl/h1_rl/controller.py` | The deployed controller (pure NumPy, no ROS, no PyTorch) |
| `src/h1_rl/h1_rl/nodes/` | ROS 2 nodes: `mujoco_sim`, `policy_controller`, `cmd_vel_demo` |
| `src/h1_rl/models/h1/` | H1 MuJoCo model, URDF for RViz, meshes |
| `src/h1_rl/policies/h1_walk.npz` | Pretrained policy used by default (weights + training settings) |
| `src/h1_msgs` | `JointCommand.msg`, the Unitree-style impedance command |
| `src/h1_rl/test/test_core.py` | The test suite |

## Train your own

```bash
source ~/h1_rl_ws/env.sh
pip install -r requirements-train.txt    # once: PyTorch, TensorBoard
cd src/h1_rl && python -m h1_rl.train     # physics on all CPU cores, PPO on the GPU if present
tensorboard --logdir logs                 # http://localhost:6006
```

```bash
python -m h1_rl.train --num-envs 256                       # more parallel robots on a bigger CPU
python -m h1_rl.train --set rewards.scales.torques=-1e-5   # override any config key
python -m h1_rl.train --resume logs/h1_walk/<run>/model_3000.pt --iterations 8000
python -m h1_rl.export --checkpoint logs/h1_walk/<run>/model_9800.pt --output policies/h1_walk.npz
python -m h1_rl.play --policy logs/h1_walk/<run>/policy_latest.npz --headless   # tracking report
```

Checkpoints are exported automatically every 100 iterations. Standing robustness varies between
nearby checkpoints, so evaluate a few late ones instead of taking the last.

**Tests** (no ROS needed): `pytest` from the repo root, or `cd src/h1_rl && python -m pytest test -q`.

## Roadmap

- **Recovery steps while standing** — let a hard shove restart the gait clock so the robot can
  step to catch itself instead of relying on ankles and hips alone.
- **Arms and torso** in the action space (currently held at their default pose).
- **Rough terrain** — height field in `scene.xml` plus a terrain curriculum.
- **Benchmark against Unitree's pretrained H1 policy** in this same simulator.
- **`unitree_ros2` bridge** — `h1_msgs/JointCommand` already mirrors `LowCmd`.

## Before you try this on a real H1

This policy is trained and validated **in simulation only**. Sim-to-real for a 51 kg humanoid is
a safety-critical step, not a config change. At minimum: motor system identification, sim-to-sim
validation against a second simulator, much wider domain randomization, a bridge with torque and
velocity limits, command ramping and a damping fallback, a real-time Linux host (not WSL), and
first tests on a gantry with a second person on the e-stop.

## Credits and license

H1 meshes and inertial parameters © Unitree Robotics, BSD-3-Clause
([`src/h1_rl/models/h1/LICENSE`](src/h1_rl/models/h1/LICENSE)), obtained via MuJoCo Menagerie /
LocoMuJoCo. Reward design follows Unitree's [`unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym);
the stepping-reference idea comes from [Humanoid-Gym](https://github.com/roboterax/humanoid-gym);
the PPO implementation follows [`rsl_rl`](https://github.com/leggedrobotics/rsl_rl) in spirit.

Everything in this repository is BSD-3-Clause — see [LICENSE](LICENSE).
