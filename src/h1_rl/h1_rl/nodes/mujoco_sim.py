"""ROS 2 node: real-time MuJoCo simulation of the Unitree H1.

Publishes
  /joint_states    sensor_msgs/JointState   19 joints: position, velocity, effort
  /imu/data        sensor_msgs/Imu          pelvis IMU: orientation, gyro, accelerometer
  /odom            nav_msgs/Odometry        ground-truth pelvis pose + body-frame twist
  /tf              odom -> pelvis
  /clock           rosgraph_msgs/Clock      simulation time (run other nodes with use_sim_time:=true)
Subscribes
  /joint_commands  h1_msgs/JointCommand     per-joint impedance command (q, dq, tau, kp, kd)
Services
  ~/reset          std_srvs/Trigger         put the robot back on its feet
  ~/set_support    std_srvs/SetBool         hold (true) / release (false) the virtual support band

Behaviour
  * After start and after every reset the pelvis hangs in a virtual support band
    and the joints hold the default pose. The band is released as soon as the
    first /joint_commands message arrives (i.e. when a controller takes over).
  * Motor watchdog: if commands stop for `command_timeout` seconds (sim time),
    all joints switch to damping mode (kp=0), like the real H1 does.
  * If the robot lies on the ground for `auto_reset_delay` seconds it is reset.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState
from std_srvs.srv import SetBool, Trigger
from tf2_ros import TransformBroadcaster

from h1_msgs.msg import JointCommand

from ..config import load_config

try:
    from ..sim import H1Sim
except ImportError as exc:  # most common setup mistake: built without the venv
    raise SystemExit(
        f"[mujoco_sim] {exc}.\nThe node runs with the Python that built the workspace. Activate the venv "
        "and rebuild:\n  source ~/h1_venv/bin/activate && cd ~/h1_rl_ws && rm -rf build install && "
        "python -m colcon build --symlink-install") from exc

KEY_END = 269  # reset from the viewer window


def stamp(t_ns: int) -> Time:
    """Integer nanoseconds -> builtin_interfaces/Time (exact, so /clock never jumps back)."""
    return Time(sec=t_ns // 1_000_000_000, nanosec=t_ns % 1_000_000_000)


class MujocoSimNode(Node):
    def __init__(self) -> None:
        super().__init__("mujoco_sim")
        number = ParameterDescriptor(dynamic_typing=True)  # accept 1 as well as 1.0
        self.declare_parameter("config", "")
        self.declare_parameter("viewer", True)
        self.declare_parameter("realtime_factor", 1.0, number)
        self.declare_parameter("publish_rate", 200.0, number)
        self.declare_parameter("odom_rate", 50.0, number)
        self.declare_parameter("publish_clock", True)
        self.declare_parameter("command_timeout", 0.25, number)
        self.declare_parameter("damping_kd", 5.0, number)
        self.declare_parameter("auto_reset", True)
        self.declare_parameter("auto_reset_delay", 2.0, number)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "pelvis")

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        cfg = load_config(p("config") or None)
        self.sim = H1Sim(cfg, visual=True)
        self.joint_names = list(self.sim.robot.joint_names)
        self.joint_index = {n: i for i, n in enumerate(self.joint_names)}
        self.viewer_enabled = bool(p("viewer"))
        self.rtf = max(1e-3, float(p("realtime_factor")))
        dt = self.sim.dt
        self.pub_every = max(1, int(round(1.0 / (float(p("publish_rate")) * dt))))
        self.odom_every = max(1, int(round(1.0 / (float(p("odom_rate")) * dt))))
        self.publish_clock = bool(p("publish_clock"))
        self.command_timeout = float(p("command_timeout"))
        self.damping_kd = float(p("damping_kd"))
        self.auto_reset = bool(p("auto_reset"))
        self.auto_reset_delay = float(p("auto_reset_delay"))
        self.odom_frame = str(p("odom_frame"))
        self.base_frame = str(p("base_frame"))

        self.lock = threading.Lock()
        self.pending_cmd = None          # latest command, applied by the physics loop
        self.last_cmd_time = None        # sim time of the last command since reset
        self.watchdog_tripped = False
        self.reset_requested = False
        self.support_request = None
        self.fallen_since = None
        self._warned = set()

        self.pub_joints = self.create_publisher(JointState, "joint_states", 10)
        self.pub_imu = self.create_publisher(Imu, "imu/data", 10)
        self.pub_odom = self.create_publisher(Odometry, "odom", 10)
        self.pub_clock = self.create_publisher(Clock, "/clock", 10) if self.publish_clock else None
        self.tf = TransformBroadcaster(self)
        self.create_subscription(JointCommand, "joint_commands", self._on_command, 10)
        self.create_service(Trigger, "~/reset", self._on_reset)
        self.create_service(SetBool, "~/set_support", self._on_set_support)

        self._joint_msg = JointState()
        self._joint_msg.name = self.joint_names
        self.get_logger().info(
            f"H1 MuJoCo sim ready: dt={dt * 1000:.1f} ms, realtime factor {self.rtf}, viewer={self.viewer_enabled}. "
            "Robot is held by the support band until the first /joint_commands message.")

    # ------------------------------------------------------------ callbacks
    def _on_command(self, msg: JointCommand) -> None:
        n = len(msg.name)
        if n == 0 or len(msg.position) != n:
            self._warn_once("bad_len", "JointCommand needs one position per name; ignoring message")
            return
        idx = []
        for name in msg.name:
            i = self.joint_index.get(name)
            if i is None:
                self._warn_once("unknown_" + name, f"unknown joint '{name}' in JointCommand (ignored)")
            idx.append(i)

        def field(values, default):
            return np.asarray(values, dtype=np.float64) if len(values) == n else np.full(n, default)

        pos, vel, eff = field(msg.position, 0.0), field(msg.velocity, 0.0), field(msg.effort, 0.0)
        kp = field(msg.kp, np.nan)
        kd = field(msg.kd, np.nan)
        with self.lock:
            s = self.sim
            q_des, dq_des, tau = s.q_des.copy(), np.zeros_like(s.q_des), np.zeros_like(s.q_des)
            kp_all, kd_all = s.kp.copy(), s.kd.copy()
            for j, i in enumerate(idx):
                if i is None:
                    continue
                q_des[i], dq_des[i], tau[i] = pos[j], vel[j], eff[j]
                if not np.isnan(kp[j]):
                    kp_all[i] = kp[j]
                if not np.isnan(kd[j]):
                    kd_all[i] = kd[j]
            self.pending_cmd = (q_des, dq_des, kp_all, kd_all, tau)

    def _on_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        with self.lock:
            self.reset_requested = True
        response.success = True
        response.message = "robot will be reset"
        return response

    def _on_set_support(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        with self.lock:
            self.support_request = bool(request.data)
        response.success = True
        response.message = "support band " + ("on" if request.data else "off")
        return response

    def _on_key(self, key: int) -> None:
        if key == KEY_END:
            with self.lock:
                self.reset_requested = True

    def _warn_once(self, key: str, text: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self.get_logger().warning(text)

    # ---------------------------------------------------------- physics loop
    def _update(self) -> None:
        """Apply requests from the ROS callbacks, then advance the physics by one step."""
        s = self.sim
        log = self.get_logger()
        with self.lock:
            if self.reset_requested:
                s.reset(support=True)
                self.reset_requested = False
                self.pending_cmd = None
                self.last_cmd_time = None
                self.watchdog_tripped = False
                self.fallen_since = None
                log.info("robot reset (held by the support band until the next command)")
            if self.support_request is not None:
                s.support = self.support_request
                self.support_request = None
            if self.pending_cmd is not None:
                s.set_command(*self.pending_cmd)  # (q_des, dq_des, kp, kd, tau_ff)
                self.pending_cmd = None
                if s.support:
                    s.support = False
                    log.info("controller connected: support band released")
                if self.watchdog_tripped:
                    log.info("joint commands resumed")
                self.last_cmd_time = s.time
                self.watchdog_tripped = False
            if (self.last_cmd_time is not None and not self.watchdog_tripped
                    and s.time - self.last_cmd_time > self.command_timeout):
                self.watchdog_tripped = True
                q, _, _ = s.joint_state()
                s.set_command(q, kp=np.zeros(len(q)), kd=np.full(len(q), self.damping_kd))
                log.warning("no joint commands: motors switched to damping mode")
            s.step()
            if self.auto_reset:
                if not s.fallen(0.55):
                    self.fallen_since = None
                elif self.fallen_since is None:
                    self.fallen_since = s.time
                elif s.time - self.fallen_since > self.auto_reset_delay:
                    self.reset_requested = True

    def run(self) -> None:
        viewer = None
        if self.viewer_enabled:
            try:
                import mujoco.viewer
                viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data, key_callback=self._on_key)
                viewer.cam.type = 1  # track the robot
                viewer.cam.trackbodyid = self.sim.pelvis
                viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 3.5, -15.0, 135.0
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"could not open the MuJoCo viewer ({exc}); running headless")
                viewer = None
        dt = self.sim.dt
        step = 0
        next_wall = time.perf_counter()
        last_sync = 0.0
        try:
            while rclpy.ok():
                if viewer is not None:
                    if not viewer.is_running():
                        break
                    with viewer.lock():
                        self._update()
                else:
                    self._update()
                step += 1
                t_ns = self.sim.time_ns
                if self.pub_clock is not None:
                    self.pub_clock.publish(Clock(clock=stamp(t_ns)))
                if step % self.pub_every == 0:
                    self._publish_state(t_ns)
                if step % self.odom_every == 0:
                    self._publish_odom(t_ns)
                now = time.perf_counter()
                if viewer is not None and now - last_sync > 1.0 / 60.0:
                    viewer.sync()
                    last_sync = now
                next_wall += dt / self.rtf
                sleep = next_wall - time.perf_counter()
                if sleep > 0.0:
                    time.sleep(sleep)
                elif sleep < -0.25:  # cannot keep up: run as fast as possible without catching up
                    next_wall = time.perf_counter()
        finally:
            if viewer is not None:
                viewer.close()

    # ------------------------------------------------------------ publishing
    def _publish_state(self, t_ns: int) -> None:
        s = self.sim
        q, dq, tau = s.joint_state()
        header_stamp = stamp(t_ns)
        js = self._joint_msg
        js.header.stamp = header_stamp
        js.position = q.tolist()
        js.velocity = dq.tolist()
        js.effort = tau.tolist()
        self.pub_joints.publish(js)

        quat, gyro, acc = s.imu()
        imu = Imu()
        imu.header.stamp = header_stamp
        imu.header.frame_id = self.base_frame
        imu.orientation.w, imu.orientation.x = float(quat[0]), float(quat[1])
        imu.orientation.y, imu.orientation.z = float(quat[2]), float(quat[3])
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = (float(v) for v in gyro)
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = (float(v) for v in acc)
        imu.orientation_covariance = [1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 0.0, 1e-6]
        imu.angular_velocity_covariance = [1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 0.0, 1e-6]
        imu.linear_acceleration_covariance = [1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 1e-4]
        self.pub_imu.publish(imu)

    def _publish_odom(self, t_ns: int) -> None:
        s = self.sim
        pos, quat = s.base_pose()
        lin, ang = s.base_twist_body()
        header_stamp = stamp(t_ns)
        odom = Odometry()
        odom.header.stamp = header_stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        pp, po = odom.pose.pose.position, odom.pose.pose.orientation
        pp.x, pp.y, pp.z = (float(v) for v in pos)
        po.w, po.x, po.y, po.z = (float(v) for v in quat)
        tl, ta = odom.twist.twist.linear, odom.twist.twist.angular
        tl.x, tl.y, tl.z = (float(v) for v in lin)
        ta.x, ta.y, ta.z = (float(v) for v in ang)
        self.pub_odom.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = header_stamp
        tf.header.frame_id = self.odom_frame
        tf.child_frame_id = self.base_frame
        tr, rot = tf.transform.translation, tf.transform.rotation
        tr.x, tr.y, tr.z = (float(v) for v in pos)
        rot.w, rot.x, rot.y, rot.z = (float(v) for v in quat)
        self.tf.sendTransform(tf)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MujocoSimNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def spin() -> None:
        try:
            executor.spin()
        except (ExternalShutdownException, KeyboardInterrupt):
            pass

    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    try:
        node.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
