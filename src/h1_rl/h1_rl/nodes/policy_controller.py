"""ROS 2 node: runs the trained walking policy on the H1.

Subscribes
  /joint_states    sensor_msgs/JointState   joint angles + velocities (matched by name)
  /imu/data        sensor_msgs/Imu          pelvis orientation + angular velocity
  /cmd_vel         geometry_msgs/Twist      linear.x, linear.y [m/s], angular.z [rad/s]
                                            (kept until the next message; set cmd_vel_timeout > 0
                                            to fall back to zero after that many seconds)
Publishes
  /joint_commands  h1_msgs/JointCommand     PD targets + gains for all 19 joints (50 Hz)
Services
  ~/enable         std_srvs/SetBool         start/stop sending commands

Nothing is published before both /joint_states and /imu/data have arrived.
If the robot tips over, the node stops commanding (the motors go to damping)
and resumes automatically once the robot stands upright again (e.g. after the
simulator resets it). With use_sim_time:=true the 50 Hz loop runs on /clock.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState
from std_srvs.srv import SetBool

from h1_msgs.msg import JointCommand

from ..config import resolve_path
from ..controller import FALLEN, PolicyController


class PolicyControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_controller")
        number = ParameterDescriptor(dynamic_typing=True)  # accept 1 as well as 1.0
        self.declare_parameter("policy", "policies/h1_walk.npz")
        self.declare_parameter("cmd_vel_timeout", 0.0, number)  # 0 = keep the last command
        self.declare_parameter("autostart", True)
        self.declare_parameter("max_state_age", 0.1, number)

        policy_path = resolve_path(self.get_parameter("policy").value)
        self.ctrl = PolicyController(str(policy_path))
        self.cmd_timeout = float(self.get_parameter("cmd_vel_timeout").value)
        self.max_state_age = float(self.get_parameter("max_state_age").value)
        self.enabled = bool(self.get_parameter("autostart").value)
        self.joint_names = self.ctrl.joint_names
        n = len(self.joint_names)

        self.q = np.zeros(n)
        self.dq = np.zeros(n)
        self.have_joints = False
        self.quat = None
        self.gyro = np.zeros(3)
        self.cmd = np.zeros(3)
        self.cmd_stamp = None
        self.state_stamp = None
        self._index_cache: tuple[tuple[str, ...], np.ndarray] | None = None
        self._was_fallen = False

        self.create_subscription(JointState, "joint_states", self._on_joints, qos_profile_sensor_data)
        self.create_subscription(Imu, "imu/data", self._on_imu, qos_profile_sensor_data)
        self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, 10)
        self.pub = self.create_publisher(JointCommand, "joint_commands", 10)
        self.create_service(SetBool, "~/enable", self._on_enable)
        self.timer = self.create_timer(self.ctrl.dt, self._tick)

        m = self.ctrl.meta
        self.get_logger().info(
            f"policy {policy_path.name} (iteration {m.get('iteration')}) | {self.ctrl.dt * 1000:.0f} ms control period | "
            f"cmd ranges vx {m['commands']['lin_vel_x']} vy {m['commands']['lin_vel_y']} "
            f"yaw {m['commands']['ang_vel_yaw']}")

    # ------------------------------------------------------------ callbacks
    def _on_joints(self, msg: JointState) -> None:
        names = tuple(msg.name)
        if self._index_cache is None or self._index_cache[0] != names:
            lookup = {name: i for i, name in enumerate(names)}
            missing = [j for j in self.joint_names if j not in lookup]
            if missing:
                self.get_logger().error(f"/joint_states is missing joints {missing}", throttle_duration_sec=5.0)
                return
            self._index_cache = (names, np.array([lookup[j] for j in self.joint_names]))
        src = self._index_cache[1]
        pos = np.asarray(msg.position, dtype=np.float64)
        vel = np.asarray(msg.velocity, dtype=np.float64)
        if len(pos) != len(names) or len(vel) != len(names):
            self.get_logger().error("/joint_states needs position and velocity for every joint",
                                    throttle_duration_sec=5.0)
            return
        self.q = pos[src]
        self.dq = vel[src]
        self.have_joints = True
        self.state_stamp = self.get_clock().now()

    def _on_imu(self, msg: Imu) -> None:
        o = msg.orientation
        self.quat = np.array([o.w, o.x, o.y, o.z])
        w = msg.angular_velocity
        self.gyro = np.array([w.x, w.y, w.z])

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self.cmd_stamp = self.get_clock().now()

    def _on_enable(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        self.enabled = bool(request.data)
        if self.enabled:
            self.ctrl.reset()
        response.success = True
        response.message = "policy " + ("enabled" if self.enabled else "disabled (no commands sent)")
        self.get_logger().info(response.message)
        return response

    # --------------------------------------------------------------- control
    def _tick(self) -> None:
        if not self.enabled:
            return
        if not self.have_joints or self.quat is None:
            self.get_logger().info("waiting for /joint_states and /imu/data ...", throttle_duration_sec=5.0)
            return
        now = self.get_clock().now()
        age = (now - self.state_stamp).nanoseconds * 1e-9
        if age > self.max_state_age:
            self.get_logger().warning(f"joint states are {age:.2f} s old: not sending commands",
                                      throttle_duration_sec=2.0)
            return
        cmd = self.cmd
        if self.cmd_stamp is None or (
                self.cmd_timeout > 0.0 and (now - self.cmd_stamp).nanoseconds * 1e-9 > self.cmd_timeout):
            cmd = np.zeros(3)

        out = self.ctrl.step(self.q, self.dq, self.quat, self.gyro, cmd)
        if self.ctrl.state == FALLEN:
            if not self._was_fallen:
                self.get_logger().warning("robot fell over: stopping commands until it is upright again")
            self._was_fallen = True
            return
        if self._was_fallen:
            self.get_logger().info("robot is upright again: policy restarted")
            self._was_fallen = False

        msg = JointCommand()
        msg.header.stamp = now.to_msg()
        msg.name = self.joint_names
        msg.position = out.position.tolist()
        msg.velocity = out.velocity.tolist()
        msg.effort = out.effort.tolist()
        msg.kp = out.kp.tolist()
        msg.kd = out.kd.tolist()
        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
