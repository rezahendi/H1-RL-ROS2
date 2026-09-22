"""ROS 2 node: publishes a scripted /cmd_vel sequence (hands-free demo / smoke test).

    ros2 run h1_rl cmd_vel_demo --ros-args -p use_sim_time:=true -p loop:=false
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from ..play import DEMO_SCRIPT


class CmdVelDemo(Node):
    def __init__(self) -> None:
        super().__init__("cmd_vel_demo")
        number = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("loop", True)
        self.declare_parameter("rate", 10.0, number)
        self.declare_parameter("start_delay", 2.0, number)
        self.loop = bool(self.get_parameter("loop").value)
        self.start_delay = float(self.get_parameter("start_delay").value)
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.t0 = None
        self.segment = -1
        self.create_timer(1.0 / float(self.get_parameter("rate").value), self._tick)

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now + self.start_delay
        t = now - self.t0
        if t < 0.0:
            return
        total = sum(seg[0] for seg in DEMO_SCRIPT)
        if t >= total:
            if not self.loop:
                self.pub.publish(Twist())
                if self.segment != len(DEMO_SCRIPT):
                    self.segment = len(DEMO_SCRIPT)
                    self.get_logger().info("demo finished")
                return
            t = t % total
        acc = 0.0
        for i, (duration, vx, vy, wz) in enumerate(DEMO_SCRIPT):
            acc += duration
            if t < acc:
                break
        if i != self.segment:
            self.segment = i
            self.get_logger().info(f"cmd_vel: vx={vx:+.2f} m/s  vy={vy:+.2f} m/s  yaw={wz:+.2f} rad/s")
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelDemo()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
