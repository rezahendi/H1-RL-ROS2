"""H1 walking demo: MuJoCo simulator + RL policy controller (+ optional RViz / scripted demo).

    ros2 launch h1_rl walk.launch.py                      # viewer window, drive with teleop
    ros2 launch h1_rl walk.launch.py demo:=true           # scripted /cmd_vel sequence
    ros2 launch h1_rl walk.launch.py rviz:=true viewer:=false
    ros2 launch h1_rl walk.launch.py policy:=/abs/path/to/policy_latest.npz

Drive it from a second terminal:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("h1_rl")
    with open(os.path.join(share, "models", "h1", "h1.urdf"), "r", encoding="utf-8") as f:
        robot_description = f.read()

    args = [
        DeclareLaunchArgument("viewer", default_value="true", description="open the MuJoCo viewer window"),
        DeclareLaunchArgument("rviz", default_value="false", description="start RViz with the robot model"),
        DeclareLaunchArgument("demo", default_value="false", description="publish a scripted /cmd_vel sequence"),
        DeclareLaunchArgument("policy", default_value=os.path.join(share, "policies", "h1_walk.npz"),
                              description="exported policy (.npz)"),
        DeclareLaunchArgument("config", default_value=os.path.join(share, "config", "h1_walk.yaml"),
                              description="robot/sim config used by the simulator"),
        DeclareLaunchArgument("realtime_factor", default_value="1.0",
                              description="simulation speed (lower it on slow machines)"),
    ]

    sim = Node(
        package="h1_rl", executable="mujoco_sim", name="mujoco_sim", output="screen",
        parameters=[{
            "config": LaunchConfiguration("config"),
            "viewer": LaunchConfiguration("viewer"),
            "realtime_factor": LaunchConfiguration("realtime_factor"),
        }],
    )
    controller = Node(
        package="h1_rl", executable="policy_controller", name="policy_controller", output="screen",
        parameters=[{"policy": LaunchConfiguration("policy"), "use_sim_time": True}],
    )
    demo = Node(
        package="h1_rl", executable="cmd_vel_demo", name="cmd_vel_demo", output="screen",
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("demo")),
    )
    state_publisher = Node(
        package="robot_state_publisher", executable="robot_state_publisher", output="log",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )
    rviz = Node(
        package="rviz2", executable="rviz2", output="log",
        arguments=["-d", os.path.join(share, "config", "h1.rviz")],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )
    return LaunchDescription(args + [sim, controller, demo, state_publisher, rviz])
