"""MuJoCo simulator only - bring your own controller on /joint_commands.

    ros2 launch h1_rl sim.launch.py viewer:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("h1_rl")
    return LaunchDescription([
        DeclareLaunchArgument("viewer", default_value="true"),
        DeclareLaunchArgument("realtime_factor", default_value="1.0"),
        DeclareLaunchArgument("config", default_value=os.path.join(share, "config", "h1_walk.yaml")),
        Node(
            package="h1_rl", executable="mujoco_sim", name="mujoco_sim", output="screen",
            parameters=[{
                "config": LaunchConfiguration("config"),
                "viewer": LaunchConfiguration("viewer"),
                "realtime_factor": LaunchConfiguration("realtime_factor"),
            }],
        ),
    ])
