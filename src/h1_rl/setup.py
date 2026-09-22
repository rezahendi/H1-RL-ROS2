from glob import glob

from setuptools import find_packages, setup

package_name = "h1_rl"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml") + glob("config/*.rviz")),
        ("share/" + package_name + "/models/h1",
         glob("models/h1/*.xml") + glob("models/h1/*.urdf") + glob("models/h1/LICENSE*")),
        ("share/" + package_name + "/models/h1/meshes", glob("models/h1/meshes/*.stl")),
        ("share/" + package_name + "/policies", glob("policies/*.npz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Reza",
    maintainer_email="rezahendi590@gmail.com",
    description="RL walking controller for the Unitree H1 (MuJoCo + ROS 2)",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "mujoco_sim = h1_rl.nodes.mujoco_sim:main",
            "policy_controller = h1_rl.nodes.policy_controller:main",
            "cmd_vel_demo = h1_rl.nodes.cmd_vel_demo:main",
            "train = h1_rl.train:main",
            "export = h1_rl.export:main",
            "play = h1_rl.play:main",
        ],
    },
)
