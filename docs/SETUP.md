# Setup

Full install for **Ubuntu 24.04 + ROS 2 Jazzy**, on WSL2 or on native Linux.
If you already have Jazzy, jump to [1.3](#13-clone-and-create-the-python-environment).

---

## 1.1 WSL2 with Ubuntu 24.04 (Windows only)

In **PowerShell (as administrator)**:

```powershell
wsl --install -d Ubuntu-24.04
wsl --update            # WSLg (GUI windows) + latest kernel
```

GPU: install the current **NVIDIA driver for Windows** (Game Ready / Studio). Do **not**
install an NVIDIA driver inside WSL — WSL2 uses the Windows one. Check inside Ubuntu:

```bash
nvidia-smi              # should list your GPU
```

A GPU is optional. It only speeds up the PPO network updates; the physics runs on the CPU.

## 1.2 ROS 2 Jazzy

**Already have Jazzy?** Only add the extras:

```bash
sudo apt install -y python3-venv python3-pip git ros-jazzy-teleop-twist-keyboard
```

Otherwise:

```bash
sudo apt update && sudo apt install -y software-properties-common curl
sudo add-apt-repository -y universe
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
sudo apt update && sudo apt upgrade -y
sudo apt install -y ros-jazzy-desktop ros-dev-tools ros-jazzy-teleop-twist-keyboard \
                    python3-venv python3-pip git
```

## 1.3 Clone and create the Python environment

On WSL, keep the workspace on the **Linux** filesystem (`~`), not on `/mnt/c` — builds are
several times faster there.

```bash
git clone https://github.com/rezahendi/h1-rl-ros2.git ~/h1_rl_ws
cd ~/h1_rl_ws

python3 -m venv --system-site-packages ~/h1_venv
source ~/h1_venv/bin/activate
pip install -r requirements.txt
python -c "import mujoco; print('mujoco', mujoco.__version__)"
```

For training, add PyTorch and friends (~3 GB with the CUDA libraries):

```bash
pip install -r requirements-train.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

> **Why a venv:** Ubuntu 24.04 blocks `pip install` into the system Python.
> `--system-site-packages` lets the venv also see ROS 2's Python packages.
> `numpy<2` is pinned because Jazzy's message libraries are compiled against NumPy 1.x.

## 1.4 Build

```bash
source ~/h1_venv/bin/activate
source /opt/ros/jazzy/setup.bash
cd ~/h1_rl_ws
python -m colcon build --symlink-install
```

Use `python -m colcon` (**not** plain `colcon`) with the venv active — that way the ROS nodes
are installed with the venv's Python, which is the one that has MuJoCo.

From then on, in **every new terminal**:

```bash
source ~/h1_rl_ws/env.sh      # venv + ROS 2 Jazzy + this workspace
```

---

## 2. Troubleshooting

* **No MuJoCo window / GLFW error (WSL).** Run `wsl --update` in PowerShell and restart WSL
  (`wsl --shutdown`). Check that `echo $DISPLAY $WAYLAND_DISPLAY` is not empty. Fallbacks:
  `ros2 launch h1_rl walk.launch.py viewer:=false rviz:=true`, or software rendering with
  `export LIBGL_ALWAYS_SOFTWARE=1`.
* **`ModuleNotFoundError: No module named 'mujoco'` when launching.** The workspace was built
  without the venv active. `source ~/h1_venv/bin/activate`, `rm -rf build install`, then
  `python -m colcon build --symlink-install`.
* **"A module that was compiled using NumPy 1.x cannot be run in NumPy 2".**
  `pip install "numpy<2"` inside the venv.
* **`torch.cuda.is_available()` is False.** Update the NVIDIA driver on Windows (the PyPI torch
  wheel targets a recent CUDA), or install the wheel for your CUDA version from pytorch.org.
  Training still works on the CPU.
* **The robot stumbles in ROS but not in `h1_rl.play`.** The machine cannot keep real time.
  Lower `realtime_factor:=0.5`; the controller runs off `/clock`, so slow motion is fine.
* **`ros2 topic list` shows nothing from the nodes.** Try
  `export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` in all terminals.
* **`colcon build` fails on `h1_rl` alone.** Build the whole workspace — `h1_rl` depends on the
  `h1_msgs` interfaces package.
