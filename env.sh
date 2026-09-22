# Source this in every new terminal:   source ~/h1_rl_ws/env.sh
# (activates the Python venv, ROS 2 Jazzy and this workspace)
H1_VENV="${H1_VENV:-$HOME/h1_venv}"
WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$H1_VENV/bin/activate" ]; then source "$H1_VENV/bin/activate"; else echo "[env.sh] venv $H1_VENV not found"; fi
source /opt/ros/jazzy/setup.bash
if [ -f "$WS_DIR/install/setup.bash" ]; then source "$WS_DIR/install/setup.bash"; else echo "[env.sh] workspace not built yet: cd $WS_DIR && python -m colcon build --symlink-install"; fi
