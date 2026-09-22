# Low-level joint command for the Unitree H1, one entry per joint (like Unitree's
# motor commands). Every joint runs an impedance controller:
#
#   tau = kp * (position - q) + kd * (velocity - dq) + effort     (clipped to motor limits)
#
# Arrays must have the same length as `name`. velocity/effort/kp/kd may be left
# empty: velocity and effort then default to 0, kp/kd keep their previous values.

std_msgs/Header header

string[] name        # joint names, e.g. left_knee_joint
float64[] position   # desired joint angle [rad]
float64[] velocity   # desired joint velocity [rad/s]
float64[] effort     # feed-forward torque [Nm]
float64[] kp         # stiffness [Nm/rad]
float64[] kd         # damping [Nm*s/rad]
