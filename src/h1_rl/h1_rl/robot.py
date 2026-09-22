"""Unitree H1 description helpers: joint tables, PD gains and MuJoCo model building."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .config import model_dir, per_joint

INTEGRATORS = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "rk4": mujoco.mjtIntegrator.mjINT_RK4,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}


@dataclass
class RobotSpec:
    """Per-joint tables resolved from the config (all arrays in joint order)."""

    joint_names: list[str]
    policy_joints: list[str]
    default_q: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    torque_limit: np.ndarray
    timestep: float
    decimation: int
    policy_idx: np.ndarray = field(init=False)
    other_idx: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.policy_idx = np.array([self.joint_names.index(j) for j in self.policy_joints])
        self.other_idx = np.array(
            [i for i in range(len(self.joint_names)) if i not in set(self.policy_idx)], dtype=int
        )

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def num_actions(self) -> int:
        return len(self.policy_joints)

    @property
    def control_dt(self) -> float:
        return self.timestep * self.decimation

    @classmethod
    def from_config(cls, cfg: dict) -> "RobotSpec":
        r = cfg["robot"]
        names = list(r["joint_names"])
        return cls(
            joint_names=names,
            policy_joints=list(r["policy_joints"]),
            default_q=per_joint(r["default_joint_pos"], names),
            kp=per_joint(r["kp"], names),
            kd=per_joint(r["kd"], names),
            torque_limit=per_joint(r["torque_limit"], names),
            timestep=float(cfg["sim"]["timestep"]),
            decimation=int(cfg["sim"]["decimation"]),
        )


def _strip_visuals(spec: mujoco.MjSpec) -> None:
    """Remove meshes, textures and visual-only geoms (training does not need them)."""
    for g in list(spec.geoms):
        if g.contype == 0 and g.conaffinity == 0:
            spec.delete(g)
    for g in spec.geoms:
        g.material = ""
    for mesh in list(spec.meshes):
        spec.delete(mesh)
    for mat in list(spec.materials):
        spec.delete(mat)
    for tex in list(spec.textures):
        spec.delete(tex)


def set_pd_gains(model: mujoco.MjModel, kp: np.ndarray, kd: np.ndarray,
                 idx: np.ndarray | slice = slice(None)) -> None:
    """Position servo: force = kp*ctrl - kp*q - kd*qdot (see h1.xml)."""
    model.actuator_gainprm[idx, 0] = kp
    model.actuator_biasprm[idx, 0] = 0.0
    model.actuator_biasprm[idx, 1] = -kp
    model.actuator_biasprm[idx, 2] = -kd


def build_model(cfg: dict, visual: bool = True) -> mujoco.MjModel:
    """Compile the H1 scene with the physics options and PD gains from the config."""
    robot = RobotSpec.from_config(cfg)
    scene = model_dir() / cfg["sim"].get("scene", "scene.xml")
    spec = mujoco.MjSpec.from_file(str(scene))
    if not visual:
        _strip_visuals(spec)
    model = spec.compile()

    sim = cfg["sim"]
    model.opt.timestep = robot.timestep
    model.opt.integrator = INTEGRATORS[sim.get("integrator", "implicitfast")]
    model.opt.iterations = int(sim.get("iterations", 10))
    model.opt.ls_iterations = int(sim.get("ls_iterations", 20))

    # The config joint order must match the MuJoCo actuator order.
    act_joints = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
        for i in range(model.nu)
    ]
    if act_joints != robot.joint_names:
        raise ValueError(f"Actuator order {act_joints} does not match config {robot.joint_names}")

    set_pd_gains(model, robot.kp, robot.kd)
    model.actuator_forcelimited[:] = 1
    model.actuator_forcerange[:, 0] = -robot.torque_limit
    model.actuator_forcerange[:, 1] = robot.torque_limit
    return model


def sensor_slices(model: mujoco.MjModel) -> dict[str, slice]:
    out = {}
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        adr, dim = int(model.sensor_adr[i]), int(model.sensor_dim[i])
        out[name] = slice(adr, adr + dim)
    return out


def joint_qpos_qvel_index(model: mujoco.MjModel, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    qadr, vadr = [], []
    for name in joint_names:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise KeyError(f"joint '{name}' not in model")
        qadr.append(model.jnt_qposadr[j])
        vadr.append(model.jnt_dofadr[j])
    return np.array(qadr), np.array(vadr)


def foot_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "_foot_" in name:
            ids.append(g)
    return np.array(ids, dtype=int)


def place_on_ground(model: mujoco.MjModel, data: mujoco.MjData,
                    feet: np.ndarray | None = None, clearance: float = 0.002) -> None:
    """Shift the floating base vertically so the lowest foot sphere is `clearance` above z=0."""
    if feet is None:
        feet = foot_geom_ids(model)
    mujoco.mj_kinematics(model, data)
    lowest = np.min(data.geom_xpos[feet, 2] - model.geom_size[feet, 0])
    data.qpos[2] += clearance - lowest


def standing_qpos(model: mujoco.MjModel, robot: RobotSpec, yaw: float = 0.0) -> np.ndarray:
    """Full qpos with the default joint angles and both feet on the floor."""
    data = mujoco.MjData(model)
    data.qpos[:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    qadr, _ = joint_qpos_qvel_index(model, robot.joint_names)
    data.qpos[qadr] = robot.default_q
    place_on_ground(model, data)
    return data.qpos.copy()
