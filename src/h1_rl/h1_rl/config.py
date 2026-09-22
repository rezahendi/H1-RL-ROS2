"""Configuration loading and file lookup.

Files (config, models, policies) are found in this order:
  1. $H1_RL_SHARE (if set)
  2. the source tree (works with `colcon build --symlink-install` and when
     running straight from `src/h1_rl` without ROS)
  3. the ROS 2 share directory of the h1_rl package (normal colcon install)
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

PACKAGE_NAME = "h1_rl"
DEFAULT_CONFIG = "config/h1_walk.yaml"


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("H1_RL_SHARE")
    if env:
        roots.append(Path(env).expanduser())
    # <src>/h1_rl/h1_rl/config.py -> <src>/h1_rl
    roots.append(Path(__file__).resolve().parent.parent)
    try:  # only available when a ROS 2 workspace is sourced
        from ament_index_python.packages import get_package_share_directory

        roots.append(Path(get_package_share_directory(PACKAGE_NAME)))
    except Exception:  # noqa: BLE001 - ament not installed / package not found
        pass
    return roots


def resolve_path(path: str | os.PathLike) -> Path:
    """Return an existing path; relative paths are looked up in the package roots."""
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        if not p.exists():
            raise FileNotFoundError(p)
        return p.resolve()
    for root in _candidate_roots():
        cand = root / p
        if cand.exists():
            return cand.resolve()
    tried = ", ".join(str(r / p) for r in _candidate_roots())
    raise FileNotFoundError(f"Could not find '{path}'. Tried: {tried}")


def model_dir() -> Path:
    return resolve_path("models/h1")


def deep_update(base: dict, other: dict) -> dict:
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _parse_override(item: str) -> tuple[list[str], Any]:
    """'ppo.num_envs=128' -> (['ppo', 'num_envs'], 128)."""
    if "=" not in item:
        raise ValueError(f"Override must look like key.sub=value, got '{item}'")
    key, value = item.split("=", 1)
    return key.strip().split("."), yaml.safe_load(value)


def load_config(path: str | os.PathLike | None = None,
                overrides: Iterable[str] | None = None) -> dict:
    """Load the YAML config, then apply 'a.b.c=value' overrides."""
    cfg_path = resolve_path(path or DEFAULT_CONFIG)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        keys, value = _parse_override(item)
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    cfg["_path"] = str(cfg_path)
    return cfg


def per_joint(table: dict, joint_names: list[str]) -> np.ndarray:
    """Expand a {substring: value} table into one value per joint (first match wins)."""
    out = np.zeros(len(joint_names))
    for i, name in enumerate(joint_names):
        for key, value in table.items():
            if str(key) in name:
                out[i] = float(value)
                break
        else:
            raise KeyError(f"No entry matches joint '{name}' in {list(table)}")
    return out
