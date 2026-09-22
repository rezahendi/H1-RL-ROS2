"""NumPy-only policy runtime (no PyTorch needed at deployment).

A policy file (.npz) is produced by `h1_rl.export` and contains the actor MLP,
the observation normalizer and a JSON `meta` blob with everything the
controller must reproduce from training (joint order, default pose, PD gains,
scales, gait clock, command ranges, timing).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ACTIVATIONS = {
    "elu": lambda x: np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0))),
    "relu": lambda x: np.maximum(x, 0.0),
    "tanh": np.tanh,
}


class Policy:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.suffix == ".pt":
            raise ValueError(f"{self.path} is a training checkpoint. Export it first:\n"
                             f"  python3 -m h1_rl.export --checkpoint {self.path} --output policy.npz\n"
                             "(training also writes policy_latest.npz next to each checkpoint)")
        with np.load(self.path, allow_pickle=False) as f:
            self.meta: dict = json.loads(str(f["meta"]))
            n_layers = int(f["num_layers"])
            self.weights = [f[f"w{i}"].astype(np.float64) for i in range(n_layers)]
            self.biases = [f[f"b{i}"].astype(np.float64) for i in range(n_layers)]
            self.obs_mean = f["obs_mean"].astype(np.float64)
            self.obs_std = f["obs_std"].astype(np.float64)
        self.activation = ACTIVATIONS[self.meta.get("activation", "elu")]
        self.obs_dim = self.weights[0].shape[1]
        self.action_dim = self.weights[-1].shape[0]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action (the mean of the Gaussian policy). obs: (obs_dim,) or (N, obs_dim)."""
        x = (np.asarray(obs, dtype=np.float64) - self.obs_mean) / self.obs_std
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w.T + b
            if i < last:
                x = self.activation(x)
        return x
