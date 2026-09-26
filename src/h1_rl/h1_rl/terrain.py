"""Procedural terrain patches for the rough-terrain curriculum.

Each environment gets its own MuJoCo height field. Every patch is **periodic**: the last row
and column continue into the first, so a robot that walks past the edge can be wrapped to the
other side mid-episode without the ground moving under its feet. That keeps episodes long
without needing a terrain the size of the distance the robot covers.

    height = generate(rng, n, level, "rough")      # (n, n) metres, >= 0, periodic
    z      = sample(height, patch, x, y)           # ground height under a point
"""

from __future__ import annotations

import numpy as np

KINDS = ("flat", "rough", "waves", "steps")


def _periodic_noise(rng: np.random.Generator, n: int, cells: int) -> np.ndarray:
    """Smooth noise that tiles seamlessly: low frequencies only, random phase."""
    field = np.zeros((n, n))
    freqs = np.fft.fftfreq(n) * n
    fx, fy = np.meshgrid(freqs, freqs, indexing="ij")
    radius = np.sqrt(fx ** 2 + fy ** 2)
    mask = (radius > 0) & (radius <= cells)
    spectrum = np.zeros((n, n), dtype=complex)
    spectrum[mask] = (rng.normal(size=mask.sum()) + 1j * rng.normal(size=mask.sum())) / radius[mask]
    field = np.fft.ifft2(spectrum).real
    peak = np.abs(field).max()
    return field / peak if peak > 0 else field


def generate(rng: np.random.Generator, n: int, level: float, kind: str = "rough",
             amplitude: float = 0.12) -> np.ndarray:
    """One periodic height patch in metres, minimum 0. `level` in [0, 1] scales the roughness."""
    level = float(np.clip(level, 0.0, 1.0))
    if kind == "flat" or level <= 0.0:
        return np.zeros((n, n))
    a = amplitude * level
    if kind == "rough":
        h = _periodic_noise(rng, n, cells=max(4, n // 8)) * a
    elif kind == "waves":
        k1, k2 = rng.integers(1, 4, size=2)
        phase = rng.uniform(0, 2 * np.pi, size=2)
        u = np.linspace(0, 2 * np.pi, n, endpoint=False)
        h = (np.sin(k1 * u + phase[0])[:, None] + np.sin(k2 * u + phase[1])[None, :]) * a * 0.6
    elif kind == "steps":
        blocks = int(rng.integers(4, 9)) * 2          # even, so the pattern tiles
        step = max(1, n // blocks)
        coarse = rng.integers(0, 3, size=(blocks, blocks)) * a
        coarse[blocks // 2:] = coarse[:blocks // 2][::-1]   # mirror: periodic in x
        coarse[:, blocks // 2:] = coarse[:, :blocks // 2][:, ::-1]
        h = np.repeat(np.repeat(coarse, step, axis=0), step, axis=1)[:n, :n]
        if h.shape != (n, n):                          # pad if n is not divisible
            h = np.pad(h, ((0, n - h.shape[0]), (0, n - h.shape[1])), mode="wrap")
    else:
        raise ValueError(f"unknown terrain kind '{kind}' (have {KINDS})")
    return h - h.min()


def sample(height: np.ndarray, patch: float, x, y) -> np.ndarray:
    """Ground height at world (x, y) by bilinear interpolation, wrapping at the patch edges."""
    n = height.shape[0]
    fx = (np.asarray(x) / patch + 0.5) * n
    fy = (np.asarray(y) / patch + 0.5) * n
    i0, j0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = fx - i0, fy - j0
    i0, j0 = i0 % n, j0 % n
    i1, j1 = (i0 + 1) % n, (j0 + 1) % n
    return (height[i0, j0] * (1 - tx) * (1 - ty) + height[i1, j0] * tx * (1 - ty)
            + height[i0, j1] * (1 - tx) * ty + height[i1, j1] * tx * ty)


def write_to_model(model, height: np.ndarray, elevation: float) -> None:
    """Copy a patch into a compiled model's height field (stored normalised to [0, 1])."""
    n = height.shape[0]
    assert model.hfield_nrow[0] == n and model.hfield_ncol[0] == n, "hfield size mismatch"
    model.hfield_data[:n * n] = np.clip(height.T.reshape(-1) / elevation, 0.0, 1.0)
