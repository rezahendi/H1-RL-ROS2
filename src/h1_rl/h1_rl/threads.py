"""Pick the number of physics threads by measuring it, not by guessing.

`os.cpu_count()` is the wrong answer on most machines: MuJoCo's Python binding holds the
GIL around every `mj_step` call and the per-step NumPy work runs on the main thread, so
throughput peaks at a handful of threads and then falls off a cliff. Measured on a 20-thread
i9-13900H, 128 robots: 15k env steps/s with 1 thread, 24k with 4, 7k with 20.

    from h1_rl.threads import choose_threads
    threads, rates = choose_threads(cfg, num_envs)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("H1_RL_CACHE", Path.home() / ".cache" / "h1_rl")) / "threads.json"


def _candidates(cpu: int, num_envs: int) -> list[int]:
    limit = max(1, min(cpu, num_envs))
    return sorted({t for t in (1, 2, 4, 6, 8, 12, max(1, cpu // 2), cpu) if t <= limit})


def measure(cfg: dict, num_envs: int, threads: int, steps: int = 20) -> float:
    """Environment steps per second with this many physics threads."""
    from .envs import H1WalkEnv

    env = H1WalkEnv(cfg, num_envs, num_threads=threads, seed=0)
    actions = np.zeros((num_envs, env.num_actions))
    env.reset_all()
    for _ in range(3):
        env.step(actions)
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(actions)
    rate = num_envs * steps / (time.perf_counter() - t0)
    env.close()
    return rate


def choose_threads(cfg: dict, num_envs: int, steps: int = 20, use_cache: bool = True,
                   verbose: bool = True) -> tuple[int, dict[int, float]]:
    """Measure a few thread counts and return the fastest. Cached per machine and env count."""
    cpu = os.cpu_count() or 1
    key = f"{cpu}x{num_envs}"
    if use_cache and CACHE.exists():
        try:
            cached = json.loads(CACHE.read_text(encoding="utf-8")).get(key)
            if cached:
                rates = {int(k): v for k, v in cached["rates"].items()}
                if verbose:
                    print(f"[threads] using cached choice: {cached['best']} threads "
                          f"({rates[cached['best']]:,.0f} env steps/s) - delete {CACHE} to redo")
                return int(cached["best"]), rates
        except (ValueError, KeyError, OSError):
            pass

    rates = {}
    for t in _candidates(cpu, num_envs):
        rates[t] = measure(cfg, num_envs, t, steps)
        if verbose:
            print(f"[threads] {t:>3} threads: {rates[t]:>9,.0f} env steps/s", flush=True)
    best = max(rates, key=rates.get)
    if verbose:
        print(f"[threads] using {best} (fastest of {len(rates)} tried)")
    if use_cache:
        try:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            all_rates = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
            all_rates[key] = {"best": best, "rates": {str(k): v for k, v in rates.items()},
                              "date": time.strftime("%Y-%m-%d %H:%M:%S")}
            CACHE.write_text(json.dumps(all_rates, indent=2), encoding="utf-8")
        except OSError:
            pass
    return best, rates
