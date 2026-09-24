"""Measure training throughput on this machine, so tuning is based on numbers.

    python -m h1_rl.bench                       # sweep env counts x physics threads
    python -m h1_rl.bench --envs 256,512 --threads 8,16,20 --steps 200
    python -m h1_rl.bench --json bench.json     # save the result (handy for comparing machines)
    python -m h1_rl.bench --no-ppo              # skip the PyTorch part

It reports environment steps per second (one step = one 50 Hz control step of one robot,
which is `decimation` physics steps), times one PPO update on the available device, and
estimates the wall-clock time of a full training run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .envs import H1WalkEnv


def machine_info() -> dict:
    import mujoco

    info = {"platform": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "mujoco": mujoco.__version__, "numpy": np.__version__}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["torch"] = None
    return info


def bench_env(cfg: dict, num_envs: int, threads: int, steps: int, warmup: int = 10) -> dict:
    """Time `steps` control steps of `num_envs` robots with random actions."""
    t0 = time.perf_counter()
    env = H1WalkEnv(cfg, num_envs, num_threads=threads, seed=0,
                    randomize=True, obs_noise=True, pushes=True)
    build_s = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    actions = rng.uniform(-0.5, 0.5, size=(num_envs, env.num_actions))
    env.reset_all()
    for _ in range(warmup):
        env.step(actions)
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(actions)
    elapsed = time.perf_counter() - t0
    env.close()
    per_s = num_envs * steps / elapsed
    return {"num_envs": num_envs, "threads": env.num_threads, "steps": steps,
            "seconds": round(elapsed, 2), "build_s": round(build_s, 2),
            "env_steps_per_s": per_s, "physics_steps_per_s": per_s * env.decimation}


def bench_ppo(cfg: dict, num_envs: int, num_obs: int, num_critic_obs: int,
              num_actions: int, device_arg: str = "auto") -> dict | None:
    """Time one PPO update with the configured network and batch size."""
    try:
        import torch
    except ImportError:
        return None
    from .ppo import PPO, ActorCritic, RolloutStorage

    pcfg = cfg["ppo"]
    device = torch.device("cuda" if (device_arg == "auto" and torch.cuda.is_available())
                          else ("cpu" if device_arg == "auto" else device_arg))
    steps = int(pcfg["num_steps_per_env"])
    policy = ActorCritic(num_obs, num_critic_obs, num_actions, pcfg["actor_hidden_dims"],
                         pcfg["critic_hidden_dims"], pcfg.get("activation", "elu"),
                         float(pcfg["init_noise_std"])).to(device)
    ppo = PPO(policy, pcfg, device)
    storage = RolloutStorage(steps, num_envs, num_obs, num_critic_obs, num_actions, device)
    for name in ("obs", "critic_obs", "actions", "log_probs", "values", "rewards", "mu"):
        getattr(storage, name).normal_()
    storage.sigma.fill_(0.8)
    storage.dones.bernoulli_(0.01)
    last_values = torch.zeros(num_envs, device=device)

    times = []
    for i in range(4):
        storage.compute_returns(last_values, ppo.gamma, ppo.lam)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ppo.update(storage)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return {"device": str(device), "num_envs": num_envs,
            "update_s": float(np.median(times[1:])),   # drop the first (warm-up) update
            "batch": steps * num_envs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--envs", default="64,128,256", help="comma separated env counts")
    ap.add_argument("--threads", default="auto", help="comma separated thread counts, or auto")
    ap.add_argument("--steps", type=int, default=100, help="timed control steps per combination")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda (for the PPO update)")
    ap.add_argument("--target-steps", type=float, default=40e6,
                    help="environment steps to estimate a full run for")
    ap.add_argument("--no-ppo", action="store_true", help="skip the PyTorch update benchmark")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cpu = os.cpu_count() or 1
    env_counts = [int(x) for x in args.envs.split(",")]
    if args.threads == "auto":
        threads = sorted({1, max(1, cpu // 4), max(1, cpu // 2), cpu})
    else:
        threads = sorted({int(x) for x in args.threads.split(",")})

    info = machine_info()
    print(f"[bench] {info['platform']}")
    print(f"[bench] {info['cpu_count']} logical CPUs | mujoco {info['mujoco']} | "
          f"torch {info['torch']} | cuda {info.get('cuda')} {info.get('gpu') or ''}")
    print(f"[bench] sweeping envs {env_counts} x threads {threads}, {args.steps} steps each\n")

    results = []
    header = "envs   " + "".join(f"{t:>12}" for t in threads)
    print(header)
    print("-" * len(header))
    for n in env_counts:
        row = f"{n:<7}"
        for t in threads:
            r = bench_env(cfg, n, t, args.steps)
            results.append(r)
            row += f"{r['env_steps_per_s']:>12,.0f}"
        print(row, flush=True)
    print("\n(environment steps per second; multiply by "
          f"{int(cfg['sim']['decimation'])} for physics steps)")

    best = max(results, key=lambda r: r["env_steps_per_s"])
    print(f"\n[bench] best: {best['num_envs']} envs x {best['threads']} threads -> "
          f"{best['env_steps_per_s']:,.0f} env steps/s "
          f"({best['physics_steps_per_s']:,.0f} physics steps/s)")

    ppo_result = None
    if not args.no_ppo:
        probe = H1WalkEnv(cfg, 2, num_threads=1, seed=0)
        dims = (probe.num_obs, probe.num_critic_obs, probe.num_actions)
        probe.close()
        ppo_result = bench_ppo(cfg, best["num_envs"], *dims, device_arg=args.device)
        if ppo_result:
            print(f"[bench] PPO update on {ppo_result['device']}: "
                  f"{ppo_result['update_s'] * 1e3:.0f} ms for a batch of {ppo_result['batch']:,}")

    steps_per_iter = best["num_envs"] * int(cfg["ppo"]["num_steps_per_env"])
    collect_s = steps_per_iter / best["env_steps_per_s"]
    update_s = ppo_result["update_s"] if ppo_result else 0.0
    iter_s = collect_s + update_s
    iterations = args.target_steps / steps_per_iter
    total_h = iterations * iter_s / 3600.0
    print(f"[bench] one iteration: {collect_s:.2f} s collect + {update_s:.2f} s update = {iter_s:.2f} s")
    print(f"[bench] {args.target_steps / 1e6:.0f}M env steps = {iterations:,.0f} iterations "
          f"~ {total_h * 60:.0f} min ({total_h:.1f} h)")
    if ppo_result and update_s > collect_s:
        print("[bench] the PPO update dominates: try a larger --num-envs, or a GPU")
    else:
        print("[bench] physics dominates: more CPU cores (or an MJX/GPU port) is what speeds this up")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"machine": info, "sweep": results, "best": best, "ppo": ppo_result,
             "estimate": {"target_steps": args.target_steps, "iterations": iterations,
                          "iteration_s": iter_s, "hours": total_h}}, indent=2), encoding="utf-8")
        print(f"[bench] json -> {args.json}")


if __name__ == "__main__":
    main()
