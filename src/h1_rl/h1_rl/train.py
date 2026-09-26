"""Train the H1 walking policy with PPO in MuJoCo (no ROS needed).

    python3 -m h1_rl.train                                  # defaults from config/h1_walk.yaml
    python3 -m h1_rl.train --num-envs 256 --threads 16      # bigger machine
    python3 -m h1_rl.train --resume logs/h1_walk/<run>/model_2000.pt
    python3 -m h1_rl.train --set rewards.scales.feet_swing_height=-10

Progress goes to the console and TensorBoard:  tensorboard --logdir logs
Every checkpoint is also exported as <run>/policy_latest.npz for the ROS 2 controller.
"""

from __future__ import annotations

import argparse
import collections
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import load_config
from .envs import H1WalkEnv
from .export import export_checkpoint
from .ppo import PPO, ActorCritic, EmpiricalNormalization, RolloutStorage


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (default: config/h1_walk.yaml)")
    ap.add_argument("--num-envs", type=int, default=None, help="parallel environments (ppo.num_envs)")
    ap.add_argument("--threads", type=int, default=None,
                    help="physics threads (default: measured once by h1_rl.threads)")
    ap.add_argument("--iterations", type=int, default=None, help="PPO iterations (ppo.max_iterations)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log-dir", default="logs/h1_walk")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None, help="checkpoint (.pt) to continue from")
    ap.add_argument("--reset-std", type=float, default=None,
                    help="when resuming/fine-tuning: set the action noise std to this value")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set ppo.entropy_coef=0.01")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    if args.num_envs:
        overrides.append(f"ppo.num_envs={args.num_envs}")
    if args.iterations:
        overrides.append(f"ppo.max_iterations={args.iterations}")
    cfg = load_config(args.config, overrides)
    resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if resume is not None and not args.config and not overrides:
        cfg = resume["cfg"]  # continue with the exact training setup
    pcfg = cfg["ppo"]

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    # CPU training: 1 torch thread while MuJoCo threads collect data, more for the PPO update
    collect_threads = 1
    update_threads = int(os.environ.get("H1_TORCH_THREADS", max(1, min(8, (os.cpu_count() or 1) // 2))))
    if device.type == "cpu":
        torch.set_num_threads(collect_threads)
    torch.manual_seed(args.seed)

    num_envs = int(pcfg["num_envs"])
    threads = args.threads
    if threads is None:
        from .threads import choose_threads

        threads, _ = choose_threads(cfg, num_envs)
    env = H1WalkEnv(cfg, num_envs=num_envs, num_threads=threads, seed=args.seed)
    # A teacher policy reads the privileged state directly; h1_rl.distill turns one into a
    # deployable student that only sees what the robot can measure.
    teacher = bool(pcfg.get("privileged_actor", False))
    actor_obs = env.num_critic_obs if teacher else env.num_obs
    policy = ActorCritic(actor_obs, env.num_critic_obs, env.num_actions,
                         pcfg["actor_hidden_dims"], pcfg["critic_hidden_dims"],
                         pcfg.get("activation", "elu"), float(pcfg["init_noise_std"])).to(device)
    use_norm = bool(pcfg.get("empirical_normalization", True))
    obs_norm = EmpiricalNormalization(actor_obs).to(device)
    critic_norm = EmpiricalNormalization(env.num_critic_obs).to(device)
    ppo = PPO(policy, pcfg, device)
    steps = int(pcfg["num_steps_per_env"])
    storage = RolloutStorage(steps, num_envs, actor_obs, env.num_critic_obs, env.num_actions, device)

    start_it = 0
    if resume is not None:
        policy.load_state_dict(resume["policy"])
        obs_norm.load_state_dict(resume["obs_norm"])
        critic_norm.load_state_dict(resume["critic_norm"])
        if resume.get("optimizer"):
            ppo.optimizer.load_state_dict(resume["optimizer"])
        ppo.lr = float(resume.get("lr", ppo.lr))
        start_it = int(resume["iteration"])
    if args.reset_std is not None:
        with torch.no_grad():
            policy.log_std.fill_(float(np.log(args.reset_std)))

    run_name = args.run_name or time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f, sort_keys=False)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(run_dir))
    except Exception as exc:  # noqa: BLE001
        print(f"[train] TensorBoard disabled ({exc})")

    print(f"[train] run dir: {run_dir}")
    print(f"[train] device: {device} | envs: {num_envs} | physics threads: {env.num_threads} | "
          f"obs: {env.num_obs} | critic obs: {env.num_critic_obs} | actions: {env.num_actions}"
          + (" | PRIVILEGED TEACHER (not deployable: distill it)" if teacher else ""))

    def save(it: int) -> Path:
        path = run_dir / f"model_{it}.pt"
        torch.save({"iteration": it, "teacher": teacher, "policy": policy.state_dict(), "obs_norm": obs_norm.state_dict(),
                    "critic_norm": critic_norm.state_dict(), "optimizer": ppo.optimizer.state_dict(),
                    "lr": ppo.lr, "cfg": cfg, "norm_eps": obs_norm.eps}, path)
        if not teacher:          # a privileged actor is not deployable: distill it first
            export_checkpoint(path, run_dir / "policy_latest.npz")
        return path

    ret_buf = collections.deque(maxlen=200)
    len_buf = collections.deque(maxlen=200)
    ep_terms: dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=50))
    cur_ret = np.zeros(num_envs)
    cur_len = np.zeros(num_envs)

    obs_np, cobs_np = env.reset_all()
    to_t = lambda x: torch.as_tensor(x, device=device)  # noqa: E731
    obs, cobs = to_t(obs_np), to_t(cobs_np)
    max_it = int(pcfg["max_iterations"])
    save_interval = int(pcfg["save_interval"])
    t_start = time.time()
    total_steps = 0

    for it in range(start_it, max_it):
        t0 = time.time()
        # ------------------------------------------------------ collect rollouts
        obs_norm.train(use_norm)
        critic_norm.train(use_norm)
        with torch.no_grad():
            for _ in range(steps):
                c = critic_norm(cobs) if use_norm else cobs
                o = c if teacher else (obs_norm(obs) if use_norm else obs)
                dist = policy.distribution(o)
                actions = dist.sample()
                values = policy.value(c)
                obs_np, cobs_np, rew_np, done_np, info = env.step(actions.cpu().numpy())
                rewards = to_t(rew_np.astype(np.float32))
                dones = to_t(done_np.astype(np.float32))
                time_outs = to_t(info["time_outs"].astype(np.float32))
                rewards = rewards + ppo.gamma * values * time_outs  # bootstrap on time limits
                storage.add(o, c, actions, dist.log_prob(actions).sum(-1), values, rewards, dones,
                            dist.mean, dist.stddev)
                obs, cobs = to_t(obs_np), to_t(cobs_np)

                cur_ret += rew_np
                cur_len += 1
                ended = np.nonzero(done_np)[0]
                if len(ended):
                    ret_buf.extend(cur_ret[ended].tolist())
                    len_buf.extend(cur_len[ended].tolist())
                    cur_ret[ended] = 0.0
                    cur_len[ended] = 0
                    for k, v in info["episode"].items():
                        ep_terms[k].append(v)
            c = critic_norm(cobs) if use_norm else cobs
            last_values = policy.value(c)
        storage.compute_returns(last_values, ppo.gamma, ppo.lam)
        t_collect = time.time() - t0

        # ------------------------------------------------------------- update
        obs_norm.eval()
        critic_norm.eval()
        t1 = time.time()
        if device.type == "cpu":
            torch.set_num_threads(update_threads)
        stats = ppo.update(storage)
        if device.type == "cpu":
            torch.set_num_threads(collect_threads)
        t_update = time.time() - t1

        total_steps += steps * num_envs
        fps = steps * num_envs / (time.time() - t0)
        mean_ret = float(np.mean(ret_buf)) if ret_buf else 0.0
        mean_len = float(np.mean(len_buf)) * env.dt if len_buf else 0.0
        if writer is not None:
            writer.add_scalar("train/mean_episode_return", mean_ret, it)
            writer.add_scalar("train/mean_episode_length_s", mean_len, it)
            writer.add_scalar("train/fps", fps, it)
            writer.add_scalar("train/learning_rate", ppo.lr, it)
            writer.add_scalar("train/action_std", policy.std.mean().item(), it)
            for k, v in stats.items():
                writer.add_scalar(f"loss/{k}", v, it)
            for k, v in ep_terms.items():
                writer.add_scalar(f"episode/{k}", float(np.mean(v)), it)
        if it % 10 == 0 or it == max_it - 1:
            elapsed = time.time() - t_start
            eta = elapsed / max(it - start_it + 1, 1) * (max_it - it - 1)
            lin = np.mean(ep_terms["rew_tracking_lin_vel"]) if ep_terms["rew_tracking_lin_vel"] else 0.0
            ang = np.mean(ep_terms["rew_tracking_ang_vel"]) if ep_terms["rew_tracking_ang_vel"] else 0.0
            print(f"[it {it:5d}] return {mean_ret:7.3f} | ep len {mean_len:5.1f}s | "
                  f"track lin {lin:.3f} ang {ang:.3f} | std {policy.std.mean().item():.3f} | "
                  f"lr {ppo.lr:.1e} | kl {stats['kl']:.4f} | {fps:6.0f} steps/s "
                  f"(collect {t_collect:.1f}s, update {t_update:.1f}s) | ETA {eta / 60:.0f} min", flush=True)
        if (it + 1) % save_interval == 0:
            save(it + 1)

    path = save(max_it)
    final = run_dir / "policy.npz"
    shutil.copy(run_dir / "policy_latest.npz", final)
    print(f"[train] done. checkpoint: {path}\n[train] policy for ROS 2: {final}")
    env.close()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
