"""Turn a privileged teacher into a deployable student (DAgger distillation).

The critic in `h1_rl.train` already sees privileged simulator state. Setting
`ppo.privileged_actor: true` gives that state to the *actor* as well, which learns faster
because it never has to infer friction, mass or its own base velocity - but it cannot run on
a robot. This script distils such a teacher into a student that sees only what the H1
measures (the same proprioceptive history the shipped policy uses):

    python -m h1_rl.train  --config config/h1_walk_teacher.yaml --run-name teacher
    python -m h1_rl.distill --teacher logs/h1_walk/teacher/model_4000.pt --iterations 1500

The student acts in the environment itself (DAgger), so it is trained on the states it
actually visits, and the teacher labels those states. The output is an ordinary checkpoint:
`h1_rl.export`, `h1_rl.eval` and the ROS 2 controller all take it as they are.
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .envs import H1WalkEnv
from .export import export_checkpoint
from .ppo import ActorCritic, EmpiricalNormalization


def load_teacher(path: str, env: H1WalkEnv, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not ck.get("teacher"):
        print("[distill] warning: that checkpoint was not trained with ppo.privileged_actor")
    cfg = ck["cfg"]
    pcfg = cfg["ppo"]
    actor_obs = env.num_critic_obs if ck.get("teacher") else env.num_obs
    policy = ActorCritic(actor_obs, env.num_critic_obs, env.num_actions,
                         pcfg["actor_hidden_dims"], pcfg["critic_hidden_dims"],
                         pcfg.get("activation", "elu"), float(pcfg["init_noise_std"])).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    norm = EmpiricalNormalization(actor_obs).to(device)
    if ck.get("obs_norm"):
        norm.load_state_dict(ck["obs_norm"])
    norm.eval()
    return policy, norm, cfg, ck


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True, help="checkpoint of the privileged policy")
    ap.add_argument("--config", default=None, help="default: the teacher's own config")
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--steps-per-env", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--mini-batches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--dagger-fraction", type=float, default=0.3,
                    help="fraction of training during which the teacher still drives")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log-dir", default="logs/h1_distill")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--save-interval", type=int, default=100)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    torch.manual_seed(args.seed)

    ck_cfg = torch.load(args.teacher, map_location="cpu", weights_only=False)["cfg"]
    cfg = load_config(args.config) if args.config else ck_cfg
    pcfg = cfg["ppo"]
    num_envs = int(args.num_envs or pcfg["num_envs"])
    threads = args.threads
    if threads is None:
        from .threads import choose_threads

        threads, _ = choose_threads(cfg, num_envs)
    env = H1WalkEnv(cfg, num_envs=num_envs, num_threads=threads, seed=args.seed)
    teacher, teacher_norm, _, _ = load_teacher(args.teacher, env, device)

    student = ActorCritic(env.num_obs, env.num_critic_obs, env.num_actions,
                          pcfg["actor_hidden_dims"], pcfg["critic_hidden_dims"],
                          pcfg.get("activation", "elu"), float(pcfg["init_noise_std"])).to(device)
    student_norm = EmpiricalNormalization(env.num_obs).to(device)
    critic_norm = EmpiricalNormalization(env.num_critic_obs).to(device)
    optimizer = torch.optim.Adam(student.actor.parameters(), lr=args.lr)

    run_dir = Path(args.log_dir) / (args.run_name or time.strftime("%Y-%m-%d_%H-%M-%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[distill] run dir: {run_dir}")
    print(f"[distill] device {device} | envs {num_envs} | teacher obs {env.num_critic_obs} -> "
          f"student obs {env.num_obs} | actions {env.num_actions}")

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(run_dir))
    except Exception as exc:  # noqa: BLE001
        print(f"[distill] TensorBoard disabled ({exc})")

    obs_np, cobs_np = env.reset_all()
    steps = int(args.steps_per_env)
    len_buf = collections.deque(maxlen=200)
    cur_len = np.zeros(num_envs)
    t_start = time.time()

    for it in range(args.iterations):
        beta = max(0.0, 1.0 - it / max(args.dagger_fraction * args.iterations, 1.0))
        obs_buf = torch.zeros(steps, num_envs, env.num_obs, device=device)
        act_buf = torch.zeros(steps, num_envs, env.num_actions, device=device)
        t0 = time.time()
        for t in range(steps):
            obs = torch.as_tensor(obs_np, device=device)
            cobs = torch.as_tensor(cobs_np, device=device)
            with torch.no_grad():
                student_norm.train(True)
                so = student_norm(obs)              # updates the running statistics
                critic_norm.train(True)
                critic_norm(cobs)
                label = teacher.actor(teacher_norm(cobs))     # what the teacher would do here
                guess = student.actor(so)
                # DAgger: the teacher drives early on, the student takes over
                mix = (torch.rand(num_envs, 1, device=device) < beta).float()
                action = mix * label + (1.0 - mix) * guess
                action = action + 0.1 * torch.randn_like(action)   # keep visiting new states
            obs_buf[t], act_buf[t] = obs, label
            obs_np, cobs_np, _, done_np, _ = env.step(action.cpu().numpy())
            cur_len += 1
            ended = np.nonzero(done_np)[0]
            if len(ended):
                len_buf.extend(cur_len[ended].tolist())
                cur_len[ended] = 0
        t_collect = time.time() - t0

        flat_obs = obs_buf.reshape(-1, env.num_obs)
        flat_act = act_buf.reshape(-1, env.num_actions)
        total = flat_obs.shape[0]
        size = max(1, total // args.mini_batches)
        losses = []
        student_norm.eval()
        for _ in range(args.epochs):
            perm = torch.randperm(total, device=device)
            for i in range(args.mini_batches):
                idx = perm[i * size:(i + 1) * size]
                loss = torch.nn.functional.mse_loss(student.actor(student_norm(flat_obs[idx])),
                                                    flat_act[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.actor.parameters(),
                                               float(pcfg.get("max_grad_norm", 1.0)))
                optimizer.step()
                losses.append(float(loss.detach()))
        loss_mean = float(np.mean(losses))
        if writer:
            writer.add_scalar("distill/loss", loss_mean, it)
            writer.add_scalar("distill/beta", beta, it)
            writer.add_scalar("train/mean_episode_length_s",
                              float(np.mean(len_buf)) * env.dt if len_buf else 0.0, it)
        if it % 10 == 0 or it == args.iterations - 1:
            ep = float(np.mean(len_buf)) * env.dt if len_buf else 0.0
            print(f"[it {it:5d}] imitation loss {loss_mean:.4f} | teacher share {beta:4.2f} | "
                  f"ep len {ep:5.1f}s | {num_envs * steps / t_collect:6.0f} steps/s", flush=True)
        if (it + 1) % args.save_interval == 0 or it == args.iterations - 1:
            path = run_dir / f"student_{it + 1}.pt"
            torch.save({"iteration": it + 1, "teacher": False, "policy": student.state_dict(),
                        "obs_norm": student_norm.state_dict(),
                        "critic_norm": critic_norm.state_dict(), "optimizer": None,
                        "cfg": cfg, "norm_eps": student_norm.eps}, path)
            export_checkpoint(path, run_dir / "policy_latest.npz")

    env.close()
    print(f"[distill] done in {(time.time() - t_start) / 60:.1f} min -> {run_dir}")
    print(f"[distill] evaluate it: python -m h1_rl.eval --policy {run_dir}/policy_latest.npz")


if __name__ == "__main__":
    main()
