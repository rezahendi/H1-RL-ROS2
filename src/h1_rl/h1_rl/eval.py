"""Reproducible evaluation suites for a trained policy (no ROS needed).

    python -m h1_rl.eval                                  # every suite, shipped policy
    python -m h1_rl.eval --policy logs/h1_walk/<run>/policy_latest.npz
    python -m h1_rl.eval --suite track,stop --jobs 8
    python -m h1_rl.eval --quick --check                  # CI gate (~1 min)
    python -m h1_rl.eval --json report.json --markdown report.md
    python -m h1_rl.eval --compare report_baseline.json   # A/B against an earlier report

Suites
    track   follow a scripted command sequence, report achieved vs commanded velocity
    stop    walk, then command zero at many points of the gait cycle: does it stop and stay up
    stand   stand still for 12 s on many randomized robots, with sensor noise / randomized dynamics
    push    shove the robot while standing and while walking, count how often it survives
    robust  20 s episodes with randomization, noise and pushes all on: falls + tracking error

The numbers in README.md come from `python -m h1_rl.eval` on the shipped policy.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .config import load_config, resolve_path
from .controller import PolicyController
from .envs import H1WalkEnv
from .play import DEMO_SCRIPT, run_script
from .policy import Policy
from .sim import H1Sim

DEFAULT_POLICY = "policies/h1_walk.npz"
SUITES = ("track", "stop", "stand", "push", "robust")

# Pass/fail gates. Rates, not counts, so they hold in --quick mode as well.
GATES = {
    "track/falls": ("<=", 0.0),
    "track/err_vx": ("<=", 0.08),
    "track/err_vy": ("<=", 0.08),
    "track/err_yaw": ("<=", 0.08),
    "stop/failure_rate": ("<=", 0.05),
    "stand/fall_rate_noise": ("<=", 0.03),
    "stand/fall_rate_random": ("<=", 0.03),
    # a corrective step now and then while standing is fine and even desirable; this gate
    # exists to catch a trigger so eager that the robot marches in place (measured at 99.8%)
    "stand/restless_fraction": ("<=", 0.05),
    "push/walking_survival_rate": (">=", 0.75),
}

# suite sizes: (full, quick)
SIZES = {
    "stop_speeds": ([0.3, 0.5, 0.7, 1.0], [0.5, 1.0]),
    "stop_phases": (12, 3),
    "stand_seeds": (3, 1),
    "stand_envs": (64, 16),
    "stand_seconds": (12.0, 5.0),
    "push_trials": (12, 4),
    "push_speeds": ([0.4, 0.8], [0.8]),
    "robust_envs": (64, 16),
    "robust_seconds": (20.0, 5.0),
}


def size(key: str, quick: bool):
    return SIZES[key][1 if quick else 0]


# --------------------------------------------------------------------------- helpers
def _rollout(policy_path, cfg, command_fn, steps, latency_steps=1, push=None, push_step=None):
    """Run one robot with the deployment controller. Returns (fell, sim, ctrl)."""
    sim = H1Sim(cfg, visual=False)
    sim.reset(support=False)
    ctrl = PolicyController(policy_path)
    dec = sim.robot.decimation
    fell = False
    for k in range(steps):
        if push is not None and k == push_step:
            sim.data.qvel[0] += push[0]
            sim.data.qvel[1] += push[1]
        q, dq, _ = sim.joint_state()
        quat, gyro, _ = sim.imu()
        out = ctrl.step(q, dq, quat, gyro, command_fn(k))
        if latency_steps:
            sim.step(latency_steps)
        sim.set_command(out.position, out.velocity, out.kp, out.kd, out.effort)
        sim.step(dec - latency_steps)
        if sim.fallen():
            fell = True
            break
    return fell, sim, ctrl


def _map(fn, jobs, items):
    """Run independent single-robot trials, in processes when it is worth it."""
    if jobs <= 1 or len(items) <= 1:
        return [fn(it) for it in items]
    with ProcessPoolExecutor(max_workers=min(jobs, len(items))) as pool:
        return list(pool.map(fn, items))


# --------------------------------------------------------------------------- suites
def suite_track(policy_path, config_path, quick=False, jobs=1, seed=0):
    """Scripted command sequence: how closely does the base follow the command."""
    cfg = load_config(config_path)
    sim = H1Sim(cfg, visual=False)
    sim.reset(support=False)
    stats = run_script(sim, PolicyController(policy_path), DEMO_SCRIPT,
                       latency_steps=1, verbose=False)
    err = stats["mean_abs_error"]
    rows = [{"command": s["cmd"], "achieved": s["achieved"], "error": s["error"]}
            for s in stats["segments"]]
    return {
        "segments": rows,
        "falls": int(stats["falls"]),
        "metrics": {"falls": float(stats["falls"]), "err_vx": float(err[0]),
                    "err_vy": float(err[1]), "err_yaw": float(err[2])},
    }


def _stop_trial(arg):
    policy_path, config_path, speed, walk_steps, stop_steps = arg
    cfg = load_config(config_path)
    fell, sim, _ = _rollout(policy_path, cfg,
                            lambda k: [speed, 0.0, 0.0] if k < walk_steps else [0.0, 0.0, 0.0],
                            walk_steps + stop_steps)
    still_moving = (not fell) and abs(sim.base_twist_body()[0][0]) > 0.15
    return {"speed": speed, "stop_tick": walk_steps, "fell": bool(fell),
            "still_moving": bool(still_moving)}


def suite_stop(policy_path, config_path, quick=False, jobs=1, seed=0):
    """Command zero at many points of the gait cycle: it must brake and stand, not topple."""
    speeds = size("stop_speeds", quick)
    phases = size("stop_phases", quick)
    step = max(1, 40 // phases)          # a gait cycle is 40 control ticks at 0.8 s / 50 Hz
    args = [(policy_path, config_path, s, 200 + i * step, 300)
            for s in speeds for i in range(phases)]
    trials = _map(_stop_trial, jobs, args)
    fails = [t for t in trials if t["fell"] or t["still_moving"]]
    n = len(trials)
    return {
        "trials": trials,
        "metrics": {"trials": float(n), "failures": float(len(fails)),
                    "falls": float(sum(t["fell"] for t in trials)),
                    "failure_rate": len(fails) / max(n, 1)},
    }


def _stand_batch(policy_path, cfg, num_envs, seconds, randomize, noise, seed):
    """Walk at 0.5 m/s, command zero, then count falls while the robots stand."""
    cfg = copy.deepcopy(cfg)
    cfg["commands"]["resample_interval_s"] = [1e4, 1e4]      # keep the command we set
    env = H1WalkEnv(cfg, num_envs, seed=seed, randomize=randomize, obs_noise=noise, pushes=False)
    policy = Policy(policy_path)
    obs, _ = env.reset_all()
    env.cmd_target[:] = env.commands[:] = [0.5, 0.0, 0.0]
    settle = int(3.0 / env.dt)
    falls, restless, ticks = 0, 0, 0
    for t in range(settle + int(seconds / env.dt)):
        if t == settle:
            env.cmd_target[:] = 0.0
        obs, _, _, done, info = env.step(policy(obs))
        if t >= settle + int(1.0 / env.dt):   # after the last step has been finished
            falls += int(info["terminated"].sum())
            restless += int(np.sum(env.clock.recovery > 0))
            ticks += env.num_envs
        ids = np.nonzero(done)[0]
        if len(ids):
            env.cmd_target[ids] = 0.0 if t >= settle else [0.5, 0.0, 0.0]
    env.close()
    return falls, restless / max(ticks, 1)


def suite_stand(policy_path, config_path, quick=False, jobs=1, seed=0):
    """Standing still: the hard case for the H1, which cannot hold a pose with fixed targets."""
    cfg = load_config(config_path)
    seeds = size("stand_seeds", quick)
    n_env = size("stand_envs", quick)
    seconds = size("stand_seconds", quick)
    out = {}
    for name, randomize, noise in (("noise", False, True), ("random", True, False)):
        runs = [_stand_batch(policy_path, cfg, n_env, seconds, randomize, noise, seed + 51 + i)
                for i in range(seeds)]
        falls = sum(r[0] for r in runs)
        robots = seeds * n_env
        out[name] = {"robots": robots, "falls": falls, "seconds": seconds,
                     "restless": float(np.mean([r[1] for r in runs]))}
    return {
        "variants": out,
        "metrics": {
            "robots": float(out["noise"]["robots"]),
            "falls_noise": float(out["noise"]["falls"]),
            "falls_random": float(out["random"]["falls"]),
            "fall_rate_noise": out["noise"]["falls"] / max(out["noise"]["robots"], 1),
            "fall_rate_random": out["random"]["falls"] / max(out["random"]["robots"], 1),
            # fraction of quiet standing time spent stepping: false recovery triggers
            "restless_fraction": max(out["noise"]["restless"], out["random"]["restless"]),
        },
    }


def _push_trial(arg):
    policy_path, config_path, standing, push, trial = arg
    cfg = load_config(config_path)
    rng = np.random.default_rng(1000 + trial)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    cmd = [0.0, 0.0, 0.0] if standing else [0.5, 0.0, 0.0]
    fell, _, _ = _rollout(policy_path, cfg, lambda k: cmd, 400,
                          push=(push * np.cos(angle), push * np.sin(angle)), push_step=250)
    return {"standing": bool(standing), "push": push, "survived": bool(not fell)}


def suite_push(policy_path, config_path, quick=False, jobs=1, seed=0):
    """Velocity kick to the pelvis, random direction, while standing and while walking."""
    trials = size("push_trials", quick)
    pushes = size("push_speeds", quick)
    args = [(policy_path, config_path, standing, p, t)
            for standing in (True, False) for p in pushes for t in range(trials)]
    results = _map(_push_trial, jobs, args)
    table, metrics = {}, {}
    for standing in (True, False):
        mode = "standing" if standing else "walking"
        for p in pushes:
            sel = [r for r in results if r["standing"] == standing and r["push"] == p]
            survived = sum(r["survived"] for r in sel)
            table[f"{mode}_{p}"] = {"survived": survived, "trials": len(sel)}
        sel = [r for r in results if r["standing"] == standing]
        metrics[f"{mode}_survival_rate"] = sum(r["survived"] for r in sel) / max(len(sel), 1)
    return {"cases": table, "metrics": metrics}


def suite_robust(policy_path, config_path, quick=False, jobs=1, seed=0):
    """Everything on at once: randomized dynamics, sensor noise and random pushes."""
    cfg = load_config(config_path)
    n_env = size("robust_envs", quick)
    seconds = size("robust_seconds", quick)
    env = H1WalkEnv(cfg, n_env, seed=seed + 123, randomize=True, obs_noise=True, pushes=True)
    policy = Policy(policy_path)
    obs, _ = env.reset_all()
    falls, err = 0, []
    for _ in range(int(seconds / env.dt)):
        obs, _, _, _, info = env.step(policy(obs))
        falls += int(info["terminated"].sum())
        err.append(np.linalg.norm(env.commands[:, :2] - env.base_lin_vel[:, :2], axis=1).mean())
    env.close()
    robot_seconds = n_env * seconds
    return {
        "robots": n_env, "seconds": seconds, "falls": falls,
        "metrics": {"falls": float(falls),
                    "falls_per_robot_minute": falls / (robot_seconds / 60.0),
                    "tracking_error_xy": float(np.mean(err))},
    }


RUNNERS = {"track": suite_track, "stop": suite_stop, "stand": suite_stand,
           "push": suite_push, "robust": suite_robust}


# --------------------------------------------------------------------------- report
def run_suites(policy_path, config_path=None, suites=SUITES, quick=False, jobs=1, seed=0):
    given, policy_path = str(policy_path), str(resolve_path(policy_path))
    meta = dict(Policy(policy_path).meta)
    report = {
        "policy": given,
        "policy_file": policy_path,
        "iteration": meta.get("iteration"),
        "trained": meta.get("created"),
        "mode": "quick" if quick else "full",
        "host": {"platform": platform.platform(), "cpus": os.cpu_count()},
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "suites": {},
        "metrics": {},
    }
    for name in suites:
        t0 = time.perf_counter()
        result = RUNNERS[name](policy_path, config_path, quick=quick, jobs=jobs, seed=seed)
        result["elapsed_s"] = round(time.perf_counter() - t0, 1)
        for key, value in result.pop("metrics").items():
            report["metrics"][f"{name}/{key}"] = value
        report["suites"][name] = result
        print(f"[eval] {name}: {result['elapsed_s']} s", flush=True)
    return report


def check_gates(metrics):
    rows = []
    for key, (op, limit) in GATES.items():
        if key not in metrics:
            continue
        value = metrics[key]
        ok = value <= limit if op == "<=" else value >= limit
        rows.append({"metric": key, "value": value, "op": op, "limit": limit, "pass": bool(ok)})
    return rows


def render_markdown(report):
    out = [f"# Evaluation report", "",
           f"policy `{report['policy']}` · iteration {report['iteration']} · "
           f"{report['mode']} mode · {report['date']}", ""]
    s = report["suites"]
    if "track" in s:
        out += ["## Command tracking", "",
                "| command vx, vy, yaw | achieved | abs error |", "|---|---|---|"]
        for r in s["track"]["segments"]:
            c, a, e = r["command"], r["achieved"], r["error"]
            out.append(f"| {c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f} | "
                       f"{a[0]:+.2f}, {a[1]:+.2f}, {a[2]:+.2f} | "
                       f"{e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f} |")
        m = report["metrics"]
        out += ["", f"falls: {int(m['track/falls'])} · mean abs error "
                    f"vx {m['track/err_vx']:.3f} m/s, vy {m['track/err_vy']:.3f} m/s, "
                    f"yaw {m['track/err_yaw']:.3f} rad/s", ""]
    if "stop" in s:
        m = report["metrics"]
        by_speed = {}
        for t in s["stop"]["trials"]:
            k = t["speed"]
            by_speed.setdefault(k, [0, 0])
            by_speed[k][1] += 1
            by_speed[k][0] += int(t["fell"] or t["still_moving"])
        out += ["## Stopping", "", "| from speed | failures |", "|---|---|"]
        for speed, (f, n) in sorted(by_speed.items()):
            out.append(f"| {speed:.1f} m/s | {f} / {n} |")
        out += ["", f"total {int(m['stop/failures'])} / {int(m['stop/trials'])} "
                    f"({m['stop/failure_rate']:.1%}), of which {int(m['stop/falls'])} falls", ""]
    if "stand" in s:
        out += ["## Standing still", "", "| condition | falls |", "|---|---|"]
        for name, v in s["stand"]["variants"].items():
            label = "sensor noise" if name == "noise" else "randomized dynamics"
            out.append(f"| {label}, {v['seconds']:.0f} s | {v['falls']} / {v['robots']} robots |")
        out.append(f"\nstepping during quiet standing (false recovery triggers): "
                   f"{report['metrics']['stand/restless_fraction']:.2%} of the time")
        out.append("")
    if "push" in s:
        out += ["## Push recovery", "", "| case | survived |", "|---|---|"]
        for case, v in s["push"]["cases"].items():
            mode, push = case.rsplit("_", 1)
            out.append(f"| {push} m/s shove while {mode} | {v['survived']} / {v['trials']} |")
        out.append("")
    if "robust" in s:
        r, m = s["robust"], report["metrics"]
        out += ["## Randomized episodes", "",
                f"{r['robots']} robots x {r['seconds']:.0f} s with randomized dynamics, sensor "
                f"noise and pushes: {r['falls']} falls "
                f"({m['robust/falls_per_robot_minute']:.2f} per robot-minute), "
                f"mean |v_xy error| {m['robust/tracking_error_xy']:.3f} m/s", ""]
    gates = check_gates(report["metrics"])
    if gates:
        out += ["## Gates", "", "| metric | value | limit | |", "|---|---|---|---|"]
        for g in gates:
            out.append(f"| {g['metric']} | {g['value']:.3f} | {g['op']} {g['limit']} | "
                       f"{'pass' if g['pass'] else 'FAIL'} |")
        out.append("")
    return "\n".join(out)


def render_compare(report, baseline):
    out = ["| metric | baseline | current | change |", "|---|---|---|---|"]
    keys = sorted(set(report["metrics"]) | set(baseline.get("metrics", {})))
    for k in keys:
        a = baseline.get("metrics", {}).get(k)
        b = report["metrics"].get(k)
        if a is None or b is None:
            out.append(f"| {k} | {'-' if a is None else f'{a:.3f}'} | "
                       f"{'-' if b is None else f'{b:.3f}'} | |")
        else:
            out.append(f"| {k} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--config", default=None)
    ap.add_argument("--suite", default="all", help=f"comma separated: all,{','.join(SUITES)}")
    ap.add_argument("--quick", action="store_true", help="small version of every suite (CI)")
    ap.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8),
                    help="parallel single-robot trials")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="write the full report as JSON")
    ap.add_argument("--markdown", default=None, help="write the report as markdown")
    ap.add_argument("--compare", default=None, help="baseline JSON report to compare against")
    ap.add_argument("--check", action="store_true", help="exit non-zero if a gate fails")
    args = ap.parse_args()

    suites = SUITES if args.suite == "all" else tuple(s.strip() for s in args.suite.split(","))
    unknown = [s for s in suites if s not in RUNNERS]
    if unknown:
        raise SystemExit(f"unknown suite(s): {', '.join(unknown)} (have: {', '.join(SUITES)})")

    report = run_suites(args.policy, args.config, suites, args.quick, args.jobs, args.seed)
    text = render_markdown(report)
    print()
    print(text)
    if args.compare:
        baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        print("\n## Compared with "
              f"{baseline.get('policy', args.compare)}\n\n{render_compare(report, baseline)}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[eval] json  -> {args.json}")
    if args.markdown:
        Path(args.markdown).write_text(text, encoding="utf-8")
        print(f"[eval] report -> {args.markdown}")
    if args.check:
        failed = [g for g in check_gates(report["metrics"]) if not g["pass"]]
        if failed:
            for g in failed:
                print(f"[eval] FAIL {g['metric']} = {g['value']:.3f}, want {g['op']} {g['limit']}")
            raise SystemExit(1)
        print("[eval] all gates passed")


if __name__ == "__main__":
    main()
