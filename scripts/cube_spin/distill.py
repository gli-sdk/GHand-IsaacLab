"""Train or evaluate a joint-history student in simulation. No hardware calls.

Online imitation (DAgger): teacher labels visited states, including student
mistakes. Evaluation always uses student actions only. This pilot does not
include calibrated dynamics randomization or PPO fine-tuning.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher", type=Path)
parser.add_argument("--init-student", type=Path, help="Warm-start training from a compatible student checkpoint")
parser.add_argument("--student", type=Path, help="Evaluation only; never executes the teacher")
parser.add_argument("--run-dir", type=Path)
parser.add_argument("--eval-output", type=Path)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--iterations", type=int, default=120)
parser.add_argument("--history-steps", type=int, default=30)
parser.add_argument("--rollout-steps", type=int, default=32)
parser.add_argument("--updates", type=int, default=16)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument(
    "--replay-capacity", type=int, default=65536, help="Number of recent teacher-labelled states retained for imitation"
)
parser.add_argument("--eval-every", type=int, default=60)
parser.add_argument("--episodes-per-env", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--randomize-cube", action="store_true", help="Override with 35–40 mm / 30–50 g independent randomization"
)
parser.add_argument("--cube-size-mm", type=float, help="Fixed evaluation size; also supply --cube-mass-g")
parser.add_argument("--cube-mass-g", type=float, help="Fixed evaluation mass; also supply --cube-size-mm")
friction = parser.add_mutually_exclusive_group()
friction.add_argument(
    "--randomize-friction",
    action="store_true",
    help="Pair static friction 0.5–1.5, dynamic/static ratio 0.7–1.0 per episode",
)
friction.add_argument(
    "--cube-friction",
    type=float,
    nargs=2,
    metavar=("STATIC", "DYNAMIC"),
    help="Fixed hand–cube friction coefficients, 0 < dynamic <= static",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.cube_friction:
    import math

    static, dynamic = args.cube_friction
    if not all(math.isfinite(v) and v > 0 for v in (static, dynamic)) or dynamic > static:
        parser.error("Friction requires finite 0 < dynamic <= static")
if (args.cube_size_mm is None) != (args.cube_mass_g is None):
    parser.error("Supply both --cube-size-mm and --cube-mass-g")
if args.randomize_cube and args.cube_size_mm is not None:
    parser.error("Choose fixed or randomized cube properties")
for name in (
    "num_envs",
    "iterations",
    "history_steps",
    "rollout_steps",
    "updates",
    "batch_size",
    "replay_capacity",
    "eval_every",
    "episodes_per_env",
):
    if getattr(args, name) < 1:
        parser.error(name + " must be positive")
if args.student:
    if not args.eval_output or args.teacher or args.run_dir or args.init_student:
        parser.error("Evaluation requires --student and --eval-output only")
    student_path = args.student.expanduser().resolve(strict=True)
else:
    if not args.teacher or not args.run_dir or args.eval_output:
        parser.error("Training requires --teacher and a new --run-dir")
    teacher_path = args.teacher.expanduser().resolve(strict=True)
    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.exists():
        parser.error("Use a new run directory; existing results are never overwritten")
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab.utils.io import dump_yaml
from ghand.common.config_io import load_config_yaml, restore_config
from ghand.common.hand_asset import USD_PATH
from ghand.tasks.cube_spin.env_cfg import env_config
from ghand.tasks.cube_spin.registry import TASK, PROJECT
from ghand.tasks.cube_spin.student import JointHistory, StudentPolicy


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def feedback(base):
    ids = base.actuated_dof_indices
    # Explicit joint-only interface. No slice of a privileged observation.
    return base.hand_dof_pos[:, ids], base.cur_targets[:, ids]


def reset(env, history, seed):
    env.unwrapped.seed(seed)
    obs = env.reset()
    history.reset(*feedback(env.unwrapped))
    return obs


def evaluate(env, student, history, seed, quota):
    base = env.unwrapped
    reset(env, history, seed)
    counts = [0] * base.num_envs
    records = []
    student.eval()
    with torch.no_grad():
        for step in range(1, (base.max_episode_length + 1) * quota + 1):
            if not simulation_app.is_running():
                break
            action = student(history.observation(base.cfg.target_speed))
            _, _, done, infos = env.step(action)
            history.advance(*feedback(base), done.bool())
            metrics = infos["spin_metrics"]
            for index in done.nonzero(as_tuple=False).flatten().cpu().tolist():
                if counts[index] >= quota:
                    continue
                row = {key: value[index].item() for key, value in metrics.items()}
                row.update(env_index=index, episode=counts[index])
                row["meets_30s_two_turns"] = (
                    not row["dropped"] and row["seconds"] >= 30.0 - base.step_dt - 1e-5 and row["net_turns"] >= 2.0
                )
                counts[index] += 1
                records.append(row)
            if min(counts) >= quota:
                break
    if min(counts) < quota:
        raise RuntimeError("Student evaluation quota incomplete")
    n = len(records)
    return {
        "policy_phase": "joint_history_student_simulation",
        "spin_axis": base.cfg.spin_axis,
        "cube_size_range_m": base.cfg.cube_size_range_m,
        "cube_mass_range_kg": base.cfg.cube_mass_range_kg,
        "cube_static_friction_range": base.cfg.cube_static_friction_range,
        "cube_dynamic_friction_ratio_range": base.cfg.cube_dynamic_friction_ratio_range,
        "teacher_control_fraction": 0.0,
        "seed": seed,
        "complete": True,
        "episode_count": n,
        "episode_seconds": base.cfg.episode_length_s,
        "mean_net_turns": sum(r["net_turns"] for r in records) / n,
        "mean_reverse_turns": sum(r["reverse_turns"] for r in records) / n,
        "mean_off_axis_radians": sum(r["off_axis_radians"] for r in records) / n,
        "drop_rate": sum(r["dropped"] for r in records) / n,
        "thirty_second_two_turn_success_rate": sum(r["meets_30s_two_turns"] for r in records) / n,
        "episodes": records,
    }


def main():
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(student_path, map_location="cpu", weights_only=False) if args.student else None
    params = (student_path.parent / "params") if checkpoint else teacher_path.parent.parent / "params"
    cfg = env_config()
    restore_config(cfg, load_config_yaml(params / "env.yaml"))
    cfg.robot_cfg.spawn.usd_path = str(USD_PATH)
    cfg.hardware_config = str(PROJECT / "config/cube_spin/hand.json")
    if args.randomize_cube:
        cfg.cube_size_range_m = (0.035, 0.040)
        cfg.cube_mass_range_kg = (0.030, 0.050)
    elif args.cube_size_mm is not None:
        cfg.cube_size_range_m = (args.cube_size_mm / 1000.0,) * 2
        cfg.cube_mass_range_kg = (args.cube_mass_g / 1000.0,) * 2
    if args.randomize_friction:
        cfg.cube_static_friction_range = (0.5, 1.5)
        cfg.cube_dynamic_friction_ratio_range = (0.7, 1.0)
    elif args.cube_friction:
        static, dynamic = args.cube_friction
        cfg.cube_static_friction_range = (static, static)
        cfg.cube_dynamic_friction_ratio_range = (dynamic / static,) * 2
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.device:
        cfg.sim.device = args.device
    if cfg.observation_space != 68 or cfg.action_space != 13 or abs(cfg.episode_length_s - 30.0) > 1e-6:
        raise ValueError("Expected the validated 30-second continuous-spin teacher environment")
    env = gym.make(TASK, cfg=cfg)
    base = env.unwrapped
    try:
        env = RlGamesVecEnvWrapper(env, cfg.sim.device, 5.0, 1.0)
        history_steps = checkpoint["history_steps"] if checkpoint else args.history_steps
        ids = base.actuated_dof_indices
        lower = base.hand_dof_lower_limits[0, ids]
        upper = base.hand_dof_upper_limits[0, ids]
        history = JointHistory(base.num_envs, history_steps, lower, upper)
        student = StudentPolicy(history_steps).to(base.device)
        if checkpoint:
            if checkpoint["joint_contract"] != base.joint_contract:
                raise ValueError("Saved student and current hand have different joint contracts")
            student.load_state_dict(checkpoint["model"])
            result = evaluate(env, student, history, args.seed, args.episodes_per_env)
            result["checkpoint"] = str(student_path)
            write_json(args.eval_output.expanduser().resolve(), result)
            print(json.dumps({k: v for k, v in result.items() if k != "episodes"}), flush=True)
            return

        if args.init_student:
            initial = torch.load(
                args.init_student.expanduser().resolve(strict=True), map_location=base.device, weights_only=False
            )
            if initial["joint_contract"] != base.joint_contract or initial["history_steps"] != history_steps:
                raise ValueError("Warm-start student must use the same joints and history length")
            student.load_state_dict(initial["model"])

        agent_cfg = load_config_yaml(params / "agent.yaml")
        agent_cfg["params"]["seed"] = args.seed
        vecenv.register("IsaacRlgWrapper", lambda name, num_actors, **kw: RlGamesGpuEnv(name, num_actors, **kw))
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
        runner = Runner()
        runner.load(agent_cfg)
        teacher = runner.create_player()
        teacher.restore(str(teacher_path))
        teacher.reset()
        teacher.model.eval()
        for parameter in teacher.model.parameters():
            parameter.requires_grad_(False)
        obs = reset(env, history, args.seed)
        teacher.get_batch_size(obs["obs"] if isinstance(obs, dict) else obs, 1)
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "params").mkdir()
        dump_yaml(str(run_dir / "params/env.yaml"), cfg)
        metadata = {
            "schema_version": 1,
            "history_steps": history_steps,
            "frame_width": 26,
            "observation_size": 26 * history_steps + 1,
            "input": "oldest-to-newest normalized joint angles + applied joint targets; then speed command",
            "history_initialization": "repeat first measured/commanded frame",
            "units": "radians before normalization",
            "step_dt": base.step_dt,
            "action_step_rad": cfg.action_step_rad,
            "target_speed": cfg.target_speed,
            "spin_axis": cfg.spin_axis,
            "joint_contract": base.joint_contract,
            "command_upper_deg": cfg.command_upper_deg,
            "command_margin_fraction": cfg.command_margin_fraction,
            "teacher": str(teacher_path),
            "teacher_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
            "hardware_ready": False,
            "domain_randomization": {
                "cube_size_range_m": cfg.cube_size_range_m,
                "cube_mass_range_kg": cfg.cube_mass_range_kg,
                "size_sampling": "independent uniform per environment at startup",
                "mass_sampling": "independent uniform per episode reset",
                "inertia": "uniform cube: mass * edge_squared / 6",
                "cube_static_friction_range": cfg.cube_static_friction_range,
                "cube_dynamic_friction_ratio_range": cfg.cube_dynamic_friction_ratio_range,
                "friction_sampling": "per-episode bucket, cube multiply / hand 1; hand self-friction unchanged"
                if cfg.cube_static_friction_range
                else None,
                "friction_bucket_count": cfg.friction_bucket_count if cfg.cube_static_friction_range else None,
                "motor_dynamics_randomized": False,
            },
            "training": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }
        write_json(run_dir / "contract.json", metadata)
        snapshot = run_dir / "source_snapshot"
        snapshot.mkdir()
        for source in (Path(__file__), PROJECT / "source/ghand/tasks/cube_spin/student.py"):
            shutil.copy2(source, snapshot / source.name)
        optimizer = torch.optim.Adam(student.parameters(), lr=3e-4)
        capacity = max(args.replay_capacity, args.num_envs * args.rollout_steps)
        replay_x = torch.empty((capacity, metadata["observation_size"]), device=base.device)
        replay_y = torch.empty((capacity, 13), device=base.device)
        pointer = count = 0
        best_score = (-1.0, -float("inf"))
        started = time.monotonic()
        write_json(run_dir / "status.json", {"phase": "training", "hardware_connected": False})
        for iteration in range(1, args.iterations + 1):
            # First quarter teacher rollouts, next half anneal to full student.
            fraction = iteration / args.iterations
            beta = min(1.0, max(0.0, (0.75 - fraction) / 0.5))
            use_teacher = torch.rand((base.num_envs, 1), device=base.device) < beta
            student.eval()
            with torch.no_grad():
                for _ in range(args.rollout_steps):
                    if not simulation_app.is_running():
                        raise RuntimeError("Simulation closed during training")
                    x = history.observation(cfg.target_speed)
                    privileged = obs["obs"] if isinstance(obs, dict) else obs
                    label = teacher.get_action(teacher.obs_to_torch(privileged), is_deterministic=True).clamp(-1.0, 1.0)
                    action = torch.where(use_teacher, label, student(x))
                    index = (torch.arange(base.num_envs, device=base.device) + pointer) % capacity
                    replay_x[index], replay_y[index] = x, label
                    pointer = (pointer + base.num_envs) % capacity
                    count = min(capacity, count + base.num_envs)
                    obs, _, done, _ = env.step(action)
                    history.advance(*feedback(base), done.bool())
            student.train()
            loss_total = 0.0
            for _ in range(args.updates):
                index = torch.randint(count, (args.batch_size,), device=base.device)
                loss = (student(replay_x[index]) - replay_y[index]).square().mean()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite student loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                loss_total += loss.item()
            row = {
                "iteration": iteration,
                "teacher_control_probability": beta,
                "action_mse": loss_total / args.updates,
                "wall_seconds": time.monotonic() - started,
            }
            if iteration % args.eval_every == 0 or iteration == args.iterations:
                result = evaluate(env, student, history, 321, 1)
                score = (result["thirty_second_two_turn_success_rate"], result["mean_net_turns"])
                if score > best_score:
                    best_score = score
                    state = {
                        **metadata,
                        "model": {k: v.detach().cpu() for k, v in student.state_dict().items()},
                        "iteration": iteration,
                        "evaluation_seed321": {k: v for k, v in result.items() if k != "episodes"},
                    }
                    torch.save(state, run_dir / "student.pt")
                    exported = StudentPolicy(history_steps).eval()
                    exported.load_state_dict(state["model"])
                    torch.jit.script(exported).save(str(run_dir / "actor.ts"))
                    write_json(run_dir / "evaluation_seed321.json", result)
                row["evaluation"] = {k: v for k, v in result.items() if k != "episodes"}
                obs = reset(env, history, args.seed + iteration)
            with (run_dir / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if iteration % 10 == 0 or "evaluation" in row:
                print(json.dumps(row, allow_nan=False), flush=True)
        write_json(
            run_dir / "status.json",
            {
                "phase": "finished",
                "iterations": args.iterations,
                "wall_seconds": time.monotonic() - started,
                "hardware_connected": False,
                "hardware_ready": False,
                "best_seed321_score": best_score,
            },
        )
    finally:
        env.close()


try:
    main()
finally:
    simulation_app.close()
