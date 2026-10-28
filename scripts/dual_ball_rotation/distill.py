"""Train or evaluate a joint-history student in simulation. No hardware calls.

Online imitation (DAgger): teacher labels visited states, including student
mistakes. Evaluation always uses student actions only. This pilot does not
include calibrated dynamics randomization or PPO fine-tuning.
"""

import argparse
import hashlib
import json
import math
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
parser.add_argument("--iterations", type=int, default=300)
parser.add_argument("--history-steps", type=int, default=30)
parser.add_argument("--rollout-steps", type=int, default=32)
parser.add_argument("--updates", type=int, default=16)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument(
    "--replay-capacity",
    type=int,
    default=262144,
    help="Number of recent teacher-labelled states retained for imitation",
)
parser.add_argument(
    "--expert-capacity",
    type=int,
    default=0,
    help="Immutable early teacher-rollout examples retained to prevent forgetting",
)
parser.add_argument(
    "--validated-handover",
    action="store_true",
    help="Reduce teacher help only after repeated student-only task success",
)
parser.add_argument(
    "--student-exploration-fraction",
    type=float,
    default=0.0,
    help="Minimum student-controlled rollout fraction after warmup; task evaluation gates remain unchanged",
)
parser.add_argument("--min-teacher-iterations", type=int, default=75)
parser.add_argument("--handover-success-rate", type=float, default=0.8)
parser.add_argument(
    "--handover-patience",
    type=int,
    default=6,
    help="Stop after this many evaluations without meaningful improvement or stage progress",
)
parser.add_argument("--eval-every", type=int, default=60)
parser.add_argument("--episodes-per-env", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.expert_capacity < 0:
    parser.error("expert-capacity must be nonnegative")
if not math.isfinite(args.student_exploration_fraction) or not 0 <= args.student_exploration_fraction < 1:
    parser.error("student-exploration-fraction must be in [0,1)")
if (
    args.min_teacher_iterations < 1
    or args.handover_patience < 1
    or not math.isfinite(args.handover_success_rate)
    or not 0 < args.handover_success_rate <= 1
):
    parser.error("Invalid validated handover settings")
if args.student and args.validated_handover:
    parser.error("Validated handover is a training option")
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
from ghand.tasks.dual_ball_rotation.env_cfg import env_config
from ghand.tasks.dual_ball_rotation.registry import TASK, ROOT as PROJECT
from ghand.tasks.dual_ball_rotation.student import JointHistory, StudentPolicy, two_relative_turns
from ghand.tasks.dual_ball_rotation.handover import GatedHandover, rollout_teacher_probability


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def feedback(base):
    ids = base.ids
    # Explicit joint-only interface. No slice of a privileged observation.
    return base.hand.data.joint_pos[:, ids], base.targets


def reset(env, history, seed):
    env.unwrapped.seed(seed)
    obs = env.reset()
    history.reset(*feedback(env.unwrapped))
    return obs


def _evaluate(env, student, history, seed, quota):
    base = env.unwrapped
    reset(env, history, seed)
    counts = [0] * base.num_envs
    records = []
    student.eval()
    with torch.no_grad():
        for step in range(1, (base.max_episode_length + 1) * quota + 1):
            if not simulation_app.is_running():
                break
            action = student(history.observation())
            _, _, done, infos = env.step(action)
            history.advance(*feedback(base), done.bool())
            metrics = infos["rotation_metrics"]
            for index in done.nonzero(as_tuple=False).flatten().cpu().tolist():
                if counts[index] >= quota:
                    continue
                row = {key: value[index].item() for key, value in metrics.items()}
                row.update(env_index=index, episode=counts[index])
                row["meets_30s_two_turns"] = two_relative_turns(row)
                counts[index] += 1
                records.append(row)
            if min(counts) >= quota:
                break
    if min(counts) < quota:
        raise RuntimeError("Student evaluation quota incomplete")
    n = len(records)
    groups = {}
    for label, flag in [("nominal", True), ("randomized", False)]:
        subset = [r for r in records if bool(r["nominal_domain"]) == flag]
        if subset:
            groups[label] = {
                "episodes": len(subset),
                "mean_net_turns": sum(r["net_turns"] for r in subset) / len(subset),
                "drop_rate": sum(r["dropped"] for r in subset) / len(subset),
                "thirty_second_two_relative_turn_rate": sum(r["meets_30s_two_turns"] for r in subset) / len(subset),
            }
    return {
        "policy_phase": "joint_history_student_simulation",
        "fixed_initial_conditions": not base.cfg.domain_randomization,
        "rotation_direction": base.cfg.rotation_direction,
        "domain_randomization": base.cfg.domain_randomization,
        "metric_note": "Task-direction relative pair rotation screen; inspect physical world sign and both balls moving.",
        "teacher_control_fraction": 0.0,
        "seed": seed,
        "complete": True,
        "episode_count": n,
        "episode_seconds": base.cfg.episode_length_s,
        "mean_net_turns": sum(r["net_turns"] for r in records) / n,
        "mean_world_signed_net_turns": sum(r["world_signed_net_turns"] for r in records) / n,
        "mean_reverse_turns": sum(r["reverse_turns"] for r in records) / n,
        "drop_rate": sum(r["dropped"] for r in records) / n,
        "thirty_second_two_relative_turn_rate": sum(r["meets_30s_two_turns"] for r in records) / n,
        "episodes": records,
        "domain_groups": groups,
    }


def evaluate(env, student, history, seed, quota):
    return _evaluate(env, student, history, seed, quota)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(student_path, map_location="cpu", weights_only=False) if args.student else None
    params = (student_path.parent / "params") if checkpoint else teacher_path.parent.parent / "params"
    cfg = env_config()
    restore_config(cfg, load_config_yaml(params / "env.yaml"))
    cfg.robot_cfg.spawn.usd_path = str(USD_PATH)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.device:
        cfg.sim.device = args.device
    if cfg.observation_space != 63 or cfg.action_space != 13 or abs(cfg.episode_length_s - 30.0) > 1e-6:
        raise ValueError("Expected the validated 30-second Dual-ball rotation teacher environment")
    env = gym.make(TASK, cfg=cfg)
    base = env.unwrapped
    try:
        env = RlGamesVecEnvWrapper(env, cfg.sim.device, 5.0, 1.0)
        history_steps = checkpoint["history_steps"] if checkpoint else args.history_steps
        ids = base.ids
        lower = base.lo
        upper = base.hi
        history = JointHistory(base.num_envs, history_steps, lower, upper)
        student = StudentPolicy(history_steps).to(base.device)
        if checkpoint:
            if checkpoint["joint_contract"] != {
                "names": [base.hand.joint_names[i] for i in ids],
                "lower": lower.cpu().tolist(),
                "upper": upper.cpu().tolist(),
            }:
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
            if (
                initial["joint_contract"]
                != {
                    "names": [base.hand.joint_names[i] for i in ids],
                    "lower": lower.cpu().tolist(),
                    "upper": upper.cpu().tolist(),
                }
                or initial["history_steps"] != history_steps
            ):
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
            "observation_size": 26 * history_steps,
            "input": "oldest-to-newest normalized joint angles + applied joint targets; no object state, goal, timer, or phase",
            "history_initialization": "repeat first measured/commanded frame",
            "units": "radians before normalization",
            "step_dt": base.step_dt,
            "action_step_rad": cfg.action_step_rad,
            "joint_contract": {
                "names": [base.hand.joint_names[i] for i in ids],
                "lower": lower.cpu().tolist(),
                "upper": upper.cpu().tolist(),
            },
            "teacher": str(teacher_path),
            "teacher_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
            "hardware_ready": False,
            "rotation_direction": cfg.rotation_direction,
            "actuator_response": None,
            "domain_randomization": {
                "enabled": cfg.domain_randomization,
                "placement_jitter_mm": cfg.placement_jitter_mm,
                "friction_scale_range": cfg.friction_scale_range,
                "nominal_fraction": cfg.nominal_fraction,
                "sampling": "placement per episode; fixed friction per environment (65 buckets), dedicated nominal environments",
            },
            "training": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }
        write_json(run_dir / "contract.json", metadata)
        snapshot = run_dir / "source_snapshot"
        snapshot.mkdir()
        for source in (Path(__file__), *(PROJECT / "source/ghand/tasks/dual_ball_rotation").glob("*.py")):
            shutil.copy2(source, snapshot / source.name)
        optimizer = torch.optim.Adam(student.parameters(), lr=3e-4)
        capacity = max(args.replay_capacity, args.num_envs * args.rollout_steps)
        replay_x = torch.empty((capacity, metadata["observation_size"]), device=base.device)
        replay_y = torch.empty((capacity, 13), device=base.device)
        pointer = count = 0
        expert_count = 0
        if args.expert_capacity:
            expert_x = torch.empty((args.expert_capacity, metadata["observation_size"]), device=base.device)
            expert_y = torch.empty((args.expert_capacity, 13), device=base.device)
        # Every valid task score starts with a nonnegative success rate.
        best_score = (-1.0, -1.0, -1.0)
        gate = (
            GatedHandover(args.min_teacher_iterations, args.handover_success_rate, args.handover_patience)
            if args.validated_handover
            else None
        )
        started = time.monotonic()
        write_json(run_dir / "status.json", {"phase": "training", "hardware_connected": False})
        for iteration in range(1, args.iterations + 1):
            # Preserve the original linear schedule unless explicitly using task gates.
            fraction = iteration / args.iterations
            beta = gate.probability if gate else min(1.0, max(0.0, (0.75 - fraction) / 0.5))
            beta = rollout_teacher_probability(
                beta, iteration, args.min_teacher_iterations, args.student_exploration_fraction
            )
            use_teacher = torch.rand((base.num_envs, 1), device=base.device) < beta
            student.eval()
            with torch.no_grad():
                for _ in range(args.rollout_steps):
                    if not simulation_app.is_running():
                        raise RuntimeError("Simulation closed during training")
                    x = history.observation()
                    privileged = obs["obs"] if isinstance(obs, dict) else obs
                    label = teacher.get_action(teacher.obs_to_torch(privileged), is_deterministic=True).clamp(-1.0, 1.0)
                    action = torch.where(use_teacher, label, student(x))
                    if beta >= 1.0 and expert_count < args.expert_capacity:
                        take = min(base.num_envs, args.expert_capacity - expert_count)
                        expert_x[expert_count : expert_count + take] = x[:take]
                        expert_y[expert_count : expert_count + take] = label[:take]
                        expert_count += take
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
                x_batch, y_batch = replay_x[index], replay_y[index]
                if expert_count:
                    take = max(1, args.batch_size // 4)
                    selected = torch.randint(expert_count, (take,), device=base.device)
                    x_batch[:take], y_batch[:take] = expert_x[selected], expert_y[selected]
                loss = (student(x_batch) - y_batch).square().mean()
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
                "expert_examples": expert_count,
                "wall_seconds": time.monotonic() - started,
            }
            if iteration % args.eval_every == 0 or iteration == args.iterations:
                result = evaluate(env, student, history, 321, 1)
                nominal = result["domain_groups"].get("nominal", result)
                randomized = result["domain_groups"].get("randomized", result)
                rate = randomized["thirty_second_two_relative_turn_rate"]
                nominal_rate = nominal["thirty_second_two_relative_turn_rate"]
                score = (
                    min(rate, nominal_rate),
                    (rate + nominal_rate) / 2,
                    (randomized["mean_net_turns"] + nominal["mean_net_turns"]) / 2,
                )
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
                    write_json(run_dir / "evaluation_nominal_seed321.json", nominal)
                row["evaluation"] = {k: v for k, v in result.items() if k != "episodes"}
                row["nominal_evaluation"] = {k: v for k, v in nominal.items() if k != "episodes"}
                if gate:
                    row["handover"] = gate.observe(iteration, min(rate, nominal_rate))
                obs = reset(env, history, args.seed + iteration)
            with (run_dir / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if iteration % 10 == 0 or "evaluation" in row:
                print(json.dumps(row, allow_nan=False), flush=True)
                write_json(
                    run_dir / "status.json",
                    {
                        "phase": "training",
                        "iteration": iteration,
                        "wall_seconds": time.monotonic() - started,
                        "hardware_connected": False,
                        "hardware_ready": False,
                        "best_seed321_score": best_score,
                        "handover": gate.state() if gate else None,
                    },
                )
            if gate and gate.stop_reason:
                break
        write_json(
            run_dir / "status.json",
            {
                "phase": "finished",
                "iterations": iteration,
                "requested_iterations": args.iterations,
                "wall_seconds": time.monotonic() - started,
                "hardware_connected": False,
                "hardware_ready": False,
                "best_seed321_score": best_score,
                "stop_reason": gate.stop_reason if gate and gate.stop_reason else "iteration_limit",
                "handover": gate.state() if gate else None,
            },
        )
    finally:
        env.close()


exit_code = 0
try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    exit_code = 1
    raise
finally:
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True, exit_code=exit_code)
