"""Evaluate the spin teacher, or smoke-test real physics without a checkpoint."""

import argparse
import json
import math
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes_per_env", type=int, default=2)
parser.add_argument("--episode_seconds", type=float)
parser.add_argument("--seed", type=int, default=321)
parser.add_argument("--max_steps", type=int, default=10000)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--video", action="store_true", help="Record environment zero to an MP4 beside the evaluation JSON")
parser.add_argument("--video_length", type=int, default=600)
parser.add_argument("--output", type=Path, required=True)
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
if args.num_envs < 1 or args.episodes_per_env < 1 or args.max_steps < 1 or args.video_length < 1:
    parser.error("Environment count, episode quota, and max_steps must be positive")
if not args.smoke and args.checkpoint is None:
    parser.error("Supply --checkpoint, or --smoke for physics validation")
if args.episode_seconds is not None and (not math.isfinite(args.episode_seconds) or args.episode_seconds <= 0):
    parser.error("episode_seconds must be finite and positive")
checkpoint = args.checkpoint.expanduser().resolve(strict=True) if args.checkpoint else None
output = args.output.expanduser().resolve()
if args.video:
    from evaluate import configure_video_launcher, create_evaluation_camera

    configure_video_launcher(args)
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from ghand.common.config_io import load_config_yaml, restore_config
from ghand.tasks.cube_spin.registry import TASK, PROJECT
from ghand.tasks.cube_spin.env_cfg import env_config
from ghand.common.hand_asset import USD_PATH


def main():
    cfg = env_config()
    agent_cfg = None
    if checkpoint:
        params = checkpoint.parent.parent / "params"
        restore_config(cfg, load_config_yaml(params / "env.yaml"))
        agent_cfg = load_config_yaml(params / "agent.yaml")
        agent_cfg["params"]["seed"] = args.seed
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
    if args.episode_seconds is not None:
        cfg.episode_length_s = args.episode_seconds
    elif args.smoke:
        cfg.episode_length_s = 3.0
    if args.video:
        cfg.viewer.resolution = (960, 720)
        cfg.viewer.env_index = 0
        cfg.viewer.eye = (0.24, -0.36, 0.80)
        cfg.viewer.lookat = (-0.025, -0.095, 0.52)
    env = gym.make(TASK, cfg=cfg, render_mode="rgb_array" if args.video else None)
    base = env.unwrapped
    records = []
    counts = [0] * args.num_envs
    started = time.monotonic()
    max_mimic_error = 0.0
    camera = None
    try:
        if args.video:
            camera = create_evaluation_camera(base)
            base.render()
            import omni.replicator.core as rep

            rep.orchestrator.step(delta_time=0.0, pause_timeline=False)
            for _ in range(10):
                base.sim.render()
            frame = base._rgb_annotator.get_data()
            if frame.size == 0 or not frame[..., :3].any():
                raise RuntimeError("Evaluation camera returned no visible RGB data")
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(output.parent / (output.stem + "_videos")),
                step_trigger=lambda step: step == 0,
                video_length=args.video_length,
                disable_logger=True,
            )
        actor = None
        if agent_cfg:
            options = agent_cfg["params"].get("env", {})
            env = RlGamesVecEnvWrapper(
                env, cfg.sim.device, options.get("clip_observations", 5.0), options.get("clip_actions", 1.0)
            )
            vecenv.register(
                "IsaacRlgWrapper", lambda name, num_actors, **kwargs: RlGamesGpuEnv(name, num_actors, **kwargs)
            )
            env_configurations.register(
                "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
            )
            runner = Runner()
            runner.load(agent_cfg)
            actor = runner.create_player()
            actor.restore(str(checkpoint))
            actor.reset()
            base.seed(args.seed)
            obs = env.reset()
            actor.get_batch_size(obs["obs"] if isinstance(obs, dict) else obs, 1)
        else:
            base.seed(args.seed)
            obs, _ = env.reset()
        assert base.cfg.observation_space == 68
        assert len(base.actuated_dof_indices) == 13
        assert abs(base.step_dt - 0.05) < 1e-9
        cube_mass = float(base.object.root_physx_view.get_masses()[0, 0])
        assert math.isfinite(cube_mass) and cube_mass > 0, cube_mass
        steps = 0
        for steps in range(1, args.max_steps + 1):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                if actor:
                    policy_obs = obs["obs"] if isinstance(obs, dict) else obs
                    actions = actor.get_action(actor.obs_to_torch(policy_obs), is_deterministic=True)
                    obs, reward, done, infos = env.step(actions)
                else:
                    actions = torch.zeros((args.num_envs, 13), device=base.device)
                    obs, reward, terminated, truncated, infos = env.step(actions)
                    done = terminated | truncated
                policy_obs = obs.get("policy", obs.get("obs")) if isinstance(obs, dict) else obs
                assert policy_obs.shape == (args.num_envs, 68), policy_obs.shape
                assert bool(torch.isfinite(policy_obs).all() & torch.isfinite(reward).all())
                ids = base.actuated_dof_indices
                assert bool((base.cur_targets[:, ids] >= base.hand_dof_lower_limits[:, ids] - 1e-6).all())
                assert bool((base.cur_targets[:, ids] <= base.hand_dof_upper_limits[:, ids] + 1e-6).all())
                q = base.hand.data.joint_pos
                residual = max(float((q[:, c] - a * q[:, p] - b).abs().max()) for c, p, a, b in base._mimic_indices)
                max_mimic_error = max(max_mimic_error, residual)
                snapshot = infos["spin_metrics"]
                for index in done.nonzero(as_tuple=False).flatten().cpu().tolist():
                    if counts[index] >= args.episodes_per_env:
                        continue
                    row = {key: value[index].item() for key, value in snapshot.items()}
                    row["env_index"] = index
                    row["episode"] = counts[index]
                    row["meets_30s_two_turns"] = (
                        not row["dropped"] and row["seconds"] >= 30.0 - base.step_dt - 1e-5 and row["net_turns"] >= 2.0
                    )
                    records.append(row)
                    counts[index] += 1
                if min(counts) >= args.episodes_per_env:
                    break
        n = len(records)
        summary = {
            "task": TASK,
            "policy_phase": "privileged_teacher_not_deployable",
            "video_camera": camera,
            "video_length_steps": args.video_length if args.video else 0,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "spin_axis": cfg.spin_axis,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "episodes_per_env": args.episodes_per_env,
            "episode_seconds": cfg.episode_length_s,
            "step_dt": base.step_dt,
            "cube_mass_kg_first_env": cube_mass,
            "cube_size_range_m": cfg.cube_size_range_m,
            "cube_mass_range_kg": cfg.cube_mass_range_kg,
            "joint_contract": base.joint_contract,
            "cube_static_friction_range": cfg.cube_static_friction_range,
            "cube_dynamic_friction_ratio_range": cfg.cube_dynamic_friction_ratio_range,
            "warmup_excluded_seconds": cfg.spin_warmup_s,
            "complete": min(counts) >= args.episodes_per_env,
            "episode_count": n,
            "mean_net_turns": sum(r["net_turns"] for r in records) / n if n else None,
            "mean_reverse_turns": sum(r["reverse_turns"] for r in records) / n if n else None,
            "drop_rate": sum(r["dropped"] for r in records) / n if n else None,
            "thirty_second_two_turn_success_rate": sum(r["meets_30s_two_turns"] for r in records) / n if n else None,
            "max_mimic_error_rad": max_mimic_error,
            "steps": steps,
            "wall_seconds": time.monotonic() - started,
            "episodes": records,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps({k: v for k, v in summary.items() if k not in ("episodes", "joint_contract")}, indent=2),
            flush=True,
        )
        if not summary["complete"]:
            raise RuntimeError("Episode quota incomplete; partial results saved")
    finally:
        env.close()


try:
    main()
finally:
    simulation_app.close()
