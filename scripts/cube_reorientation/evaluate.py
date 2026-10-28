# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate an RL Games in-hand checkpoint with equal episode quotas per environment.

Example (from this project)::

    bash python.sh scripts/cube_reorientation/evaluate.py --task Isaac-Reorient-Cube-GHand-v0 \
        --checkpoint models/cube_reorientation/teacher/nn/policy.pth --num_envs 64 \
        --episodes_per_env 5 --seed 42 --output runs/evaluation.json --headless

For final evaluation of a curriculum checkpoint, use --relative_goal_probability 0 --tolerance 0.1.
Only the command entry point imports Isaac Sim. Unit tests can import this module.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from ghand.common.config_io import load_config_yaml, restore_config
from ghand.tasks.cube_reorientation.metrics import (
    EpisodeAccumulator,
    apply_reset_mode,
    capture_terminal_successes,
)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes_per_env", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument(
        "--relative_goal_probability",
        type=probability_value,
        default=None,
        help="Override the saved relative-goal probability (0..1); 0 restores full-random goals.",
    )
    parser.add_argument("--saved_env_config", type=Path, default=None)
    parser.add_argument(
        "--reset_mode",
        choices=("saved", "corrected", "legacy"),
        default="saved",
        help="Restore checkpoint reset behavior (default), or explicitly override it.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="Save environment-zero physics poses to NPZ for independent trajectory replay.",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=None,
        help="Save all-environment pre-reward states before goal/episode resets to NPZ.",
    )
    parser.add_argument(
        "--episode_seconds",
        type=float,
        default=None,
        help="Explicit diagnostic episode-duration override; recorded in provenance.",
    )
    parser.add_argument(
        "--stochastic", action="store_true", help="Diagnostic only: sample policy actions instead of using their mean."
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_length", type=int, default=600)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Optional global step limit; exhausted quota is reported as incomplete.",
    )
    return parser


def probability_value(value):
    """Parse a finite probability before starting the simulator."""
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("relative_goal_probability must be finite and between 0 and 1")
    return value


def apply_goal_probability(config, saved, override):
    """Apply an explicit goal-distribution override after saved configuration restoration."""
    supported = hasattr(config, "relative_goal_probability")
    if override is not None:
        if not supported:
            raise ValueError("Selected task does not support relative_goal_probability overrides")
        config.relative_goal_probability = probability_value(override)
    effective = float(getattr(config, "relative_goal_probability", 0.0))
    source = (
        "explicit_override"
        if override is not None
        else "saved_config"
        if supported and "relative_goal_probability" in saved
        else "task_default"
    )
    return {
        "goal_mode": "full_random" if effective == 0.0 else "relative" if effective == 1.0 else "mixed",
        "goal_mode_source": source,
        "saved_relative_goal_probability": saved.get("relative_goal_probability", 0.0),
        "effective_relative_goal_probability": effective,
        "relative_goal_probability_override": override,
    }


def find_saved_env(checkpoint):
    candidates = [checkpoint.parent / "params" / "env.yaml", checkpoint.parent.parent / "params" / "env.yaml"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No saved environment config found: {candidates}; pass --saved_env_config")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class TrajectoryRecorder:
    """Store real simulator poses; this data is independent of RTX rendering."""

    def __init__(self, metadata):
        self.metadata = metadata
        self.frames = []

    def append(self, frame):
        self.frames.append(frame)

    def save(self, path):
        import numpy as np

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: np.asarray(value) for name, value in self.metadata.items()}
        if self.frames:
            arrays.update({name: np.asarray([frame[name] for frame in self.frames]) for name in self.frames[0]})
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)


def trajectory_frame(base):
    """Copy env-zero poses, including auto-reset discontinuities, to CPU memory."""
    frame = {
        "body_pos_w": base.hand.data.body_pos_w[0].detach().cpu().numpy().copy(),
        "body_quat_w": base.hand.data.body_quat_w[0].detach().cpu().numpy().copy(),
        "object_pos_w": base.object.data.root_pos_w[0].detach().cpu().numpy().copy(),
        "object_quat_w": base.object.data.root_quat_w[0].detach().cpu().numpy().copy(),
        "goal_rot": base.goal_rot[0].detach().cpu().numpy().copy(),
    }
    optional = {
        "hand_joint_pos": getattr(base.hand.data, "joint_pos", None),
        "hand_joint_vel": getattr(base.hand.data, "joint_vel", None),
        "hand_joint_pos_target": getattr(base.hand.data, "joint_pos_target", getattr(base, "cur_targets", None)),
        "actions": getattr(base, "actions", None),
    }
    for name, value in optional.items():
        if value is not None:
            frame[name] = value[0].detach().cpu().numpy().copy()
    return frame


def trajectory_metadata(base):
    from pxr import Usd, UsdGeom
    from isaaclab.sim import get_current_stage

    cube_scale = base.cfg.object_cfg.spawn.scale or (1.0, 1.0, 1.0)
    prim_path = base.cfg.object_cfg.prim_path.replace("env_.*", "env_0")
    prim = get_current_stage().GetPrimAtPath(prim_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bounds = cache.ComputeUntransformedBound(prim).ComputeAlignedRange().GetSize()
    return {
        "body_names": base.hand.body_names,
        "hand_joint_names": base.hand.joint_names,
        "env_origin": base.scene.env_origins[0].detach().cpu().numpy().copy(),
        "step_dt": base.step_dt,
        "cube_scale": cube_scale,
        "cube_dimensions": [float(bounds[i]) * float(cube_scale[i]) for i in range(3)],
        "cube_usd_path": base.cfg.object_cfg.spawn.usd_path,
        "quaternion_order": "wxyz",
        "frame_semantics": "initial reset then each post-step; includes automatic episode resets",
    }


def runtime_physics_snapshot(base):
    """Record solver values; authored USD mass can override configured density."""

    def first(value):
        return value[0].detach().cpu().tolist()

    return {
        "sample_environment_index": 0,
        "object_mass_kg": first(base.object.root_physx_view.get_masses()),
        "object_inertia_kg_m2": first(base.object.root_physx_view.get_inertias()),
        "joint_names": list(base.hand.joint_names),
        "joint_velocity_limits_rad_s": first(base.hand.data.joint_vel_limits),
        "joint_effort_limits_Nm": first(base.hand.data.joint_effort_limits),
    }


def diagnostic_frame(base):
    """Snapshot the evaluated target and terminal state before reward resamples goals."""
    values = {
        "joint_pos": base.hand.data.joint_pos,
        "joint_vel": base.hand.data.joint_vel,
        "joint_target": base.cur_targets,
        "actions": base.actions,
        "object_pos": base.object_pos,
        "object_rot": base.object_rot,
        "object_velocities": base.object_velocities,
        "goal_rot": base.goal_rot,
        "fingertip_pos": base.fingertip_pos,
        "episode_step": base.episode_length_buf,
        "successes_before_reward": base.successes,
        "fall": base.reset_terminated,
        "timeout": base.reset_time_outs,
    }
    return {name: value.detach().cpu().numpy().copy() for name, value in values.items()}


def local_file_sha256(path):
    """Hash an existing local asset; remote or missing assets have no local digest."""
    path = Path(path).expanduser()
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def camera_world_pose(viewer, env_origin):
    """Translate the evaluation camera from environment-local to world coordinates."""
    eye = tuple(float(value) + float(offset) for value, offset in zip(viewer.eye, env_origin))
    target = tuple(float(value) + float(offset) for value, offset in zip(viewer.lookat, env_origin))
    return eye, target


def create_evaluation_camera(base):
    """Create a real USD camera without relying on a GUI viewport camera prim."""
    from pxr import Gf, UsdGeom
    from isaaclab.sim import get_current_stage

    viewer = base.cfg.viewer
    origin = base.scene.env_origins[viewer.env_index].detach().cpu().tolist()
    eye, target = camera_world_pose(viewer, origin)
    camera_path = "/World/InHandEvaluationCamera"
    camera = UsdGeom.Camera.Define(get_current_stage(), camera_path)
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(20.955 * viewer.resolution[1] / viewer.resolution[0])
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    camera.MakeMatrixXform().Set(transform)
    viewer.cam_prim_path = camera_path
    # Match Isaac Lab Camera registration so reset waits for textures and steps render sensors.
    base.sim.set_setting("/isaaclab/render/rtx_sensors", True)
    base.sim.set_setting("/exts/omni.replicator.core/Orchestrator/enabled", True)
    return {"prim_path": camera_path, "eye_world": eye, "target_world": target}


def configure_hand_asset(checkpoint):
    """Select the bundle or project USD before importing the hand configuration."""
    import os

    bundle_asset = checkpoint.parent.parent / "assets/ghand5_right.usd" if checkpoint else None
    project_asset = Path(__file__).resolve().parents[2] / "assets/ghand5_right.usd"
    usd_path = bundle_asset if bundle_asset and bundle_asset.is_file() else project_asset
    if usd_path and Path(usd_path).expanduser().is_file():
        os.environ["GHAND_USD_PATH"] = str(Path(usd_path).expanduser().resolve())


def evaluate(args, simulation_app):
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    saved_env = (
        args.saved_env_config.expanduser().resolve(strict=True) if args.saved_env_config else find_saved_env(checkpoint)
    )
    saved_agent = saved_env.with_name("agent.yaml")
    agent_cfg = load_config_yaml(saved_agent)
    saved_env_data = load_config_yaml(saved_env)
    configure_hand_asset(checkpoint)

    # Simulator-dependent imports must follow AppLauncher.
    import gymnasium as gym
    import torch
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
    import isaaclab_tasks  # noqa: F401
    import ghand.tasks.cube_reorientation  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    restore_report = restore_config(env_cfg, {k: v for k, v in saved_env_data.items() if k != "correct_joint_reset"})
    from ghand.common.hand_asset import USD_PATH

    env_cfg.robot_cfg.spawn.usd_path = str(USD_PATH)
    reset_provenance = apply_reset_mode(env_cfg, saved_env_data, args.reset_mode)
    goal_provenance = apply_goal_probability(env_cfg, saved_env_data, args.relative_goal_probability)
    if "correct_joint_reset" in saved_env_data:
        restore_report["restored"].append("correct_joint_reset")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device
    if args.tolerance is not None:
        env_cfg.success_tolerance = args.tolerance
    if args.episode_seconds is not None:
        if not math.isfinite(args.episode_seconds) or args.episode_seconds <= 0:
            raise ValueError("episode_seconds must be positive and finite")
        env_cfg.episode_length_s = args.episode_seconds
    if args.video:
        # Frame environment zero in environment-local coordinates; works at either hand height.
        x, y, z = env_cfg.object_cfg.init_state.pos
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (x + 0.27, y - 0.36, z + 0.29)
        env_cfg.viewer.lookat = (x - 0.08, y + 0.04, z - 0.03)
        env_cfg.viewer.resolution = (960, 720)
    agent_cfg["params"]["seed"] = args.seed
    agent_cfg["params"]["config"]["device"] = env_cfg.sim.device
    agent_cfg["params"]["config"]["device_name"] = env_cfg.sim.device
    agent_cfg["params"]["config"]["num_actors"] = args.num_envs
    agent_cfg["params"]["config"]["multi_gpu"] = False
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(checkpoint)
    agent_cfg["params"]["config"].setdefault("player", {})["deterministic"] = not args.stochastic

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "task": args.task,
        "checkpoint": str(checkpoint),
        **reset_provenance,
        **goal_provenance,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "robot_usd_sha256": local_file_sha256(env_cfg.robot_cfg.spawn.usd_path),
        "saved_env_config": str(saved_env),
        "saved_agent_config": str(saved_agent),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "episodes_per_env": args.episodes_per_env,
        "tolerance": env_cfg.success_tolerance,
        "deterministic_actor": not args.stochastic,
        "success_capture": "post-reward counter before auto-reset",
        "configuration_restore": restore_report,
        "resolved_env_config": env_cfg.to_dict(),
        "resolved_agent_config": agent_cfg,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "video_length_steps": args.video_length if args.video else 0,
        "torch_version": torch.__version__,
        "launcher_experience": args.experience,
        "launcher_kit_args": args.kit_args,
        "trajectory_file": str(args.trajectory.expanduser().resolve()) if args.trajectory else None,
        "diagnostics_file": str(args.diagnostics.expanduser().resolve()) if args.diagnostics else None,
        "episode_seconds_override": args.episode_seconds,
    }
    env = None
    tracker = None
    trajectory = None
    diagnostics = None
    steps = 0
    started = time.monotonic()
    error = None
    stop_reason = "incomplete"
    numerical_events = []
    try:
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        base = env.unwrapped
        provenance["runtime_physics"] = runtime_physics_snapshot(base)
        if not hasattr(base, "successes"):
            raise TypeError("Evaluator requires a direct in-hand environment with a successes counter")
        capture = capture_terminal_successes(base)
        original_reward = base._get_rewards

        if args.diagnostics:
            diagnostics = TrajectoryRecorder(
                {
                    "joint_names": base.hand.joint_names,
                    "active_joint_indices": base.actuated_dof_indices,
                    "lower_limits": base.hand_dof_lower_limits[0].cpu().numpy().copy(),
                    "upper_limits": base.hand_dof_upper_limits[0].cpu().numpy().copy(),
                    "step_dt": base.step_dt,
                    "success_tolerance": base.cfg.success_tolerance,
                    "episodes_per_env": args.episodes_per_env,
                    "frame_semantics": "after physics and done detection, before reward and goal/episode resets",
                }
            )

        def checked_reward():
            if diagnostics is not None:
                frame = diagnostic_frame(base)
                if "raw_policy_mean" in capture:
                    frame["raw_policy_mean"] = capture["raw_policy_mean"].detach().cpu().numpy().copy()
                diagnostics.append(frame)
            reward = original_reward()
            finite = torch.isfinite(reward).reshape(args.num_envs, -1).all(dim=1)
            for state in (
                base.object_pos,
                base.object_rot,
                base.object_velocities,
                base.hand_dof_pos,
                base.hand_dof_vel,
            ):
                finite &= torch.isfinite(state).reshape(args.num_envs, -1).all(dim=1)
            capture["numerical_failures"] = ~finite
            return reward

        base._get_rewards = checked_reward
        tracker = EpisodeAccumulator(args.num_envs, args.episodes_per_env, base.step_dt)
        if args.video:
            provenance["video_camera"] = create_evaluation_camera(base)
            provenance["resolved_env_config"] = base.cfg.to_dict()
            base.render()
            import omni.replicator.core as rep

            rep.orchestrator.step(delta_time=0.0, pause_timeline=False)
            for _ in range(10):
                base.sim.render()
            frame = base._rgb_annotator.get_data()
            if frame.size == 0 or not frame[..., :3].any():
                raise RuntimeError(
                    "Video camera produced no visible RGB data after warmup; refusing to save black video"
                )
            provenance["video_camera"]["initial_rgb_shape"] = list(frame.shape)
            print(f"[EVAL CAMERA] Ready: RGB shape={frame.shape}", flush=True)
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(output.parent / (output.stem + "_videos")),
                step_trigger=lambda step: step == 0,
                video_length=args.video_length,
                disable_logger=True,
            )
        options = agent_cfg["params"].get("env", {})
        env = RlGamesVecEnvWrapper(
            env,
            env_cfg.sim.device,
            options.get("clip_observations", math.inf),
            options.get("clip_actions", math.inf),
            options.get("obs_groups"),
            options.get("concate_obs_groups", True),
        )
        vecenv.register("IsaacRlgWrapper", lambda name, num_actors, **kwargs: RlGamesGpuEnv(name, num_actors, **kwargs))
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
        runner = Runner()
        runner.load(agent_cfg)
        agent = runner.create_player()
        agent.restore(str(checkpoint))
        if diagnostics is not None:

            def capture_policy_mean(module, inputs, result):
                capture["raw_policy_mean"] = result["mus"].detach().clone()

            agent.model.register_forward_hook(capture_policy_mean)
        agent.reset()
        obs = env.reset()
        if args.trajectory:
            trajectory = TrajectoryRecorder(trajectory_metadata(base))
            trajectory.append(trajectory_frame(base))
        policy_obs = obs["obs"] if isinstance(obs, dict) else obs
        agent.get_batch_size(policy_obs, 1)
        if agent.is_rnn:
            agent.init_rnn()

        def finite_rows(value):
            if isinstance(value, dict):
                rows = [finite_rows(item) for item in value.values()]
                return torch.stack(rows).all(dim=0)
            return torch.isfinite(value).reshape(args.num_envs, -1).all(dim=1)

        while simulation_app.is_running() and not tracker.complete:
            if args.max_steps and steps >= args.max_steps:
                stop_reason = "max_steps"
                break
            with torch.inference_mode():
                policy_obs = obs["obs"] if isinstance(obs, dict) else obs
                policy_obs = agent.obs_to_torch(policy_obs)
                obs_finite = finite_rows(policy_obs)
                if not bool(obs_finite.all()):
                    numerical_events.append(
                        {"step": steps, "source": "observation", "env_ids": (~obs_finite).nonzero().flatten().tolist()}
                    )
                    raise FloatingPointError("Non-finite policy observation; evaluation aborted before physics step")
                actions = agent.get_action(policy_obs, is_deterministic=not args.stochastic)
                action_finite = finite_rows(actions)
                if not bool(action_finite.all()):
                    numerical_events.append(
                        {"step": steps, "source": "action", "env_ids": (~action_finite).nonzero().flatten().tolist()}
                    )
                    raise FloatingPointError("Non-finite policy action; evaluation aborted before physics step")
                obs, _, dones, _ = env.step(actions)
                steps += 1
                if trajectory is not None:
                    trajectory.append(trajectory_frame(base))
                bad = capture["numerical_failures"].cpu().tolist()
                tracker.step(
                    capture["successes"].cpu().tolist(),
                    base.reset_terminated.cpu().tolist(),
                    base.reset_time_outs.cpu().tolist(),
                    bad,
                )
                if any(bad):
                    numerical_events.append(
                        {
                            "step": steps,
                            "source": "simulation_state_or_reward",
                            "env_ids": [i for i, value in enumerate(bad) if value],
                        }
                    )
                    raise FloatingPointError("Non-finite simulation state or reward; partial results saved")
                if agent.is_rnn and agent.states is not None:
                    for state in agent.states:
                        state[:, dones.bool(), :] = 0.0
            if steps % 500 == 0:
                print(
                    f"[EVAL] steps={steps} episodes={len(tracker.records)}/{args.num_envs * args.episodes_per_env}",
                    flush=True,
                )
        if tracker.complete:
            stop_reason = "quota_complete"
        elif stop_reason == "incomplete":
            stop_reason = "simulation_closed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        stop_reason = "error"
        raise
    finally:
        if trajectory is not None:
            trajectory.save(args.trajectory.expanduser().resolve())
        if diagnostics is not None:
            diagnostics.save(args.diagnostics.expanduser().resolve())
        result = {
            "schema_version": 1,
            "provenance": provenance,
            "summary": tracker.summary() if tracker else {"complete": False},
            "episodes": tracker.records if tracker else [],
            "stop_reason": stop_reason,
            "error": error,
            "numerical_failure_events": numerical_events,
            "trajectory_frames": len(trajectory.frames) if trajectory is not None else 0,
            "simulation_steps": steps,
            "wall_seconds": time.monotonic() - started,
        }
        # An aborted run is never advertised as a complete evaluation, even if its last event fills a quota.
        result["summary"]["complete"] = stop_reason == "quota_complete"
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n")
        temporary.replace(output)
        print(f"[EVAL] Results saved to {output}", flush=True)
        print(json.dumps(result["summary"], indent=2), flush=True)
        if env is not None:
            env.close()
    if stop_reason != "quota_complete":
        raise RuntimeError(f"Evaluation incomplete: {stop_reason}; partial results saved to {output}")


def configure_video_launcher(args):
    """Enable the renderer dependency absent from some minimal headless experiences."""
    args.enable_cameras = True
    if "omni.kit.viewport.window" not in args.kit_args:
        args.kit_args = (args.kit_args + " --enable omni.kit.viewport.window").strip()


def main():
    from isaaclab.app import AppLauncher

    parser = make_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.episodes_per_env < 1 or args.seed < 0:
        parser.error("num_envs and episodes_per_env must be positive; seed must be nonnegative")
    if args.video_length < 1 or args.max_steps < 0:
        parser.error("video_length must be positive; max_steps must be nonnegative")
    if args.tolerance is not None and (not math.isfinite(args.tolerance) or args.tolerance <= 0):
        parser.error("tolerance must be finite and positive")
    if args.video:
        configure_video_launcher(args)
    # AppLauncher consumes fields from its input mapping; retain CLI values for provenance.
    launcher = AppLauncher(vars(args).copy())
    try:
        evaluate(args, launcher.app)
    except Exception:
        # Some simulator versions exit the process in close(); preserve the actual failure in logs.
        import traceback

        traceback.print_exc()
        raise
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
