"""GUI policy viewer or headless episode evaluation; no hardware interface."""

import argparse
import hashlib
import math
import json
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", type=Path)
p.add_argument("--student", type=Path, help="Joint-history policy; no privileged actor inputs")
p.add_argument("--hold", action="store_true")
p.add_argument(
    "--tilt-range-deg",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    help="Forward palm inclination, sampled once per episode",
)
p.add_argument(
    "--side-tilt-deg", type=float, default=None, help="Palm side inclination; negative lowers right-hand thumb side"
)
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--episodes", type=int, default=1)
p.add_argument("--seconds", type=float, default=None)
p.add_argument(
    "--ball-diameter-mm",
    type=float,
    default=None,
    help="Size comparison: also preserve bottom heights and goal-orbit surface clearance; mass unchanged",
)
p.add_argument("--ball-mass-g", type=float, default=None, help="Optional per-ball mass override in grams")
p.add_argument("--seed", type=int, default=123)
p.add_argument("--domain", choices=("saved", "nominal", "randomized"), default="saved")
p.add_argument(
    "--placement-jitter-mm", type=float, default=None, help="Common shift of both balls, uniform +/- bound per XY axis"
)
p.add_argument("--perturb-seed", type=int, default=20261019)
p.add_argument("--friction", type=float, nargs=2, metavar=("STATIC", "DYNAMIC"))
p.add_argument("--output", type=Path, default=Path("results/view_latest.json"))
p.add_argument("--snapshot", type=Path)
p.add_argument("--video", type=Path, help="Record real simulator frames at the control rate (1x playback)")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if sum((bool(a.checkpoint), bool(a.student), a.hold)) != 1:
    p.error("Choose one of --checkpoint, --student, --hold")
if a.num_envs < 1 or a.episodes < 1:
    p.error("Counts must be positive")
if a.ball_diameter_mm is not None and (not math.isfinite(a.ball_diameter_mm) or a.ball_diameter_mm <= 0):
    p.error("Ball diameter must be finite and positive")
if a.ball_mass_g is not None and (not math.isfinite(a.ball_mass_g) or a.ball_mass_g <= 0):
    p.error("Ball mass must be finite and positive")
if a.placement_jitter_mm is not None and (not math.isfinite(a.placement_jitter_mm) or a.placement_jitter_mm < 0):
    p.error("Invalid placement bound")
if a.friction and (not all(math.isfinite(x) and x > 0 for x in a.friction) or a.friction[1] > a.friction[0]):
    p.error("Require finite 0 < dynamic friction <= static friction")
if a.snapshot or a.video:
    a.enable_cameras = True
    a.kit_args = (a.kit_args + " --enable omni.kit.viewport.window").strip()
app = AppLauncher(a).app
import gymnasium as gym
import torch
from ghand.tasks.dual_ball_rotation.registry import TASK
from ghand.tasks.dual_ball_rotation.env_cfg import env_config
from ghand.common.config_io import load_config_yaml, restore_config
from ghand.common.hand_asset import USD_PATH
from ghand.tasks.dual_ball_rotation.student import JointHistory, StudentPolicy, two_relative_turns
from ghand.tasks.dual_ball_rotation.perturbations import placement_offsets, resized_ball_geometry


def main():
    cfg = env_config()
    agent = None
    if a.checkpoint:
        params = a.checkpoint.resolve().parent.parent / "params"
        restore_config(cfg, load_config_yaml(params / "env.yaml"))
        agent = load_config_yaml(params / "agent.yaml")
    if a.student:
        restore_config(cfg, load_config_yaml(a.student.resolve().parent / "params/env.yaml"))
    cfg.robot_cfg.spawn.usd_path = str(USD_PATH)
    if a.tilt_range_deg is not None:
        cfg.palm_tilt_range_deg = tuple(a.tilt_range_deg)
    if a.side_tilt_deg is not None:
        cfg.palm_side_tilt_deg = a.side_tilt_deg
    cfg.scene.num_envs = a.num_envs
    if a.domain != "saved":
        cfg.domain_randomization = a.domain == "randomized"
    if a.placement_jitter_mm is not None or a.friction:
        if a.domain == "randomized":
            raise ValueError("Use saved/randomized domains or fixed perturbations, not both")
        cfg.domain_randomization = False
    offsets = placement_offsets(a.num_envs, a.placement_jitter_mm or 0.0, a.perturb_seed)
    cfg.initial_xy_offsets_m = offsets.tolist() if a.placement_jitter_mm else None
    if a.friction:
        cfg.ball_static_friction, cfg.ball_dynamic_friction = a.friction
    cfg.ball_radius, cfg.orbit_radii, cfg.initial_ball_heights = resized_ball_geometry(
        cfg.ball_radius, cfg.orbit_radii, cfg.initial_ball_heights, a.ball_diameter_mm
    )
    if a.ball_mass_g is not None:
        cfg.ball_mass = a.ball_mass_g / 1000.0
    cfg.seed = a.seed
    if a.seconds is not None:
        cfg.episode_length_s = a.seconds
    if a.device:
        cfg.sim.device = a.device
    env = gym.make(TASK, cfg=cfg, render_mode="rgb_array" if (a.snapshot or a.video) else None)
    base = env.unwrapped
    records = []
    counts = [0] * a.num_envs
    trace = []
    max_mimic = 0.0
    min_sep = 1.0
    min_z = 1.0
    max_speed = 0.0
    # Episode metrics exclude the first two seconds of settling. No reward changes here.
    previous_net = torch.zeros(a.num_envs, device=base.device)
    samples = torch.zeros_like(previous_net)
    slow_samples = torch.zeros_like(previous_net)
    reverse_samples = torch.zeros_like(previous_net)
    slow_run = torch.zeros_like(previous_net)
    longest_slow = torch.zeros_like(previous_net)
    writer = None
    video_frames = 0
    try:
        actor = None
        student = None
        history = None
        if agent:
            from rl_games.common import env_configurations, vecenv
            from rl_games.torch_runner import Runner
            from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

            opts = agent["params"].get("env", {})
            env = RlGamesVecEnvWrapper(
                env, cfg.sim.device, opts.get("clip_observations", 5.0), opts.get("clip_actions", 1.0)
            )
            vecenv.register("IsaacRlgWrapper", lambda name, num_actors, **kw: RlGamesGpuEnv(name, num_actors, **kw))
            env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
            runner = Runner()
            runner.load(agent)
            actor = runner.create_player()
            actor.restore(str(a.checkpoint.resolve()))
            actor.reset()
            obs = env.reset()
            actor.get_batch_size(obs["obs"] if isinstance(obs, dict) else obs, 1)
        else:
            obs, _ = env.reset()
        if a.student:
            saved = torch.load(a.student, map_location="cpu", weights_only=False)
            contract = {
                "names": [base.hand.joint_names[i] for i in base.ids],
                "lower": base.lo.cpu().tolist(),
                "upper": base.hi.cpu().tolist(),
            }
            if (
                saved["joint_contract"] != contract
                or abs(saved["step_dt"] - base.step_dt) > 1e-8
                or abs(saved["action_step_rad"] - cfg.action_step_rad) > 1e-8
            ):
                raise ValueError("Student control contract does not match environment")
            student = StudentPolicy(saved["history_steps"]).to(base.device).eval()
            student.load_state_dict(saved["model"])
            history = JointHistory(base.num_envs, saved["history_steps"], base.lo, base.hi)
            history.reset(base.hand.data.joint_pos[:, base.ids], base.targets)
        assert len(base.ids) == 13 and abs(base.step_dt - 0.05) < 1e-8
        measured_materials = []
        for b in base.balls:
            props = b.root_physx_view.get_material_properties()
            expected = (
                props.new_tensor([cfg.ball_static_friction, cfg.ball_dynamic_friction, 0.0])[None, None]
                * base.friction_scales[:, None, None]
            )
            assert torch.allclose(props, expected.expand_as(props), atol=1e-6), "Friction not applied to physics"
            measured_materials.append(props[0].tolist())
            assert abs(float(b.root_physx_view.get_masses()[0, 0]) - cfg.ball_mass) < 1e-6
        assert ((base.pos[:, 0] - base.pos[:, 1]).norm(dim=-1) >= 2 * cfg.ball_radius - 1e-6).all(), (
            "Initial sphere overlap"
        )
        from pxr import UsdGeom

        physics_radii = [
            float(
                UsdGeom.Sphere(base.sim.stage.GetPrimAtPath(f"/World/envs/env_0/Ball{i}/geometry/mesh"))
                .GetRadiusAttr()
                .Get()
            )
            for i in range(2)
        ]
        assert all(abs(radius - cfg.ball_radius) < 1e-8 for radius in physics_radii), "Sphere radius not applied"
        initial_surface_gap = float(((base.pos[:, 0] - base.pos[:, 1]).norm(dim=-1) - 2 * cfg.ball_radius).min())
        initial = base.pos[0].cpu().tolist()
        initial_all = base.pos.cpu().tolist()
        initial_offsets = base.episode_offsets.cpu().tolist()
        initial_scales = base.friction_scales.tolist()
        expected = torch.tensor([cfg.orbit_center[:2], cfg.orbit_center[:2]], device=base.device)
        expected[0, 0] += cfg.orbit_radii[0]
        expected[1, 0] -= cfg.orbit_radii[0]
        assert torch.allclose(base.pos[..., :2], expected[None] + base.episode_offsets[:, None], atol=1e-6), (
            "Initial perturbation not applied"
        )
        if a.snapshot or a.video:
            from pxr import Gf, UsdGeom

            camera = UsdGeom.Camera.Define(base.sim.stage, "/World/EvaluationCamera")
            camera.CreateFocalLengthAttr(24.0)
            camera.CreateHorizontalApertureAttr(20.955)
            camera.CreateVerticalApertureAttr(15.716)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
            origin = base.scene.env_origins[0].cpu().tolist()
            eye = tuple(v + o for v, o in zip(cfg.viewer.eye, origin))
            look = tuple(v + o for v, o in zip(cfg.viewer.lookat, origin))
            camera.MakeMatrixXform().Set(
                Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(0, 0, 1)).GetInverse()
            )
            base.cfg.viewer.cam_prim_path = "/World/EvaluationCamera"
            base.sim.set_setting("/isaaclab/render/rtx_sensors", True)
            base.sim.set_setting("/exts/omni.replicator.core/Orchestrator/enabled", True)
            base.render()
            import omni.replicator.core as rep

            rep.orchestrator.step(delta_time=0.0, pause_timeline=False)
            for _ in range(10):
                base.sim.render()
            import imageio.v3 as iio

            frame = base.render()
            if frame is None or not frame.any():
                raise RuntimeError("No RGB frame")
            if a.snapshot:
                a.snapshot.parent.mkdir(parents=True, exist_ok=True)
                iio.imwrite(a.snapshot, frame)
            if a.video:
                import imageio.v2 as video_io
                import os
                import shutil

                encoder = shutil.which("ffmpeg")
                if not encoder:
                    raise RuntimeError("ffmpeg is required for video encoding")
                os.environ["IMAGEIO_FFMPEG_EXE"] = encoder
                a.video.parent.mkdir(parents=True, exist_ok=True)
                writer = video_io.get_writer(
                    str(a.video),
                    fps=round(1 / base.step_dt),
                    codec="libx264",
                    quality=8,
                    macro_block_size=2,
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                )
                writer.append_data(frame[..., :3])
                video_frames += 1
        for step in range(int(cfg.episode_length_s / base.step_dt + 2) * a.episodes + 10):
            if not app.is_running():
                break
            with torch.inference_mode():
                if actor:
                    po = obs["obs"] if isinstance(obs, dict) else obs
                    act = actor.get_action(actor.obs_to_torch(po), is_deterministic=True)
                    obs, rew, done, info = env.step(act)
                    po = obs["obs"] if isinstance(obs, dict) else obs
                else:
                    act = (
                        student(history.observation())
                        if student is not None
                        else torch.zeros((a.num_envs, 13), device=base.device)
                    )
                    obs, rew, term, trunc, info = env.step(act)
                    done = term | trunc
                    po = obs["policy"]
                    if history is not None:
                        history.advance(base.hand.data.joint_pos[:, base.ids], base.targets, done.bool())
                assert po.shape == (a.num_envs, 63) and torch.isfinite(po).all() and torch.isfinite(rew).all()
                assert ((base.targets >= base.lo - 1e-6) & (base.targets <= base.hi + 1e-6)).all()
                q = info["terminal_q"]
                pos = info["terminal_positions"]
                max_mimic = max(
                    max_mimic, max(float((q[:, c] - m * q[:, j] - b).abs().max()) for c, j, m, b in base.mimics)
                )
                min_sep = min(min_sep, float((pos[:, 0] - pos[:, 1]).norm(dim=-1).min()))
                min_z = min(min_z, float(pos[..., 2].min()))
                max_speed = max(max_speed, float(base.vel.norm(dim=-1).max()))
                if counts[0] == 0:
                    trace.append(
                        {
                            "time": (step + 1) * base.step_dt,
                            "positions": pos[0].cpu().tolist(),
                            "q": q[0, base.ids].cpu().tolist(),
                            "targets": info["terminal_targets"][0].cpu().tolist(),
                            "actions": info["terminal_actions"][0].cpu().tolist(),
                        }
                    )
                net = info["rotation_metrics"]["net_turns"] * (2 * math.pi)
                speed = (net - previous_net) / base.step_dt
                measured = info["rotation_metrics"]["seconds"] > 2.0
                slow = measured & (speed.abs() < 0.5)
                samples += measured.float()
                slow_samples += slow.float()
                reverse_samples += (measured & (speed < 0)).float()
                slow_run = torch.where(slow, slow_run + base.step_dt, 0.0)
                longest_slow = torch.maximum(longest_slow, slow_run)
                previous_net = torch.where(done.bool(), 0.0, net)
                for i in done.nonzero().flatten().tolist():
                    if counts[i] >= a.episodes:
                        continue
                    row = {k: v[i].item() for k, v in info["rotation_metrics"].items()}
                    row["slow_fraction_after_2s"] = float(slow_samples[i] / samples[i].clamp_min(1))
                    row["reverse_fraction_after_2s"] = float(reverse_samples[i] / samples[i].clamp_min(1))
                    row["longest_slow_seconds_after_2s"] = float(longest_slow[i])
                    row["continuity_samples"] = int(samples[i])
                    row["env"] = i
                    row["two_relative_turns_30s"] = two_relative_turns(row)
                    full = not row["dropped"] and row["seconds"] >= cfg.episode_length_s - base.step_dt - 1e-5
                    row["one_orbit"] = (
                        full and row["net_turns"] >= 1.0 and min(row["ball0_turns"], row["ball1_turns"]) >= 0.75
                    )
                    row["two_orbits"] = (
                        full and row["net_turns"] >= 2.0 and min(row["ball0_turns"], row["ball1_turns"]) >= 1.75
                    )
                    records.append(row)
                    counts[i] += 1
                for buffer in (samples, slow_samples, reverse_samples, slow_run, longest_slow):
                    buffer[done.bool()] = 0.0
                # Stop at the first recorded episode boundary: never splice a reset into the video.
                if writer is not None:
                    if bool(done[0]):
                        writer.close()
                        writer = None
                    else:
                        frame = base.render()
                        if frame is None or not frame.any():
                            raise RuntimeError("Empty video frame")
                        writer.append_data(frame[..., :3])
                        video_frames += 1
                if min(counts) >= a.episodes:
                    break
        n = len(records)
        result = {
            "palm_tilt_range_deg": list(cfg.palm_tilt_range_deg),
            "palm_side_tilt_deg": cfg.palm_side_tilt_deg,
            "position_coordinate_frame": "untilted_palm_reference",
            "complete": min(counts) >= a.episodes,
            "episodes": n,
            "seed": a.seed,
            "policy": "teacher" if actor else ("joint_history_student" if student is not None else "zero_action"),
            "checkpoint": str(a.checkpoint or a.student) if (a.checkpoint or a.student) else None,
            "checkpoint_sha256": hashlib.sha256((a.checkpoint or a.student).read_bytes()).hexdigest()
            if (a.checkpoint or a.student)
            else None,
            "ball_geometry": {
                "diameter_mm": 2000 * cfg.ball_radius,
                "mass_kg_per_ball": cfg.ball_mass,
                "orbit_radii_m": list(cfg.orbit_radii),
                "initial_heights_m": list(cfg.initial_ball_heights),
                "size_override_mm": a.ball_diameter_mm,
                "physics_radii_m": physics_radii,
                "initial_minimum_surface_gap_m": initial_surface_gap,
            },
            "initial_ball_positions": initial,
            "max_mimic_error_rad": max_mimic,
            "minimum_ball_distance_m": min_sep,
            "minimum_ball_height_m": min_z,
            "max_ball_speed_m_s": max_speed,
            "mean_net_turns": sum(r["net_turns"] for r in records) / max(n, 1),
            "mean_world_signed_net_turns": sum(r["world_signed_net_turns"] for r in records) / max(n, 1),
            "drop_rate": sum(r["dropped"] for r in records) / max(n, 1),
            "fixed_initial_conditions": not cfg.domain_randomization
            and not a.placement_jitter_mm
            and cfg.palm_tilt_range_deg[0] == cfg.palm_tilt_range_deg[1],
            "rotation_direction": cfg.rotation_direction,
            "domain_randomization": cfg.domain_randomization,
            "actuator_response": None,
            "contact_pair_friction": base.contact_pair_friction,
            "perturbation": {
                "xy_bound_mm_per_axis": a.placement_jitter_mm,
                "seed": a.perturb_seed,
                "ball_static_friction": cfg.ball_static_friction,
                "ball_dynamic_friction": cfg.ball_dynamic_friction,
                "offsets_m": initial_offsets,
                "initial_friction_scales": initial_scales,
                "physics_materials_env0": measured_materials,
            },
            "all_initial_ball_positions": initial_all,
            "thirty_second_two_relative_turn_rate": sum(r["two_relative_turns_30s"] for r in records) / max(n, 1),
            "continuity": {
                "warmup_seconds": 2.0,
                "slow_speed_threshold_rad_s": 0.5,
                "mean_slow_fraction": sum(r["slow_fraction_after_2s"] for r in records) / max(n, 1),
                "mean_reverse_fraction": sum(r["reverse_fraction_after_2s"] for r in records) / max(n, 1),
                "max_slow_seconds": max((r["longest_slow_seconds_after_2s"] for r in records), default=0.0),
            },
            "shared_lift": {
                "tolerance_m": cfg.shared_lift_tolerance_m,
                "mean_fraction": sum(r["shared_lift_fraction"] for r in records) / max(n, 1),
                "max_common_rise_m": max((r["max_shared_lift_m"] for r in records), default=0.0),
            },
            "metric_note": "Relative pair rotation permits midpoint translation; old one/two_orbit_success fields require the original fixed center. Inspect both balls moving.",
            "one_orbit_success": sum(r["one_orbit"] for r in records) / max(n, 1),
            "two_orbit_success": sum(r["two_orbits"] for r in records) / max(n, 1),
            "video": str(a.video) if a.video else None,
            "video_frames": video_frames,
            "video_fps": round(1 / base.step_dt) if a.video else None,
            "records": records,
            "first_episode_trace": trace,
        }
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2))
        print(
            "BAODING_RESULT",
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ["records", "first_episode_trace", "all_initial_ball_positions", "perturbation"]
                }
            ),
            flush=True,
        )
        if not result["complete"]:
            raise RuntimeError("Evaluation quota incomplete")
    finally:
        if writer is not None:
            writer.close()
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
    app.close(wait_for_replicator=False, skip_cleanup=True, exit_code=exit_code)
