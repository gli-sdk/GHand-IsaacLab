"""Simulator-free tests for checkpoint evaluation accounting and safe configuration."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/cube_reorientation"))
from ghand.tasks.cube_reorientation import metrics
from ghand.common import config_io


class EpisodeTests(unittest.TestCase):
    def test_fixed_quota_ignores_extra_fast_episodes_and_counts_terminal_success(self):
        tracker = metrics.EpisodeAccumulator(2, 1, 0.1)
        tracker.step([2, 0], [True, False], [False, False])
        tracker.step([9, 0], [True, False], [False, False])
        tracker.step([9, 3], [True, False], [False, True])
        self.assertTrue(tracker.complete)
        self.assertEqual([r["successes"] for r in tracker.records], [2, 3])
        self.assertEqual([r["steps"] for r in tracker.records], [1, 3])
        summary = tracker.summary()
        self.assertEqual(summary["mean_successes"], 2.5)
        self.assertEqual(summary["successes_standard_error"], 0.5)
        self.assertEqual(summary["at_least_one_success_rate"], 1.0)
        self.assertEqual(summary["fall_count"], 1)
        self.assertEqual(summary["timeout_count"], 1)
        self.assertEqual(summary["fall_rate"], 0.5)
        self.assertEqual(summary["timeout_rate"], 0.5)

    def test_numerical_failure_is_separate_from_fall_and_timeout(self):
        tracker = metrics.EpisodeAccumulator(1, 1, 0.1)
        tracker.step([float("nan")], [False], [False], numerical_failures=[True])
        self.assertEqual(tracker.records[0]["termination"], "numerical_failure")
        self.assertIsNone(tracker.records[0]["successes"])
        self.assertEqual(tracker.summary()["numerical_failure_count"], 1)
        self.assertIsNone(tracker.summary()["mean_successes"])

    def test_both_fall_and_timeout_preserves_both_flags(self):
        tracker = metrics.EpisodeAccumulator(1, 1, 0.1)
        tracker.step([0], [True], [True])
        self.assertEqual(tracker.records[0]["termination"], "fall")
        self.assertTrue(tracker.records[0]["timeout"])

    def test_success_times_record_increments_once_and_include_terminal_step(self):
        tracker = metrics.EpisodeAccumulator(1, 1, 0.5)
        tracker.step([1], [False], [False])
        tracker.step([1], [False], [False])
        tracker.step([3], [False], [True])
        self.assertEqual(tracker.records[0]["success_times_seconds"], [0.5, 1.5, 1.5])
        self.assertEqual(tracker.records[0]["successes"], 3)
        self.assertEqual(tracker.summary()["at_least_two_success_rate"], 1.0)
        self.assertEqual(tracker.summary()["mean_successes_after_1s"], 2.0)
        self.assertEqual(tracker.summary()["episodes_with_success_after_1s_rate"], 1.0)

    def test_success_times_reset_between_episodes_and_ignore_over_quota(self):
        tracker = metrics.EpisodeAccumulator(1, 2, 0.5)
        tracker.step([2], [True], [False])
        tracker.step([0], [False], [False])
        tracker.step([1], [False], [True])
        tracker.step([99], [True], [False])
        self.assertEqual([r["success_times_seconds"] for r in tracker.records], [[0.5, 0.5], [1.0]])
        self.assertEqual([r["successes"] for r in tracker.records], [2, 1])
        self.assertEqual(tracker.summary()["at_least_two_success_rate"], 0.5)
        # Exactly one second is not after one second.
        self.assertEqual(tracker.summary()["mean_successes_after_1s"], 0.0)
        self.assertEqual(tracker.summary()["episodes_with_success_after_1s_rate"], 0.0)

    def test_success_time_summaries_exclude_failed_and_unfinished_episodes(self):
        tracker = metrics.EpisodeAccumulator(4, 1, 0.5)
        tracker.step([1, 1, 1, 0], [False] * 4, [False] * 4)
        tracker.step([2, 1, 1, 0], [False] * 4, [False] * 4)
        tracker.step(
            [3, 1, 2, 9], [False] * 4, [True, True, False, False], numerical_failures=[False, False, True, False]
        )
        summary = tracker.summary()
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["valid_episode_count"], 2)
        self.assertEqual(summary["at_least_two_success_rate"], 0.5)
        self.assertEqual(summary["mean_successes_after_1s"], 0.5)
        self.assertEqual(summary["episodes_with_success_after_1s_rate"], 0.5)
        self.assertEqual(summary["mean_successes"], 2.0)

    def test_success_time_summaries_are_null_without_valid_finished_episodes(self):
        tracker = metrics.EpisodeAccumulator(2, 1, 0.5)
        tracker.step([float("nan"), 2], [False, False], [False, False])
        summary = tracker.summary()
        self.assertIsNone(summary["at_least_two_success_rate"])
        self.assertIsNone(summary["mean_successes_after_1s"])
        self.assertIsNone(summary["episodes_with_success_after_1s_rate"])

    def test_invalid_quotas_rejected(self):
        for args in [(0, 1, 0.1), (1, 0, 0.1), (1, 1, 0)]:
            with self.assertRaises(ValueError):
                metrics.EpisodeAccumulator(*args)


class ConfigurationTests(unittest.TestCase):
    def test_tuple_yaml_is_safe_but_object_construction_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("position: !!python/tuple [1, 2, 3]\n")
            self.assertEqual(config_io.load_config_yaml(path)["position"], (1, 2, 3))
            path.write_text("evil: !!python/object/apply:os.system [echo forbidden]\n")
            with self.assertRaises(Exception):
                config_io.load_config_yaml(path)

    def test_restore_preserves_callables_and_objects_but_restores_primitives(self):
        def fn():
            return None

        cfg = SimpleNamespace(
            success_tolerance=0.2,
            robot=SimpleNamespace(func=fn, pos=(0, 0, 0)),
            weights={"old": 1},
            class_type=SimpleNamespace,
        )
        report = config_io.restore_config(
            cfg,
            {
                "success_tolerance": 0.1,
                "robot": {"func": "evil:callable", "pos": [1, 2, 3]},
                "weights": {"new": 2},
                "class_type": "evil:class",
                "unknown": 5,
            },
        )
        self.assertEqual(cfg.success_tolerance, 0.1)
        self.assertIs(cfg.robot.func, fn)
        self.assertIs(cfg.class_type, SimpleNamespace)
        self.assertEqual(cfg.robot.pos, (1, 2, 3))
        self.assertEqual(cfg.weights, {"new": 2})
        self.assertIn("robot.func", report["skipped"])
        self.assertIn("unknown", report["skipped"])

    def test_saved_corrected_reset_is_restored_on_legacy_task(self):
        cfg = SimpleNamespace()
        report = metrics.apply_reset_mode(cfg, {"correct_joint_reset": True}, "saved")
        self.assertIs(cfg.correct_joint_reset, True)
        self.assertEqual(
            report, {"saved_reset_mode": "corrected", "effective_reset_mode": "corrected", "reset_mode_override": None}
        )

    def test_missing_saved_reset_uses_legacy_even_on_training_task(self):
        cfg = SimpleNamespace(correct_joint_reset=True)
        report = metrics.apply_reset_mode(cfg, {}, "saved")
        self.assertIs(cfg.correct_joint_reset, False)
        self.assertEqual(report["saved_reset_mode"], "legacy")
        self.assertEqual(report["effective_reset_mode"], "legacy")

    def test_reset_mode_intentional_overrides_are_recorded(self):
        for saved, override in [(False, "corrected"), (True, "legacy")]:
            with self.subTest(override=override):
                cfg = SimpleNamespace(correct_joint_reset=saved)
                report = metrics.apply_reset_mode(cfg, {"correct_joint_reset": saved}, override)
                self.assertIs(cfg.correct_joint_reset, override == "corrected")
                self.assertEqual(report["reset_mode_override"], override)
                self.assertEqual(report["effective_reset_mode"], override)

    def test_reward_capture_is_independent_of_later_reset(self):
        class Buffer(list):
            def clone(self):
                return Buffer(self)

        env = SimpleNamespace(successes=Buffer([0]))

        def reward():
            env.successes[0] += 1
            return 3

        env._get_rewards = reward
        capture = metrics.capture_terminal_successes(env)
        self.assertEqual(env._get_rewards(), 3)
        env.successes[0] = 0
        self.assertEqual(capture["successes"], [1])


class GoalDistributionTests(unittest.TestCase):
    def test_default_keeps_saved_curriculum_and_reports_source(self):
        import evaluate as evaluate_inhand

        cfg = SimpleNamespace(relative_goal_probability=1.0)
        report = evaluate_inhand.apply_goal_probability(cfg, {"relative_goal_probability": 1.0}, None)
        self.assertEqual(cfg.relative_goal_probability, 1.0)
        self.assertEqual(report["goal_mode"], "relative")
        self.assertEqual(report["goal_mode_source"], "saved_config")
        self.assertIsNone(report["relative_goal_probability_override"])

    def test_explicit_zero_restores_full_random_and_records_saved_value(self):
        import evaluate as evaluate_inhand

        cfg = SimpleNamespace(relative_goal_probability=1.0)
        report = evaluate_inhand.apply_goal_probability(cfg, {"relative_goal_probability": 1.0}, 0.0)
        self.assertEqual(cfg.relative_goal_probability, 0.0)
        self.assertEqual(report["goal_mode"], "full_random")
        self.assertEqual(report["goal_mode_source"], "explicit_override")
        self.assertEqual(report["saved_relative_goal_probability"], 1.0)
        self.assertEqual(report["relative_goal_probability_override"], 0.0)

    def test_mixed_goal_probability_is_identified(self):
        import evaluate as evaluate_inhand

        cfg = SimpleNamespace(relative_goal_probability=0.0)
        report = evaluate_inhand.apply_goal_probability(cfg, {}, 0.4)
        self.assertEqual(cfg.relative_goal_probability, 0.4)
        self.assertEqual(report["goal_mode"], "mixed")

    def test_legacy_task_accepts_default_but_rejects_explicit_override(self):
        import evaluate as evaluate_inhand

        cfg = SimpleNamespace()
        report = evaluate_inhand.apply_goal_probability(cfg, {}, None)
        self.assertEqual(report["goal_mode"], "full_random")
        self.assertEqual(report["goal_mode_source"], "task_default")
        self.assertFalse(hasattr(cfg, "relative_goal_probability"))
        with self.assertRaisesRegex(ValueError, "does not support"):
            evaluate_inhand.apply_goal_probability(cfg, {}, 0.0)

    def test_probability_parser_rejects_nonfinite_and_out_of_bounds(self):
        import argparse
        import evaluate as evaluate_inhand

        for value in ("nan", "inf", "-0.1", "1.1"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                evaluate_inhand.probability_value(value)
        self.assertEqual(evaluate_inhand.probability_value("0"), 0.0)
        self.assertEqual(evaluate_inhand.probability_value("1"), 1.0)


class TrajectoryTests(unittest.TestCase):
    def test_trajectory_saves_initial_and_step_frames_without_pickle(self):
        import numpy as np
        import evaluate as evaluate_inhand

        tracker = evaluate_inhand.TrajectoryRecorder(
            {"body_names": ["palm", "finger"], "env_origin": [0, 0, 0], "step_dt": 0.02, "cube_scale": [0.6, 0.6, 0.6]}
        )
        frame = {
            "body_pos_w": [[0, 0, 1], [0, 0, 2]],
            "body_quat_w": [[1, 0, 0, 0], [1, 0, 0, 0]],
            "object_pos_w": [0, 0, 3],
            "object_quat_w": [1, 0, 0, 0],
            "goal_rot": [1, 0, 0, 0],
        }
        tracker.append(frame)
        tracker.append({**frame, "object_pos_w": [0, 0, 4]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.npz"
            tracker.save(path)
            with np.load(path, allow_pickle=False) as trace:
                self.assertEqual(trace["body_pos_w"].shape, (2, 2, 3))
                self.assertEqual(trace["body_quat_w"].shape, (2, 2, 4))
                self.assertEqual(trace["object_pos_w"].tolist(), [[0, 0, 3], [0, 0, 4]])
                self.assertEqual(trace["body_names"].dtype.kind, "U")
                self.assertEqual(float(trace["step_dt"]), 0.02)


class RuntimePhysicsTests(unittest.TestCase):
    def test_reports_solver_mass_and_limits_instead_of_nominal_configuration(self):
        import evaluate as evaluate_inhand

        class Tensor:
            def __init__(self, values):
                self.values = values

            def __getitem__(self, key):
                return Tensor(self.values[key])

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        base = SimpleNamespace(
            cfg=SimpleNamespace(
                object_cfg=SimpleNamespace(spawn=SimpleNamespace(mass_props=SimpleNamespace(mass=0.026, density=567)))
            ),
            object=SimpleNamespace(
                root_physx_view=SimpleNamespace(
                    get_masses=lambda: Tensor([[0.216]]), get_inertias=lambda: Tensor([[1.0, 2.0, 3.0]])
                )
            ),
            hand=SimpleNamespace(
                joint_names=["joint"],
                data=SimpleNamespace(joint_vel_limits=Tensor([[4.0]]), joint_effort_limits=Tensor([[0.1]])),
            ),
        )
        record = evaluate_inhand.runtime_physics_snapshot(base)
        self.assertEqual(record["object_mass_kg"], [0.216])
        self.assertEqual(record["object_inertia_kg_m2"], [1.0, 2.0, 3.0])
        self.assertEqual(record["joint_velocity_limits_rad_s"], [4.0])
        self.assertEqual(record["joint_effort_limits_Nm"], [0.1])
        self.assertEqual(record["sample_environment_index"], 0)


class EvaluatorCliTests(unittest.TestCase):
    def test_import_and_parser_do_not_import_simulator(self):
        import evaluate as evaluate_inhand

        args = evaluate_inhand.make_parser().parse_args(
            [
                "--task",
                "test",
                "--checkpoint",
                "test.pth",
                "--output",
                "result.json",
                "--num_envs",
                "4",
                "--seed",
                "12",
                "--episodes_per_env",
                "2",
                "--tolerance",
                "0.1",
            ]
        )
        self.assertEqual(args.episodes_per_env, 2)
        self.assertEqual(args.seed, 12)
        self.assertEqual(args.reset_mode, "saved")
        self.assertIsNone(args.relative_goal_probability)
        explicit = evaluate_inhand.make_parser().parse_args(
            [
                "--task",
                "test",
                "--checkpoint",
                "test.pth",
                "--output",
                "result.json",
                "--relative_goal_probability",
                "0",
                "--tolerance",
                "0.1",
            ]
        )
        self.assertEqual(explicit.relative_goal_probability, 0.0)
        self.assertNotIn("isaacsim", sys.modules)

    def test_hand_asset_is_pinned_before_task_import(self):
        import os
        from unittest.mock import patch
        import evaluate as evaluate_inhand

        with tempfile.TemporaryDirectory() as tmp:
            usd = Path(tmp) / "assets/ghand5_right.usd"
            usd.parent.mkdir()
            usd.write_text("#usda 1.0")
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GHAND_USD_PATH", None)
                evaluate_inhand.configure_hand_asset(Path(tmp) / "nn/policy.pth")
                self.assertEqual(os.environ["GHAND_USD_PATH"], str(usd))
                usd.unlink()
                evaluate_inhand.configure_hand_asset(Path(tmp) / "nn/policy.pth")
                self.assertEqual(
                    Path(os.environ["GHAND_USD_PATH"]), Path(__file__).resolve().parents[2] / "assets/ghand5_right.usd"
                )

    def test_video_launcher_enables_missing_render_dependency_without_replacing_args(self):
        import evaluate as evaluate_inhand

        args = SimpleNamespace(kit_args="--/existing/setting=true", enable_cameras=False)
        evaluate_inhand.configure_video_launcher(args)
        self.assertTrue(args.enable_cameras)
        self.assertIn("--/existing/setting=true", args.kit_args)
        self.assertIn("--enable omni.kit.viewport.window", args.kit_args)
        evaluate_inhand.configure_video_launcher(args)
        self.assertEqual(args.kit_args.count("omni.kit.viewport.window"), 1)

    def test_camera_world_pose_adds_environment_origin(self):
        import evaluate as evaluate_inhand

        cfg = SimpleNamespace(eye=(0.45, -0.59, 0.92), lookat=(0, -0.09, 0.51))
        eye, target = evaluate_inhand.camera_world_pose(cfg, (3, -2, 0))
        self.assertEqual(eye, (3.45, -2.59, 0.92))
        self.assertEqual(target, (3, -2.09, 0.51))

    def test_local_usd_hash_and_remote_usd_have_explicit_provenance(self):
        import evaluate as evaluate_inhand

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "robot.usda"
            path.write_bytes(b"abc")
            self.assertEqual(
                evaluate_inhand.local_file_sha256(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )
            self.assertIsNone(evaluate_inhand.local_file_sha256(path.with_name("missing.usd")))
            self.assertIsNone(evaluate_inhand.local_file_sha256("https://example.com/robot.usd"))

    def test_checkpoint_config_lookup_supports_rl_games_nn_directory(self):
        import evaluate as evaluate_inhand

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nn").mkdir()
            (root / "params").mkdir()
            (root / "params" / "env.yaml").write_text("seed: 0\n")
            self.assertEqual(evaluate_inhand.find_saved_env(root / "nn" / "model.pth"), root / "params" / "env.yaml")


if __name__ == "__main__":
    unittest.main()
