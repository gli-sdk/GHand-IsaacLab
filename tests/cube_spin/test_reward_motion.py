"""Exercise the real reward method without starting the GPU simulator."""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).parents[2] / "source/ghand/tasks/cube_spin"
spec = importlib.util.spec_from_file_location("spin_core", ROOT / "core.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
tree = ast.parse((ROOT / "env.py").read_text())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GHandSpinEnv")
methods = [
    node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in ("_get_rewards", "_get_dones")
]
scope = {"torch": torch, "math": math, **vars(core)}
exec(compile(ast.Module(body=methods, type_ignores=[]), str(ROOT / "env.py"), "exec"), scope)


def environment(angle=0.0, reported_speed=1.0):
    zeros = torch.zeros(1)
    return SimpleNamespace(
        previous_object_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        object_rot=torch.tensor([[math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]]),
        axis=torch.tensor([0.0, 0.0, 1.0]),
        object_angvel=torch.tensor([[0.0, 0.0, reported_speed]]),
        object_linvel=torch.zeros(1, 3),
        object_pos=torch.zeros(1, 3),
        in_hand_pos=torch.zeros(1, 3),
        action_delta=torch.zeros(1, 13),
        episode_length_buf=torch.tensor([20]),
        step_dt=0.05,
        cfg=SimpleNamespace(
            spin_position_penalty_scale=0.5,
            spin_terminate_on_distance=True,
            fall_dist=0.12,
            cube_static_friction_range=None,
            cube_size_range_m=None,
            spin_warmup_s=0.5,
            target_speed=1.0,
            spin_survival_reward=0.1,
            spin_drop_penalty=-10.0,
        ),
        _spin_dropped=torch.tensor([False]),
        net_angle=zeros.clone(),
        reverse_angle=zeros.clone(),
        off_axis_angle=zeros.clone(),
        extras={},
    )


class ResetParent:
    def _reset_idx(self, env_ids):
        self.cur_targets[env_ids] = self.hand_dof_upper_limits[env_ids]
        self.prev_targets[env_ids] = self.cur_targets[env_ids]
        self.hand_dof_targets[env_ids] = self.cur_targets[env_ids]


integration_methods = [
    node
    for node in cls.body
    if isinstance(node, ast.FunctionDef) and node.name in ("_pre_physics_step", "_reset_idx", "_get_observations")
]
integration_cls = ast.ClassDef(
    name="BoundedSpin",
    bases=[ast.Name(id="ResetParent", ctx=ast.Load())],
    keywords=[],
    body=integration_methods,
    decorator_list=[],
)
integration_scope = {
    "torch": torch,
    "math": math,
    "ResetParent": ResetParent,
    "unscale": lambda x, low, high: 2.0 * (x - low) / (high - low) - 1.0,
}
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=[integration_cls], type_ignores=[])),
        str(ROOT / "env.py"),
        "exec",
    ),
    integration_scope,
)


class SpinCommandIntegrationTests(unittest.TestCase):
    def make_env(self):
        env = integration_scope["BoundedSpin"]()
        env.cfg = SimpleNamespace(
            cube_static_friction_range=None, cube_size_range_m=None, action_step_rad=0.075, target_speed=1.0
        )
        env.num_envs = 2
        env.device = "cpu"
        env.actuated_dof_indices = list(range(13))
        env.hand_dof_lower_limits = torch.zeros(2, 13)
        env.hand_dof_upper_limits = torch.full((2, 13), math.pi / 2)
        env.command_lower_limits = torch.full((13,), math.radians(4.5))
        env.command_upper_limits = env.hand_dof_upper_limits[0].clone()
        env.command_upper_limits[[1, 3]] = torch.tensor([math.radians(64.0), math.radians(80.0)])
        env.cur_targets = env.hand_dof_upper_limits.clone()
        env.prev_targets = env.cur_targets.clone()
        env.hand_dof_targets = env.cur_targets.clone()
        env.hand_dof_pos = torch.full((2, 13), math.pi / 4)
        env.hand_dof_vel = torch.zeros(2, 13)
        env.actions = torch.zeros(2, 13)
        env.action_delta = torch.zeros(2, 13)
        env.object_pos = env.object_linvel = env.object_angvel = env.in_hand_pos = torch.zeros(2, 3)
        env.object_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
        env.fingertip_pos = torch.zeros(2, 5, 3)
        env.previous_object_quat = env.object_rot.clone()
        env.net_angle = torch.ones(2)
        env.reverse_angle = torch.ones(2)
        env.off_axis_angle = torch.ones(2)
        env._spin_dropped = torch.ones(2, dtype=torch.bool)
        env.hand = SimpleNamespace(_ALL_INDICES=torch.arange(2), data=SimpleNamespace(root_quat_w=env.object_rot))
        env.object = SimpleNamespace(data=SimpleNamespace(root_quat_w=env.object_rot))
        env.hand.set_joint_position_target = lambda target, **kw: setattr(env, "sent_targets", target.clone())
        return env

    def test_increment_clamps_commands_without_changing_physical_or_normalization_limits(self):
        env = self.make_env()
        upper = env.hand_dof_upper_limits.clone()
        position = env.hand_dof_pos.clone()
        env._pre_physics_step(torch.ones(2, 13))
        torch.testing.assert_close(env.cur_targets, env.command_upper_limits.expand(2, 13))
        torch.testing.assert_close(env.prev_targets, env.cur_targets)
        torch.testing.assert_close(env.hand_dof_upper_limits, upper)
        torch.testing.assert_close(env.hand_dof_pos, position)
        obs = env._get_observations()["policy"]
        torch.testing.assert_close(obs[:, :13], torch.zeros(2, 13))
        expected = 2.0 * env.command_upper_limits / (math.pi / 2) - 1.0
        torch.testing.assert_close(obs[:, 26:39], expected.expand(2, 13))
        env.cur_targets.zero_()
        env._pre_physics_step(-torch.ones(2, 13))
        torch.testing.assert_close(env.cur_targets, env.command_lower_limits.expand(2, 13))
        torch.testing.assert_close(env.hand_dof_lower_limits, torch.zeros(2, 13))

    def test_partial_reset_caps_targets_but_preserves_measured_state_and_physical_limits(self):
        env = self.make_env()
        position, upper = env.hand_dof_pos.clone(), env.hand_dof_upper_limits.clone()
        untouched = env.cur_targets[1].clone()
        env._reset_idx(torch.tensor([0]))
        torch.testing.assert_close(env.cur_targets[0], env.command_upper_limits)
        torch.testing.assert_close(env.cur_targets[1], untouched)
        torch.testing.assert_close(env.prev_targets[0], env.command_upper_limits)
        torch.testing.assert_close(env.hand_dof_targets[0], env.command_upper_limits)
        torch.testing.assert_close(env.sent_targets[0], env.command_upper_limits)
        torch.testing.assert_close(env.hand_dof_pos, position)
        torch.testing.assert_close(env.hand_dof_upper_limits, upper)


class SpinMotionRewardTests(unittest.TestCase):
    def test_stationary_pose_cannot_earn_spin_reward_from_reported_velocity(self):
        env = environment(angle=0.0, reported_speed=1.0)
        reward = scope["_get_rewards"](env)
        self.assertAlmostEqual(reward.item(), env.cfg.spin_survival_reward, places=6)
        self.assertAlmostEqual(env.extras["log"]["spin/reward_rotation"].item(), 0.0, places=6)

    def test_real_rotation_is_rewarded_even_when_reported_velocity_is_zero(self):
        env = environment(angle=0.05, reported_speed=0.0)
        reward = scope["_get_rewards"](env)
        self.assertAlmostEqual(reward.item(), 2.1, places=5)
        self.assertAlmostEqual(env.net_angle.item(), 0.05, places=6)

    def test_real_reverse_motion_is_penalized_despite_positive_velocity_readout(self):
        env = environment(angle=-0.05, reported_speed=1.0)
        reward = scope["_get_rewards"](env)
        self.assertAlmostEqual(reward.item(), -1.9, places=5)
        self.assertAlmostEqual(env.reverse_angle.item(), 0.05, places=6)

    def test_free_position_motion_has_equal_reward_after_translation(self):
        rewards = []
        for offset in ([0.0, 0.0, 0.0], [0.04, -0.03, 0.02]):
            env = environment(angle=0.05, reported_speed=0.0)
            env.cfg.spin_position_penalty_scale = 0.0
            env.object_pos[:] = torch.tensor(offset)
            rewards.append(scope["_get_rewards"](env))
            self.assertEqual(env.extras["log"]["spin/reward_position"].item(), 0.0)
        torch.testing.assert_close(rewards[0], rewards[1])

    def test_legacy_position_penalty_remains_available(self):
        env = environment(angle=0.05, reported_speed=0.0)
        env.object_pos[:, 0] = 0.05
        reward = scope["_get_rewards"](env)
        self.assertAlmostEqual(reward.item(), 1.6, places=5)
        self.assertAlmostEqual(env.extras["log"]["spin/reward_position"].item(), -0.5, places=6)

    def test_position_penalty_coefficient_is_applied(self):
        env = environment(angle=0.05, reported_speed=0.0)
        env.cfg.spin_position_penalty_scale = 0.25
        env.object_pos[:, 0] = 0.05
        self.assertAlmostEqual(scope["_get_rewards"](env).item(), 1.85, places=5)

    def test_warmup_and_drop_never_earn_rotation_credit(self):
        for dropped in (False, True):
            env = environment(angle=0.05)
            env._spin_dropped[:] = dropped
            if not dropped:
                env.episode_length_buf[:] = 1
            reward = scope["_get_rewards"](env)
            self.assertAlmostEqual(reward.item(), 0.1 - 10 * dropped, places=5)
            self.assertEqual(env.net_angle.item(), 0.0)


class SpinFreePositionTerminationTests(unittest.TestCase):
    def make_env(self):
        env = environment()
        env.cfg.spin_terminate_on_distance = False
        env._compute_intermediate_values = lambda: None
        env.hand_dof_pos = torch.zeros(1, 18)
        env.hand_dof_vel = torch.zeros(1, 18)
        env.object_velocities = torch.zeros(1, 6)
        env.max_episode_length = 600
        return env

    def test_horizontal_translation_does_not_end_free_position_episode(self):
        env = self.make_env()
        env.object_pos[:, 0] = 0.14
        dropped, timeout = scope["_get_dones"](env)
        self.assertFalse(dropped.item())
        self.assertFalse(timeout.item())
        env.cfg.spin_terminate_on_distance = True
        self.assertTrue(scope["_get_dones"](env)[0].item())

    def test_below_hand_drop_and_time_limit_remain(self):
        env = self.make_env()
        env.object_pos[:, 2] = -0.056
        self.assertTrue(scope["_get_dones"](env)[0].item())
        env.object_pos.zero_()
        env.episode_length_buf[:] = 599
        dropped, timeout = scope["_get_dones"](env)
        self.assertFalse(dropped.item())
        self.assertTrue(timeout.item())


if __name__ == "__main__":
    unittest.main()
