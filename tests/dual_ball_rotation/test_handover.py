import unittest
import math
import importlib.util


class HandoverTests(unittest.TestCase):
    def gate(self, **kwargs):
        spec = importlib.util.find_spec("ghand.tasks.dual_ball_rotation.handover")
        self.assertIsNotNone(spec, "Validated teacher handover has not been implemented")
        from ghand.tasks.dual_ball_rotation.handover import GatedHandover

        return GatedHandover(**kwargs)

    def test_poor_student_never_loses_teacher_due_to_iteration_count(self):
        gate = self.gate(patience=20)
        for iteration in (30, 60, 90, 120, 180, 300):
            gate.observe(iteration, 0.3)
            self.assertEqual(gate.probability, 1.0)
        self.assertFalse(gate.complete)

    def test_two_passes_after_minimum_training_are_required(self):
        gate = self.gate()
        gate.observe(30, 0.9)
        gate.observe(60, 0.9)
        gate.observe(90, 0.9)
        self.assertEqual(gate.probability, 1.0)
        gate.observe(120, 0.9)
        self.assertAlmostEqual(gate.probability, 0.8)
        self.assertFalse(gate.complete)

    def test_failed_evaluation_breaks_pass_streak(self):
        gate = self.gate(min_teacher_iterations=1)
        for iteration, rate in [(1, 0.9), (2, 0.7), (3, 0.9)]:
            gate.observe(iteration, rate)
        self.assertEqual(gate.probability, 1.0)
        gate.observe(4, 0.9)
        self.assertAlmostEqual(gate.probability, 0.8)

    def test_severe_regression_restores_teacher_help(self):
        gate = self.gate(min_teacher_iterations=1)
        gate.observe(1, 0.9)
        gate.observe(2, 0.9)
        gate.observe(3, 0.2)
        self.assertEqual(gate.probability, 1.0)
        self.assertFalse(gate.complete)

    def test_stalled_student_stops_without_being_called_successful(self):
        gate = self.gate(min_teacher_iterations=1, patience=3)
        for iteration in range(1, 5):
            gate.observe(iteration, 0.3)
        self.assertEqual(gate.stop_reason, "student_stalled")
        self.assertFalse(gate.complete)
        self.assertEqual(gate.probability, 1.0)

    def test_completion_requires_passes_after_teacher_probability_is_zero(self):
        gate = self.gate(min_teacher_iterations=1)
        for iteration in range(1, 11):
            gate.observe(iteration, 0.9)
        self.assertEqual(gate.probability, 0.0)
        self.assertFalse(gate.complete)
        gate.observe(11, 0.9)
        self.assertFalse(gate.complete)
        gate.observe(12, 0.9)
        self.assertTrue(gate.complete)
        self.assertEqual(gate.stop_reason, "student_validated")

    def test_invalid_configuration_and_metrics_are_rejected(self):
        for kwargs in (
            {"min_teacher_iterations": 0},
            {"success_rate": math.nan},
            {"success_rate": 1.1},
            {"patience": 0},
        ):
            with self.assertRaises(ValueError):
                self.gate(**kwargs)
        gate = self.gate()
        for rate in (math.nan, -0.1, 1.1):
            with self.assertRaises(ValueError):
                gate.observe(30, rate)

    def test_optional_student_exploration_after_warmup(self):
        from ghand.tasks.dual_ball_rotation.handover import rollout_teacher_probability as prob

        self.assertEqual(prob(1.0, 30, 30, 0.2), 1.0)
        self.assertEqual(prob(1.0, 31, 30, 0.2), 0.8)
        self.assertEqual(prob(0.6, 31, 30, 0.2), 0.6)
        self.assertEqual(prob(0.0, 31, 30, 0.2), 0.0)
        self.assertEqual(prob(1.0, 31, 30, 0.0), 1.0)
        for fraction in (-0.1, 1.0, math.nan):
            with self.assertRaises(ValueError):
                prob(1.0, 31, 30, fraction)


if __name__ == "__main__":
    unittest.main()
