"""Teacher assistance controlled by student-only task evaluations."""

import math


class GatedHandover:
    def __init__(self, min_teacher_iterations=75, success_rate=0.8, patience=6):
        if min_teacher_iterations < 1 or patience < 1 or not math.isfinite(success_rate) or not 0 < success_rate <= 1:
            raise ValueError("Positive minimum/patience and a success threshold in (0,1] required")
        self.min_teacher_iterations = min_teacher_iterations
        self.success_rate = success_rate
        self.patience = patience
        self.probability = 1.0
        self.complete = False
        self.stop_reason = None
        self.pass_streak = 0
        self.stale_evaluations = 0
        self.best_rate = -1.0
        self.last_iteration = 0

    def state(self):
        return {key: value for key, value in vars(self).items()}

    def observe(self, iteration, rate):
        if not math.isfinite(rate) or not 0 <= rate <= 1 or iteration <= self.last_iteration:
            raise ValueError("Finite success rate and increasing evaluation iterations required")
        self.last_iteration = iteration
        if self.stop_reason:
            return self.state()
        if rate >= self.best_rate + 0.02:
            self.best_rate = rate
            self.stale_evaluations = 0
        else:
            self.stale_evaluations += 1
        if iteration < self.min_teacher_iterations:
            self.pass_streak = 0
            self.stale_evaluations = 0
            return self.state()
        self.pass_streak = self.pass_streak + 1 if rate >= self.success_rate else 0
        if self.pass_streak >= 2:
            if self.probability == 0.0:
                self.complete = True
                self.stop_reason = "student_validated"
            else:
                self.probability = round(max(0.0, self.probability - 0.2), 8)
            self.pass_streak = 0
            self.stale_evaluations = 0
        elif rate < 0.5 and self.probability < 1.0:
            self.probability = round(min(1.0, self.probability + 0.2), 8)
        if self.stale_evaluations >= self.patience and not self.complete:
            self.stop_reason = "student_stalled"
        return self.state()


def rollout_teacher_probability(scheduled, iteration, warmup_iterations, student_exploration=0.0):
    """Optional student rollouts collect correction labels even before task gates pass."""
    if not math.isfinite(student_exploration) or not 0 <= student_exploration < 1:
        raise ValueError("Student exploration must be in [0, 1)")
    if iteration <= warmup_iterations:
        return scheduled
    return min(scheduled, 1.0 - student_exploration)
