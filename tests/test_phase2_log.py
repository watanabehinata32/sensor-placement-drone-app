"""反復探索のログ（steps.hybrid.iteration_log / totals）のテスト."""
import itertools
import random
import unittest
from unittest.mock import patch

import numpy as np

from core.phase2_assignment import build_assignment_model
from core.phase2_hybrid import solve_hybrid
from test_phase2_dp_v01 import oracle_assignments, random_instance

KEYS = {"iteration", "stage", "deadline", "candidates", "dp_calls", "dp_seconds", "dp_states", "dp_transitions",
        "cache_hits", "best_end", "elapsed_seconds"}
QUBO_KEYS = {"lower_bound_proven", "lower_bound_search", "cuts_total", "cuts_active", "qubo", "anneal_seconds",
             "anneal_calls_total", "solutions_returned", "solutions_valid", "violations"}


class IterationLogTest(unittest.TestCase):
    def run_case(self, seed):
        inst, _ = random_instance(random.Random(seed), n_sensors=3, n_launch=3, closed=True, bases=("B1", "B2"))
        with patch("core.phase2_hybrid.solve_assignment", side_effect=oracle_assignments):
            return solve_hybrid(inst, lambda i, r: None, deadline_step_min=0.5, max_iterations=20)

    def test_every_iteration_has_the_requested_items(self):
        for seed in range(10):
            hybrid = self.run_case(seed)["steps"]["hybrid"]
            log = hybrid["iteration_log"]
            with self.subTest(seed=seed):
                self.assertEqual(log[0]["stage"], "段1 最大フロー")
                for record in log:
                    self.assertTrue(KEYS <= record.keys(), KEYS - record.keys())
                    if record["iteration"] > 0:                       # 割当 QUBO を解いた回
                        self.assertTrue(QUBO_KEYS <= record.keys(), QUBO_KEYS - record.keys())
                        size = record["qubo"]
                        self.assertGreater(size["variables"], 0)
                        self.assertGreaterEqual(size["auxiliary"], 0)
                        self.assertGreaterEqual(size["quadratic"], 0)
                        if record["lower_bound_proven"] is not None:
                            self.assertLessEqual(record["lower_bound_proven"], record["lower_bound_search"] + 1e-9)
                    for candidate in record["candidates"]:
                        self.assertIn("status", candidate)
                        if candidate["status"] != "検証済み":
                            self.assertIsNone(candidate["end"])       # 実際の時刻が無いものは null
                # 経過時間は増えるだけ、段2の最良時刻は減るだけ
                elapsed = [r["elapsed_seconds"] for r in log]
                self.assertEqual(elapsed, sorted(elapsed))
                stage2 = [r["best_end"] for r in log if r["deadline"] is not None]
                self.assertEqual(stage2, sorted(stage2, reverse=True))
                # 合計は各行の合計と一致する
                totals = hybrid["totals"]
                self.assertEqual(totals["dp_states"], sum(r["dp_states"] for r in log))
                self.assertEqual(totals["dp_transitions"], sum(r["dp_transitions"] for r in log))
                self.assertEqual(totals["anneal_calls"], sum(1 for r in log if r["iteration"] > 0))


class ViolationReportTest(unittest.TestCase):
    def test_violations_explain_exactly_the_rejected_bit_strings(self):
        """決まりを破った解だけが「違反あり」になり、その内容が返る."""
        problem = build_assignment_model(
            {"a": [0, 1], "b": [0], "c": [1]}, {("a", 0): 1, ("a", 1): 2, ("b", 0): 1, ("c", 1): 1},
            allow_unserved=True, cuts=[(0, {"a", "b"})], capacities={0: 1, 1: 1}, unserved_count=1,
        )
        for bits in itertools.product((0, 1), repeat=len(problem.pairs)):
            reasons = problem.violations(bits)
            self.assertEqual(problem.decode_bits(bits) is None, bool(reasons))
        too_many = tuple(1 if vehicle == 0 else 0 for _, vehicle in problem.pairs)
        self.assertTrue(problem.violations(too_many))

    def test_size_counts_auxiliary_and_quadratic_terms(self):
        problem = build_assignment_model({"a": [0, 1], "b": [0, 1]}, {("a", 0): 1, ("a", 1): 1, ("b", 0): 1,
                                                                     ("b", 1): 1},
                                         allow_unserved=False, capacities={0: 1, 1: 1})
        size = problem.size()
        self.assertEqual(size["variables"], 4)
        self.assertEqual(size["auxiliary"], 2)          # 上限 1 の不等式ごとに補助変数 1 個
        self.assertGreater(size["quadratic"], 0)


if __name__ == "__main__":
    unittest.main()
