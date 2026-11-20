"""最大フローで求める「回復できる最大の数 N*」のテスト（予備バッテリーの配分も含む）."""
import itertools
import random
import unittest

from core.phase2_maxflow import max_recovery


def brute_force(eligible, base_capacity, spare_total, capacity_per_spare):
    """全部の担当と予備の配分を試して、最大の回復数を数える（小さい問題用の答え合わせ）."""
    sensors = list(eligible)
    vehicles = list(base_capacity)
    best = 0
    for n_spares in range(min(spare_total, len(vehicles)) + 1):
        for spares in itertools.combinations(vehicles, n_spares):
            capacity = {v: base_capacity[v] + (capacity_per_spare if v in spares else 0) for v in vehicles}
            # 各センサーは「担当しない(None)」か「担当できる車のどれか」
            for choice in itertools.product(*[[None] + list(eligible[s]) for s in sensors]):
                used = {}
                for v in choice:
                    if v is not None:
                        used[v] = used.get(v, 0) + 1
                if all(used[v] <= capacity[v] for v in used):
                    best = max(best, sum(v is not None for v in choice))
    return best


class MaxRecoveryTest(unittest.TestCase):
    def test_capacity_limits_the_count(self):
        result = max_recovery({'a': [0], 'b': [0], 'c': [0]}, {0: 2})
        self.assertEqual(result['count'], 2)

    def test_spare_goes_to_the_vehicle_that_needs_it(self):
        eligible = {'a': [0], 'b': [0], 'c': [0], 'd': [0], 'e': [1]}
        no_spare = max_recovery(eligible, {0: 2, 1: 2})
        with_spare = max_recovery(eligible, {0: 2, 1: 2}, spare_total=1, capacity_per_spare=2)
        self.assertEqual(no_spare['count'], 3)
        self.assertEqual(with_spare['count'], 5)
        self.assertEqual(with_spare['spares'], [0])

    def test_assignment_respects_eligibility_and_capacity(self):
        eligible = {'a': [0, 1], 'b': [1], 'c': [], 'd': [0]}
        result = max_recovery(eligible, {0: 1, 1: 1})
        self.assertEqual(result['count'], 2)
        for vehicle, sensors in result['assignment'].items():
            self.assertLessEqual(len(sensors), 1)
            for sensor in sensors:
                self.assertIn(vehicle, eligible[sensor])

    def test_matches_brute_force_on_random_small_cases(self):
        rng = random.Random(0)
        for _ in range(60):
            n_vehicles = rng.randint(1, 3)
            sensors = [f's{i}' for i in range(rng.randint(0, 5))]
            eligible = {s: [v for v in range(n_vehicles) if rng.random() < 0.6] for s in sensors}
            base = {v: rng.randint(0, 2) for v in range(n_vehicles)}
            spare_total = rng.randint(0, 2)
            per_spare = rng.randint(0, 2)
            group = {v: rng.choice(['B1', 'B2']) for v in range(n_vehicles)}
            result = max_recovery(eligible, base, spare_total=spare_total, capacity_per_spare=per_spare,
                                  vehicle_group=group)
            self.assertEqual(result['count'], brute_force(eligible, base, spare_total, per_spare))


if __name__ == '__main__':
    unittest.main()
