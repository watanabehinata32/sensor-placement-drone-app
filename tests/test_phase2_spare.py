"""v0.2：予備バッテリー（1 台 0〜1 個・全体 B 個・積むと飛べる回数が増える）と上位の割当のテスト."""
import copy
import itertools
import random
import unittest
from unittest.mock import patch

from core.phase2_assignment import build_assignment_model
from core.phase2_hybrid import solve_hybrid
from core.routing import Instance, evaluate
from test_phase2_dp import enumerate_routes
from test_phase2_dp_v01 import floyd, make_instance, oracle_assignments, random_instance
from test_phase2_hybrid import decision_energies


def with_fleet(inst, **fleet):
    """inst の車・バッテリーの設定だけを変えた写しを作る."""
    inst = copy.copy(inst)
    inst.cfg = copy.deepcopy(inst.cfg)
    for key, value in fleet.items():
        setattr(inst.cfg.fleet, key, value)
    return inst


def exact_optimum_with_spares(inst):
    """全部の担当の分け方 × 予備の積み方 × 全部の順番を試して、(最大回復数, その数での最短の最後の回復) を出す."""
    fleet = inst.cfg.fleet
    n_vehicles = len(inst.vehicles)
    best = (0, 0.0)
    for owners in itertools.product([None] + list(range(n_vehicles)), repeat=len(inst.targets)):
        for n_spares in range(min(fleet.spare_total, n_vehicles) + 1):
            for spares in itertools.combinations(range(n_vehicles), n_spares):
                ends, count, ok = [], 0, True
                for vehicle in range(n_vehicles):
                    mine = {s for s, o in zip(inst.targets, owners) if o == vehicle}
                    if not mine:
                        continue
                    limit = fleet.flights_builtin + (fleet.flights_per_spare if vehicle in spares else 0)
                    single = Instance(**{**inst.__dict__, "vehicles": [inst.vehicles[vehicle]]})
                    single = with_fleet(single, flights_builtin=limit)
                    valid = enumerate_routes(single, mine)
                    if not valid:
                        ok = False
                        break
                    ends.append(min(valid.values()))
                    count += len(mine)
                if not ok:
                    continue
                candidate = (count, max(ends, default=0.0))
                if candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                    best = candidate
    return best


def oracle_hybrid(inst, **kwargs):
    with patch("core.phase2_hybrid.solve_assignment", side_effect=oracle_assignments):
        return solve_hybrid(inst, lambda i, r: None, deadline_step_min=0.01, max_iterations=60, **kwargs)


class SpareQuboTest(unittest.TestCase):
    """割当 QUBO：担当 x と予備 b を同時に決める。罰則 0 ⇔ 決まりを守っている."""

    def test_penalty_is_zero_exactly_for_valid_assignments(self):
        rng = random.Random(10)
        for case in range(25):
            sensors = ["a", "b", "c", "d"][:rng.randint(2, 4)]
            eligible = {s: [v for v in (0, 1) if rng.random() < 0.8] for s in sensors}
            costs = {(s, v): 0 for s, vs in eligible.items() for v in vs}
            problem = build_assignment_model(eligible, costs, allow_unserved=False,
                                             capacities={0: 1, 1: 1}, spare_capacity={0: 2, 1: 2},
                                             spare_total=rng.randint(0, 2))
            with self.subTest(case=case):
                for decisions, energy in decision_energies(problem):
                    valid = problem.decode_bits(decisions) is not None
                    self.assertEqual(abs(energy) < 1e-9, valid, (decisions, energy))

    def test_spare_variables_are_counted_in_the_size(self):
        problem = build_assignment_model({"a": [0], "b": [0], "c": [0]}, {("a", 0): 0, ("b", 0): 0, ("c", 0): 0},
                                         allow_unserved=False, capacities={0: 1}, spare_capacity={0: 2},
                                         spare_total=1)
        self.assertEqual(problem.size()["spare_variables"], 1)
        self.assertIsNotNone(problem.decode_bits((1, 1, 1)))      # 予備を積めば 3 基担当できる


class SpareHybridTest(unittest.TestCase):
    def base_case(self):
        # 1 台・内蔵 2 回・予備で +2 回。4 基を回るには予備が要る
        drive = floyd([[0, 3, 3, 3, 3], [3, 0, 2, 2, 2], [3, 2, 0, 2, 2], [3, 2, 2, 0, 2], [3, 2, 2, 2, 0]])
        flights = [(s, i, 6, 3) for i, s in enumerate("ABCD")]
        return make_instance(drive, flights, charge=5, setup=1, max_flights=2)

    def test_spare_lets_one_vehicle_recover_more(self):
        no_spare = oracle_hybrid(with_fleet(self.base_case(), spare_total=0, flights_per_spare=2))
        with_spare = oracle_hybrid(with_fleet(self.base_case(), spare_total=1, flights_per_spare=2))
        self.assertEqual(no_spare["steps"]["hybrid"]["max_recovered"], 2)
        hybrid = with_spare["steps"]["hybrid"]
        self.assertEqual(hybrid["max_recovered"], 4)
        self.assertTrue(hybrid["max_recovered_proven"])
        self.assertEqual(hybrid["spares"], ["B-1"])
        self.assertTrue(with_spare["eval"]["battery_ok"])
        self.assertTrue(with_spare["eval"]["vehicles"][0]["spare"])

    def test_evaluate_rejects_too_many_flights_or_spares(self):
        inst = with_fleet(self.base_case(), spare_total=1, flights_per_spare=2)
        self.assertFalse(evaluate(inst, {0: [0, 1, 2]})["battery_ok"])               # 予備なしで 3 回
        self.assertTrue(evaluate(inst, {0: [0, 1, 2]}, spares={0})["battery_ok"])
        inst = with_fleet(inst, spare_total=0)
        self.assertFalse(evaluate(inst, {0: [0, 1, 2]}, spares={0})["battery_ok"])   # 予備は 0 個

    def test_random_cases_match_exact_optimum(self):
        """2 台・予備あり／なしの小さい問題：最大回復数、その数での最短、証明済み下界 ≤ 最短、カットの正しさ."""
        rng = random.Random(11)
        checked = 0
        for case in range(60):
            inst, _ = random_instance(rng, n_sensors=rng.randint(2, 4), n_launch=rng.randint(2, 3), closed=True,
                                      bases=("B1", "B2"))
            inst = with_fleet(inst, flights_builtin=rng.randint(1, 2), spare_total=rng.randint(0, 1),
                              flights_per_spare=rng.randint(1, 2))
            count, optimum = exact_optimum_with_spares(inst)
            out = oracle_hybrid(inst)
            hybrid = out["steps"]["hybrid"]
            with self.subTest(case=case):
                self.assertEqual(hybrid["max_recovered"], count)
                self.assertTrue(out["eval"]["battery_ok"])
                if count:
                    checked += 1
                    self.assertLessEqual(hybrid["lower_bound_proven"], optimum + 1e-9)
                    self.assertAlmostEqual(out["eval"]["makespan_recovery"], optimum, delta=0.02)
                # カットは「飛べる回数を最大にしても、その締切に間に合わない」組だけ
                for cut in hybrid["cut_list"]:
                    vehicle = [v.vehicle_id for v in inst.vehicles].index(cut["vehicle"])
                    single = Instance(**{**inst.__dict__, "vehicles": [inst.vehicles[vehicle]]})
                    most = inst.cfg.fleet.flights_builtin + (inst.cfg.fleet.flights_per_spare
                                                             if inst.cfg.fleet.spare_total else 0)
                    single = with_fleet(single, flights_builtin=most)
                    limit = {s: cut["deadline"] for s in cut["sensors"]} if cut["deadline"] is not None else None
                    self.assertEqual(enumerate_routes(single, set(cut["sensors"]), deadlines=limit), {})
        self.assertGreater(checked, 30)

    def test_log_has_vehicle_details(self):
        out = oracle_hybrid(with_fleet(self.base_case(), spare_total=1, flights_per_spare=2))
        for record in out["steps"]["hybrid"]["iteration_log"]:
            for candidate in record["candidates"]:
                for vehicle in candidate.get("vehicles", []):
                    self.assertTrue({"vehicle", "assigned", "spare", "dp_seconds", "dp_states",
                                     "dp_transitions", "status"} <= vehicle.keys())


class UnprovenMaxRecoveryTest(unittest.TestCase):
    def test_more_recovery_is_still_searched_when_n_star_is_not_proven(self):
        # 最短経路の表になっていないので、最大フローは 1 基と見積もる（本当は 2 基回れる）
        drive = [[0, 1, float("inf")], [1, 0, 1], [1, 1, 0]]
        inst = make_instance(drive, [("A", 0, 6, 3), ("B", 1, 6, 3)], charge=0, setup=0)
        out = oracle_hybrid(inst)
        hybrid = out["steps"]["hybrid"]
        self.assertEqual(hybrid["max_recovered_by_flow"], 1)
        self.assertEqual(hybrid["max_recovered"], 2)      # アニーリングが 2 基の担当を見つけたら、N* を上げる
        self.assertEqual(out["missing"], [])


if __name__ == "__main__":
    unittest.main()
