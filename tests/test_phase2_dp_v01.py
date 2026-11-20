"""v0.1：1 台の巡回 DP（core/phase2_dp.py）の正しさを、いろいろな角度から確かめるテスト.

答え合わせには、DP とは別に書いた「全部の順番を試す」計算（test_phase2_dp.enumerate_routes）を使う。
小さい問題をたくさん乱数で作り、DP の答えがいつも全列挙の最短と同じになるかを見る。
"""
import itertools
import math
import random
import unittest
from unittest.mock import patch

import numpy as np

from core.config import Config, FleetSpec
from core.phase2_bounds import drive_is_shortest_path_closed, vehicle_lower_bound
from core.phase2_dp import solve_vehicle
from core.phase2_hybrid import solve_hybrid
from core.routing import Flight, Instance, Vehicle, evaluate, route_times
from test_phase2_dp import enumerate_routes

INF = math.inf


def make_instance(drive, flights, *, charge=30, setup=10, max_flights=5, end_mode="recovery", bases=("B",)):
    """小さい問題を作る. drive の行・列は「拠点（bases の順）→ 離陸地点 L0, L1, ...」.

    flights: [(センサー, 離陸地点の番号, 飛行の分, 離陸から回復までの分), ...]（1 飛行 1 基）
    """
    n_launch = len(drive) - len(bases)
    labels = list(bases) + [f"L{i}" for i in range(n_launch)]
    flight_objs = [Flight(i, f"F{i}", f"L{lp}", [s], float(fm), [float(ro)])
                   for i, (s, lp, fm, ro) in enumerate(flights)]
    vehicles = [Vehicle(i, f"{b}-1", b, b) for i, b in enumerate(bases)]
    return Instance(
        vehicles=vehicles, flights=flight_objs, targets=sorted({f[0] for f in flights}), unreachable=[],
        drive=np.array(drive, dtype=float), labels=labels,
        point_of_base={b: i for i, b in enumerate(bases)},
        point_of_flight=[len(bases) + lp for _, lp, _, _ in flights],
        cfg=Config(fleet=FleetSpec(charge_min=charge, flights_builtin=max_flights, launch_setup_min=setup)),
        end_mode=end_mode,
    )


def floyd(drive):
    """走行時間の表を「最短経路の表」にする（実データの前処理と同じ性質にそろえる）."""
    d = np.array(drive, dtype=float)
    for k in range(len(d)):
        d = np.minimum(d, d[:, [k]] + d[[k], :])
    return d


def random_instance(rng, *, n_sensors, n_launch, closed=False, end_mode="recovery", bases=("B",)):
    """乱数で小さい問題を作る. 向きのある道路（行きと帰りで時間が違う）と、通れない区間（inf）を含む."""
    size = len(bases) + n_launch
    drive = [[0 if a == b else (INF if rng.random() < 0.15 else rng.randint(1, 30)) for b in range(size)]
             for a in range(size)]
    if closed:
        drive = floyd(drive)
    flights = []
    sensors = [chr(ord("a") + i) for i in range(n_sensors)]
    for s in sensors:
        for lp in rng.sample(range(n_launch), rng.randint(1, min(3, n_launch))):   # 1 基に離陸候補 1〜3 か所
            recover = rng.randint(3, 9)
            flights.append((s, lp, recover + rng.randint(0, 6), recover))
    return make_instance(drive, flights, charge=rng.choice([0, 5, 30]), setup=rng.choice([0, 2, 10]),
                         max_flights=rng.randint(1, 4), end_mode=end_mode, bases=bases), sensors


class DPAgreesWithEnumerationTest(unittest.TestCase):
    """① 小さい問題で、DP の最短 ＝ 全列挙の最短（できない ⇔ 全列挙でも 0 本）."""

    def check(self, inst, assigned, **constraints):
        valid = enumerate_routes(inst, set(assigned), **constraints)
        result = solve_vehicle(inst, 0, assigned, **constraints)
        if valid:
            self.assertEqual(result.status, "FEASIBLE")
            self.assertAlmostEqual(valid[result.route], min(valid.values()), places=9)
        else:
            self.assertEqual(result.status, "PROVEN_INFEASIBLE")
        return result, valid

    def test_random_small_cases(self):
        rng = random.Random(1)
        for case in range(300):
            end_mode = rng.choice(["recovery", "return"])
            inst, sensors = random_instance(rng, n_sensors=rng.randint(0, 3), n_launch=rng.randint(1, 4),
                                            closed=rng.random() < 0.5, end_mode=end_mode)
            assigned = set(rng.sample(sensors, rng.randint(0, len(sensors))))
            constraints = {}
            if rng.random() < 0.4:   # 締切つき（等号はセーフ）
                constraints["deadlines"] = {s: rng.randint(10, 120) for s in assigned}
            if rng.random() < 0.3:
                constraints["return_deadline"] = rng.randint(10, 200)
            with self.subTest(case=case):
                self.check(inst, assigned, **constraints)


class LaunchPointChoiceTest(unittest.TestCase):
    """② 離陸候補が複数あるとき、「その 1 基だけ見て最速」ではなく、全体で最速の地点を選ぶ."""

    def test_farther_launch_point_wins_because_the_next_one_is_close(self):
        # A は L0（拠点から 1 分）と L1（拠点から 5 分）のどちらからでも飛べる。B は L2 からだけ。
        # L0→L2 は 20 分、L1→L2 は 1 分。A だけ見れば L0 が早いが、B まで含めると L1 が早い。
        drive = [[0, 1, 5, 30],
                 [1, 0, 30, 20],
                 [5, 30, 0, 1],
                 [30, 20, 1, 0]]
        inst = make_instance(drive, [("A", 0, 6, 3), ("A", 1, 6, 3), ("B", 2, 6, 3)], charge=0, setup=0)
        result = solve_vehicle(inst, 0, {"A", "B"})
        self.assertEqual(result.route, (1, 2))                      # L1 → L2
        _, end_recovery, _ = route_times(inst, 0, list(result.route))
        self.assertEqual(end_recovery, 15)                          # 5 + 6 + 1 + 3
        self.assertEqual(min(enumerate_routes(inst, {"A", "B"}).values()), 15)


class TimeOverlapTest(unittest.TestCase):
    """③ 次の離陸 ＝ 着陸 ＋ max(充電, 走行 ＋ 準備)。充電は走行・準備と並行して進む."""

    def two_flights(self, drive_between, charge, setup):
        drive = [[0, 4, 4], [4, 0, drive_between], [4, drive_between, 0]]
        inst = make_instance(drive, [("A", 0, 10, 5), ("B", 1, 10, 5)], charge=charge, setup=setup)
        launches, _, _ = route_times(inst, 0, [0, 1])
        return launches

    def test_charge_longer_than_drive_and_setup(self):
        # 着陸 4+10+10=24。走行 5＋準備 10＝15 より充電 30 が長い → 次の離陸は 24＋30＝54
        self.assertEqual(self.two_flights(5, 30, 10), [14, 54])

    def test_drive_and_setup_longer_than_charge(self):
        # 走行 25＋準備 10＝35 が充電 30 より長い → 24＋35＝59（充電待ちは無い）
        self.assertEqual(self.two_flights(25, 30, 10), [14, 59])

    def test_first_flight_waits_only_for_drive_and_setup(self):
        # 出発時は満充電なので、最初の離陸は 走行 4 ＋ 準備 10 ＝ 14（充電は待たない）
        self.assertEqual(self.two_flights(5, 30, 10)[0], 14)

    def test_dp_uses_the_same_rule(self):
        for between, charge, setup in [(5, 30, 10), (25, 30, 10), (0, 0, 0), (3, 7, 2)]:
            drive = [[0, 4, 4], [4, 0, between], [4, between, 0]]
            inst = make_instance(drive, [("A", 0, 10, 5), ("B", 1, 10, 5)], charge=charge, setup=setup)
            with self.subTest(between=between, charge=charge, setup=setup):
                result = solve_vehicle(inst, 0, {"A", "B"})
                self.assertAlmostEqual(min(enumerate_routes(inst, {"A", "B"}).values()),
                                       route_times(inst, 0, list(result.route))[1])


class RecoveryLandingReturnTest(unittest.TestCase):
    """④ 「最後の回復」「着陸」「帰着」は別の時刻。目的は最後の回復（帰着は別に出すだけ）."""

    def test_three_times_are_different(self):
        drive = [[0, 6], [7, 0]]                                     # 行き 6 分、帰り 7 分
        inst = make_instance(drive, [("A", 0, 9, 4)], charge=0, setup=10)
        ev = evaluate(inst, {0: [0]})
        times = {e["kind"]: e["t"] for e in ev["vehicles"][0]["events"]}
        self.assertEqual(times["離陸"], 16)
        self.assertEqual(times["回復"], 20)                            # 離陸 ＋ 回復まで 4
        self.assertEqual(times["着陸"], 25)                            # 離陸 ＋ 飛行 9
        self.assertEqual(times["帰着"], 32)                            # 着陸 ＋ 帰り 7
        self.assertEqual(ev["makespan_recovery"], 20)
        self.assertEqual(ev["makespan_return"], 32)

    def test_best_recovery_route_is_not_the_best_return_route(self):
        # A は L0（回復は早いが帰りが遠い）と L1（回復は遅いが帰りが近い）から飛べる
        drive = [[0, 2, 5], [40, 0, 40], [5, 40, 0]]
        inst = make_instance(drive, [("A", 0, 9, 4), ("A", 1, 9, 4)], charge=0, setup=0)
        self.assertEqual(solve_vehicle(inst, 0, {"A"}).route, (0,))
        inst.end_mode = "return"
        self.assertEqual(solve_vehicle(inst, 0, {"A"}).route, (1,))


class RoadCutAndDirectionTest(unittest.TestCase):
    """⑤ 道路の切断（inf）と、向きのある道路（行きと帰りで違う）."""

    def test_one_way_road_forces_the_order(self):
        # L0→L1 は通れるが L1→L0 は切断。A（L0）→ B（L1）の順しかできない
        drive = [[0, 3, 3], [3, 0, 2], [3, INF, 0]]
        inst = make_instance(drive, [("A", 0, 6, 3), ("B", 1, 6, 3)], charge=0, setup=0)
        self.assertEqual(solve_vehicle(inst, 0, {"A", "B"}).route, (0, 1))
        self.assertEqual(set(enumerate_routes(inst, {"A", "B"})), {(0, 1)})

    def test_cut_return_road_makes_it_infeasible(self):
        drive = [[0, 3], [INF, 0]]                                   # 行けるが帰れない
        inst = make_instance(drive, [("A", 0, 6, 3)], charge=0, setup=0)
        self.assertEqual(solve_vehicle(inst, 0, {"A"}).status, "PROVEN_INFEASIBLE")

    def test_cut_road_between_launch_points(self):
        # A と B の離陸地点の間がどちら向きも切断 → 1 台で両方は回れない（1 つずつなら回れる）
        drive = [[0, 3, 3], [3, 0, INF], [3, INF, 0]]
        inst = make_instance(drive, [("A", 0, 6, 3), ("B", 1, 6, 3)], charge=0, setup=0)
        self.assertEqual(solve_vehicle(inst, 0, {"A", "B"}).status, "PROVEN_INFEASIBLE")
        self.assertEqual(solve_vehicle(inst, 0, {"A"}).status, "FEASIBLE")


class LowerBoundTest(unittest.TestCase):
    """⑥ 下界は DP の厳密な最短を超えない（超えたら「証明済み下界」が嘘になる）."""

    def test_vehicle_lower_bound_never_exceeds_dp(self):
        rng = random.Random(2)
        for case in range(300):
            inst, sensors = random_instance(rng, n_sensors=rng.randint(1, 3), n_launch=rng.randint(1, 4),
                                            closed=rng.random() < 0.5)
            assigned = set(rng.sample(sensors, rng.randint(1, len(sensors))))
            result = solve_vehicle(inst, 0, assigned)
            if result.status != "FEASIBLE":
                continue
            optimum = route_times(inst, 0, list(result.route))[1]
            with self.subTest(case=case):
                self.assertLessEqual(vehicle_lower_bound(inst, assigned), optimum + 1e-9)


def exact_two_vehicle_optimum(inst):
    """2 台の小さい問題で、全部の担当の分け方 × 全部の順番を試して、(最大回復数, その数での最短の最後の回復) を出す."""
    best = (0, 0.0)
    targets = inst.targets
    for owners in itertools.product([None, 0, 1], repeat=len(targets)):
        ends, count, ok = [], 0, True
        for vehicle in (0, 1):
            mine = {s for s, o in zip(targets, owners) if o == vehicle}
            if not mine:
                continue
            single = Instance(**{**inst.__dict__, "vehicles": [inst.vehicles[vehicle]]})
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


def oracle_assignments(problem, client):
    """「アニーリングが、条件を満たす担当を全部見つけてくれた」場合の代わり（小さい問題だけ）."""
    found = []
    for bits in itertools.product((0, 1), repeat=len(problem.pairs)):
        assignment = problem.decode_bits(bits)
        if assignment is not None:
            found.append(assignment)
    return found


class HybridBoundsAndCutsTest(unittest.TestCase):
    """⑥⑦ 2 台の小さい問題で：
    ・証明済み下界 ≤ 本当の最短 ≤ 見つけた計画
    ・N* を「証明済み」と言ったときは、本当に最大
    ・アニーリングが全部見つける場合、締切での除外・カットで最適解を取りこぼさない（二分探索が本当の最短に届く）
    """

    def test_random_two_vehicle_cases(self):
        rng = random.Random(3)
        checked = 0
        for case in range(80):
            inst, _ = random_instance(rng, n_sensors=rng.randint(1, 3), n_launch=rng.randint(2, 3),
                                      closed=True, bases=("B1", "B2"))
            count, optimum = exact_two_vehicle_optimum(inst)
            with patch("core.phase2_hybrid.solve_assignment", side_effect=oracle_assignments):
                out = solve_hybrid(inst, lambda i, r: None, deadline_step_min=0.01, max_iterations=50)
            hybrid = out["steps"]["hybrid"]
            with self.subTest(case=case):
                self.assertEqual(hybrid["max_recovered"], count)
                if hybrid["max_recovered_proven"]:
                    self.assertEqual(hybrid["max_recovered"], count)
                if count > 0:
                    checked += 1
                    self.assertLessEqual(hybrid["lower_bound_proven"], optimum + 1e-9)
                    self.assertGreaterEqual(out["eval"]["makespan_recovery"], optimum - 0.01)
                    # 全部見つかるなら、二分探索は本当の最短（刻み 0.01 分）まで届く
                    self.assertLessEqual(out["eval"]["makespan_recovery"], optimum + 0.02)
        self.assertGreater(checked, 30)

    def test_superset_of_a_too_slow_set_is_also_too_slow(self):
        """部分集合カット（設計書 8 節）が安全である根拠：最短経路の表なら、担当を増やして早くなることは無い."""
        rng = random.Random(4)
        for case in range(200):
            inst, sensors = random_instance(rng, n_sensors=3, n_launch=rng.randint(1, 4), closed=True)
            inst.cfg.fleet.flights_builtin = 3
            times = {}
            for size in range(1, 4):
                for subset in itertools.combinations(sensors, size):
                    valid = enumerate_routes(inst, set(subset))
                    times[frozenset(subset)] = min(valid.values()) if valid else INF
            with self.subTest(case=case):
                for small, small_time in times.items():
                    for big, big_time in times.items():
                        if small < big:
                            self.assertLessEqual(small_time, big_time + 1e-9)


class MaxRecoveryProofConditionTest(unittest.TestCase):
    """N* を「証明済み」と言ってよい条件（ひなたさん指示：往復・離陸地点間の移動・活動時間の上限）."""

    def test_shortest_path_check(self):
        self.assertTrue(drive_is_shortest_path_closed(floyd([[0, 1, 5], [1, 0, 1], [5, 1, 0]])))
        self.assertFalse(drive_is_shortest_path_closed([[0, 1, 5], [1, 0, 1], [5, 1, 0]]))   # 0→2 は 0→1→2 の 2 が最短
        self.assertFalse(drive_is_shortest_path_closed([[0, 1, INF], [1, 0, 1], [1, 1, 0]]))

    def test_not_proven_when_the_table_is_not_shortest_paths(self):
        # 拠点 → L1 は直接は inf だが、L0 を経由すれば行ける表（最短経路の表になっていない）。
        # 最大フローは B を担当できないと見て N* = 1 と出すが、本当は A → B で 2 基回れる。
        # → この表では「証明済み」と言ってはいけない
        drive = [[0, 1, INF], [1, 0, 1], [1, 1, 0]]
        inst = make_instance(drive, [("A", 0, 6, 3), ("B", 1, 6, 3)], charge=0, setup=0)
        self.assertEqual(solve_vehicle(inst, 0, {"A", "B"}).status, "FEASIBLE")
        with patch("core.phase2_hybrid.solve_assignment", return_value=[]):
            out = solve_hybrid(inst, lambda i, r: None)
        hybrid = out["steps"]["hybrid"]
        self.assertEqual(hybrid["max_recovered_by_flow"], 1)
        self.assertFalse(hybrid["max_recovered_proven"])

    def test_not_proven_when_return_is_impossible(self):
        # 行けるが帰れない離陸地点しかない → 最大フローは 1 と言うが、DP で計画にできない → 未証明
        drive = floyd([[0, 3], [INF, 0]])
        inst = make_instance(drive, [("A", 0, 6, 3)], charge=0, setup=0)
        with patch("core.phase2_hybrid.solve_assignment", return_value=[]):
            out = solve_hybrid(inst, lambda i, r: None)
        hybrid = out["steps"]["hybrid"]
        self.assertEqual(hybrid["max_recovered_by_flow"], 1)
        self.assertEqual(hybrid["max_recovered"], 0)
        self.assertFalse(hybrid["max_recovered_proven"])

    def test_not_proven_when_activity_time_limit_blocks_the_flow_plan(self):
        # 回復の締切（活動時間の上限）があると、最大フローの担当がそのままでは間に合わないことがある
        drive = floyd([[0, 3, 3], [3, 0, 50], [3, 50, 0]])
        inst = make_instance(drive, [("A", 0, 6, 3), ("B", 1, 6, 3)], charge=0, setup=0)
        with patch("core.phase2_hybrid.solve_assignment", return_value=[]):
            out = solve_hybrid(inst, lambda i, r: None, deadlines={"A": 10, "B": 10})
        hybrid = out["steps"]["hybrid"]
        self.assertEqual(hybrid["max_recovered_by_flow"], 2)
        self.assertLess(hybrid["max_recovered"], 2)
        self.assertFalse(hybrid["max_recovered_proven"])


class RestoredRouteValidationTest(unittest.TestCase):
    """⑧ DP が返したルートを、時刻表（routing.evaluate）に直して、DP を使わずにルールどおりか確かめる."""

    def test_random_routes_follow_all_rules(self):
        rng = random.Random(5)
        checked = 0
        for case in range(300):
            inst, sensors = random_instance(rng, n_sensors=rng.randint(1, 3), n_launch=rng.randint(1, 4),
                                            closed=rng.random() < 0.5)
            assigned = set(rng.sample(sensors, rng.randint(1, len(sensors))))
            result = solve_vehicle(inst, 0, assigned)
            if result.status != "FEASIBLE":
                continue
            checked += 1
            with self.subTest(case=case):
                self.validate(inst, assigned, list(result.route))
        self.assertGreater(checked, 100)

    def validate(self, inst, assigned, route):
        fleet = inst.cfg.fleet
        ev = evaluate(inst, {0: route})
        events = ev["vehicles"][0]["events"]
        times = [e["t"] for e in events]
        self.assertEqual(times, sorted(times))                              # 時刻が順に並ぶ
        self.assertLessEqual(len(route), fleet.flights_builtin)               # 飛べる回数以内
        self.assertEqual(set(ev["recover_time"]), assigned)                   # 担当をちょうど 1 回ずつ
        self.assertEqual(ev["duplicated"], [])
        point = inst.point_of_base["B"]
        landed = None
        for k, f in enumerate(route):
            flight = inst.flights[f]
            launch = [e["t"] for e in events if e["kind"] == "離陸"][k]
            left = 0.0 if landed is None else landed
            arrive = left + inst.drive[point, inst.point_of_flight[f]]
            self.assertTrue(math.isfinite(arrive))                            # 通れる道だけ
            self.assertGreaterEqual(launch + 1e-9, arrive + fleet.launch_setup_min)   # 準備が終わってから離陸
            if landed is not None:
                self.assertGreaterEqual(launch + 1e-9, landed + fleet.charge_min)     # 充電が終わってから離陸
            self.assertLessEqual(flight.flight_min, inst.cfg.drone.max_flight_min + 1e-9)   # 1 回 15 分以内
            self.assertAlmostEqual(ev["recover_time"][flight.sensors[0]], launch + flight.recover_off[0])
            landed = launch + flight.flight_min
            point = inst.point_of_flight[f]
        back = landed + inst.drive[point, inst.point_of_base["B"]]
        self.assertTrue(math.isfinite(back))                                  # 拠点に帰れる
        self.assertAlmostEqual(ev["makespan_return"], round(back, 2))
        self.assertAlmostEqual(ev["makespan_recovery"], round(max(ev["recover_time"].values()), 2))
        self.assertAlmostEqual(ev["makespan_recovery"], round(min(enumerate_routes(inst, assigned).values()), 2))


if __name__ == "__main__":
    unittest.main()
