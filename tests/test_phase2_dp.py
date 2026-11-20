import itertools
import math
import time
import unittest

import numpy as np

from core.config import Config, FleetSpec
from core.phase2_dp import solve_vehicle
from core.routing import Flight, Instance, Vehicle


def small_instance(*, end_mode="recovery", flights_builtin=2):
    labels = ["base", "L0", "L1", "L2", "L3"]
    return Instance(
        vehicles=[Vehicle(0, "base-1", "base", "Base")],
        flights=[
            Flight(0, "F0", "L0", ["A"], 6, [2]),
            Flight(1, "F1", "L1", ["A"], 4, [2]),
            Flight(2, "F2", "L2", ["B"], 4, [1]),
            Flight(3, "F3", "L3", ["A", "B"], 9, [2, 7]),
        ],
        targets=["A", "B"],
        unreachable=[],
        drive=np.array([
            [0, 1, 3, 4, 5],
            [15, 0, 6, 8, 3],
            [1, 5, 0, 1, 3],
            [2, 2, 4, 0, 4],
            [3, 7, 5, 2, 0],
        ], dtype=float),
        labels=labels,
        point_of_base={"base": 0},
        point_of_flight=[1, 2, 3, 4],
        cfg=Config(fleet=FleetSpec(charge_min=3, flights_builtin=flights_builtin, launch_setup_min=2)),
        end_mode=end_mode,
    )


def enumerate_routes(inst, assigned, *, deadlines=None, return_deadline=None, forbidden_transitions=()):
    """Compute valid routes and their objective values directly from the raw timings."""
    deadlines = deadlines or {}
    return_deadline = math.inf if return_deadline is None else return_deadline
    fleet = inst.cfg.fleet
    valid = {}
    for count in range(fleet.flights_builtin + 1):
        for route in itertools.permutations(range(len(inst.flights)), count):
            sensors = [sensor for index in route for sensor in inst.flights[index].sensors]
            if len(sensors) != len(assigned) or set(sensors) != assigned:
                continue
            transitions = set(zip(route, route[1:]))
            if transitions.intersection(forbidden_transitions):
                continue

            point = inst.point_of_base[inst.vehicles[0].base_id]
            landed_at = 0
            recovery = {}
            for index in route:
                flight = inst.flights[index]
                launch_point = inst.point_of_flight[index]
                ready_at = landed_at + inst.drive[point, launch_point] + fleet.launch_setup_min
                if recovery:
                    ready_at = max(ready_at, landed_at + fleet.charge_min)
                recovery.update({sensor: ready_at + offset for sensor, offset in zip(flight.sensors, flight.recover_off)})
                landed_at = ready_at + flight.flight_min
                point = launch_point

            returned_at = landed_at + inst.drive[point, inst.point_of_base[inst.vehicles[0].base_id]]
            if not math.isfinite(returned_at) or returned_at > return_deadline:
                continue
            if any(at > deadlines.get(sensor, math.inf) for sensor, at in recovery.items()):
                continue
            valid[route] = returned_at if inst.end_mode == "return" else max(recovery.values(), default=0)
    return valid


class VehicleDPTests(unittest.TestCase):
    def assert_agrees_with_enumeration(self, inst, assigned, **constraints):
        valid = enumerate_routes(inst, assigned, **constraints)
        result = solve_vehicle(inst, 0, assigned, **constraints)
        if valid:
            self.assertEqual(result.status, "FEASIBLE")
            self.assertIn(result.route, valid)
            self.assertEqual(valid[result.route], min(valid.values()))
        else:
            self.assertEqual(result.status, "PROVEN_INFEASIBLE")
            self.assertEqual(result.route, ())
        return result

    def test_objective_can_choose_different_launch_points(self):
        for end_mode, expected in [("recovery", (0,)), ("return", (1,))]:
            with self.subTest(end_mode=end_mode):
                result = self.assert_agrees_with_enumeration(small_instance(end_mode=end_mode), {"A"})
                self.assertEqual(result.route, expected)

    def test_recovery_and_return_deadlines_include_equality(self):
        inst = small_instance()
        result = self.assert_agrees_with_enumeration(inst, {"A", "B"}, deadlines={"A": 7, "B": 13}, return_deadline=18)
        self.assertEqual(result.route, (1, 2))
        self.assert_agrees_with_enumeration(inst, {"A", "B"}, deadlines={"A": 7, "B": 12.9})
        self.assert_agrees_with_enumeration(inst, {"A", "B"}, deadlines={"A": 7, "B": 13}, return_deadline=17.9)

    def test_directed_transition_ban_can_change_the_best_route(self):
        result = self.assert_agrees_with_enumeration(
            small_instance(), {"A", "B"}, forbidden_transitions={(1, 2)},
        )
        self.assertEqual(result.route, (3,))

    def test_combined_flight_respects_each_sensor_deadline(self):
        inst = small_instance(flights_builtin=1)
        result = self.assert_agrees_with_enumeration(inst, {"A", "B"}, deadlines={"A": 9, "B": 14})
        self.assertEqual(result.route, (3,))
        self.assert_agrees_with_enumeration(inst, {"A", "B"}, deadlines={"A": 8.9, "B": 14})

    def test_route_must_allow_return_even_when_optimizing_recovery(self):
        inst = small_instance(flights_builtin=1)
        inst.drive[inst.point_of_flight[3], 0] = math.inf
        self.assert_agrees_with_enumeration(inst, {"A", "B"})

    def test_slower_label_can_preserve_a_flight_for_the_remaining_sensor(self):
        inst = small_instance(flights_builtin=3)
        inst.flights.extend([
            Flight(4, "F4", "L2", ["C"], 2, [1]),
            Flight(5, "F5", "L1", ["D"], 2, [1]),
        ])
        inst.targets.extend(["C", "D"])
        inst.point_of_flight.extend([3, 2])
        # F1 → F2 reaches F4 earlier than F3 alone but uses an extra flight.
        blocked = {(3, 5), (4, 3), (5, 3), (5, 4)}
        result = self.assert_agrees_with_enumeration(inst, {"A", "B", "C", "D"}, forbidden_transitions=blocked)
        self.assertEqual(result.route, (3, 4, 5))

    def test_empty_assignment(self):
        result = self.assert_agrees_with_enumeration(small_instance(), set(), return_deadline=0)
        self.assertEqual(result.route, ())

    def test_search_budgets_return_unknown(self):
        inst = small_instance()
        result = solve_vehicle(inst, 0, {"A", "B"}, max_states=0)
        self.assertEqual(result.status, "UNKNOWN")
        self.assertEqual(result.states, 0)
        result = solve_vehicle(inst, 0, {"A", "B"}, stop_at=time.monotonic() - 1)
        self.assertEqual(result.status, "UNKNOWN")
