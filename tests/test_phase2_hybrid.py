import itertools
from math import prod
import unittest
from unittest.mock import patch

import numpy as np

from core.config import Config, FleetSpec
from core.local_client import LocalSAClient
from core.phase2_assignment import build_assignment_model
from core.phase2_hybrid import solve_hybrid
from core.routing import Flight, Instance, Vehicle


def two_vehicle_instance():
    return Instance(
        vehicles=[Vehicle(0, "west-1", "west", "West"), Vehicle(1, "east-1", "east", "East")],
        flights=[
            Flight(0, "alpha-flight", "alpha-launch", ["alpha"], 5, [3]),
            Flight(1, "beta-flight", "beta-launch", ["beta"], 5, [3]),
        ],
        targets=["alpha", "beta"],
        unreachable=[],
        drive=np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ]),
        labels=["west", "east", "alpha-launch", "beta-launch"],
        point_of_base={"west": 0, "east": 1},
        point_of_flight=[2, 3],
        cfg=Config(fleet=FleetSpec(charge_min=0, flights_builtin=2, launch_setup_min=0)),
    )


def decision_energies(problem):
    """Minimize the generated polynomial over its auxiliary bits for each decision."""
    coefficients = problem.model.as_dict()
    decision_ids = [variable.variables[0].id for variable in problem.variables]
    auxiliary_ids = sorted({idx for key in coefficients for idx in key} - set(decision_ids))
    for decisions in itertools.product((0, 1), repeat=len(decision_ids)):
        energy = float("inf")
        for slack in itertools.product((0, 1), repeat=len(auxiliary_ids)):
            bits = dict(zip(decision_ids + auxiliary_ids, decisions + slack))
            energy = min(energy, sum(coefficient * prod(bits[idx] for idx in key)
                                     for key, coefficient in coefficients.items()))
        yield decisions, energy


class AssignmentModelTest(unittest.TestCase):
    def test_exact_cut_does_not_forbid_superset(self):
        eligible = {"alpha": (0, 1), "beta": (0, 1)}
        costs = {(sensor, vehicle): 0 for sensor, vehicles in eligible.items() for vehicle in vehicles}
        problem = build_assignment_model(
            eligible, costs, allow_unserved=False, cuts=[(0, frozenset({"alpha"}))],
        )
        energies = dict(decision_energies(problem))

        def decision(alpha_vehicle, beta_vehicle):
            selected = {("alpha", alpha_vehicle), ("beta", beta_vehicle)}
            return tuple(int(pair in selected) for pair in problem.pairs)

        forbidden = decision(0, 1)
        superset = decision(0, 0)
        alternative = decision(1, 0)
        self.assertIsNone(problem.decode_bits(forbidden))
        self.assertEqual(problem.decode_bits(superset), {0: frozenset({"alpha", "beta"})})
        self.assertEqual(problem.decode_bits(alternative), {0: frozenset({"beta"}), 1: frozenset({"alpha"})})
        self.assertEqual(energies[superset], energies[alternative])
        self.assertGreater(energies[forbidden], energies[superset])

    def test_generated_energy_prefers_more_recovery_to_any_proxy_cost(self):
        eligible = {"alpha": (0,), "beta": (0,)}
        costs = {("alpha", 0): 50, ("beta", 0): 100}
        problem = build_assignment_model(eligible, costs, allow_unserved=True)
        all_recovered = {0: frozenset({"alpha", "beta"})}
        candidates = list(decision_energies(problem))
        best_energy = min(energy for _, energy in candidates)
        winners = [problem.decode_bits(bits) for bits, energy in candidates if energy == best_energy]
        self.assertTrue(winners)
        self.assertTrue(all(assignment == all_recovered for assignment in winners))

    def test_generated_energy_enforces_capacity_even_with_unserved_penalty(self):
        eligible = {"alpha": (0,), "beta": (0,), "gamma": (0,)}
        costs = {(sensor, 0): 0 for sensor in eligible}
        problem = build_assignment_model(eligible, costs, allow_unserved=True, capacities={0: 1})

        candidates = list(decision_energies(problem))
        best_energy = min(energy for _, energy in candidates)
        winners = [problem.decode_bits(bits) for bits, energy in candidates if energy == best_energy]

        self.assertEqual(len(winners), 3)
        for assignment in winners:
            self.assertIsNotNone(assignment)
            self.assertEqual(len(assignment[0]), 1)


    def test_fixed_unserved_count_keeps_recovery_number(self):
        # 回復数を 1 基に固定すると、2 基とも回る割当も 0 基の割当も罰則がかかる
        eligible = {"alpha": (0,), "beta": (0,)}
        costs = {("alpha", 0): 50, ("beta", 0): 100}
        problem = build_assignment_model(eligible, costs, allow_unserved=True, unserved_count=1)
        candidates = list(decision_energies(problem))
        best_energy = min(energy for _, energy in candidates)
        winners = [problem.decode_bits(bits) for bits, energy in candidates if energy == best_energy]
        self.assertEqual(winners, [{0: frozenset({"alpha"})}])
        for bits, _ in candidates:
            assignment = problem.decode_bits(bits)
            if assignment is not None:
                self.assertEqual(sum(len(s) for s in assignment.values()), 1)


class HybridSearchTest(unittest.TestCase):
    def test_deadline_binary_search_shortens_end_time(self):
        # 段1では 1 台に 2 基とも任せる（最後の回復 8 分）。段2の二分探索で 2 台に分ける計画（4 分）に縮む
        one_car = {0: frozenset({"alpha", "beta"})}
        two_cars = {0: frozenset({"alpha"}), 1: frozenset({"beta"})}

        def sampled_assignments(problem, client):
            if sampled.call_count == 1:
                order = [one_car]
            else:
                order = [two_cars, one_car]
            for candidate in order:
                selected = {(sensor, vehicle) for vehicle, sensors in candidate.items() for sensor in sensors}
                if not selected <= set(problem.pairs):
                    continue
                bits = tuple(int(pair in selected) for pair in problem.pairs)
                if problem.decode_bits(bits) == candidate:
                    return [candidate]
            return []

        with patch("core.phase2_hybrid.solve_assignment", side_effect=sampled_assignments) as sampled:
            result = solve_hybrid(
                two_vehicle_instance(), lambda iteration, remaining_seconds: None, allow_unserved=False,
            )

        hybrid = result["steps"]["hybrid"]
        self.assertEqual(result["eval"]["recover_time"], {"alpha": 3, "beta": 4})
        self.assertEqual(hybrid["max_recovered"], 2)
        self.assertEqual(hybrid["deadline_upper"], 4)
        self.assertTrue(hybrid["deadline_history"][0]["found"])
        self.assertEqual(hybrid["stop_reason"], "DEADLINE_CONVERGED")

    def test_local_annealer_and_dp_respect_forbidden_assignments(self):
        result = solve_hybrid(
            two_vehicle_instance(), lambda iteration, remaining_seconds: LocalSAClient(sweeps=30),
            allow_unserved=False, deadlines={"alpha": 4, "beta": 4},
            forbidden_assignments=[("alpha", 1), ("beta", 0)], max_iterations=1,
        )

        self.assertEqual(result["eval"]["recover_time"], {"alpha": 3, "beta": 4})
        self.assertEqual(result["missing"], [])
        self.assertTrue(result["steps"]["hybrid"]["has_solution"])

    def test_failed_deadline_assignment_is_cut_and_reassigned(self):
        instance = two_vehicle_instance()
        first = {0: frozenset({"alpha", "beta"})}
        second = {0: frozenset({"alpha"}), 1: frozenset({"beta"})}

        def sampled_assignments(problem, client):
            candidate = first if sampled.call_count == 1 else second
            selected = {(sensor, vehicle) for vehicle, sensors in candidate.items() for sensor in sensors}
            bits = tuple(int(pair in selected) for pair in problem.pairs)
            self.assertEqual(problem.decode_bits(bits), candidate)
            if sampled.call_count == 2:
                old = {(sensor, 0) for sensor in first[0]}
                self.assertIsNone(problem.decode_bits(tuple(int(pair in old) for pair in problem.pairs)))
            return [candidate]

        with patch("core.phase2_hybrid.solve_assignment", side_effect=sampled_assignments) as sampled:
            result = solve_hybrid(
                instance, lambda iteration, remaining_seconds: None, allow_unserved=False,
                deadlines={"alpha": 4, "beta": 4}, max_iterations=2,
            )

        self.assertEqual(sampled.call_count, 2)
        self.assertEqual({vehicle: tuple(route) for vehicle, route in result["routes"].items()},
                         {0: (0,), 1: (1,)})
        self.assertEqual(result["eval"]["recover_time"], {"alpha": 3, "beta": 4})
        self.assertEqual(result["missing"], [])
        self.assertFalse(result["violations"])
        self.assertEqual(result["steps"]["hybrid"]["cuts"], 1)

    def test_unresolved_dp_candidate_does_not_create_a_cut(self):
        candidate = {0: frozenset({"alpha"}), 1: frozenset({"beta"})}

        def sampled_assignments(problem, client):
            selected = {(sensor, vehicle) for vehicle, sensors in candidate.items() for sensor in sensors}
            bits = tuple(int(pair in selected) for pair in problem.pairs)
            self.assertEqual(problem.decode_bits(bits), candidate)
            return [candidate]

        with patch("core.phase2_hybrid.solve_assignment", side_effect=sampled_assignments):
            result = solve_hybrid(
                two_vehicle_instance(), lambda iteration, remaining_seconds: None,
                max_iterations=2, max_states=0,
            )

        self.assertFalse(result["routes"])
        self.assertCountEqual(result["missing"], ["alpha", "beta"])
        self.assertEqual(result["steps"]["hybrid"]["cuts"], 0)
        self.assertGreater(result["steps"]["hybrid"]["unresolved"], 0)


if __name__ == "__main__":
    unittest.main()
