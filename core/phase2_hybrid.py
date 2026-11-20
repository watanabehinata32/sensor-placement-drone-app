"""Search sensor-to-vehicle assignments and schedule each vehicle independently."""

import math
import time

from .phase2_assignment import build_assignment_model, solve_assignment
from .phase2_dp import TIME_EPSILON, solve_vehicle
from .routing import evaluate, route_times


def solve_hybrid(
    inst, client_factory, *, allow_unserved=True, deadlines=None, return_deadlines=None,
    forbidden_assignments=(), forbidden_transitions=None, max_iterations=12,
    max_states=50_000, time_limit_s=15, progress=None,
):
    """Return route data and search diagnostics for the supplied flight candidates.

    Constraints use minutes from departure, sensor IDs, and vehicle/flight indices
    in this Instance. A failed DP can exclude only its exact vehicle assignment.
    """
    started = time.monotonic()
    stop_at = started + time_limit_s
    deadlines = deadlines or {}
    return_deadlines = return_deadlines or {}
    forbidden_assignments = set(forbidden_assignments)
    forbidden_transitions = {
        vehicle: frozenset(transitions)
        for vehicle, transitions in (forbidden_transitions or {}).items()
    }

    # Direct recovery time is a ranking proxy. An infinite direct leg does
    # not by itself exclude the assignment in a supplied travel-time table.
    eligible = {sensor: [] for sensor in inst.targets}
    costs = {}
    capacities = {}
    for vehicle in range(len(inst.vehicles)):
        max_sensors = 0
        for flight_index, flight in enumerate(inst.flights):
            if any((sensor, vehicle) in forbidden_assignments or sensor not in eligible for sensor in flight.sensors):
                continue
            max_sensors = max(max_sensors, len(flight.sensors))
            launch = inst.drive_from_base(vehicle, flight_index)
            for sensor, offset in zip(flight.sensors, flight.recover_off):
                key = (sensor, vehicle)
                costs[key] = min(costs.get(key, math.inf), launch + offset)
        capacities[vehicle] = inst.max_flights() * max_sensors
        for sensor in eligible:
            key = (sensor, vehicle)
            if key in costs and capacities[vehicle]:
                eligible[sensor].append(vehicle)
    finite_costs = [cost for cost in costs.values() if math.isfinite(cost)]
    fallback_cost = max(finite_costs, default=0.0) + 1.0
    costs = {key: cost if math.isfinite(cost) else fallback_cost for key, cost in costs.items()}

    cuts = set()
    seen = set()
    best = {} if (allow_unserved or not inst.targets) and all(
        deadline >= 0 for deadline in return_deadlines.values()
    ) else None
    best_score = (len(inst.targets), 0.0)
    iterations = states = variables = unresolved = 0
    stop_reason = 'NO_TARGETS' if not inst.targets else 'ITERATION_LIMIT'
    for iteration in range(1, max_iterations + 1) if inst.targets else ():
        if time.monotonic() >= stop_at:
            stop_reason = 'TIME_LIMIT'
            break
        iterations = iteration
        if progress:
            progress(f'割当と車両別DPを計算しています（{iteration} / {max_iterations}）')
        problem = build_assignment_model(
            eligible, costs, allow_unserved=allow_unserved,
            cuts=sorted(cuts, key=lambda cut: (cut[0], tuple(sorted(cut[1])))), capacities=capacities,
        )
        variables = len(problem.pairs)
        if problem.statically_infeasible:
            stop_reason = 'NO_ASSIGNMENT_IN_MODEL'
            break
        remaining = stop_at - time.monotonic()
        if remaining <= 0:
            stop_reason = 'TIME_LIMIT'
            break
        try:
            assignments = solve_assignment(problem, client_factory(iteration, remaining))
        except RuntimeError:
            if not best:
                raise
            stop_reason = 'ANNEALER_ERROR'
            break
        for assignment in assignments:
            if time.monotonic() >= stop_at:
                break
            key = tuple((v, tuple(sorted(sensors))) for v, sensors in sorted(assignment.items()))
            if key in seen:
                continue
            seen.add(key)

            routes = {}
            all_scheduled = True
            for vehicle, sensors in key:
                result = solve_vehicle(
                    inst, vehicle, sensors, deadlines=deadlines,
                    return_deadline=return_deadlines.get(vehicle),
                    forbidden_transitions=forbidden_transitions.get(vehicle, ()),
                    max_states=max_states, stop_at=stop_at,
                )
                states += result.states
                if result.status == 'PROVEN_INFEASIBLE':
                    cuts.add((vehicle, frozenset(sensors)))
                elif result.status == 'UNKNOWN':
                    unresolved += 1
                else:
                    routes[vehicle] = result.route
                all_scheduled &= result.status == 'FEASIBLE'
            if not all_scheduled:
                continue

            score = _validate_routes(
                inst, routes, assignment, deadlines, return_deadlines,
                forbidden_assignments, forbidden_transitions, allow_unserved,
            )
            if score is not None and (best is None or score < best_score):
                best, best_score = routes, score
        if time.monotonic() >= stop_at:
            stop_reason = 'TIME_LIMIT'
            break

    routes = best if best is not None else {}
    ev = evaluate(inst, routes)
    return dict(
        inst=inst, routes=routes, eval=ev, missing=ev['missing'], violations=[],
        steps=dict(hybrid=dict(
            iterations=iterations, cuts=len(cuts), assignment_variables=variables,
            flight_candidates=len(inst.flights), dp_states=states, unresolved=unresolved,
            stop_reason=stop_reason, has_solution=best is not None,
        )),
        seconds=round(time.monotonic() - started, 2),
    )


def _validate_routes(
    inst, routes, assignment, deadlines, return_deadlines,
    forbidden_assignments, forbidden_transitions, allow_unserved,
):
    """Recompute the schedule from restored flight indices and score valid plans."""
    if routes.keys() != assignment.keys():
        return None
    if any(vehicle not in routes and deadline < 0 for vehicle, deadline in return_deadlines.items()):
        return None
    recovered = set()
    end_time = 0.0
    for vehicle, route in routes.items():
        launches, end_recovery, end_return = route_times(inst, vehicle, route)
        end = end_return if inst.end_mode == 'return' else end_recovery
        if not math.isfinite(end) or not math.isfinite(end_return) or len(route) > inst.max_flights():
            return None
        if end_return > return_deadlines.get(vehicle, math.inf) + TIME_EPSILON:
            return None
        if forbidden_transitions.get(vehicle, frozenset()).intersection(zip(route, route[1:])):
            return None
        served = set()
        for flight_index, launch in zip(route, launches):
            flight = inst.flights[flight_index]
            for sensor, offset in zip(flight.sensors, flight.recover_off):
                if (
                    sensor in served or sensor in recovered or (sensor, vehicle) in forbidden_assignments
                    or launch + offset > deadlines.get(sensor, math.inf) + TIME_EPSILON
                ):
                    return None
                served.add(sensor)
        if served != assignment[vehicle]:
            return None
        recovered.update(served)
        end_time = max(end_time, end)
    if not recovered <= set(inst.targets) or (not allow_unserved and len(recovered) != len(inst.targets)):
        return None
    return len(inst.targets) - len(recovered), end_time
