"""Find a route for one vehicle using a fixed assignment and flight pool."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import inf, isfinite, isnan
from operator import index
from time import monotonic

from .routing import Instance


TIME_EPSILON = 1e-9  # Floating-point tolerance in minutes.


@dataclass(frozen=True)
class VehicleResult:
    status: str
    route: tuple[int, ...] = ()
    states: int = 0


def solve_vehicle(
    inst: Instance,
    vehicle: int,
    assigned: Iterable[str],
    *,
    deadlines: Mapping[str, float] | None = None,
    return_deadline: float | None = None,
    forbidden_transitions: Iterable[tuple[int, int]] = (),
    max_states: int | None = None,
    stop_at: float | None = None,
) -> VehicleResult:
    """Minimize the selected end time over routes that recover the assignment exactly once.

    Deadlines are inclusive and use minutes from departure. Missing sensor deadlines
    are unlimited. A state is a distinct combination of flight count, recovered
    sensors, and last flight. If a search bound interrupts the DP, the result is
    UNKNOWN. An infeasibility result applies to the supplied flight pool.
    """
    if max_states is not None:
        max_states = index(max_states)
        if max_states < 0:
            raise ValueError('max_states must be nonnegative')
    if stop_at is not None and isnan(stop_at):
        raise ValueError('stop_at must be a monotonic deadline')

    def time_expired():
        return stop_at is not None and monotonic() >= stop_at

    if time_expired():
        return VehicleResult('UNKNOWN')

    return_limit = inf if return_deadline is None else return_deadline
    sensor_bits = {sensor: 1 << i for i, sensor in enumerate(dict.fromkeys(assigned))}
    if not sensor_bits:
        status = 'FEASIBLE' if 0 <= return_limit else 'PROVEN_INFEASIBLE'
        return VehicleResult(status)

    max_flights = min(inst.max_flights(), len(sensor_bits))
    if max_flights <= 0:
        return VehicleResult('PROVEN_INFEASIBLE')

    deadlines = {} if deadlines is None else deadlines
    forbidden = set(forbidden_transitions)
    full_mask = (1 << len(sensor_bits)) - 1
    pool = []
    available_mask = 0
    for flight_index, flight in enumerate(inst.flights):
        if time_expired():
            return VehicleResult('UNKNOWN')
        sensors = set(flight.sensors)
        if not sensors or len(sensors) != len(flight.sensors) or not sensors <= sensor_bits.keys():
            continue
        mask = sum(sensor_bits[sensor] for sensor in sensors)
        recovery_limits = tuple(
            (offset, deadlines[sensor])
            for sensor, offset in zip(flight.sensors, flight.recover_off, strict=True)
            if sensor in deadlines
        )
        pool.append((flight_index, mask, recovery_limits))
        available_mask |= mask
    if time_expired():
        return VehicleResult('UNKNOWN')
    if available_mask != full_mask:
        return VehicleResult('PROVEN_INFEASIBLE')

    # At a fixed flight count, arriving earlier at the same flight can only
    # improve the remaining options. Separate layers preserve the battery budget.
    layer = {(0, None): (0.0, ())}
    states = 0
    best_route = ()
    best_end = inf
    for _ in range(max_flights):
        next_layer = {}
        for (covered, last), (launch, route) in layer.items():
            if covered == full_mask:
                continue
            for flight_index, mask, recovery_limits in pool:
                if time_expired():
                    return VehicleResult('UNKNOWN', states=states)
                if covered & mask or (last, flight_index) in forbidden:
                    continue
                next_launch = (
                    inst.drive_from_base(vehicle, flight_index)
                    if last is None else launch + inst.gap(last, flight_index)
                )
                if not isfinite(next_launch):
                    continue
                if not all(next_launch + offset <= limit + TIME_EPSILON for offset, limit in recovery_limits):
                    continue

                next_covered = covered | mask
                key = (next_covered, flight_index)
                previous = next_layer.get(key)
                if previous is not None and previous[0] <= next_launch:
                    continue
                if previous is None:
                    if max_states is not None and states == max_states:
                        return VehicleResult('UNKNOWN', states=states)
                    states += 1
                next_route = route + (flight_index,)
                next_layer[key] = (next_launch, next_route)

                if next_covered == full_mask:
                    flight = inst.flights[flight_index]
                    returned_at = next_launch + flight.flight_min + inst.drive_to_base(vehicle, flight_index)
                    if isfinite(returned_at) and returned_at <= return_limit + TIME_EPSILON:
                        end = returned_at if inst.end_mode == 'return' else next_launch + flight.last_recover_off
                        if end < best_end:
                            best_end = end
                            best_route = next_route

        if not next_layer:
            break
        layer = next_layer

    if time_expired():
        return VehicleResult('UNKNOWN', states=states)
    if best_route:
        return VehicleResult('FEASIBLE', best_route, states)
    return VehicleResult('PROVEN_INFEASIBLE', states=states)
