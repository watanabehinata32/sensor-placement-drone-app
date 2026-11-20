"""Phase 2 の担当割当QUBOと、サンプルの検証。"""

from dataclasses import dataclass
from math import fsum, isfinite
from operator import index

import numpy as np
from amplify import Poly, PolyArray, VariableGenerator, solve


@dataclass
class AssignmentModel:
    model: Poly
    variables: PolyArray
    pairs: tuple
    sensors: tuple
    cuts: tuple
    capacities: dict
    statically_infeasible: bool

    def decode(self, values):
        """エネルギー計算に現れない変数は0として、担当だけを読み出す。"""
        return self.decode_bits(self.variables.evaluate(values, 0))

    def decode_bits(self, bits):
        """担当変数を検証し、使う車両の担当集合を返す。"""
        bits = np.asarray(bits, dtype=float)
        rounded = np.rint(bits)
        if (self.statically_infeasible or bits.shape != (len(self.pairs),)
                or not np.allclose(bits, rounded, rtol=0, atol=1e-6)
                or not np.isin(rounded, (0, 1)).all()):
            return None

        seen = set()
        assigned = {}
        for (sensor, vehicle), chosen in zip(self.pairs, rounded):
            if not chosen:
                continue
            if sensor in seen:
                return None
            seen.add(sensor)
            if vehicle is not None:
                assigned.setdefault(vehicle, set()).add(sensor)

        if len(seen) != len(self.sensors):
            return None
        for vehicle, limit in self.capacities.items():
            if len(assigned.get(vehicle, ())) > limit:
                return None
        for vehicle, failed_set in self.cuts:
            if assigned.get(vehicle, frozenset()) == failed_set:
                return None
        return {vehicle: frozenset(sensors) for vehicle, sensors in assigned.items()}


def _bounded_slack(generator, limit):
    """0からlimitまでを表せる二進補助変数の和。"""
    weights = []
    remaining = limit
    while remaining:
        weight = min(1 << len(weights), remaining)
        weights.append(weight)
        remaining -= weight
    slack = Poly()
    for bit, weight in zip(generator.array('Binary', len(weights)), weights):
        slack += weight * bit
    return slack


def build_assignment_model(eligible, costs, *, allow_unserved, cuts=(), capacities=None):
    """担当変数と、任意の未回収変数・担当上限・完全一致カットを組み立てる。"""
    cuts = tuple(dict.fromkeys((vehicle, frozenset(sensors)) for vehicle, sensors in cuts))
    try:
        capacities = {vehicle: index(limit) for vehicle, limit in (capacities or {}).items()}
    except TypeError as exc:
        raise ValueError('担当上限は整数で指定してください') from exc
    if any(limit < 0 for limit in capacities.values()):
        raise ValueError('担当上限は0以上で指定してください')

    choices = {
        sensor: tuple(dict.fromkeys(vehicles)) + ((None,) if allow_unserved else ())
        for sensor, vehicles in eligible.items()
    }
    pairs = tuple((sensor, vehicle) for sensor, vehicles in choices.items() for vehicle in vehicles)
    weights = [float(costs[sensor, vehicle]) for sensor, vehicle in pairs if vehicle is not None]
    if any(not isfinite(weight) for weight in weights):
        raise ValueError('担当の代理費用は有限値で指定してください')

    # 代理費用の総変動幅を1以下にする。未回収1基のペナルティ2がその幅を上回る。
    scale = max((abs(weight) for weight in weights), default=0) or 1
    scaled_weights = [weight / scale for weight in weights]
    span = fsum(abs(weight) for weight in scaled_weights) or 1
    proxy_weights = iter(weight / span for weight in scaled_weights)
    # 不正なビット列を含めた目的値の総変動幅より、ハード罰則1件を重くする。
    hard_weight = 2 + (2 * len(choices) if allow_unserved else 0)

    generator = VariableGenerator()
    variables = generator.array('Binary', len(pairs))
    model = Poly()
    totals = {sensor: Poly() for sensor in choices}
    by_vehicle = {}
    for (sensor, vehicle), bit in zip(pairs, variables):
        totals[sensor] += bit
        if vehicle is None:
            model += 2 * bit
        else:
            model += next(proxy_weights) * bit
            by_vehicle.setdefault(vehicle, {})[sensor] = bit
    for total in totals.values():
        model += hard_weight * (total - 1) ** 2

    for vehicle, limit in capacities.items():
        members = by_vehicle.get(vehicle, {})
        if len(members) <= limit:
            continue
        total = sum(members.values(), Poly())
        model += hard_weight * (total + _bounded_slack(generator, limit) - limit) ** 2

    impossible = any(not options for options in choices.values())
    for vehicle, failed_set in cuts:
        members = by_vehicle.get(vehicle, {})
        if not failed_set.issubset(members):
            continue
        if not members:
            impossible = True
            continue
        distance = Poly(len(failed_set))
        for sensor, bit in members.items():
            distance += -bit if sensor in failed_set else bit
        # 差分数が1以上なら、差分数 - 1を補助変数で表現できる。
        model += hard_weight * (distance - 1 - _bounded_slack(generator, len(members) - 1)) ** 2

    return AssignmentModel(model, variables, pairs, tuple(choices), cuts, capacities, impossible)


def solve_assignment(problem, client):
    """アニーリングの候補から、制約を満たす重複のない担当を返す。"""
    if problem.statically_infeasible:
        return []
    if not problem.model.variables:
        assignment = problem.decode_bits(np.zeros(len(problem.pairs)))
        return [] if assignment is None else [assignment]

    assignments = []
    seen = set()
    for solution in solve(problem.model, client, filter_solution=False):
        assignment = problem.decode(solution.values)
        if assignment is None:
            continue
        key = frozenset(assignment.items())
        if key not in seen:
            assignments.append(assignment)
            seen.add(key)
    return assignments
