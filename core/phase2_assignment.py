"""Phase 2 の担当割当QUBOと、サンプルの検証。"""

from collections import Counter
from dataclasses import dataclass, field
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
    unserved_count: int | None = None   # 未回収にする基数を決めたとき、その数（決めないなら None）
    last_report: dict = field(default_factory=dict)   # 直前の solve_assignment の記録（ログ用）
    spare_capacity: dict = field(default_factory=dict)   # 予備を積むと増える担当の数 {車: 数}
    spare_total: int = 0                                  # 予備バッテリーの個数（全体）
    spare_variables: dict = field(default_factory=dict)  # 予備を積むかの変数 b[車]
    max_unserved: int | None = None                       # 未回収を「この数以下」にするとき、その数

    def spares_needed(self, assignment):
        """担当から、予備バッテリーが要る車を求める（担当が予備なしの上限を超える車だけが予備を使う）.

        予備は「飛べる回数」だけを増やすモデルなので、上限を超えない車に積んでも計画は変わらない。
        だから担当が決まれば、意味のある予備の配分も決まる。
        """
        return frozenset(v for v, sensors in assignment.items() if len(sensors) > self.capacities.get(v, len(sensors)))

    def size(self):
        """QUBO の大きさ（ログ用）：担当の変数、補助変数（不等式のための変数）、二次の結合の数."""
        n_total = len(self.model.variables)
        n_quadratic = sum(1 for key in self.model.as_dict() if len(set(key)) == 2)
        n_spare = len(self.spare_variables)
        return dict(variables=len(self.pairs), spare_variables=n_spare,
                    auxiliary=n_total - len(self.pairs) - n_spare, quadratic=n_quadratic)

    def violations(self, bits):
        """担当のビット列が、どの決まりを破っているかを日本語で返す（破っていなければ空）."""
        bits = np.asarray(bits, dtype=float)
        rounded = np.rint(bits)
        if self.statically_infeasible:
            return ['どの担当も選べない問題になっている']
        if (bits.shape != (len(self.pairs),) or not np.allclose(bits, rounded, rtol=0, atol=1e-6)
                or not np.isin(rounded, (0, 1)).all()):
            return ['変数が 0 か 1 になっていない']
        reasons = []
        chosen_count = Counter()
        assigned = {}
        for (sensor, vehicle), chosen in zip(self.pairs, rounded):
            if chosen:
                chosen_count[sensor] += 1
                if vehicle is not None:
                    assigned.setdefault(vehicle, set()).add(sensor)
        if any(chosen_count[sensor] != 1 for sensor in self.sensors):
            reasons.append('担当（か未回収の印）がちょうど 1 つでないセンサーがある')
        n_assigned = sum(len(sensors) for sensors in assigned.values())
        if self.unserved_count is not None and len(self.sensors) - n_assigned != self.unserved_count:
            reasons.append('未回収の数が、決めた数と違う')
        if self.max_unserved is not None and len(self.sensors) - n_assigned > self.max_unserved:
            reasons.append('未回収の数が、決めた上限を超える')
        if any(len(assigned.get(vehicle, ())) > limit + self.spare_capacity.get(vehicle, 0)
               for vehicle, limit in self.capacities.items()):
            reasons.append('担当の数が、予備を積んでも上限を超える車がある')
        if len(self.spares_needed(assigned)) > self.spare_total:
            reasons.append('予備バッテリーが要る車が、予備の個数より多い')
        if any(assigned.get(vehicle, frozenset()) == failed_set for vehicle, failed_set in self.cuts):
            reasons.append('禁止制約（カット）に当たる担当がある')
        return reasons

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
        # 「回復数を固定」したときは、未回収の基数がちょうどその数か確かめる
        n_assigned = sum(len(sensors) for sensors in assigned.values())
        if self.unserved_count is not None and len(self.sensors) - n_assigned != self.unserved_count:
            return None
        if self.max_unserved is not None and len(self.sensors) - n_assigned > self.max_unserved:
            return None
        # 担当の数：予備なしの上限を超える車は予備が要る。要る車が予備の個数以下で、予備込みの上限以内ならよい
        for vehicle, limit in self.capacities.items():
            if len(assigned.get(vehicle, ())) > limit + self.spare_capacity.get(vehicle, 0):
                return None
        if len(self.spares_needed(assigned)) > self.spare_total:
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


def build_assignment_model(eligible, costs, *, allow_unserved, cuts=(), capacities=None, unserved_count=None,
                           spare_capacity=None, spare_total=0, max_unserved=None):
    """担当変数と、任意の未回収変数・担当上限・完全一致カットを組み立てる。

    unserved_count … 締切 D の段で使う。「未回収はちょうど この数」という決まりを足す
    （一段目で見つけた最大の回復数を、二段目で減らさないようにするため）。
    max_unserved   … 「未回収は この数以下」という決まり（N* が未証明のとき、もっと多く回復できる答えを残すため）
    spare_capacity … 予備バッテリーを積むと増える担当の数 {車: 数}。ここにある車には変数 b[車]（積むなら 1）を作り、
                     担当の上限を「予備なしの上限 ＋ 増える数 × b[車]」にする
    spare_total    … 予備バッテリーの個数。「b の合計 ≤ この数」という決まりを足す
    担当 x と予備 b を同じ QUBO で同時に決める（2026-11-20 ひなたさん指示）。
    """
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

    # 予備バッテリーの変数 b[車]：予備の個数が 1 以上で、予備なしの上限より担当できるセンサーが多い車にだけ作る
    #（担当できる数が上限以下の車は、予備を積んでも意味がないため）
    spare_capacity = {v: int(extra) for v, extra in (spare_capacity or {}).items() if extra > 0}
    spare_total = int(spare_total or 0)
    spare_variables = {}
    if spare_total > 0:
        for vehicle in sorted(spare_capacity, key=str):
            if len(by_vehicle.get(vehicle, {})) > capacities.get(vehicle, 0):
                spare_variables[vehicle] = generator.array('Binary', 1)[0]

    for vehicle, limit in capacities.items():
        members = by_vehicle.get(vehicle, {})
        if len(members) <= limit:
            continue
        total = sum(members.values(), Poly())
        if vehicle in spare_variables:
            # 担当の数 ＋ 補助 ＝ 予備なしの上限 ＋ 増える数 × b   （補助は 0 〜 上限＋増える数）
            extra = spare_capacity[vehicle]
            model += hard_weight * (total + _bounded_slack(generator, limit + extra) - limit
                                    - extra * spare_variables[vehicle]) ** 2
        else:
            model += hard_weight * (total + _bounded_slack(generator, limit) - limit) ** 2
    # 予備の合計 ≤ 予備の個数
    if len(spare_variables) > spare_total:
        spare_sum = sum(spare_variables.values(), Poly())
        model += hard_weight * (spare_sum + _bounded_slack(generator, spare_total) - spare_total) ** 2

    impossible = any(not options for options in choices.values())

    # 回復数を固定する決まり：（未回収の印の合計 − 決めた数）² が 0 になるように罰則をかける
    if allow_unserved and unserved_count is not None:
        unserved_total = Poly()
        for (sensor, vehicle), bit in zip(pairs, variables):
            if vehicle is None:
                unserved_total += bit
        model += hard_weight * (unserved_total - unserved_count) ** 2
        # 担当できる車が 1 台もいないセンサーが多すぎて、決めた数まで回復できない場合
        n_can_serve = sum(1 for options in choices.values() if any(v is not None for v in options))
        if n_can_serve < len(choices) - unserved_count:
            impossible = True
    # 未回収の数を「max_unserved 以下」にする決まり（不等式なので補助変数を使う）
    if allow_unserved and max_unserved is not None:
        unserved_total = Poly()
        n_marks = 0
        for (sensor, vehicle), bit in zip(pairs, variables):
            if vehicle is None:
                unserved_total += bit
                n_marks += 1
        if n_marks > max_unserved:
            model += hard_weight * (unserved_total + _bounded_slack(generator, max_unserved) - max_unserved) ** 2
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

    if not allow_unserved:
        unserved_count = None   # 未回収を許さないときは、もともと全部回復が決まりになっている
    return AssignmentModel(model, variables, pairs, tuple(choices), cuts, capacities, impossible, unserved_count,
                           spare_capacity=spare_capacity, spare_total=spare_total, spare_variables=spare_variables,
                           max_unserved=max_unserved if allow_unserved else None)


def solve_assignment(problem, client):
    """アニーリングの候補から、制約を満たす重複のない担当を返す。"""
    if problem.statically_infeasible:
        return []
    if not problem.model.variables:
        assignment = problem.decode_bits(np.zeros(len(problem.pairs)))
        return [] if assignment is None else [assignment]

    assignments = []
    seen = set()
    report = dict(solutions=0, valid=0, violations=Counter())   # ログ用：返ってきた解の数と、破った決まり
    problem.last_report = report
    for solution in solve(problem.model, client, filter_solution=False):
        report['solutions'] += 1
        assignment = problem.decode(solution.values)
        if assignment is None:
            report['violations'].update(problem.violations(problem.variables.evaluate(solution.values, 0)))
            continue
        report['valid'] += 1
        key = frozenset(assignment.items())
        if key not in seen:
            assignments.append(assignment)
            seen.add(key)
    return assignments
