"""二段階目のハイブリッド方式：車への担当割当（QUBO）＋ 1 台ごとの巡回順（動的計画法）。

流れ（目的は「①回復できる数を最大 → ②その数のまま、最後の回復時刻を最小」の順）
  段1 締切なしで解く …… 回復できる最大の数 N* と、そのときの終了時刻（上の端）を求める
  段2 締切 D の二分探索 … 「N* 基を回復し、全部 D 分以内に終わる計画はあるか」を、
                          D を上の端と下の端の真ん中に置いて調べ、範囲を半分ずつ絞る
                          ・見つかった → 上の端 = その計画の終了時刻（もっと短い D を試す）
                          ・見つからない → 下の端 = D（もっと長い D を試す）
各 D の中では「割当 QUBO → 車ごとの DP」をくり返し、DP が「この担当は D に間に合わない」と
示した組（車, 担当の組）は次の QUBO で選べないようにする（カット）。

注意：アニーリングで見つからなかっただけでは「D では無理」と証明できない。
だから二分探索は「ヒューリスティック（経験的なやり方）」として使う（2026-11-20 ひなたさん決定・設計書第10節）。
下の端は 2 つに分けて持つ。
  ・証明済み下界 …… 計算で「これより早くは絶対に終わらない」と言える時刻。探索の失敗では上げない
  ・探索上の暫定下限 … 二分探索で使う下の端。D で見つからなければ D まで上げる（証明ではない）
答えは「制限時間内に見つけた、検証済みの最良の計画」。最短だと言えるのは
「上の端 − 証明済み下界」が刻み幅以下になったときだけで、そうでなければ「最適性は未証明」と返す。
"""

import math
import time

from .phase2_assignment import build_assignment_model, solve_assignment
from .phase2_dp import TIME_EPSILON, solve_vehicle
from .routing import evaluate, route_times


def solve_hybrid(
    inst, client_factory, *, allow_unserved=True, deadlines=None, return_deadlines=None,
    forbidden_assignments=(), forbidden_transitions=None, max_iterations=12,
    max_states=50_000, time_limit_s=60, deadline_step_min=1.0, progress=None,
):
    """段1（最大回復数）と段2（締切 D の二分探索）で計画を探し、ルートと探索の記録を返す。

    max_iterations     … 1 つの D（段1は締切なし）で QUBO を解き直す最大の回数
    time_limit_s       … 全体の計算時間の上限（秒）
    deadline_step_min  … 上の端と下の端の差がこれ以下（分）になったら二分探索をやめる
    deadlines / return_deadlines … 画面や API で指定された締切（出発からの分）
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

    # 各（センサー, 車）について「その車が最短で回復できる時刻」を調べる。
    # ・QUBO の目的（代理の費用）に使う
    # ・締切 D のときに「どうやっても D に間に合わない担当」を外すのに使う
    # ・二分探索の下の端に使う
    eligible = {sensor: [] for sensor in inst.targets}
    earliest = {}
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
                earliest[key] = min(earliest.get(key, math.inf), launch + offset)
        capacities[vehicle] = inst.max_flights() * max_sensors
        for sensor in eligible:
            key = (sensor, vehicle)
            if key in earliest and capacities[vehicle]:
                eligible[sensor].append(vehicle)
    # 走って行けない（inf）組は、費用だけ「いちばん遅い値 + 1」にしておく（QUBO の係数を有限にするため）
    finite_costs = [cost for cost in earliest.values() if math.isfinite(cost)]
    fallback_cost = max(finite_costs, default=0.0) + 1.0
    costs = {key: cost if math.isfinite(cost) else fallback_cost for key, cost in earliest.items()}

    n_targets = len(inst.targets)
    cuts = []            # (車, 担当の組, そのときの D)。D 以下の締切ならいつでも「無理」のまま使える
    stats = dict(iterations=0, states=0, variables=0, unresolved=0)

    def search(deadline, unserved_count, label, stop_when_found):
        """締切 deadline（None なら締切なし）で、割当 QUBO → 車ごとの DP をくり返す。

        戻り値: (いちばん良い計画 {車: ルート} か None, その評価 (未回収数, 終了時刻) か None, 止まった理由)
        """
        d_value = math.inf if deadline is None else deadline
        # センサーごとの回復の締切 = 指定された締切 と D の早いほう（「最後の回復」で測るとき）
        sensor_limits = dict(deadlines)
        vehicle_limits = dict(return_deadlines)
        if deadline is not None:
            if inst.end_mode == 'return':
                for vehicle in range(len(inst.vehicles)):
                    vehicle_limits[vehicle] = min(vehicle_limits.get(vehicle, math.inf), deadline)
            else:
                for sensor in inst.targets:
                    sensor_limits[sensor] = min(sensor_limits.get(sensor, math.inf), deadline)
        # D に絶対間に合わない担当（最短でも D を過ぎる組）は、はじめから候補に入れない。
        # 走行時間が inf の組は判断できないので外さない（DP に任せる）
        usable = {
            sensor: [v for v in vehicles
                     if deadline is None or not math.isfinite(earliest[sensor, v])
                     or earliest[sensor, v] <= deadline + TIME_EPSILON]
            for sensor, vehicles in eligible.items()
        }

        best_routes, best_score = None, None
        seen = set()
        for _ in range(max_iterations):
            if time.monotonic() >= stop_at:
                return best_routes, best_score, 'TIME_LIMIT'
            # この D で使ってよいカット = 同じか、より長い D で「無理」と分かった組
            active_cuts = sorted(
                {(v, sensors) for v, sensors, at in cuts if at >= d_value},
                key=lambda cut: (cut[0], tuple(sorted(cut[1]))),
            )
            problem = build_assignment_model(
                usable, costs, allow_unserved=allow_unserved, cuts=active_cuts,
                capacities=capacities, unserved_count=unserved_count,
            )
            stats['variables'] = len(problem.pairs)
            if problem.statically_infeasible:
                return best_routes, best_score, 'NO_ASSIGNMENT_IN_MODEL'
            remaining = stop_at - time.monotonic()
            if remaining <= 0:
                return best_routes, best_score, 'TIME_LIMIT'
            stats['iterations'] += 1
            if progress:
                progress(f'{label}：割当 QUBO と車両別 DP（通算 {stats["iterations"]} 回目）')
            try:
                assignments = solve_assignment(problem, client_factory(stats['iterations'], remaining))
            except RuntimeError:
                return best_routes, best_score, 'ANNEALER_ERROR'

            new_cut = False
            for assignment in assignments:
                if time.monotonic() >= stop_at:
                    break
                key = tuple((v, tuple(sorted(sensors))) for v, sensors in sorted(assignment.items()))
                if key in seen:
                    continue
                seen.add(key)

                # 車ごとに、決まった担当を回るいちばん早い順番を DP で求める
                routes = {}
                all_scheduled = True
                for vehicle, sensors in key:
                    result = solve_vehicle(
                        inst, vehicle, sensors, deadlines=sensor_limits,
                        return_deadline=vehicle_limits.get(vehicle),
                        forbidden_transitions=forbidden_transitions.get(vehicle, ()),
                        max_states=max_states, stop_at=stop_at,
                    )
                    stats['states'] += result.states
                    if result.status == 'PROVEN_INFEASIBLE':
                        # この車にこの担当の組は、この D では無理 → 次の QUBO で選べないようにする
                        cuts.append((vehicle, frozenset(sensors), d_value))
                        new_cut = True
                    elif result.status == 'UNKNOWN':
                        stats['unresolved'] += 1
                    else:
                        routes[vehicle] = result.route
                    all_scheduled &= result.status == 'FEASIBLE'
                if not all_scheduled:
                    continue

                score = _validate_routes(
                    inst, routes, assignment, sensor_limits, vehicle_limits,
                    forbidden_assignments, forbidden_transitions, allow_unserved,
                )
                if score is not None and (best_score is None or score < best_score):
                    best_routes, best_score = routes, score

            if best_routes is not None and stop_when_found:
                return best_routes, best_score, 'FOUND'
            if not new_cut:
                # QUBO の中身が前と同じになるので、解き直しても同じ答えになりやすい → ここでやめる
                return best_routes, best_score, 'NO_NEW_CUT'
        return best_routes, best_score, 'ITERATION_LIMIT'

    # ---- 段1：締切なしで、回復できる最大の数を求める --------------------------------
    # 何も回らない計画（全部未回収）は、未回収を許すときだけ「計画あり」とみなす
    empty_ok = (allow_unserved or not inst.targets) and all(d >= 0 for d in return_deadlines.values())
    best = {} if empty_ok else None
    best_score = (n_targets, 0.0)
    stop_reason = 'NO_TARGETS'
    history = []
    max_recovered = 0
    if inst.targets:
        routes, score, stop_reason = search(None, None, '段1（回復数を最大に）', stop_when_found=False)
        if routes is not None and score < best_score:
            best, best_score = routes, score
        if stop_reason == 'ANNEALER_ERROR' and best is None:
            raise RuntimeError('割当 QUBO を解けませんでした')
        max_recovered = n_targets - best_score[0]

    # ---- 段2：回復数 N* のまま、締切 D を二分探索して最後の時刻を縮める ------------------
    upper = None
    proven_lower = None      # 証明済み下界（計算で言える「これより早くは無理」）
    search_lower = None      # 探索上の暫定下限（二分探索の下の端。証明ではない）
    if best and max_recovered > 0 and stop_reason not in ('TIME_LIMIT', 'ANNEALER_ERROR'):
        upper = best_score[1]
        # 下の端：N* 基を回復するなら、どれか 1 基は「各センサーの最短回復時刻」の N* 番目以降になる
        fastest = sorted(
            min((earliest[sensor, v] for v in vehicles), default=0.0) if vehicles else math.inf
            for sensor, vehicles in eligible.items()
        )
        # これは「証明済み下界」：どの計画でもこの時刻より前には N* 基目が回復できない
        proven_lower = fastest[max_recovered - 1]
        if not math.isfinite(proven_lower):
            proven_lower = 0.0
        proven_lower = min(proven_lower, upper)
        search_lower = proven_lower   # 二分探索はここから始める
        unserved_count = n_targets - max_recovered if allow_unserved else None
        while upper - search_lower > deadline_step_min:
            deadline = (search_lower + upper) / 2
            routes, score, why = search(
                deadline, unserved_count, f'段2（締切 D = {deadline:.1f} 分）', stop_when_found=True,
            )
            if routes is not None and score[0] == best_score[0]:
                best, best_score = routes, score
                upper = score[1]   # 見つかった：実際の終了時刻まで上の端を下げる
                history.append(dict(deadline=round(deadline, 2), found=True, end=round(score[1], 2)))
                continue
            history.append(dict(deadline=round(deadline, 2), found=False, reason=why))
            if why in ('TIME_LIMIT', 'ANNEALER_ERROR'):
                stop_reason = why
                break
            # 見つからない：暫定の下限だけ D に上げて、もっと長い D を試す。
            # 「D では無理」と証明できたわけではないので、証明済み下界（proven_lower）は上げない
            search_lower = deadline
        else:
            stop_reason = 'DEADLINE_CONVERGED'

    # 最適性の証明状態（最短と言えるのは、上の端と「証明済み」下界の差が刻み幅以下のときだけ）
    if best is None:
        optimality = 'NO_SOLUTION'
    elif max_recovered == 0:
        optimality = 'PROVEN' if not inst.targets else 'NOT_PROVEN'
    elif upper is not None and proven_lower is not None and upper - proven_lower <= deadline_step_min:
        optimality = 'PROVEN'
    else:
        optimality = 'NOT_PROVEN'
    # 回復数の最大（N*）も、全部回復できたとき以外はアニーリングで見つけた数なので証明ではない
    max_recovered_proven = best is not None and max_recovered == n_targets

    routes = best if best is not None else {}
    ev = evaluate(inst, routes)
    return dict(
        inst=inst, routes=routes, eval=ev, missing=ev['missing'], violations=[],
        steps=dict(hybrid=dict(
            iterations=stats['iterations'], cuts=len({(v, s) for v, s, _ in cuts}),
            assignment_variables=stats['variables'], flight_candidates=len(inst.flights),
            dp_states=stats['states'], unresolved=stats['unresolved'],
            stop_reason=stop_reason, has_solution=best is not None,
            max_recovered=max_recovered,
            max_recovered_proven=max_recovered_proven,
            deadline_upper=None if upper is None else round(upper, 2),
            lower_bound_proven=None if proven_lower is None else round(proven_lower, 2),
            lower_bound_search=None if search_lower is None else round(search_lower, 2),
            optimality=optimality,
            deadline_history=history,
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
