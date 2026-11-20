"""二段階目のハイブリッド方式：車への担当割当（QUBO）＋ 1 台ごとの巡回順（動的計画法）。

流れ（目的は「①回復できる数を最大 → ②その数のまま、最後の回復時刻を最小」の順）
  段1 最大フロー（古典）… 回復できる最大の数 N* と担当を厳密に求め、DP で時刻表にして
                          初めの計画（上の端）にする（phase2_maxflow.py）
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
from .phase2_bounds import drive_is_shortest_path_closed
from .phase2_maxflow import max_recovery
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
    capacities = {}          # 予備なしで担当できる数の上限 {車: 数}
    spare_capacity = {}      # 予備を積むと増える担当の数 {車: 数}
    spare_total = max(int(inst.cfg.fleet.spare_total), 0)   # 予備バッテリーの個数（全体）
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
        # 予備を積むと増える担当の数（1 飛行で max_sensors 基まで × 増える回数）
        spare_capacity[vehicle] = (inst.max_flights(has_spare=True) - inst.max_flights()) * max_sensors
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

    # ---- 探索のログ（性能検証と、問題が起きたときの原因探し用。2026-11-20 ひなたさん依頼）----
    # 1 回の「割当 QUBO → 車ごとの DP」ごとに 1 行（dict）を iteration_log に足す。
    # 段1の最大フローも 1 行として残す。結果の steps.hybrid.iteration_log で取り出せる。
    iteration_log = []
    now_state = dict(proven_lower=None, search_lower=None, best_end=None)   # 外側（段1・段2）が書き換える
    totals = dict(anneal_calls=0, anneal_seconds=0.0, dp_calls=0, dp_seconds=0.0, dp_states=0, dp_transitions=0)

    def elapsed():
        """探索を始めてからの秒数."""
        return round(time.monotonic() - started, 4)

    def run_dp(vehicle, sensors, sensor_limits, return_limit, record, has_spare=False, vehicle_log=None):
        """solve_vehicle を呼び、かかった時間・状態数・遷移数を record と totals（と vehicle_log）に足す.

        has_spare: 予備バッテリーを積むなら True（飛べる回数が増える）
        """
        t0 = time.monotonic()
        result = solve_vehicle(
            inst, vehicle, sensors, deadlines=sensor_limits, return_deadline=return_limit,
            forbidden_transitions=forbidden_transitions.get(vehicle, ()), max_states=max_states, stop_at=stop_at,
            max_flights=inst.max_flights(has_spare=has_spare),
        )
        seconds = time.monotonic() - t0
        for key, value in (('dp_calls', 1), ('dp_seconds', seconds), ('dp_states', result.states),
                           ('dp_transitions', result.transitions)):
            record[key] = record.get(key, 0) + value
            totals[key] += value
            if vehicle_log is not None:
                vehicle_log[key] = vehicle_log.get(key, 0) + value
        stats['states'] += result.states
        return result

    def schedule(assignment, sensor_limits, vehicle_limits, record):
        """担当 {車: センサーの集合} を、車ごとの DP で時刻表にする（予備バッテリーの配分も決める）.

        予備：担当が予備なしの上限を超える車に積む（予備は回数だけを増やすので、それ以外の車に積んでも意味がない）。
        DP が「無理」と言ったら、理由を 2 つに分ける：
          ・容量（バッテリー）不足 … 予備を積めば回れる。カットは作らない（容量の制約で扱う）
          ・時間・道路の理由     … 予備を積んでも（飛べる回数を最大にしても）回れない。これだけをカットにする
        戻り値: (routes {車: 飛行の並び} か None, spares 予備を積む車の集合, new_cuts [(車, 組)], vehicle_logs, 失敗の理由)
        """
        spares = set(v for v, sensors in assignment.items() if len(sensors) > capacities.get(v, len(sensors)))
        routes, new_cuts, vehicle_logs, failure = {}, [], [], None
        for vehicle, sensors in sorted(assignment.items()):
            sensors = tuple(sorted(sensors))
            vlog = dict(vehicle=inst.vehicles[vehicle].vehicle_id, assigned=len(sensors), spare=vehicle in spares)
            vehicle_logs.append(vlog)
            result = run_dp(vehicle, sensors, sensor_limits, vehicle_limits.get(vehicle), record,
                            has_spare=vehicle in spares, vehicle_log=vlog)
            if result.status == 'PROVEN_INFEASIBLE' and vehicle not in spares and spare_total > 0:
                # 予備なしでは無理。予備を積めば（飛べる回数を最大にすれば）回れるかを調べて、理由を分ける
                retry = run_dp(vehicle, sensors, sensor_limits, vehicle_limits.get(vehicle), record,
                               has_spare=True, vehicle_log=vlog)
                if retry.status == 'FEASIBLE':
                    if len(spares) < spare_total:     # 予備が余っていれば積む
                        spares.add(vehicle)
                        vlog['spare'] = True
                        result = retry
                    else:
                        vlog['status'] = 'CAPACITY'
                        failure = failure or '予備バッテリーが足りない（容量不足。カットは作らない）'
                        continue
                else:
                    result = retry
            vlog['status'] = result.status
            if result.status == 'PROVEN_INFEASIBLE':
                # 飛べる回数を最大にしても無理 ＝ 時間・道路の理由 → この D ではカットにしてよい
                new_cuts.append((vehicle, frozenset(sensors)))
                failure = failure or 'DP で間に合わない車がある'
            elif result.status == 'UNKNOWN':
                stats['unresolved'] += 1
                failure = failure or 'DP が時間・状態数の上限で止まった'
            else:
                routes[vehicle] = result.route
        for vlog in vehicle_logs:
            for key in ('dp_seconds',):
                if key in vlog:
                    vlog[key] = round(vlog[key], 6)
        return (None if failure else routes), frozenset(spares), new_cuts, vehicle_logs, failure

    def search(deadline, unserved_count, label, stop_when_found, max_unserved=None):
        """締切 deadline（None なら締切なし）で、割当 QUBO → 車ごとの DP をくり返す。

        戻り値: (いちばん良い計画 dict(routes, spares) か None, その評価 (未回収数, 終了時刻) か None, 止まった理由)
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
                capacities=capacities, unserved_count=unserved_count, max_unserved=max_unserved,
                spare_capacity=spare_capacity, spare_total=spare_total,
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
            record = dict(
                iteration=stats['iterations'], stage=label,
                deadline=None if deadline is None else round(deadline, 4),
                lower_bound_proven=now_state['proven_lower'], lower_bound_search=now_state['search_lower'],
                cuts_total=len({(v, c) for v, c, _ in cuts}), cuts_active=len(active_cuts),
                qubo=problem.size(), candidates=[], dp_calls=0, dp_seconds=0.0, dp_states=0, dp_transitions=0,
                cache_hits=None,   # キャッシュはまだ作っていない（v0.3 で入れたら数を入れる）
            )
            iteration_log.append(record)
            t0 = time.monotonic()
            totals['anneal_calls'] += 1
            try:
                assignments = solve_assignment(problem, client_factory(stats['iterations'], remaining))
            except RuntimeError as error:
                record.update(anneal_seconds=round(time.monotonic() - t0, 4), anneal_error=str(error),
                              elapsed_seconds=elapsed())
                totals['anneal_seconds'] += time.monotonic() - t0
                return best_routes, best_score, 'ANNEALER_ERROR'
            anneal_seconds = time.monotonic() - t0
            totals['anneal_seconds'] += anneal_seconds
            report = problem.last_report or {}
            record.update(
                anneal_seconds=round(anneal_seconds, 4), anneal_calls_total=totals['anneal_calls'],
                solutions_returned=report.get('solutions'), solutions_valid=report.get('valid'),
                violations=dict(report.get('violations', {})),   # 決まりを破った解の、破った内容ごとの数
            )

            new_cut = False
            for assignment in assignments:
                if time.monotonic() >= stop_at:
                    break
                key = tuple((v, tuple(sorted(sensors))) for v, sensors in sorted(assignment.items()))
                if key in seen:
                    record['candidates'].append(dict(status='前と同じ担当', end=None))
                    continue
                seen.add(key)

                # 車ごとに、決まった担当を回るいちばん早い順番を DP で求める（予備の配分もここで決まる）
                routes, spares, new_cuts, vehicle_logs, failure = schedule(
                    assignment, sensor_limits, vehicle_limits, record)
                for vehicle, sensors in new_cuts:
                    # この車にこの担当の組は、この D では（予備を積んでも）無理 → 次の QUBO で選べないようにする
                    cuts.append((vehicle, sensors, d_value))
                    new_cut = True
                if routes is None:
                    record['candidates'].append(dict(status=failure, end=None, vehicles=vehicle_logs))
                    continue

                score = _validate_routes(
                    inst, routes, assignment, sensor_limits, vehicle_limits,
                    forbidden_assignments, forbidden_transitions, allow_unserved, spares,
                )
                if score is None:
                    record['candidates'].append(dict(status='独立検証で不合格', end=None, vehicles=vehicle_logs))
                    continue
                record['candidates'].append(dict(status='検証済み', end=round(score[1], 4), unserved=score[0],
                                                 vehicles=vehicle_logs))
                if best_score is None or score < best_score:
                    best_routes, best_score = dict(routes=routes, spares=spares), score

            # この反復が終わった時点の「最良の最後の回復時刻」と経過時間
            if best_score is not None and (now_state['best_end'] is None or best_score[1] < now_state['best_end']) \
                    and deadline is not None:
                now_state['best_end'] = round(best_score[1], 4)
            record.update(best_end=now_state['best_end'], elapsed_seconds=elapsed())
            if best_routes is not None and stop_when_found:
                return best_routes, best_score, 'FOUND'
            if not new_cut:
                # QUBO の中身が前と同じになるので、解き直しても同じ答えになりやすい → ここでやめる
                return best_routes, best_score, 'NO_NEW_CUT'
        return best_routes, best_score, 'ITERATION_LIMIT'

    # ---- 段1：締切なしで、回復できる最大の数を求める --------------------------------
    # 何も回らない計画（全部未回収）は、未回収を許すときだけ「計画あり」とみなす
    empty_ok = (allow_unserved or not inst.targets) and all(d >= 0 for d in return_deadlines.values())
    best = dict(routes={}, spares=frozenset()) if empty_ok else None
    best_score = (n_targets, 0.0)
    stop_reason = 'NO_TARGETS'
    history = []
    max_recovered = 0
    max_recovered_by_flow = None   # 最大フローで出した N*
    max_recovered_proven = False   # N* が「本当に最大」と言えるか
    if inst.targets:
        # 段1-a：最大フローで N* と担当を求める（古典の厳密な計算）
        if progress:
            progress('段1（最大フローで回復できる最大の数を計算）')
        single_sensor_flights = all(len(flight.sensors) == 1 for flight in inst.flights)
        flow_eligible = {
            sensor: [v for v in vehicles
                     if math.isfinite(earliest[sensor, v])
                     and earliest[sensor, v] <= deadlines.get(sensor, math.inf) + TIME_EPSILON]
            for sensor, vehicles in eligible.items()
        }
        # 1 回の飛行で 1 基なら「担当できる数 ＝ 飛べる回数」。そうでなければ capacities（大きめの上限）を使う
        base_capacity = {v: (inst.max_flights() if single_sensor_flights else capacities[v])
                         for v in range(len(inst.vehicles))}
        # 予備バッテリーの配分も含めて最大を求める（予備を積む車の選び方を全部試す。phase2_maxflow.py）。
        # 1 飛行 1 基でないときの「増える数」は車ごとに違いうるので、いちばん大きい値を使う（上限としては正しい）
        flow = max_recovery(flow_eligible, base_capacity, spare_total=spare_total,
                            capacity_per_spare=(inst.max_flights(has_spare=True) - inst.max_flights()
                                                if single_sensor_flights else max(spare_capacity.values(), default=0)),
                            vehicle_group={v: veh.base_id for v, veh in enumerate(inst.vehicles)})
        max_recovered_by_flow = flow['count']
        # 段1-b：最大フローの担当を DP で時刻表にする（初めの実行可能な計画と、上の端）
        # 段1で決めた予備の配分は「初めの計画」にだけ使う。段2では固定せず、アニーリングが担当と一緒に決め直す
        flow_record = dict(iteration=0, stage='段1 最大フロー', deadline=None, max_recovered_by_flow=flow['count'],
                           spare_allocations_tried=flow['allocations_tried'], spares_by_flow=sorted(
                               inst.vehicles[v].vehicle_id for v in flow['spares']), candidates=[],
                           dp_calls=0, dp_seconds=0.0, dp_states=0, dp_transitions=0, cache_hits=None)
        iteration_log.append(flow_record)
        routes, spares, _, vehicle_logs, failure = schedule(flow['assignment'], dict(deadlines),
                                                            dict(return_deadlines), flow_record)
        score = None
        if routes is not None:
            score = _validate_routes(inst, routes, flow['assignment'], deadlines, return_deadlines,
                                     forbidden_assignments, forbidden_transitions, allow_unserved, spares)
        flow_record['candidates'].append(
            dict(status='検証済み', end=round(score[1], 4), unserved=score[0], vehicles=vehicle_logs)
            if score is not None
            else dict(status=f'最大フローの担当を時刻表にできない（{failure or "独立検証で不合格"}）', end=None,
                      vehicles=vehicle_logs))
        if score is not None:
            best, best_score = dict(routes=routes, spares=spares), score
            stop_reason = 'MAX_FLOW'
            now_state['best_end'] = round(score[1], 4)
            flow_record.update(best_end=now_state['best_end'], elapsed_seconds=elapsed())
        else:
            flow_record.update(best_end=None, elapsed_seconds=elapsed())
            # 最大フローの担当が時刻表にできなかった（締切の指定などで前提が崩れた）とき：
            # アニーリングで「N* 基を回復する担当」を探し、それも無理なら回復数を決めずに探す
            unserved = n_targets - max_recovered_by_flow
            routes, score, stop_reason = search(None, unserved if allow_unserved else None,
                                                '段1（N* 基を回復する担当を探す）', stop_when_found=True)
            if routes is None:
                routes, score, stop_reason = search(None, None, '段1（回復数を最大に）', stop_when_found=False)
            if routes is not None and score < best_score:
                best, best_score = routes, score
            if stop_reason == 'ANNEALER_ERROR' and best is None:
                raise RuntimeError('割当 QUBO を解けませんでした')
        max_recovered = n_targets - best_score[0]

    # N* が証明済みか（段2で「回復数をちょうど N*」に固定してよいか）を、ここで判断しておく
    flow_is_upper_bound = max_recovered_by_flow is not None and drive_is_shortest_path_closed(inst.drive)
    n_star_proven = best is not None and (
        max_recovered == n_targets or (flow_is_upper_bound and max_recovered == max_recovered_by_flow))

    # N* が未証明なら、段2に入る前に「もっと多く回復できる担当」を、時間の締切なしで探しておく
    #（見つかれば N* を 1 つずつ上げる。アニーリングで見つからなくても「無理」とは言わない）
    while best is not None and not n_star_proven and allow_unserved and max_recovered < n_targets \
            and stop_reason not in ('TIME_LIMIT', 'ANNEALER_ERROR'):
        plan, score, why = search(None, None, f'段1（{max_recovered + 1} 基以上を回復できるか）',
                                  stop_when_found=True, max_unserved=n_targets - max_recovered - 1)
        if plan is None or score[0] >= best_score[0]:
            if why in ('TIME_LIMIT', 'ANNEALER_ERROR'):
                stop_reason = why
            break
        best, best_score = plan, score
        max_recovered = n_targets - score[0]
        n_star_proven = max_recovered == n_targets or (flow_is_upper_bound and max_recovered == max_recovered_by_flow)

    # ---- 段2：回復数 N* のまま、締切 D を二分探索して最後の時刻を縮める ------------------
    # N* が未証明なら、回復数は「N* 以上」にして、もっと多く回復できる答えも残す（見つかったら N* を上げる）
    upper = None
    proven_lower = None      # 証明済み下界（計算で言える「これより早くは無理」）
    search_lower = None      # 探索上の暫定下限（二分探索の下の端。証明ではない）
    if best is not None and max_recovered > 0 and stop_reason not in ('TIME_LIMIT', 'ANNEALER_ERROR'):
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
        now_state.update(proven_lower=round(proven_lower, 4), search_lower=round(search_lower, 4),
                         best_end=round(upper, 4))
        def fix_or_cap():
            """段2の QUBO に渡す「未回収の数」の決まり：(ちょうどの数, 上限)."""
            if not allow_unserved:
                return None, None
            if n_star_proven:
                return n_targets - max_recovered, None          # 証明済み：ちょうど N* 基
            return None, n_targets - max_recovered              # 未証明：N* 基以上（未回収はこの数以下）

        while upper - search_lower > deadline_step_min:
            deadline = (search_lower + upper) / 2
            exact_unserved, cap_unserved = fix_or_cap()
            routes, score, why = search(
                deadline, exact_unserved, f'段2（締切 D = {deadline:.1f} 分）', stop_when_found=True,
                max_unserved=cap_unserved,
            )
            if routes is not None and score[0] < best_score[0]:
                # N* より多く回復できる計画が見つかった（N* が未証明だったとき）→ N* を上げて探し直す
                best, best_score = routes, score
                max_recovered = n_targets - score[0]
                upper = score[1]
                proven_lower = min(fastest[max_recovered - 1], upper) if math.isfinite(fastest[max_recovered - 1]) else 0.0
                search_lower = proven_lower
                now_state.update(proven_lower=round(proven_lower, 4), search_lower=round(search_lower, 4),
                                 best_end=round(upper, 4))
                history.append(dict(deadline=round(deadline, 2), found=True, end=round(score[1], 2),
                                    more_recovered=max_recovered))
                continue
            if routes is not None and score[0] == best_score[0]:
                best, best_score = routes, score
                upper = score[1]   # 見つかった：実際の終了時刻まで上の端を下げる
                now_state['best_end'] = round(upper, 4)
                history.append(dict(deadline=round(deadline, 2), found=True, end=round(score[1], 2)))
                continue
            history.append(dict(deadline=round(deadline, 2), found=False, reason=why))
            if why in ('TIME_LIMIT', 'ANNEALER_ERROR'):
                stop_reason = why
                break
            # 見つからない：暫定の下限だけ D に上げて、もっと長い D を試す。
            # 「D では無理」と証明できたわけではないので、証明済み下界（proven_lower）は上げない
            search_lower = deadline
            now_state['search_lower'] = round(search_lower, 4)
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
    # 回復数の最大（N*）が証明済みと言える条件
    #   最大フローは「行きに走って行ける組なら担当できる」「担当できる数は飛べる回数 × 1 飛行の最大基数」という
    #   ゆるめた（＝本当より有利な）条件で解いているので、その数は本当の最大の「上限」になる。
    #   ただしこれは、走行時間の表が最短経路の表（三角不等式が成り立つ）のときだけ。そうでないと
    #   「拠点から直接は行けないが、途中を経由すれば行ける」点を見落とし、上限にならないことがある。
    #   実際に検証済みの計画がその上限と同じ数を回復できていれば、それが最大だと言える。
    #   （帰着できない・離陸地点間を走れない・締切に間に合わない、といった理由で上限まで届かなければ未証明）
    max_recovered_proven = best is not None and (
        max_recovered == n_targets or (flow_is_upper_bound and max_recovered == max_recovered_by_flow))

    routes = best['routes'] if best is not None else {}
    spares = best['spares'] if best is not None else frozenset()
    ev = evaluate(inst, routes, spares=spares)
    return dict(
        inst=inst, routes=routes, eval=ev, missing=ev['missing'], violations=[],
        steps=dict(hybrid=dict(
            iterations=stats['iterations'], cuts=len({(v, s) for v, s, _ in cuts}),
            assignment_variables=stats['variables'], flight_candidates=len(inst.flights),
            dp_states=stats['states'], unresolved=stats['unresolved'],
            stop_reason=stop_reason, has_solution=best is not None,
            max_recovered=max_recovered,
            max_recovered_proven=max_recovered_proven,
            max_recovered_by_flow=max_recovered_by_flow,
            spare_total=spare_total,
            cut_list=[dict(vehicle=inst.vehicles[v].vehicle_id, sensors=sorted(c),
                           deadline=None if not math.isfinite(at) else round(at, 4)) for v, c, at in cuts],
            spares=sorted(inst.vehicles[v].vehicle_id for v in spares),
            deadline_upper=None if upper is None else round(upper, 2),
            lower_bound_proven=None if proven_lower is None else round(proven_lower, 2),
            lower_bound_search=None if search_lower is None else round(search_lower, 2),
            optimality=optimality,
            iteration_log=iteration_log,
            totals=dict(
                anneal_calls=totals['anneal_calls'], anneal_seconds=round(totals['anneal_seconds'], 4),
                dp_calls=totals['dp_calls'], dp_seconds=round(totals['dp_seconds'], 4),
                dp_states=totals['dp_states'], dp_transitions=totals['dp_transitions'],
                cuts=len({(v, c) for v, c, _ in cuts}), cache_hits=None, elapsed_seconds=elapsed(),
            ),
            deadline_history=history,
        )),
        seconds=round(time.monotonic() - started, 2),
    )


def _validate_routes(
    inst, routes, assignment, deadlines, return_deadlines,
    forbidden_assignments, forbidden_transitions, allow_unserved, spares=frozenset(),
):
    """Recompute the schedule from restored flight indices and score valid plans."""
    if routes.keys() != assignment.keys():
        return None
    if any(vehicle not in routes and deadline < 0 for vehicle, deadline in return_deadlines.items()):
        return None
    if len(set(spares)) > max(int(inst.cfg.fleet.spare_total), 0):
        return None               # 予備を、ある数より多く積んでいる
    recovered = set()
    end_time = 0.0
    for vehicle, route in routes.items():
        launches, end_recovery, end_return = route_times(inst, vehicle, route)
        end = end_return if inst.end_mode == 'return' else end_recovery
        if not math.isfinite(end) or not math.isfinite(end_return) or len(route) > inst.max_flights(vehicle in spares):
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
