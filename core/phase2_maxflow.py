"""二段階目：回復できる最大の数 N* を、最大フロー（古典の厳密な計算）で求める.

なぜ最大フローか（2026-11-20 ひなたさん決定）
  「どのセンサーを、どの車が担当するか」だけを考えると、
    ・センサーは 1 台の車にだけ担当させる（各センサー 1 本）
    ・車は「飛べる回数」までしか担当できない（1 回の飛行で 1 基なので、回数 ＝ 担当できる数）
  という形の「割り振り」の問題になる。これは最大フロー（水道管の問題）で、厳密に・一瞬で解ける。
  アニーリングは、この N* を保ったまま担当を入れ替えて、最後の回復時刻を縮めるのに使う。

予備バッテリー（v0.2 で入れる）への備え
  予備を積んだ車は担当できる数が増える（例：5 回 → 5 ＋ 4 回）。どの車に予備を積むかで N* が変わるので、
  「予備を積む車の選び方」を全部試し、選び方ごとに最大フローを解いて、いちばん多いものを取る。
  （予備の個数を B とすると、試す数は「予備が役に立つ車」から B 台を選ぶ組み合わせの数。
    同じ拠点で担当できるセンサーも同じ車は、どれに積んでも同じなので 1 通りにまとめる）
  予備が 0 個のときは、最大フローを 1 回解くだけ。

前提（これが崩れたら N* は「上限」でしかなくなる。その場合は proven=False を返す）
  ・1 回の飛行で 1 基だけ回る（複数基を回る飛行があると、回数 ＝ 担当数 が成り立たない）
  ・車が担当できるセンサー（候補 eligible）は、実際に行って帰ってこられるものだけ
  ・活動時間の上限（締切）や、通れない順番の指定がない
"""
import itertools

import networkx as nx


def max_flow_assignment(eligible, capacity):
    """1 回の最大フローで、担当の割り振りを求める.

    eligible: {センサー: [担当できる車の番号, ...]}
    capacity: {車の番号: 担当できる最大の数}
    戻り値: {車の番号: センサーの集合}（担当なしの車は入らない）
    """
    graph = nx.DiGraph()
    # 出発点 → センサー（1 基は 1 回だけ）→ 車（担当できるなら 1）→ 終点（車ごとの担当できる数まで）
    for sensor, vehicles in eligible.items():
        graph.add_edge('出発点', ('センサー', sensor), capacity=1)
        for vehicle in vehicles:
            graph.add_edge(('センサー', sensor), ('車', vehicle), capacity=1)
    for vehicle, limit in capacity.items():
        graph.add_edge(('車', vehicle), '終点', capacity=limit)
    if '出発点' not in graph or '終点' not in graph:
        return {}
    _, flow = nx.maximum_flow(graph, '出発点', '終点')
    assignment = {}
    for sensor in eligible:
        for (_, vehicle), amount in flow.get(('センサー', sensor), {}).items():
            if amount > 0.5:                 # 流れた（＝その車が担当する）
                assignment.setdefault(vehicle, set()).add(sensor)
    return {vehicle: frozenset(sensors) for vehicle, sensors in assignment.items()}


def max_recovery(eligible, base_capacity, *, spare_total=0, capacity_per_spare=0, vehicle_group=None):
    """予備バッテリーの配分も含めて、回復できる最大の数 N* と、そのときの担当・予備の配分を求める.

    eligible:           {センサー: [担当できる車の番号, ...]}
    base_capacity:      {車の番号: 予備なしで担当できる数}
    spare_total:        予備バッテリーの個数（全体）。1 台に積めるのは 0〜1 個
    capacity_per_spare: 予備を 1 個積むと増える、担当できる数
    vehicle_group:      {車の番号: 拠点の ID など}。同じグループで担当できるセンサーも同じ車は同じ扱いにする
    戻り値: dict(count=N*, assignment={車: センサーの集合}, spares=[予備を積む車], allocations_tried=試した数)
    """
    vehicle_group = vehicle_group or {}
    degree = {vehicle: 0 for vehicle in base_capacity}   # 車ごとの「担当できるセンサーの数」
    for vehicles in eligible.values():
        for vehicle in vehicles:
            degree[vehicle] = degree.get(vehicle, 0) + 1

    # 予備が役に立つのは「担当できるセンサーが、予備なしの上限より多い」車だけ
    helpful = []
    seen_kinds = {}
    for vehicle in sorted(base_capacity):
        if spare_total <= 0 or capacity_per_spare <= 0 or degree.get(vehicle, 0) <= base_capacity[vehicle]:
            continue
        kind = (vehicle_group.get(vehicle, vehicle),
                frozenset(s for s, vs in eligible.items() if vehicle in vs))
        seen_kinds.setdefault(kind, []).append(vehicle)
        helpful.append(vehicle)

    # 予備は多く積んで損はないので、積める数（予備の個数と、役に立つ車の数の小さいほう）だけ積む
    n_spares = min(spare_total, len(helpful))
    best = None
    tried = 0
    tried_patterns = set()
    for chosen in itertools.combinations(helpful, n_spares):
        # 同じ種類の車の入れ替えだけ違う配分は 1 回だけ試す
        pattern = tuple(sorted(
            (kind, sum(1 for v in chosen if v in members)) for kind, members in seen_kinds.items()
        ))
        if pattern in tried_patterns:
            continue
        tried_patterns.add(pattern)
        tried += 1
        capacity = {vehicle: base_capacity[vehicle] + (capacity_per_spare if vehicle in chosen else 0)
                    for vehicle in base_capacity}
        assignment = max_flow_assignment(eligible, capacity)
        count = sum(len(sensors) for sensors in assignment.values())
        if best is None or count > best['count']:
            best = dict(count=count, assignment=assignment, spares=list(chosen))
    best['allocations_tried'] = tried
    return best
