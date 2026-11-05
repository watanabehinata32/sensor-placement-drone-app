"""二段階目（災害時のドローン巡回）を、QUBO 2 つで解く.

お題 PDF スライド 7 の「解き方の見通し」どおり、次の順に進める。

  ① 前処理（普通のコード）     … 通行止めを除いた道路網で走行時間を求め、各センサーに届く飛行を数え上げる。
                                   どの離陸地点からも届かないセンサーは「対応不能」。（preprocess.py）
  ② 飛行の選択（QUBO その 1）  … 通信断センサーを全部覆う飛行の組を選ぶ（集合被覆）。
                                   選ばれた飛行の離陸地点が、③で使ってよい離陸地点になる。
  ③ ルートの数え上げ（普通のコード）
                               … 車ごとに「拠点 → 離陸地点で飛行 → … → 拠点」のルートを数え上げる。
                                   ②で選んだ離陸地点だけ使う／予備バッテリーの有無で飛べる回数が変わる／
                                   回るセンサーの組が同じルートは、いちばん早く終わる 1 本だけ残す。
  ④ ルートの選択（QUBO その 2）… 各車に 1 本ずつルートを選ぶ（何もしない「空ルート」も選べる）。
  ⑤ 後処理（普通のコード）     … 選んだルートを時刻つきの予定表にし、決まりを守れているか確かめる。

ソルバーは Fixstars Amplify AE だけを使う（トークンは .env から読み、ここには書かない）。
QUBO の答えで回れなかったセンサーを、あとから別の方法で付け足すことはしない（画面で「⚠ 未回収」と出す）。

④ の式（時間は「時間」単位。分のままだと 2 乗が大きくなりすぎるため）:
  H ＝ λ1 ×（各車ちょうど 1 ルートの罰則）
     ＋ λ2 ×（各センサーは「どれか 1 ルートで回収」か「u ＝ 1（未回収の印）」のちょうど 1 つ、の罰則）
     ＋ λ3 ×（予備バッテリーを使うルートが 2 本同時に選ばれた組の数）
     ＋ A ×（u ＝ 1 の数）
     ＋（選んだルートの終わる時刻² の合計）
  重みの大きさは λ ＞ A ＞ 時刻² にする（決まり ＞ 回れるだけ回る ＞ 早く終わる、の順に大事なので）。
"""
import time
from itertools import combinations

import numpy as np
from amplify import ConstraintList, Poly, VariableGenerator, greater_equal, less_equal, one_hot, solve

from .routing import build_instance, evaluate, route_times


def add_up(variables, numbers, weights=None):
    """variables のうち numbers 番目の変数を足し合わせた式（weights があれば重みを掛けて足す）."""
    total = Poly()
    for k, i in enumerate(numbers):
        total += variables[i] if weights is None else float(weights[k]) * variables[i]
    return total


# ============================================================
# Amplify で解く（クライアントは solvers.make_amplify_client で作る）
# ============================================================
def solve_with_amplify(model, client):
    """Amplify にモデルを渡して、いちばん良い答えの変数の値を返す.

    filter_solution=False: 罰則（決まり）を守れていない答えも返してもらう。
    守れていない所は⑤の後処理で見つけて画面に出す（あとから勝手に直さない）。
    """
    result = solve(model, client, filter_solution=False)
    if len(result) == 0:
        raise RuntimeError('Amplify から答えが返ってきませんでした')
    return result.best.values


def read_values(variables, values):
    """変数の配列に、答えの値（0 か 1）を入れて取り出す."""
    return np.round(variables.evaluate(values)).astype(int)


# ============================================================
# ② 飛行の選択（集合被覆の QUBO）
# ============================================================
def build_flight_selection(inst, penalty=1.0):
    """② のモデルを作る.

    変数  z[f] = 1 … 飛行 f を選ぶ（f は①で数え上げた飛行。どの離陸地点から・どのセンサーを回るか）

    H1 ＝（選んだ飛行の本数）
        ＋ ε ×（選んだ飛行の「近い拠点からの走行時間 ＋ 飛行時間」の合計）
        ＋ λ_cover ×（どの選んだ飛行にも入っていないセンサーの罰則）

    ・本数を少なくする … 離陸する回数（＝充電の回数）が少ないほど早く終わりやすい
    ・ε の項 … 本数が同じなら、拠点から近くて飛行の短いほうを選ぶ（ε は全部足しても 1 未満になる小ささ）
    ・λ_cover の項 … 「各センサーを 1 本以上の飛行が覆う」という決まり（集合被覆）
    """
    flights = inst.flights
    n_vehicles = len(inst.vehicles)
    # 各飛行の「いちばん近い拠点から走って準備するまでの時間 ＋ 飛行時間」（分）
    time_of_flight = np.array([min(inst.drive_from_base(v, f) for v in range(n_vehicles)) + flights[f].flight_min
                               for f in range(len(flights))])
    # ε: 対象のセンサーの数だけ飛行を選んでも、ε の項の合計が 1 未満になる大きさ
    epsilon = 1.0 / (max(len(inst.targets), 1) * float(time_of_flight.max()) + 1.0)
    cost = 1.0 + epsilon * time_of_flight          # 1 本あたりの費用（1 〜 2 未満）
    lam_cover = 2.0 * penalty                      # 費用の最大（2 未満）より大きくする

    gen = VariableGenerator()
    z = gen.array('Binary', len(flights))
    total_cost = add_up(z, range(len(flights)), cost)                  # 選んだ飛行の費用の合計

    penalty_cover = ConstraintList()
    for s in inst.targets:
        covering = [f for f, flight in enumerate(flights) if s in flight.sensors]
        if covering:   # 届く飛行が 1 本も無いセンサーは①で「対応不能」になっているので、ここには来ない
            penalty_cover.append(lam_cover * greater_equal(add_up(z, covering), 1))

    model = total_cost + penalty_cover
    return model, z, dict(n_variables=len(flights), lam_cover=lam_cover, epsilon=epsilon)


def decode_flight_selection(inst, z, values):
    """② の答えから、選んだ飛行と、③で使ってよい離陸地点を取り出す."""
    chosen = [f for f, v in enumerate(read_values(z, values)) if v == 1]
    covered = {s for f in chosen for s in inst.flights[f].sensors}
    launch_ids = sorted({inst.flights[f].launch_id for f in chosen})
    uncovered = [s for s in inst.targets if s not in covered]   # 罰則を守れず、覆われなかったセンサー
    return dict(chosen_flights=chosen, launch_ids=launch_ids, uncovered=uncovered)


# ============================================================
# ③ ルートの数え上げ（普通のコード）
# ============================================================
def enumerate_routes(inst, launch_ids, max_flights_per_route):
    """車ごとのルートを数え上げる.

    ルート = 拠点を出て、離陸地点で飛行を順に行い、拠点に戻る道のり（時刻のルールは routing.route_times）。
    使う飛行: ②で選んだ離陸地点から出る飛行すべて（同じ地点から別のセンサーの組を回る飛行も使える）。
    決まり:
      ・1 本のルートの中で、同じセンサーを 2 回回らない
      ・飛ぶ回数は、予備なしなら flights_builtin 回まで、予備ありなら + flights_per_spare 回まで
        （予備は全体で spare_total 個。0 個なら予備ありのルートは作らない）
      ・画面の「1 台が回る飛行の最大回数」(max_flights_per_route) も超えない（数え上げが増えすぎないため）
    減らし方:
      ・同じ拠点の車は同じ場所から出るので、ルートは拠点ごとに 1 回だけ数える
      ・途中までの並びは「回ったセンサーの組 ＋ 最後の飛行」が同じなら、いちばん早いものだけ伸ばす
      ・最後に「回ったセンサーの組 ＋ 予備の有無」が同じなら、いちばん早く終わる 1 本だけ残す

    戻り値: [{"vehicle": 車の番号, "flights": [飛行の番号...], "sensors": frozenset, "spare": bool, "end_min": 分}, ...]
    """
    fleet = inst.cfg.fleet
    allowed = set(launch_ids)
    pool = [f for f, flight in enumerate(inst.flights) if flight.launch_id in allowed]
    longest = min(max_flights_per_route, inst.max_flights(fleet.spare_total > 0))

    routes = []
    first_vehicle_of_base = {}
    for v, veh in enumerate(inst.vehicles):
        first_vehicle_of_base.setdefault(veh.base_id, v)
    for base_id, v0 in first_vehicle_of_base.items():
        # 途中までの並び: (センサーの組, 最後の飛行) → (最後の飛行の離陸時刻, 飛行の並び)
        partial = {}
        for f in pool:
            t = inst.drive_from_base(v0, f)
            if np.isfinite(t):
                key = (frozenset(inst.flights[f].sensors), f)
                if key not in partial or t < partial[key][0]:
                    partial[key] = (t, [f])
        finished = list(partial.values())
        for _ in range(longest - 1):        # 飛行を 1 回ずつ付け足す
            grown = {}
            for (done, last), (t, seq) in partial.items():
                for g in pool:
                    if done & set(inst.flights[g].sensors):
                        continue            # 同じセンサーを 2 回回らない
                    t_next = t + inst.gap(last, g)
                    if not np.isfinite(t_next):
                        continue
                    key = (done | frozenset(inst.flights[g].sensors), g)
                    if key not in grown or t_next < grown[key][0]:
                        grown[key] = (t_next, seq + [g])
            partial = grown
            finished += list(partial.values())

        # 「センサーの組 ＋ 予備の有無」ごとに、いちばん早く終わる 1 本を残す
        best = {}
        for _, seq in finished:
            _, end_recovery, end_return = route_times(inst, v0, seq)
            end = end_recovery if inst.end_mode == 'recovery' else end_return
            if not np.isfinite(end_return):
                continue                    # 拠点に帰れないルートは使わない
            spare = len(seq) > fleet.flights_builtin
            key = (frozenset(s for f in seq for s in inst.flights[f].sensors), spare)
            if key not in best or end < best[key][0]:
                best[key] = (end, seq)
        # 同じ拠点の車それぞれに、同じルートの一覧を配る
        for v, veh in enumerate(inst.vehicles):
            if veh.base_id != base_id:
                continue
            for (sensors, spare), (end, seq) in best.items():
                routes.append(dict(vehicle=v, flights=seq, sensors=sensors, spare=spare, end_min=float(end)))
    return routes


# ============================================================
# ④ ルートの選択（QUBO その 2）
# ============================================================
def build_route_selection(inst, routes, penalty=1.0):
    """④ のモデルを作る.

    変数  r[i] = 1 … ルート i（ある車の 1 本）を選ぶ。各車に「空ルート」（出発しない）の変数も 1 つ用意する
          u[s] = 1 … センサー s を回収しない（未回収の印。電池や車が足りず回れないときに使う）

    H ＝ λ1 ×（各車ちょうど 1 ルートの罰則）＋ λ2 ×（各センサーは 1 ルートで回収か u ＝ 1 のちょうど 1 つ、の罰則）
       ＋ λ3 ×（予備ありルートが 2 本同時に選ばれた組の数）＋ A ×（u ＝ 1 の数）＋（選んだルートの終わる時刻²の合計）
    """
    n_vehicles = len(inst.vehicles)
    # 空ルートを各車に 1 本ずつ足す（終わる時刻 0、センサーなし）
    all_routes = list(routes) + [dict(vehicle=v, flights=[], sensors=frozenset(), spare=False, end_min=0.0)
                                 for v in range(n_vehicles)]
    end_hours = np.array([r['end_min'] / 60.0 for r in all_routes])     # 時間単位
    time_squared = end_hours ** 2

    # 重みの決め方（λ ＞ A ＞ 時刻²）
    #   A: センサーを 1 基多く回ると、ある車のルートが長くなり時刻²がいくらか増える。その増え方の最大
    #      （いちばん遅いルートの時刻²）より A を大きくすれば、「回れるのに回らない」ほうが損になる。
    #   λ: 決まりを 1 つ破ると、A か時刻²のどちらかを最大でその分だけ得できる。それより大きくする。
    biggest_square = float(time_squared.max()) if len(all_routes) else 1.0
    A = biggest_square + 1.0
    lam = penalty * (A + biggest_square + 1.0)
    lam1 = lam2 = lam3 = lam

    gen = VariableGenerator()
    r = gen.array('Binary', len(all_routes))
    u = gen.array('Binary', len(inst.targets))

    # 目的: 選んだルートの終わる時刻² の合計 ＋ A ×（未回収の数）
    objective = add_up(r, range(len(all_routes)), time_squared) + A * add_up(u, range(len(inst.targets)))

    # λ1: 各車ちょうど 1 ルート（空ルートも 1 ルートと数える）
    penalty_vehicle = ConstraintList()
    for v in range(n_vehicles):
        mine = [i for i, route in enumerate(all_routes) if route['vehicle'] == v]
        penalty_vehicle.append(lam1 * one_hot(add_up(r, mine)))

    # λ2: 各センサーは「どれか 1 ルートで回収」か「u ＝ 1」のちょうど 1 つ
    penalty_sensor = ConstraintList()
    for k, s in enumerate(inst.targets):
        collecting = [i for i, route in enumerate(all_routes) if s in route['sensors']]
        penalty_sensor.append(lam2 * one_hot(add_up(r, collecting) + u[k]))

    # λ3: 予備バッテリーを使うルート
    spare_routes = [i for i, route in enumerate(all_routes) if route['spare']]
    spare_total = inst.cfg.fleet.spare_total
    penalty_spare = Poly()          # 予備ありルートの罰則（式で書く場合）
    spare_limit = ConstraintList()  # 予備ありルートの罰則（予備が 2 個以上のとき、制約で書く場合）
    if spare_routes and spare_total == 1:
        # 予備は全体で 1 個 → 予備ありルートが 2 本同時に選ばれた組の数を罰則にする
        # （s = 予備ありルートの選んだ本数 とすると、組の数 = s(s−1)/2。0/1 の変数は r×r = r なので式で書ける）
        s = add_up(r, spare_routes)
        penalty_spare = lam3 * (s * s - s) / 2
    elif spare_routes and spare_total < n_vehicles:
        # 予備が 2 個以上のとき（初期値ではここは通らない）: 予備ありルートの本数 ≦ 予備の総数
        spare_limit.append(lam3 * less_equal(add_up(r, spare_routes), spare_total))

    model = objective + penalty_spare + penalty_vehicle + penalty_sensor + spare_limit
    info = dict(n_routes=len(routes), n_variables=len(all_routes) + len(inst.targets),
                A=round(A, 3), lam=round(lam, 3))
    return model, r, u, all_routes, info


def decode_route_selection(inst, r, u, all_routes, values):
    """④ の答えから、各車のルートと、決まりを守れていない所を取り出す."""
    picked = read_values(r, values)
    marked = read_values(u, values)
    violations = []
    routes, spares = {}, {}
    for v, veh in enumerate(inst.vehicles):
        mine = [i for i, route in enumerate(all_routes) if route['vehicle'] == v and picked[i] == 1]
        if len(mine) != 1:
            violations.append(f'{veh.vehicle_id}: ルートが {len(mine)} 本選ばれました（決まりは 1 本）')
        if mine:
            # 2 本以上選ばれたときは、センサーの多いほうを使う（ほかは使わない。直して付け足すことはしない）
            i = max(mine, key=lambda i: len(all_routes[i]['sensors']))
            routes[v] = list(all_routes[i]['flights'])
            spares[v] = all_routes[i]['spare']
    for k, s in enumerate(inst.targets):
        n = sum(1 for i, route in enumerate(all_routes) if picked[i] == 1 and s in route['sensors']) + marked[k]
        if n != 1:
            violations.append(f'{s}: 「回収」と「未回収の印」が合わせて {n} 個（決まりは 1 個）')
    n_spare = sum(1 for i, route in enumerate(all_routes) if picked[i] == 1 and route['spare'])
    if n_spare > inst.cfg.fleet.spare_total:
        violations.append(f'予備バッテリーを使うルートが {n_spare} 本（予備は {inst.cfg.fleet.spare_total} 個）')
    return routes, spares, violations, [s for k, s in enumerate(inst.targets) if marked[k] == 1]


# ============================================================
# ①〜⑤ を通して解く
# ============================================================
def solve_phase2(pre, cfg, client, end_mode='recovery', penalty=1.0, max_flights_per_route=3, progress=None):
    """① の前処理の結果 pre から、②〜⑤ を順に行う.

    client: Amplify AE のクライアント（solvers.make_amplify_client で作る）
    progress: 進み具合を画面に伝えるための関数（文字列を受け取る）
    """
    def say(text):
        if progress:
            progress(text)

    t_start = time.time()
    inst = build_instance(pre, cfg, end_mode=end_mode, cand_per_base=None)   # ①の飛行を全部そのまま使う
    if not inst.targets:
        raise ValueError('ドローンで回復できる通信断センサーがありません（被害を設定してください）')
    if not inst.vehicles:
        raise ValueError('車が 0 台です（詳しい設定で台数を決めてください）')

    # ② 飛行の選択
    say(f'② 飛行の選択を Amplify で解いています（飛行の候補 {len(inst.flights)} 本）')
    model2, z, info2 = build_flight_selection(inst, penalty)
    selection = decode_flight_selection(inst, z, solve_with_amplify(model2, client))

    # ③ ルートの数え上げ
    say(f'③ ルートを数え上げています（使う離陸地点 {len(selection["launch_ids"])} か所）')
    routes = enumerate_routes(inst, selection['launch_ids'], max_flights_per_route)

    # ④ ルートの選択
    say(f'④ ルートの選択を Amplify で解いています（ルートの候補 {len(routes)} 本）')
    model4, r, u, all_routes, info4 = build_route_selection(inst, routes, penalty)
    chosen, spares, violations, marked = decode_route_selection(inst, r, u, all_routes, solve_with_amplify(model4, client))

    # ⑤ 後処理: 時刻つきの予定表と、決まりの確認（バッテリー・走れるか・重複）は routing.evaluate で行う
    say('⑤ 予定表を作っています')
    ev = evaluate(inst, chosen, spares)
    if ev['duplicated']:
        violations.append(f'2 回回ったセンサー: {"、".join(ev["duplicated"])}')
    if not ev['battery_ok']:
        violations.append('飛ぶ回数か予備バッテリーの数が決まりを超えています')
    if not ev['finite']:
        violations.append('走って行けない区間を含むルートがあります')
    return dict(
        inst=inst, routes=chosen, eval=ev, violations=violations,
        missing=ev['missing'],                       # 回収できなかったセンサー（画面で ⚠ 未回収）
        marked_unrecovered=marked,                   # ④で u ＝ 1（未回収の印）になったセンサー
        uncovered_in_step2=selection['uncovered'],   # ②でどの飛行にも覆われなかったセンサー
        steps=dict(
            step2=dict(candidates=len(inst.flights), chosen=len(selection['chosen_flights']),
                       launch_points=len(selection['launch_ids']), **info2),
            step4=info4,
        ),
        seconds=round(time.time() - t_start, 1),
    )
