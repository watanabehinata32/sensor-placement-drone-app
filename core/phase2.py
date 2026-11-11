"""二段階目（災害時のドローン巡回）を、QUBO 2 つで解く.

お題 PDF スライド 7 の「解き方の見通し」どおり、次の順に進める。

  ① 前処理（普通のコード）     … 通行止めを除いた道路網で走行時間を求め、各センサーに届く飛行を数え上げる。
                                   どの離陸地点からも届かないセンサーは「対応不能」。（preprocess.py）
  ② 飛行の選択（QUBO その 1）  … 通信断センサーを全部覆う飛行の組を選ぶ（集合被覆）。
                                   今は使わない（use_flight_selection=False が初期値。2026-11-11）。
  ③ ルートの数え上げ（普通のコード）
                               … 車ごとに「拠点 → 離陸地点で飛行 → … → 拠点」のルートを数え上げる。
                                   使う離陸地点は①の全部（②を使うときは②で選んだ地点だけ）／
                                   回るセンサーの組が同じルートは、いちばん早く終わる 1 本だけ残す。
  ④ ルートの選択（QUBO その 2）… 拠点ごとに、台数までのルートを選ぶ（どの車が走るかは⑤で割り当てる）。
  ⑤ 後処理（普通のコード）     … 選んだルートを時刻つきの予定表にし、決まりを守れているか確かめる。

ソルバーは Fixstars Amplify AE だけを使う（トークンは .env から読み、ここには書かない）。
QUBO の答えで回れなかったセンサーを、あとから別の方法で付け足すことはしない（画面で「⚠ 未回収」と出す）。

④ の式（時間は「時間」単位。分のままだと 2 乗が大きくなりすぎるため）:
  H ＝ λ1 ×（各拠点で選ぶルートが台数以下、の罰則）
     ＋ λ2 ×（各センサーは「どれか 1 ルートで回収」か「u ＝ 1（未回収の印）」のちょうど 1 つ、の罰則）
     ＋ A ×（u ＝ 1 の数）
     ＋（選んだルートの終わる時刻³ の合計）
  予備バッテリーは扱わない（2026-11-11 に外した）。
  重みの大きさは λ ＞ A ＞ 時刻³ にする（決まり ＞ 回れるだけ回る ＞ 早く終わる、の順に大事なので）。
"""
import time

import numpy as np
from amplify import ConstraintList, Poly, VariableGenerator, greater_equal, less_equal, one_hot, solve

from .routing import build_instance, evaluate, route_times


# ④の目的で、終わる時刻を何乗するか（2 だと合計を平均的に小さくし、最後の回復が遅れることがあった）
TIME_POWER = 3


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
# 1 台が飛ぶ回数の上限（自動で決めるときの最大）。5 回を超えるとルートの候補が数万本になり重くなるため
AUTO_MAX_FLIGHTS_CAP = 5


def auto_max_flights(n_targets, n_vehicles):
    """1 台が飛ぶ最大回数を、被害の大きさから自動で決める（2026-11-11 ひなたさんと決めた）.

    最大回数 ＝（回れるセンサー数 ÷ 台数 の切り上げ）＋ 1、ただし 5 回まで。
      ・切り上げ … 全部の車で同じ数ずつ回れば、全センサーを回れる最小の回数
      ・＋ 1     … センサーが偏っている所では、近くの車が多めに回った方が早く終わるので 1 回だけ余裕を持たせる
    今のデータでは 1 回の飛行で回れるセンサーは 1 基なので、回数 ＝ 回れるセンサー数 と考えてよい。
    """
    if n_vehicles <= 0:
        return 1
    needed = -(-n_targets // n_vehicles)          # 切り上げの割り算
    return max(1, min(needed + 1, AUTO_MAX_FLIGHTS_CAP))


def enumerate_routes(inst, launch_ids, max_flights_per_route):
    """車ごとのルートを数え上げる.

    ルート = 拠点を出て、離陸地点で飛行を順に行い、拠点に戻る道のり（時刻のルールは routing.route_times）。
    使う飛行: ②で選んだ離陸地点から出る飛行すべて（同じ地点から別のセンサーの組を回る飛行も使える）。
    決まり:
      ・1 本のルートの中で、同じセンサーを 2 回回らない
      ・飛ぶ回数は、内蔵バッテリーで飛べる flights_builtin 回まで
      ・画面の「1 台が回る飛行の最大回数」(max_flights_per_route) も超えない（数え上げが増えすぎないため）
    減らし方:
      ・同じ拠点の車は同じ場所から出るので、ルートは拠点ごとに 1 回だけ数える
      ・途中までの並びは「回ったセンサーの組 ＋ 最後の飛行」が同じなら、いちばん早いものだけ伸ばす
      ・最後に「回ったセンサーの組」が同じなら、いちばん早く終わる 1 本だけ残す

    戻り値: [{"base": 拠点の ID, "flights": [飛行の番号...], "sensors": frozenset, "end_min": 分}, ...]
    """
    allowed = set(launch_ids)
    pool = [f for f, flight in enumerate(inst.flights) if flight.launch_id in allowed]
    longest = min(max_flights_per_route, inst.max_flights())

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

        # 「センサーの組」ごとに、いちばん早く終わる 1 本を残す
        best = {}
        for _, seq in finished:
            _, end_recovery, end_return = route_times(inst, v0, seq)
            end = end_recovery if inst.end_mode == 'recovery' else end_return
            if not np.isfinite(end_return):
                continue                    # 拠点に帰れないルートは使わない
            key = frozenset(s for f in seq for s in inst.flights[f].sensors)   # 回ったセンサーの組
            if key not in best or end < best[key][0]:
                best[key] = (end, seq)
        # ルートは「拠点ごと」に 1 つの一覧にする（2026-11-11 案A）。
        # 同じ拠点の車は出発地点も性能も同じなので、どの車が走っても中身は同じ。車ごとに配ると
        # 「車を入れ替えただけの同じ答え」がたくさんでき、変数も台数倍になるため。どの車が走るかは⑤で決める。
        for sensors, (end, seq) in best.items():
            routes.append(dict(base=base_id, flights=seq, sensors=sensors, end_min=float(end)))
    return routes


# ============================================================
# ④ ルートの選択（QUBO その 2）
# ============================================================
def build_route_selection(inst, routes, penalty=1.0):
    """④ のモデルを作る.

    変数  r[i] = 1 … ルート i（ある拠点の 1 本）を選ぶ
          u[s] = 1 … センサー s を回収しない（未回収の印。電池や車が足りず回れないときに使う）

    H ＝ λ1 ×（各拠点で選ぶルートが台数以下、の罰則）＋ λ2 ×（各センサーは 1 ルートで回収か u ＝ 1 のちょうど 1 つ、の罰則）
       ＋ A ×（u ＝ 1 の数）＋（選んだルートの終わる時刻³の合計）
    """
    # 拠点ごとの台数（選べるルートの本数の上限）
    cars_at_base = {}
    for veh in inst.vehicles:
        cars_at_base[veh.base_id] = cars_at_base.get(veh.base_id, 0) + 1
    # 「出発しない車」は、その拠点でルートを台数より少なく選ぶことで表すので、空ルートの変数は要らない
    all_routes = list(routes)
    end_hours = np.array([r['end_min'] / 60.0 for r in all_routes])     # 時間単位
    # 終わる時刻を TIME_POWER 乗する（2026-11-11 に 2 乗 → 3 乗へ）。
    # 乗数が大きいほど「いちばん遅い車」を重く見るので、最後の回復時刻を小さくする方向に近づく。
    # ルートごとに決まった数を掛けるだけなので、何乗にしても式は 2 次（QUBO）のまま。
    time_squared = end_hours ** TIME_POWER

    # 重みの決め方（λ ＞ A ＞ 時刻²）
    #   A: センサーを 1 基多く回ると、ある車のルートが長くなり時刻²がいくらか増える。その増え方の最大
    #      （いちばん遅いルートの時刻²）より A を大きくすれば、「回れるのに回らない」ほうが損になる。
    #   λ: 決まりを 1 つ破ると、A か時刻²のどちらかを最大でその分だけ得できる。それより大きくする。
    biggest_square = float(time_squared.max()) if len(all_routes) else 1.0
    A = biggest_square + 1.0
    lam = penalty * (A + biggest_square + 1.0)
    lam1 = lam2 = lam

    gen = VariableGenerator()
    r = gen.array('Binary', len(all_routes))
    u = gen.array('Binary', len(inst.targets))

    # 目的: 選んだルートの終わる時刻³（TIME_POWER 乗）の合計 ＋ A ×（未回収の数）
    objective = add_up(r, range(len(all_routes)), time_squared) + A * add_up(u, range(len(inst.targets)))

    # λ1: 各拠点で選ぶルートの本数は、その拠点の台数以下（例: 県庁 4 本以下）
    penalty_vehicle = ConstraintList()
    for base_id, n_cars in cars_at_base.items():
        mine = [i for i, route in enumerate(all_routes) if route['base'] == base_id]
        if len(mine) > n_cars:      # 候補が台数以下なら、決まりは必ず守られるので入れない
            penalty_vehicle.append(lam1 * less_equal(add_up(r, mine), n_cars))

    # λ2: 各センサーは「どれか 1 ルートで回収」か「u ＝ 1」のちょうど 1 つ
    penalty_sensor = ConstraintList()
    for k, s in enumerate(inst.targets):
        collecting = [i for i, route in enumerate(all_routes) if s in route['sensors']]
        penalty_sensor.append(lam2 * one_hot(add_up(r, collecting) + u[k]))

    model = objective + penalty_vehicle + penalty_sensor
    info = dict(n_routes=len(routes), n_variables=len(all_routes) + len(inst.targets),
                A=round(A, 3), lam=round(lam, 3))
    return model, r, u, all_routes, info


def decode_route_selection(inst, r, u, all_routes, values):
    """④ の答えから、各車のルートと、決まりを守れていない所を取り出す."""
    picked = read_values(r, values)
    marked = read_values(u, values)
    violations = []
    routes = {}
    # 拠点ごとに、選ばれたルートをその拠点の車へ 1 号車から順に割り当てる
    for base_id in dict.fromkeys(veh.base_id for veh in inst.vehicles):
        cars = [v for v, veh in enumerate(inst.vehicles) if veh.base_id == base_id]
        chosen = [i for i, route in enumerate(all_routes) if route['base'] == base_id and picked[i] == 1]
        # 終わるのが遅いルートから順に並べる（どの順でも結果は同じ。表示をそろえるため）
        chosen.sort(key=lambda i: -all_routes[i]['end_min'])
        if len(chosen) > len(cars):
            violations.append(f'{base_id}: ルートが {len(chosen)} 本選ばれました（台数は {len(cars)} 台）')
            # 台数を超えた分は使わない（直して付け足すことはしない）
        for v, i in zip(cars, chosen):
            routes[v] = list(all_routes[i]['flights'])
    for k, s in enumerate(inst.targets):
        n = sum(1 for i, route in enumerate(all_routes) if picked[i] == 1 and s in route['sensors']) + marked[k]
        if n != 1:
            violations.append(f'{s}: 「回収」と「未回収の印」が合わせて {n} 個（決まりは 1 個）')
    return routes, violations, [s for k, s in enumerate(inst.targets) if marked[k] == 1]


# ============================================================
# ①〜⑤ を通して解く
# ============================================================
def solve_phase2(pre, cfg, client, end_mode='recovery', penalty=1.0, max_flights_per_route=None, progress=None,
                 use_flight_selection=False):
    """① の前処理の結果 pre から、②〜⑤ を順に行う.

    use_flight_selection: True なら ② 飛行の選択（集合被覆の QUBO）で離陸地点を絞ってから ③ に進む。
        False（今の初期値・2026-11-11）なら ② を飛ばし、①で見つけた離陸地点を全部 ③ で使う。
        ② は走行を見ずに離陸地点を選ぶので、全体で一番良いルートを取りこぼすことがあるため。
        「全部回る」という決まりは ④ の「センサー担当の罰則」に入っているので、② が無くても守られる。

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

    if use_flight_selection:
        # ② 飛行の選択（集合被覆の QUBO）で、使う離陸地点を絞る
        say(f'② 飛行の選択を Amplify で解いています（飛行の候補 {len(inst.flights)} 本）')
        model2, z, info2 = build_flight_selection(inst, penalty)
        selection = decode_flight_selection(inst, z, solve_with_amplify(model2, client))
        launch_ids = selection['launch_ids']
        uncovered_in_step2 = selection['uncovered']
        step2_info = dict(used=True, candidates=len(inst.flights), chosen=len(selection['chosen_flights']),
                          launch_points=len(launch_ids), **info2)
    else:
        # ② は飛ばす：①で見つけた離陸地点を全部使う
        launch_ids = sorted({flight.launch_id for flight in inst.flights})
        uncovered_in_step2 = []
        step2_info = dict(used=False, candidates=len(inst.flights), launch_points=len(launch_ids))

    # ③ ルートの数え上げ
    # 1 台が飛ぶ最大回数: 指定が無ければ（None か 0）、回れるセンサー数と台数から自動で決める
    if not max_flights_per_route:
        max_flights_per_route = auto_max_flights(len(inst.targets), len(inst.vehicles))
    say(f'③ ルートを数え上げています（使う離陸地点 {len(launch_ids)} か所・1台 {max_flights_per_route} 回まで）')
    routes = enumerate_routes(inst, launch_ids, max_flights_per_route)

    # ④ ルートの選択
    say(f'④ ルートの選択を Amplify で解いています（ルートの候補 {len(routes)} 本）')
    model4, r, u, all_routes, info4 = build_route_selection(inst, routes, penalty)
    chosen, violations, marked = decode_route_selection(inst, r, u, all_routes, solve_with_amplify(model4, client))

    # ⑤ 後処理: 時刻つきの予定表と、決まりの確認（バッテリー・走れるか・重複）は routing.evaluate で行う
    say('⑤ 予定表を作っています')
    ev = evaluate(inst, chosen)
    if ev['duplicated']:
        violations.append(f'2 回回ったセンサー: {"、".join(ev["duplicated"])}')
    if not ev['battery_ok']:
        violations.append('飛ぶ回数がバッテリーで飛べる回数を超えています')
    if not ev['finite']:
        violations.append('走って行けない区間を含むルートがあります')
    return dict(
        inst=inst, routes=chosen, eval=ev, violations=violations,
        missing=ev['missing'],                       # 回収できなかったセンサー（画面で ⚠ 未回収）
        marked_unrecovered=marked,                   # ④で u ＝ 1（未回収の印）になったセンサー
        uncovered_in_step2=uncovered_in_step2,       # ②でどの飛行にも覆われなかったセンサー（② を飛ばしたときは空）
        steps=dict(
            step2=step2_info,
            step4=dict(max_flights_per_route=max_flights_per_route, **info4),
        ),
        seconds=round(time.time() - t_start, 1),
    )
