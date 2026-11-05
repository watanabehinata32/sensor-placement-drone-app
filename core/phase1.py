"""一段階目：センサーの配置を QUBO で解く.

phase2_app/page_phase1.py（元は webapp/app.py）の式を、画面と切り離してここに移したもの。
2026-11-05 カバー率を「報酬」から「制約条件（罰金項 penalty_cover）」に変えた。
初期値は Render に公開している前回のアプリ（GitHub の Optimal-Sensor-Placement-in-Two-Dimensions の
app.py）に合わせた: λ_cov = 4、制限時間 5000 ミリ秒、マス目 500 m、中央市のカバー率下限 95%。
前回のアプリは罰則（1 地点 1 台・α・β）に重みを付けていない（Amplify AE に任せる）ので、λ_one = λ_ab = 1 にしている。
λ_cov はいまはカバー率の制約（罰金項）の重み。

使い方:
    data = load_phase1_data('data/data_transformed.json')
    covers = make_covers(data, grid=500, cross=False)
    model, x = build_model(data, covers)
    place = solve_placement(model, x, client)      # client は Amplify AE（か手元の確認用）
    rows = check_placement(data, covers, place)    # 市町村ごとに要求を満たしたか
"""
import json

import numpy as np
from amplify import VariableGenerator, greater_equal, less_equal, solve

# 罰金の重み・マス目・制限時間の初期値（前回のアプリ = Render 版の初期値と同じ）
DEFAULTS = dict(grid=500, lam_one=1.0, lam_ab=1.0, lam_cov=4.0, cross=False, time_limit_ms=5000)


def load_phase1_data(path):
    """配布データを読み、計算に使う配列をそろえる.

    戻り値の dict:
      munis, specs   … 市町村（8）とセンサーの種類（3）の元データ
      points         … 候補地点（86）を 1 つのリストにしたもの
      px, py         … 候補地点の座標（m）
      pm             … 候補地点がどの市町村か（市町村の番号）
      cost, radius, alpha, beta     … 種類 k の値（radius は m に直したもの）
      need_a, need_b, need_c        … 市町村 m の α・β 必要量とカバー率下限
    """
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    munis = raw['municipalities']
    specs = raw['sensor_specifications']
    # 中央市（M06）のカバー率下限は元データでは 100% だが、ほぼ達成できないので 95% にする（元のページと同じ）
    munis[5]['required_coverage_rate'] = 0.95

    points = [c for m in munis for c in m['candidate_locations']]
    return dict(
        raw=raw, munis=munis, specs=specs, points=points,
        px=np.array([c['x'] for c in points], float),
        py=np.array([c['y'] for c in points], float),
        pm=np.array([m for m, muni in enumerate(munis) for _ in muni['candidate_locations']]),
        cost=np.array([s['unit_procurement_cost'] for s in specs], float),
        radius=np.array([s['effective_coverage_radius'] for s in specs], float) * 1000,   # km → m
        alpha=np.array([s['alpha_field_response'] for s in specs], float),
        beta=np.array([s['beta_density_resolution'] for s in specs], float),
        need_a=np.array([m['required_alpha_response'] for m in munis], float),
        need_b=np.array([m['required_beta_resolution'] for m in munis], float),
        need_c=np.array([m['required_coverage_rate'] for m in munis], float),
    )


def with_changes(data, sensors=None, munis=None):
    """画面の表で変えた値（センサーの性能・市町村の必要値）を入れた、data のコピーを返す.

    sensors: 種類ごとの [{"cost", "radius_km", "alpha", "beta"}, ...]（None なら元のまま）
    munis:   市町村ごとの [{"need_alpha", "need_beta", "need_coverage"}, ...]（None なら元のまま）
    元の data は書き換えない（ほかの人の計算に影響しないように）。
    """
    changed = dict(data)
    if sensors:
        changed['cost'] = np.array([s['cost'] for s in sensors], float)
        changed['radius'] = np.array([s['radius_km'] for s in sensors], float) * 1000   # km → m
        changed['alpha'] = np.array([s['alpha'] for s in sensors], float)
        changed['beta'] = np.array([s['beta'] for s in sensors], float)
    if munis:
        changed['need_a'] = np.array([m['need_alpha'] for m in munis], float)
        changed['need_b'] = np.array([m['need_beta'] for m in munis], float)
        changed['need_c'] = np.array([m['need_coverage'] for m in munis], float)
    return changed


def make_covers(data, grid=500, cross=False):
    """カバー率を数えるマス目.

    covers[m][g, i, k] = True なら、市町村 m の g 番目のマス目を、地点 i に置いた種類 k のセンサーが覆う。
    マス目の中心点がセンサーの円に入っていれば「覆われている」と数える。
    cross = False なら、隣の市町村に置いたセンサーのはみ出しは数えない。
    """
    px, py, pm, radius = data['px'], data['py'], data['pm'], data['radius']
    covers = []
    for m, muni in enumerate(data['munis']):
        poly = np.array(muni['polygon_vertices'])
        gx, gy = np.meshgrid(np.arange(poly[:, 0].min(), poly[:, 0].max(), grid),
                             np.arange(poly[:, 1].min(), poly[:, 1].max(), grid))
        gx, gy = gx.ravel(), gy.ravel()
        inside = np.zeros(len(gx), dtype=bool)  # マス目の中心が市町村の中にあるか（レイキャスティング法）
        for j in range(len(poly)):
            (x1, y1), (x2, y2) = poly[j - 1], poly[j]
            inside ^= ((y1 > gy) != (y2 > gy)) & (gx < (x2 - x1) * (gy - y1) / (y2 - y1 + 1e-12) + x1)
        gx, gy = gx[inside], gy[inside]
        dist = np.sqrt((gx[:, None] - px[None, :]) ** 2 + (gy[:, None] - py[None, :]) ** 2)
        if not cross:
            dist[:, pm != m] = np.inf  # はみ出しを数えないときは、他の市町村のセンサーは覆わないことにする
        covers.append(dist[:, :, None] <= radius[None, None, :])
    return covers


def build_model(data, covers, lam_one=1.0, lam_ab=1.0, lam_cov=4.0):
    """QUBO のモデルを作る. 戻り値: (model, x)  x[i, k] = 1 なら「地点 i に種類 k のセンサーを置く」."""
    cost, alpha, beta, pm = data['cost'], data['alpha'], data['beta'], data['pm']
    need_a, need_b, need_c = data['need_a'], data['need_b'], data['need_c']
    N, K, M = len(data['points']), len(data['specs']), len(data['munis'])

    # 変数を発行
    gen = VariableGenerator()
    x = gen.array('Binary', shape=(N, K))

    # コスト関数：置いたセンサーのコストの合計（ほかの項と大きさをそろえるため 1/100 にする）
    total_cost = (cost[np.newaxis, :] / 100 * x).sum()

    # 配置の変数を1列に並べる（j = i * K + k 番目が x[i, k]）
    xf = x.flatten()

    # 罰金項：市町村ごとに「覆われたマス目の数 ≥ 必要な数」（カバー率を制約条件として入れる）
    #   「覆われている」を表す 0/1 の変数 y を作り、次の2つを罰金にする。
    #   ① y ≤ そのマス目を覆えるセンサーの台数（1台も覆っていないのに「覆われている」にはできない）
    #   ② 覆われたマス目の数 ≥ 必要カバー率 × マス目の数（小数は切り上げ）
    #   「覆われているのに y = 0」は②の罰金を増やすだけで得がないので、y ≥ x の制約は入れなくてよい。
    #   報酬と違い、同じマス目を何台で重ねて覆っても得は増えない（y は 0 か 1 だから）。
    #
    #   マス目は約 2.5 万個あり、1 個ずつ y を作ると変数が多すぎて解けない（手元で試すとメモリ不足）。
    #   そこで「覆えるセンサーの組み合わせがまったく同じマス目」を1つのグループにまとめ、
    #   グループごとに y を1つ作る（約 2600 個になる）。同じグループのマス目は必ず一緒に覆われるので、結果は変わらない。
    penalty_cover = 0
    for m in range(M):
        cells = covers[m].reshape(len(covers[m]), N * K)  # マス目 × 変数 の True/False の表
        need_cells = int(np.ceil(need_c[m] * len(cells) - 1e-9))  # 必要なマス目の数（1e-9 は小数の誤差よけ）
        if need_cells == 0:
            continue
        # 覆えるセンサーの組み合わせごとにまとめる（だれも覆えないマス目は、どうやっても覆われないので除く）
        coverable = cells[cells.any(axis=1)]
        patterns, cell_counts = np.unique(coverable, axis=0, return_counts=True)
        y = gen.array('Binary', shape=(len(patterns),))  # y[n] = 1 なら「グループ n のマス目は覆われている」
        for n in range(len(patterns)):
            # グループ n を覆っているセンサーの台数（覆える変数だけ 1 を掛けて足す）
            covering_count = (patterns[n].astype(int) * xf).sum()
            penalty_cover += lam_cov * less_equal(y[n] - covering_count, 0)                  # ①
        penalty_cover += lam_cov * greater_equal((cell_counts * y).sum(), need_cells)        # ②

    # 罰金項：1つの地点には1台まで
    penalty_one = lam_one * less_equal(x, 1, axis=1)

    # 罰金項：市町村ごとに α の合計 ≥ 必要量、β の合計 ≥ 必要量
    penalty_ab = 0
    for m in range(M):
        idx = np.nonzero(pm == m)[0]       # 市町村 m の地点の番号（連続して並んでいる）
        xm = x[idx[0]:idx[-1] + 1]         # 市町村 m の地点の変数
        penalty_ab += lam_ab * greater_equal((alpha[np.newaxis, :] * xm).sum(), need_a[m])
        penalty_ab += lam_ab * greater_equal((beta[np.newaxis, :] * xm).sum(), need_b[m])

    # 目的は総コストの最小化。カバー率・1地点1台・α・β はすべて制約（罰金項）として足す
    model = total_cost + penalty_one + penalty_ab + penalty_cover
    return model, x


def solve_placement(model, x, client):
    """モデルを解いて、配置（N × K の 0/1 の表）を返す.

    filter_solution=False：制約を満たさない解も返してもらい、どこが足りないかを表示できるようにする。
    """
    result = solve(model, client, filter_solution=False)
    if len(result) == 0:
        raise RuntimeError('解が得られませんでした')
    return np.round(x.evaluate(result.best.values)).astype(int)


def check_placement(data, covers, place):
    """市町村ごとに α・β・カバー率が必要量に届いているか. 戻り値: 市町村ごとの dict のリスト."""
    alpha, beta, pm = data['alpha'], data['beta'], data['pm']
    rows = []
    for m, muni in enumerate(data['munis']):
        a_sum = float((alpha * place[pm == m]).sum())
        b_sum = float((beta * place[pm == m]).sum())
        rate = float((covers[m] & (place[None, :, :] == 1)).any(axis=(1, 2)).mean())  # 覆われたマス目の割合
        # 下限に少しでも届かなければ不合格（元のページの「0.1 ポイントまでおまけ」はやめた）。
        # 1e-9 は小数の計算誤差（0.9 が 0.8999999… になるなど）を吸収するためだけの余裕
        ok = a_sum >= data['need_a'][m] and b_sum >= data['need_b'][m] and rate >= data['need_c'][m] - 1e-9
        rows.append(dict(id=muni['id'], name=muni['name'].split(' (')[0], count=int(place[pm == m].sum()),
                         alpha=a_sum, need_alpha=float(data['need_a'][m]),
                         beta=b_sum, need_beta=float(data['need_b'][m]),
                         coverage=rate, need_coverage=float(data['need_c'][m]), ok=bool(ok)))
    return rows


def placement_to_list(data, place):
    """N × K の表を、二段階目に渡す形 [{"id": 地点 ID, "type_name": 種類の名前}, ...] に直す."""
    return [{'id': data['points'][i]['id'], 'type_name': data['specs'][k]['name']}
            for i in range(len(data['points'])) for k in range(len(data['specs'])) if place[i, k] == 1]


def list_to_placement(data, placed):
    """[{"id", "type_name"}, ...] を N × K の 0/1 の表に直す（保存した配置や厳密解の配置を読むとき）."""
    point_index = {c['id']: i for i, c in enumerate(data['points'])}
    type_index = {s['name']: k for k, s in enumerate(data['specs'])}
    place = np.zeros((len(data['points']), len(data['specs'])), dtype=int)
    for p in placed:
        place[point_index[p['id']], type_index[p['type_name']]] = 1
    return place
