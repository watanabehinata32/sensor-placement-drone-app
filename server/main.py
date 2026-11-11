"""センサー配置と災害時ドローン巡回の Web サイト（計算の窓口・FastAPI）.

起動（sensor_app フォルダで）:
    uvicorn server.main:app --reload
ブラウザで http://127.0.0.1:8000 を開く。

しくみ:
- 画面（web/ の HTML と JavaScript）は、ここにある「窓口（/api/...）」に条件を送り、答えを JSON で受け取る。
- サーバーは「いまの配置」や「いまの被害」を覚えない。毎回、画面から全部の条件を送ってもらう
  （開いている人ごとに状態が混ざらず、作りも簡単になるため）。
- 覚えておくのは、起動時に 1 回だけ読むデータ（道路網など）と、二段階目の計算の進み具合だけ。
- ソルバーは Amplify AE。トークンは .env（環境変数 AMPLIFY_TOKEN）から読み、画面にもログにも出さない。
  一段階目だけは、トークンがないとき確認用に手元のタブーサーチで解く（画面に「確認用」と出す）。
  二段階目は Amplify だけで解く（トークンがなければ、その旨を画面に出して止める）。
"""
import json
import os
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core import phase1
from core.config import Config, FleetSpec
from core.damage import make_damage
from core.data_io import load_roads, load_scenario, load_sensors
from core.phase2 import solve_phase2 as solve_phase2_steps
from core.preprocess import RoadNetwork
from core.solvers import make_amplify_client
from server.settings import APP_DIR, amplify_token

DATA_DIR = os.path.join(APP_DIR, 'data')
# 道路データ: 先方（国際航業）から届いた本データ（2026-11-04）。中身は phase2_roads_company/ と同じ。
# 前の仮データ（資料の図から作った phase2_roads_v2）に戻すときは 'roads_company' を 'roads' にし、
# scenario を 'scenario_eq_tonan_west.json'、loader_config を None にする。
ROADS_DIR = os.path.join(DATA_DIR, 'roads_company')
WEB_DIR = os.path.join(APP_DIR, 'web')
PATHS = dict(
    sensors=os.path.join(DATA_DIR, 'data_transformed.json'),          # 一段階目の配布データ（候補地点・市町村）
    exact=os.path.join(DATA_DIR, 'phase1_exact_placement.json'),      # 一段階目の厳密解の配置（読むだけ）
    segments=os.path.join(ROADS_DIR, 'road_segments.geojson'),
    nodes=os.path.join(ROADS_DIR, 'road_nodes.geojson'),
    facilities=os.path.join(ROADS_DIR, 'facilities.geojson'),
    scenario=os.path.join(ROADS_DIR, 'scenario.json'),                # 先方のサンプルの地震（最初に表示する被害）
    loader_config=os.path.join(ROADS_DIR, 'loader_config.json'),     # 道路の読み込み設定（車の速度は先方シナリオの値）
    style=os.path.join(ROADS_DIR, 'style.json'),                      # 資料の凡例の色
)


def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# 起動時に 1 回だけ読むデータ
# ============================================================
print('データを読み込んでいます…')
PHASE1_DATA = phase1.load_phase1_data(PATHS['sensors'])
SEGMENTS, NODES = load_roads(PATHS['segments'], PATHS['nodes'], config=PATHS['loader_config'])
ALL_SENSORS = load_sensors(PATHS['sensors'])          # 候補地点 86 か所 {ID: (x, y, 市町村 ID)}
SCENARIO = load_scenario(PATHS['scenario'])
# 道路網（離陸候補点）は候補地点 86 か所すべてで 1 回だけ作る。
# どの配置でも、設置したセンサーの分はこの中に含まれるので、配置が変わっても作り直さなくてよい。
NETWORK = RoadNetwork(SEGMENTS, NODES, ALL_SENSORS, Config())
SEGMENT_INFO = {f['properties']['segment_id']: f['properties'] for f in read_json(PATHS['segments'])['features']}
print('読み込みおわり')

app = FastAPI(title='センサー配置と災害時ドローン巡回')


def solver_label():
    return 'Amplify AE' if amplify_token() else '手元のタブーサーチ（確認用：トークン未設定）'


# ============================================================
# 2-1 地図データ
# ============================================================
@app.get('/api/map')
def get_map():
    """地図に描くもの全部（県・市町村・候補地点・道路・拠点・凡例の色）と、サンプルの被害・初期の設定."""
    raw = PHASE1_DATA['raw']
    roads = [dict(id=s['segment_id'], cls=s['highway'], name=s['name'] or '',
                  cls_ja=SEGMENT_INFO.get(s['segment_id'], {}).get('road_class_ja') or SEGMENT_INFO.get(s['segment_id'], {}).get('road_type', ''),
                  coords=[[round(x, 1), round(y, 1)] for x, y in s['coords']])
             for s in SEGMENTS]
    facilities = [dict(f['properties'], x=f['geometry']['coordinates'][0], y=f['geometry']['coordinates'][1])
                  for f in read_json(PATHS['facilities'])['features']]
    fleet = Config().fleet
    return dict(
        boundary=raw['prefecture_outer_boundary']['vertices'],
        municipalities=[dict(id=m['id'], name=m['name'].split(' (')[0], polygon=m['polygon_vertices'])
                        for m in raw['municipalities']],
        candidates=[dict(id=c['id'], x=c['x'], y=c['y'], muni=m['id'])
                    for m in raw['municipalities'] for c in m['candidate_locations']],
        # 一段階目の「詳しい設定」の表の初期値（センサーの性能と市町村の必要値。中央市のカバー率下限は 95%）
        sensor_types=[dict(name=s['name'], cost=float(PHASE1_DATA['cost'][k]), radius_km=float(PHASE1_DATA['radius'][k]) / 1000,
                           alpha=float(PHASE1_DATA['alpha'][k]), beta=float(PHASE1_DATA['beta'][k]))
                      for k, s in enumerate(PHASE1_DATA['specs'])],
        muni_needs=[dict(name=m['name'].split(' (')[0], n_candidates=len(m['candidate_locations']),
                         need_alpha=float(PHASE1_DATA['need_a'][i]), need_beta=float(PHASE1_DATA['need_b'][i]),
                         need_coverage=float(PHASE1_DATA['need_c'][i]))
                    for i, m in enumerate(PHASE1_DATA['munis'])],
        roads=roads,
        facilities=facilities,
        style=read_json(PATHS['style']),
        sample=dict(title=SCENARIO['title'], closed=sorted(SCENARIO['closed_segments']), down=SCENARIO['comm_down'],
                    epicenter=SCENARIO['raw'].get('epicenter'), bases=SCENARIO['bases']),
        defaults=dict(phase1=phase1.DEFAULTS,
                      fleet=dict(charge_min=fleet.charge_min, flights_builtin=fleet.flights_builtin),
                      damage=dict(sensor_radius_km=20.0, sensor_strength=1.0, road_radius_km=20.0, road_strength=0.5)),
        solver=solver_label(),
    )


# ============================================================
# 2-2 一段階目を解く ／ 配置の確認 ／ 2-3 厳密解の配置
# ============================================================
class Phase1Request(BaseModel):
    grid: float = 500
    lam_one: float = 1.0
    lam_ab: float = 1.0
    lam_cov: float = 4.0
    cross: bool = False
    time_limit_ms: int = 5000
    sensors: list = None   # 表で変えたセンサーの性能 [{"cost", "radius_km", "alpha", "beta"}]（None なら元の値）
    munis: list = None     # 表で変えた市町村の必要値 [{"need_alpha", "need_beta", "need_coverage"}]


class PlacementRequest(BaseModel):
    placed: list          # [{"id": 地点 ID, "type_name": 種類の名前}, ...]
    grid: float = 500
    cross: bool = False
    sensors: list = None
    munis: list = None


def placement_summary(data, place, covers, source):
    """配置 1 つ分の答え（画面に出すもの）をまとめる."""
    rows = phase1.check_placement(data, covers, place)
    return dict(placed=phase1.placement_to_list(data, place),
                total_cost=float((data['cost'] * place).sum()),
                count=int(place.sum()), rows=rows, all_ok=all(r['ok'] for r in rows),
                multi=int((place.sum(axis=1) > 1).sum()),     # 1 地点に 2 台以上置いた所の数（罰金が効いていない印）
                source=source)


@app.post('/api/phase1/solve')
def solve_phase1(req: Phase1Request):
    data = phase1.with_changes(PHASE1_DATA, req.sensors, req.munis)
    covers = phase1.make_covers(data, grid=req.grid, cross=req.cross)
    model, x = phase1.build_model(data, covers, req.lam_one, req.lam_ab, req.lam_cov)
    token = amplify_token()
    if token:
        client = make_amplify_client(token, req.time_limit_ms)
    else:
        from core.local_client import LocalSAClient
        client = LocalSAClient(method='tabu')
    started = time.time()
    try:
        place = phase1.solve_placement(model, x, client)
    except Exception as error:   # 通信の失敗などは画面に伝える（トークンの値は含まれない）
        raise HTTPException(500, f'一段階目の計算に失敗しました: {error}')
    out = placement_summary(data, place, covers, f'一段階目で計算した配置（{solver_label()}）')
    out['seconds'] = round(time.time() - started, 1)
    return out


@app.post('/api/phase1/check')
def check_phase1(req: PlacementRequest):
    """保存した配置などを読み込んだとき、コストと市町村ごとの達成状況を計算し直す."""
    data = phase1.with_changes(PHASE1_DATA, req.sensors, req.munis)
    covers = phase1.make_covers(data, grid=req.grid, cross=req.cross)
    try:
        place = phase1.list_to_placement(data, req.placed)
    except KeyError as error:
        raise HTTPException(400, f'配置に知らない地点か種類があります: {error}')
    return placement_summary(data, place, covers, '読み込んだ配置')


@app.get('/api/phase1/exact')
def exact_placement():
    """厳密解（総コスト 6,240）の配置. 保存してあるファイルを読むだけ."""
    exact = read_json(PATHS['exact'])
    covers = phase1.make_covers(PHASE1_DATA)
    place = phase1.list_to_placement(PHASE1_DATA, exact['placed'])
    return placement_summary(PHASE1_DATA, place, covers, f'厳密解の配置（総コスト {exact["total_cost"]:,}）')


# ============================================================
# 2-4 被害をつくる ／ 2-5 前処理
# ============================================================
class DamageRequest(BaseModel):
    placed_ids: list       # 設置したセンサーの地点 ID
    epicenter: list        # [x, y]
    sensor_radius_km: float = 20.0
    sensor_strength: float = 1.0
    road_radius_km: float = 20.0
    road_strength: float = 0.5
    seed: int = 0


@app.post('/api/damage')
def damage(req: DamageRequest):
    sensors = {s: ALL_SENSORS[s] for s in req.placed_ids if s in ALL_SENSORS}
    down, closed = make_damage(tuple(req.epicenter), sensors, SEGMENTS, req.sensor_radius_km, req.sensor_strength,
                               req.road_radius_km, req.road_strength, seed=req.seed)
    return dict(down=sorted(down), closed=sorted(closed))


class Conditions(BaseModel):
    closed: list           # 通行止めの区間 ID
    down: list             # 通信断のセンサー ID
    bases: list            # [{"base_id", "node_id", "name", "vehicles"}]


@app.post('/api/preprocess')
def preprocess(req: Conditions):
    """ドローンで届かないセンサー（対応不能）の判定と、飛べる飛行の数. 被害を変えるたびに画面から呼ぶ（0.2 秒ほど）."""
    res = NETWORK.run(req.closed, req.down, req.bases)
    return dict(unreachable=res['unreachable'], reachable=sorted(s for s, v in res['sensors'].items()
                                                                 if v['status'] == 'reachable'),
                n_flights=len(res['flights']), n_launch_points=len(res['launch_points']))


# ============================================================
# 2-6 二段階目を解く（受付番号を返し、画面が進み具合を問い合わせる）
# ============================================================
class Phase2Request(Conditions):
    end_mode: str = 'recovery'          # 'recovery' = 最後の回復時刻（資料の目的②）、'return' = 全車帰着
    charge_min: float = 30.0
    flights_builtin: int = 5
    penalty: float = 1.0                # 罰則の強さ（倍率）
    time_limit_ms: int = 3000           # Amplify の 1 回あたりの制限時間（②と④で 1 回ずつ解く）


JOBS = {}                 # 受付番号 → {"status", "stage", "result", "error"}


def route_result(out, closed, bases):
    """解いたルートを、地図と時刻表に描ける形にする."""
    inst = out['inst']
    legs = []
    for v, route in out['routes'].items():
        if route:
            stops = [inst.vehicles[v].base_id] + [inst.flights[f].launch_id for f in route] + [inst.vehicles[v].base_id]
            legs += list(zip(stops[:-1], stops[1:]))
    paths = NETWORK.drive_paths(closed, legs, bases) if legs else {}
    ev = out['eval']
    vehicles = []
    for v, veh in enumerate(inst.vehicles):
        route = out['routes'].get(v, [])
        stops = [veh.base_id] + [inst.flights[f].launch_id for f in route] + [veh.base_id]
        # 走行の 1 区切り（拠点 → 離陸点 → … → 拠点）ごとの道路上の折れ線. アニメーションで車を動かすのに使う
        # 各点は [x, y, 出発からの走行秒数]（高速道路は速く、一般道は遅く動かすため）
        legs_xy = [[[round(x, 1), round(y, 1), round(sec, 1)] for x, y, sec in paths.get((a, b), [])]
                   for a, b in zip(stops[:-1], stops[1:])]
        line = [p[:2] for leg in legs_xy for p in leg]
        flights = [dict(order=i + 1, launch=[inst.flights[f].x, inst.flights[f].y], launch_id=inst.flights[f].launch_id,
                        sensors=inst.flights[f].sensors) for i, f in enumerate(route)]
        table = ev['vehicles'][v]
        # 画面に出す名前（例: B1-2 → 「新緑県庁 2号車」）
        # 拠点名のかっこ書き（例:「（災害対策本部）」）は時刻表などでは長すぎるので外す. 地図の拠点ラベルはそのまま
        base_short = veh.base_name.split('（')[0]
        name = f"{base_short} {veh.vehicle_id.split('-')[-1]}号車"
        vehicles.append(dict(id=veh.vehicle_id, name=name, base_id=veh.base_id, path=line if route else [], legs=legs_xy if route else [], flights=flights,
                             events=table['events'], end_recovery=table['end_recovery'], end_return=table['end_return']))
    return dict(vehicles=vehicles, recover_time={s: round(t, 2) for s, t in ev['recover_time'].items()},
                targets=inst.targets, unreachable=inst.unreachable, missing=out['missing'],
                violations=out['violations'], steps=out['steps'],
                makespan=ev['makespan'], makespan_recovery=ev['makespan_recovery'],
                makespan_return=ev['makespan_return'], feasible=ev['feasible'],
                hover_min=inst.cfg.drone.hover_min, seconds=out['seconds'], solver=solver_label())


def run_phase2(job_id, req):
    """別のスレッドで二段階目を解く（画面を待たせないため）. ①前処理 → ②〜⑤ は core/phase2.py."""
    job = JOBS[job_id]
    try:
        token = amplify_token()
        if not token:
            raise ValueError('Amplify のトークンが設定されていません（二段階目は Amplify だけで解きます。'
                             'sensor_app/.env に AMPLIFY_TOKEN を書いてください）')
        cfg = Config()
        cfg.fleet = FleetSpec(charge_min=req.charge_min, flights_builtin=req.flights_builtin)
        job['stage'] = '① 前処理（走行時間と飛行の数え上げ）'
        pre = NETWORK.run(req.closed, req.down, req.bases)
        out = solve_phase2_steps(pre, cfg, make_amplify_client(token, req.time_limit_ms), end_mode=req.end_mode,
                                 penalty=req.penalty,   # 1 台が飛ぶ最大回数は solve_phase2 の中で自動で決める
                                 progress=lambda text: job.update(stage=text))
        job['result'] = route_result(out, req.closed, req.bases)
        job['status'] = 'done'
    except Exception as error:
        job['error'] = str(error)
        job['status'] = 'error'


@app.post('/api/phase2/solve')
def solve_phase2(req: Phase2Request):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = dict(status='running', stage='受付', result=None, error=None)
    threading.Thread(target=run_phase2, args=(job_id, req), daemon=True).start()
    return dict(job_id=job_id)


@app.get('/api/phase2/jobs/{job_id}')
def phase2_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, 'その受付番号はありません（サーバーを起動し直した可能性があります）')
    job = JOBS[job_id]
    return dict(status=job['status'], stage=job['stage'], result=job['result'], error=job['error'])


# ============================================================
# 画面（web/ のファイル）
# ============================================================
app.mount('/web', StaticFiles(directory=WEB_DIR), name='web')


@app.get('/')
def index():
    return FileResponse(os.path.join(WEB_DIR, 'index.html'))
