"""Phase 2 ステップ 2・3 の共通部品: 問題の組み立て（Instance）と、ルートから時刻表を作る評価.

前処理（preprocess.RoadNetwork.run）の結果を受け取り、
「どの車が・どの離陸点から・どの順に・どのセンサーへ飛ぶか」を解くための数値をそろえる。
二段階目の計算（phase2.py）は、この Instance の時間の数え方と評価（evaluate）を使う。

時刻のルール（仮定は README 参照）:
- 車は時刻 0 に拠点を出て、離陸点まで最短経路で走る。
- 離陸点に着いたら（ドローンが充電済みなら）すぐ飛ぶ。車はドローンが戻るまでその場で待つ。
- 着陸後、次の飛行まで充電 30 分。充電は走行中もできるので、次の離陸は
  「着陸 + max(充電 30 分, 次の離陸点までの走行時間)」。
- センサーの回復時刻 = そのセンサー上空での 3 分滞空が終わった時刻。
- 最後の飛行のあと車は拠点へ戻る。
- 終了時刻の定義は 2 通り（画面で切り替え）:
    "recovery" … 最後のセンサー回復時刻（資料どおり）
    "return"   … 全車が拠点に帰り着く時刻
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .config import Config

END_MODES = {"recovery": "最後のセンサー回復時刻", "return": "全車が拠点に帰り着く時刻"}
INF = math.inf


@dataclass
class Flight:
    """1 回の飛行（離陸点から、いくつかのセンサーを順に回って同じ離陸点に戻る）."""
    idx: int                 # Instance.flights の中での番号
    flight_id: str           # 前処理で付けた ID（例 "F00012"）
    launch_id: str           # 離陸点の ID（例 "L04464"）
    sensors: list            # 回る順のセンサー
    flight_min: float        # 離陸から着陸までの分
    recover_off: list        # 離陸から各センサーの回復（3 分滞空の終わり）までの分（sensors と同じ順）
    x: float = 0.0           # 離陸点の座標（地図用）
    y: float = 0.0

    @property
    def last_recover_off(self):
        """離陸から、この飛行で最後のセンサーが回復するまでの分."""
        return self.recover_off[-1]


@dataclass
class Vehicle:
    """ドック搭載車 1 台（ドローン 1 機を積む）."""
    idx: int
    vehicle_id: str          # 例 "B1-1"（拠点 B1 の 1 台目）
    base_id: str
    base_name: str


@dataclass
class Instance:
    """ステップ 2・3 で解く問題 1 つ分の数値."""
    vehicles: list
    flights: list            # 候補の飛行（絞り込み後）
    targets: list            # 回復の対象（ドローンで届く通信断）センサー
    unreachable: list        # 対応不能のセンサー
    drive: np.ndarray        # 走行時間の表 [点, 点]（分）。点 = 拠点 + 離陸点（labels の順）。行けなければ inf
    labels: list
    point_of_base: dict      # 拠点 ID → drive の番号
    point_of_flight: list    # 飛行の番号 → その離陸点の drive の番号
    cfg: Config
    end_mode: str = "recovery"
    sensor_xy: dict = field(default_factory=dict)
    base_xy: dict = field(default_factory=dict)
    n_flights_all: int = 0   # 絞り込む前の飛行の本数

    # ---- よく使う時間 ----
    def drive_from_base(self, v, f):
        """車 v の拠点を出てから、飛行 f の離陸点で離陸できるまでの時間 = 走行 + 離陸の準備（行けなければ inf）."""
        return self.drive[self.point_of_base[self.vehicles[v].base_id], self.point_of_flight[f]] + self.cfg.fleet.launch_setup_min

    def drive_between(self, f, g):
        """飛行 f の離陸点から、飛行 g の離陸点までの走行時間."""
        return self.drive[self.point_of_flight[f], self.point_of_flight[g]]

    def drive_to_base(self, v, f):
        """飛行 f の離陸点から、車 v の拠点までの走行時間."""
        return self.drive[self.point_of_flight[f], self.point_of_base[self.vehicles[v].base_id]]

    def gap(self, f, g):
        """f の離陸から、次の g の離陸までの最短時間 = f の飛行 + max(充電, f→g の走行 + 離陸の準備).
        充電は走行中・準備中も進むので、長いほうだけ待てばよい."""
        d = self.drive_between(f, g)
        if not np.isfinite(d):
            return INF
        return self.flights[f].flight_min + max(self.cfg.fleet.charge_min, d + self.cfg.fleet.launch_setup_min)

    def time_after_last(self, v, f):
        """最後の飛行 f の離陸から、終了時刻までの分（終了時刻の定義による）."""
        flight = self.flights[f]
        if self.end_mode == "return":
            return flight.flight_min + self.drive_to_base(v, f)    # 着陸して拠点に帰るまで
        return flight.last_recover_off                             # 最後のセンサーの回復まで

    def max_flights(self, spare):
        """1 台が飛べる回数（予備バッテリーを積むかどうかで変わる）."""
        fleet = self.cfg.fleet
        return fleet.flights_builtin + (fleet.flights_per_spare if spare else 0)


def build_instance(res, cfg=None, end_mode="recovery", cand_per_base=1, extra_cand=1):
    """前処理の結果 res から Instance を作る.

    cand_per_base: 同じセンサーの組を回る飛行のうち、各拠点から見て「走行 + 最後の回復まで」が短いものを何本残すか
    extra_cand:    それとは別に、飛行時間（= 次の飛行までの待ちに効く）が短いものを何本残すか
    （QUBO の変数を減らすための絞り込み。cand_per_base が 0 以下や None なら全部残す）
    """
    cfg = cfg or Config()
    assert end_mode in END_MODES
    meters_per_min = cfg.drone.speed_kmh * 1000 / 60
    hover = cfg.drone.hover_min
    sensors = res["sensors"]
    targets = sorted(s for s, v in sensors.items() if v["status"] == "reachable")
    unreachable = sorted(res["unreachable"])
    launch_point = {p["launch_id"]: p for p in res["launch_points"]}
    labels = res["travel_time_min"]["labels"]
    index_of_label = {label: i for i, label in enumerate(labels)}
    drive = np.array([[INF if v is None else v for v in row] for row in res["travel_time_min"]["matrix"]], float)

    # 車: 拠点ごとに台数分
    vehicles = []
    for b in res["bases"]:
        for k in range(int(b.get("vehicles", 1))):
            vehicle_id = f"{b['base_id']}-{k + 1}"
            vehicles.append(Vehicle(len(vehicles), vehicle_id, b["base_id"], b.get("name") or b["base_id"]))
    point_of_base = {b["base_id"]: index_of_label[b["base_id"]] for b in res["bases"]}

    all_flights = res["flights"]

    def recover_offsets(f):
        """飛行 f の、離陸から各センサーの回復までの分（直線で飛び、各センサーで 3 分滞空）."""
        p = launch_point[f["launch_id"]]
        here = np.array([p["x"], p["y"]])
        t, out = 0.0, []
        for s in f["sensors"]:
            there = np.array([sensors[s]["x"], sensors[s]["y"]])
            t += float(np.hypot(*(there - here))) / meters_per_min + hover
            out.append(round(t, 3))
            here = there
        return out

    # 候補の絞り込み（QUBO の変数を減らすため）
    # 「回るセンサーの組」が同じ飛行どうしで比べ、
    #   ・各拠点から見て「拠点からの走行 + 最後の回復まで」が短いものを cand_per_base 本
    #   ・飛行時間（= 次の飛行までの待ちに効く）が短いものを extra_cand 本
    # だけ残す。組ごとに比べるので、2 基以上を回る飛行も、その組の中で良いものは必ず残る。
    # （絞った問題の最適値は、全部の飛行を使った最適値より少しだけ悪くなることがある）
    if cand_per_base and cand_per_base > 0:
        by_set = {}
        for f in all_flights:
            by_set.setdefault(frozenset(f["sensors"]), []).append(f)
        keep = set()
        for same_set in by_set.values():
            for b in res["bases"]:
                base_point = point_of_base[b["base_id"]]

                def time_to_recover_all(f, base_point=base_point):
                    return drive[base_point, index_of_label[f["launch_id"]]] + recover_offsets(f)[-1]
                for f in sorted(same_set, key=time_to_recover_all)[:cand_per_base]:
                    if np.isfinite(time_to_recover_all(f)):
                        keep.add(f["flight_id"])
            for f in sorted(same_set, key=lambda f: f["flight_min"])[:max(extra_cand or 0, 0)]:
                keep.add(f["flight_id"])
        chosen = [f for f in all_flights if f["flight_id"] in keep]
    else:
        chosen = list(all_flights)

    flights = []
    for f in chosen:
        p = launch_point[f["launch_id"]]
        flights.append(Flight(len(flights), f["flight_id"], f["launch_id"], list(f["sensors"]), float(f["flight_min"]),
                              recover_offsets(f), p["x"], p["y"]))
    point_of_flight = [index_of_label[f.launch_id] for f in flights]
    return Instance(
        vehicles=vehicles, flights=flights, targets=targets, unreachable=unreachable, drive=drive, labels=labels,
        point_of_base=point_of_base, point_of_flight=point_of_flight, cfg=cfg, end_mode=end_mode,
        sensor_xy={s: (v["x"], v["y"]) for s, v in sensors.items()},
        n_flights_all=len(all_flights),
    )


# ----------------------------------------------------------------------
def route_times(inst, v, route):
    """車 v が飛行 route（飛行の番号の列）をこの順に行うときの時刻.

    戻り値: (各飛行の離陸時刻の列, 最後の回復時刻, 拠点に帰り着く時刻)
    """
    if not route:
        return [], 0.0, 0.0
    t = inst.drive_from_base(v, route[0])
    launches = [t]
    for prev_f, next_f in zip(route[:-1], route[1:]):
        t = t + inst.gap(prev_f, next_f)
        launches.append(t)
    last = inst.flights[route[-1]]
    end_recovery = t + last.last_recover_off
    end_return = t + last.flight_min + inst.drive_to_base(v, route[-1])
    return launches, end_recovery, end_return


def evaluate(inst, routes, spares=None):
    """routes = {車の番号: [飛行の番号, ...]} を時刻表にして評価する.

    戻り値 dict:
      recovered（回復できた数）/ missing（回れていない）/ duplicated（2 回回った）センサー、
      recover_time（センサー → 回復時刻）、makespan_recovery / makespan_return / makespan（選んだ定義）、
      vehicles（車ごとの時刻表）、battery_ok、finite（走って行けない区間がないか）、
      feasible（全部回れて、重複なし、走れて、バッテリーも足りる）
    """
    fleet = inst.cfg.fleet
    spares = dict(spares or {})
    recover_time, duplicated = {}, []
    vehicle_tables = []
    makespan_recovery = makespan_return = 0.0
    finite = True
    battery_ok = True
    for v, veh in enumerate(inst.vehicles):
        route = list(routes.get(v, []))
        launches, end_recovery, end_return = route_times(inst, v, route)
        if route and not (np.isfinite(end_recovery) and np.isfinite(end_return)):
            finite = False
        if len(route) > fleet.flights_builtin and not spares.get(v, False):
            spares[v] = True       # 予備なしの回数を超えるなら予備を積んだとして数え、総数オーバーは下で判定
        if len(route) > inst.max_flights(spares.get(v, False)):
            battery_ok = False

        # 時刻表（できごとの列）
        events = []
        if route:
            events.append(dict(t=0.0, kind="出発", where=veh.base_id, note=veh.base_name))
            prev_point = inst.point_of_base[veh.base_id]
            for k, (f, launch) in enumerate(zip(route, launches)):
                flight = inst.flights[f]
                # 前の場所を出た時刻（最初は 0、2 回目からは前の飛行の着陸）+ 走行 = ここに着いた時刻
                left_at = 0.0 if k == 0 else launches[k - 1] + inst.flights[route[k - 1]].flight_min
                arrive = left_at + inst.drive[prev_point, inst.point_of_flight[f]]
                events.append(dict(t=arrive, kind="到着", where=flight.launch_id, note=""))
                # 着いてから離陸まで: まず離陸の準備（10 分）。準備が終わっても充電が終わっていなければ、残りは充電待ち
                setup = inst.cfg.fleet.launch_setup_min
                if setup > 1e-6:
                    events.append(dict(t=arrive, kind="準備", where=flight.launch_id, note=f"離陸の準備 {setup:.0f} 分"))
                if launch - (arrive + setup) > 1e-6:
                    events.append(dict(t=arrive + setup, kind="充電待ち", where=flight.launch_id,
                                       note=f"充電が終わるまで {launch - arrive - setup:.1f} 分"))
                events.append(dict(t=launch, kind="離陸", where=flight.launch_id, note=f"{k + 1} 回目の飛行"))
                for s, offset in zip(flight.sensors, flight.recover_off):
                    events.append(dict(t=launch + offset, kind="回復", where=s, note="3 分滞空おわり"))
                    if s in recover_time:
                        duplicated.append(s)
                    else:
                        recover_time[s] = launch + offset
                events.append(dict(t=launch + flight.flight_min, kind="着陸", where=flight.launch_id, note=""))
                prev_point = inst.point_of_flight[f]
            events.append(dict(t=end_return, kind="帰着", where=veh.base_id, note=veh.base_name))
            makespan_recovery = max(makespan_recovery, end_recovery)
            makespan_return = max(makespan_return, end_return)
        vehicle_tables.append(dict(
            vehicle_id=veh.vehicle_id, base_id=veh.base_id, route=[inst.flights[f].flight_id for f in route],
            flights=len(route), spare=bool(spares.get(v, False)),
            end_recovery=round(end_recovery, 2), end_return=round(end_return, 2), events=events))

    spares_used = sum(1 for t in vehicle_tables if t["spare"])
    if spares_used > fleet.spare_total:
        battery_ok = False
    missing = [s for s in inst.targets if s not in recover_time]
    makespan = makespan_recovery if inst.end_mode == "recovery" else makespan_return
    return dict(
        recovered=len(recover_time), missing=missing, duplicated=sorted(set(duplicated)), recover_time=recover_time,
        makespan_recovery=round(makespan_recovery, 2), makespan_return=round(makespan_return, 2),
        makespan=round(makespan, 2), spares_used=spares_used, battery_ok=battery_ok, finite=finite,
        feasible=(not missing and not duplicated and finite and battery_ok),
        vehicles=vehicle_tables,
    )
