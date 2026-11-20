"""Phase 2 の前提パラメータ（企業資料と 2026-10-23 の仮定）.

本データや条件が変わったらここだけ直す。
"""
from dataclasses import dataclass, field, asdict


@dataclass
class DroneSpec:
    max_flight_min: float = 15.0      # 1 回の最大飛行時間
    hover_min: float = 3.0            # センサー地点での滞空（中身は未定。地点に 3 分いるとだけ扱う）
    speed_kmh: float = 40.0           # 飛行速度
    max_sensors_per_flight: int = 4   # 3 分×5 基 = 15 分で移動ゼロになるので実質 4 が上限

    @property
    def one_way_reach_m(self):
        """1 基だけ回るときの片道到達距離（40 km/h × (15−3) 分 ÷ 2 = 4 km）."""
        return self.speed_kmh * 1000 / 60 * (self.max_flight_min - self.hover_min) / 2


@dataclass
class RoadRule:
    # 高速は走れるが停車（離陸）はできない（先方シナリオ: 離陸できるのは trunk・primary・secondary だけ）
    no_launch_highways: tuple = ("motorway", "motorway_link")
    # 道路上の離陸候補点の間隔 [m]。区間の両端に加えてこの間隔で区間上に点を置く
    launch_spacing_m: float = 250.0
    # 各センサーから各道路区間への最寄り点も候補に入れる（1 基飛行の到達判定を連続のルールと一致させる）
    add_sensor_projections: bool = True
    # 走行速度の倍率（災害時の徐行などを入れたいとき。1.0 = 規制速度どおり）
    speed_factor: float = 1.0


@dataclass
class FleetSpec:
    """ドック搭載車とバッテリー（ステップ 2・3 で使う）."""
    charge_min: float = 30.0          # 着陸後、次に飛べるまでの充電時間。走行中も充電できる（停車して待つ必要はない）
    flights_builtin: int = 5          # 1 台が飛べる回数（出発時に満充電 1 回 + 内蔵バッテリーで充電 4 回。解釈は仮定）
    # 予備バッテリー（2026-11-20 ひなたさん判断で設計書どおり戻した。11/11 の「扱わない」を上書き）
    #   ・1 台に積めるのは 0〜1 個、全体で spare_total 個まで（先方シナリオ scenario.json の fleet.total_spare_batteries）
    #   ・積んだ車は飛べる回数が flights_per_spare 回増える（scenario.json の dock.charges_per_spare_battery）
    #   ・仮のモデル：予備は「回数」だけを増やし、充電時間などの時刻は変えない（交換式か充電用かは未確認）
    #   ・今の従来方式（ルート列挙）は予備を使わない。使うのはハイブリッド方式だけ
    spare_total: int = 0
    flights_per_spare: int = 4
    launch_setup_min: float = 10.0    # 離陸点に着いてから離陸するまでの準備時間（先方シナリオ）。準備中も充電は進む


@dataclass
class Config:
    drone: DroneSpec = field(default_factory=DroneSpec)
    road: RoadRule = field(default_factory=RoadRule)
    fleet: FleetSpec = field(default_factory=FleetSpec)

    def to_dict(self):
        d = asdict(self)
        d["drone"]["one_way_reach_m"] = self.drone.one_way_reach_m
        return d
