"""震源と半径から、地震の被害（通信断のセンサー・通行止めの区間）をまとめてつくる.

画面で 1 つずつクリックして選ぶのは大変なので、「震源」と「範囲」と「強さ」を決めると、
震源に近いものほど被害を受けやすいようにランダムに選ぶ。

センサー: 範囲の中なら「強さ」の確率で通信断（強さ 1.0 なら範囲の中は全部通信断）。範囲より遠いものは 0。
          先方サンプルの説明「震源から約 20 km 以内のセンサーが通信断」に合わせた（2026-11-05）。
道路:     強さ × (1 − 震源からの距離 / 範囲)   （範囲より遠いものは 0）
          震源の真上なら「強さ」の確率、範囲のふちに近づくほど 0 に近づく。
乱数の種（seed）が同じなら、何度押しても同じ被害になる（説明や比べるときに便利なため）。
"""
import math
import random


def damage_probability(distance_m, radius_km, strength):
    """震源から distance_m [m] 離れたものが被害を受ける確率."""
    radius_m = radius_km * 1000
    if radius_m <= 0 or distance_m >= radius_m:
        return 0.0
    return strength * (1 - distance_m / radius_m)


def sensor_down_probability(distance_m, radius_km, strength):
    """震源から distance_m [m] 離れたセンサーが通信断になる確率（範囲の中は一律「強さ」）."""
    return strength if distance_m < radius_km * 1000 else 0.0


def segment_middle(segment):
    """道路区間の真ん中あたりの点（折れ線の真ん中の頂点）."""
    coords = segment['coords']
    return coords[len(coords) // 2]


def make_damage(epicenter, sensors, segments, sensor_radius_km, sensor_strength, road_radius_km, road_strength,
                seed=0):
    """被害をつくる.

    epicenter: 震源の座標 (x, y) [m]
    sensors:   {センサー ID: (x, y, ...)}  … 一段階目で設置したセンサーだけを渡す
    segments:  道路区間の一覧（segment_id と coords を持つ dict）
    戻り値: (通信断のセンサー ID の set, 通行止めの区間 ID の set)
    """
    rng = random.Random(seed)
    ex, ey = epicenter
    down = set()
    for sensor_id in sorted(sensors):                  # 並びを固定して、同じ seed なら同じ結果にする
        x, y = sensors[sensor_id][:2]
        p = sensor_down_probability(math.hypot(x - ex, y - ey), sensor_radius_km, sensor_strength)
        if rng.random() < p:
            down.add(sensor_id)
    closed = set()
    for segment in sorted(segments, key=lambda s: s['segment_id']):
        x, y = segment_middle(segment)
        p = damage_probability(math.hypot(x - ex, y - ey), road_radius_km, road_strength)
        if rng.random() < p:
            closed.add(segment['segment_id'])
    return down, closed
