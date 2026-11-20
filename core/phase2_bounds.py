"""二段階目の「下界」と、証明に使う前提のチェック.

下界 ＝「どんな計画でも、これより早くは終わらない」と計算で言える時刻。
アニーリングの結果とは関係なく成り立つので「証明済み」として使える。
"""
import math

import numpy as np


def vehicle_lower_bound(inst, sensors):
    """1 台がセンサーの組 sensors を回るときの、最後の回復時刻の下界（設計書 7.1 節）.

    下界 ＝ 準備 ＋（各センサーの「離陸から回復まで」の最短の合計）＋ 充電 ×（基数 − 1）
    なぜ成り立つか：
      ・最初の離陸は「走行 ＋ 準備」以降なので、準備より前にはならない
      ・次の離陸までは「飛行 ＋ max(充電, 走行 ＋ 準備)」かかるので、少なくとも「回復まで ＋ 充電」はかかる
    1 回の飛行で 1 基だけ回る前提の式。空の組は 0。どれかのセンサーに飛行が無ければ inf（回れない）。
    """
    sensors = list(sensors)
    if not sensors:
        return 0.0
    fleet = inst.cfg.fleet
    total = fleet.launch_setup_min + fleet.charge_min * (len(sensors) - 1)
    for sensor in sensors:
        fastest = min((offset for flight in inst.flights
                       for s, offset in zip(flight.sensors, flight.recover_off) if s == sensor), default=math.inf)
        total += fastest
    return total


def drive_is_shortest_path_closed(drive, tolerance=1e-9):
    """走行時間の表が「最短経路の表」になっているか（三角不等式 d[a,c] ≤ d[a,b] + d[b,c] がすべて成り立つか）.

    これが成り立てば、「拠点から直接行けない点には、途中を経由しても行けない」と言える。
    最大回復数の証明や、締切で担当を外すときの「拠点からの最短」は、この前提に頼っている。
    前処理（Dijkstra で作った表）なら必ず成り立つ。手で作った表や、一部だけ書き換えた表では崩れることがある。
    """
    drive = np.asarray(drive, dtype=float)
    n = len(drive)
    for middle in range(n):
        via = drive[:, [middle]] + drive[[middle], :]      # a → middle → c の時間
        if np.any(drive > via + tolerance):
            return False
    return True
