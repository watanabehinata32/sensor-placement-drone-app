"""QUBO を解く部品.

- make_amplify_client … Fixstars Amplify AE のクライアントを作る（本番用）。トークンはコードに書かず、
                        .env（環境変数 AMPLIFY_TOKEN）から渡す
- TabuSearch / LocalAnnealer … 手元で動く確認用の解き方。一段階目でトークンが無いときだけ使う
                        （二段階目は Amplify だけで解くので使わない）

手元の 2 つは solve(prob) → (x: 0/1 の配列, info: dict) という形。prob は下の QuboProblem。

手元の 2 つは「ある変数 i を反転したらエネルギーがいくつ変わるか」を毎回すばやく求めるため、
field[i] = linear[i] + Σ_j W[i, j] × x[j]  を持ち歩く。x[i] を 0→1 にするとエネルギーは field[i] 増え、
1→0 にすると field[i] 減る。変数を 1 つ反転したら、field に W のその行を足す（引く）だけで更新できる。
"""
import time
from datetime import timedelta

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QuboProblem:
    """係数の表で表した QUBO. エネルギー = const + Σ linear[i]·x_i + Σ quadratic[(i, j)]·x_i·x_j."""
    n: int                    # 変数の数
    linear: np.ndarray        # 1 次の係数（長さ n）
    quadratic: dict           # 2 次の係数 {(i, j): 係数}（i < j）
    const: float              # 定数
    sensor_groups: list = field(default_factory=list)   # 焼きなましの「付け替え」に使うまとまり（無くてよい）
    meta: dict = field(default_factory=dict)

    def matrix(self):
        """対称行列 W（W[i, j] = W[j, i] = 2 次の係数）."""
        W = np.zeros((self.n, self.n))
        for (i, j), c in self.quadratic.items():
            W[i, j] += c
            W[j, i] += c
        return W


class TabuSearch:
    """タブーサーチ.

    毎回「反転するとエネルギーがいちばん下がる（または上がり方がいちばん小さい）変数」を 1 つ反転する。
    同じ所を行ったり来たりしないよう、直前に反転した変数はしばらく（tenure 回）反転を禁止する（タブー）。
    ただし、禁止中でもこれまでの最良を更新できるなら許す。
    QUBO の係数だけを使う汎用の解き方で、焼きなましより制約の多い QUBO に強いことが多い。
    """
    name = "手元のタブーサーチ"

    def __init__(self, iters_per_var=40, restarts=4, tenure=None, seed=0):
        self.iters_per_var = iters_per_var   # 繰り返し回数 = これ × 変数の数
        self.restarts = restarts             # やり直しの回数
        self.tenure = tenure                 # 反転を禁止する回数（None なら変数の数から決める）
        self.seed = seed

    def solve(self, prob):
        t_start = time.time()
        W = prob.matrix()
        linear = prob.linear
        n = prob.n
        rng = np.random.default_rng(self.seed)
        tenure = self.tenure or max(7, n // 50)
        x = np.zeros(n)
        best_x, best_e = x.copy(), 0.0          # すべて 0 のときのエネルギー（定数を除く）は 0
        for restart in range(self.restarts):
            if restart > 0:
                # 2 回目からは、いちばん良い答えの 5% ほどの変数を反転させてから始める
                x = best_x.copy()
                flip = rng.random(n) < 0.05
                x[flip] = 1 - x[flip]
            field = linear + W @ x
            e = float(linear @ x + 0.5 * x @ W @ x)   # W は対称で各項を 2 回数えるので 1/2
            tabu_until = np.zeros(n, int)            # 変数 i は tabu_until[i] 回目まで反転禁止
            for it in range(self.iters_per_var * n):
                change = (1 - 2 * x) * field         # 各変数を反転したときのエネルギーの変化
                allowed = (tabu_until <= it) | (e + change < best_e - 1e-12)
                change_allowed = np.where(allowed, change, np.inf)
                i = int(np.argmin(change_allowed))
                if not np.isfinite(change_allowed[i]):
                    break                            # 動かせる変数がない
                sign = 1 - 2 * x[i]                  # 0→1 なら +1、1→0 なら −1
                x[i] += sign
                field += sign * W[i]
                e += change[i]
                tabu_until[i] = it + tenure + int(rng.integers(0, tenure))
                if e < best_e - 1e-12:
                    best_x, best_e = x.copy(), e
        return best_x.astype(int), dict(solver=self.name, energy=float(best_e + prob.const),
                                        seconds=round(time.time() - t_start, 3), iters=self.iters_per_var * n,
                                        restarts=self.restarts)


class LocalAnnealer:
    """焼きなまし法. 温度を高い所から下げながら、次の 2 種類の動きを試す.

    1. 1 つの変数の 0/1 を反転する（エネルギーが上がる動きも、温度が高いうちはある確率で受け入れる）
    2. 付け替え: あるセンサーを担当している変数を 0 にし、同じセンサーの別の変数（別の車・順番・離陸点）を 1 にする
       （「各センサーちょうど 1 回」を保ったまま動けるので、罰金の山を越えずに済む）
    """
    name = "手元の焼きなまし法"

    def __init__(self, sweeps=600, restarts=3, t_hot=None, t_cold=1e-3, seed=0):
        self.sweeps = sweeps        # 温度を下げる段階の数
        self.restarts = restarts    # 最初からやり直す回数
        self.t_hot = t_hot          # 最初の温度（None なら罰金の大きさ）
        self.t_cold = t_cold        # 最後の温度
        self.seed = seed

    def solve(self, prob):
        t_start = time.time()
        W = prob.matrix()
        linear = prob.linear
        n = prob.n
        rng = np.random.default_rng(self.seed)
        t_hot = self.t_hot or max(prob.meta.get("A", 1.0), 1.0)
        groups = [np.array(g, int) for g in prob.sensor_groups if len(g) > 1]
        best_x, best_e = np.zeros(n), np.inf
        for restart in range(self.restarts):
            x = np.zeros(n)
            field = linear.copy()
            e = 0.0
            # はじめに各センサーへ 1 つずつ変数を置く（付け替えの動きが使えるように）
            for g in groups:
                i = int(rng.choice(g))
                if x[i] == 0:
                    e += field[i]
                    x[i] = 1
                    field += W[i]
            for temperature in np.geomspace(t_hot, self.t_cold, self.sweeps):
                # 動き 1: 全変数をばらばらの順に 1 回ずつ試す
                for i, u in zip(rng.permutation(n), rng.random(n)):
                    change = field[i] * (1 - 2 * x[i])
                    if change <= 0 or u < np.exp(-change / temperature):
                        sign = 1 - 2 * x[i]
                        x[i] += sign
                        field += sign * W[i]
                        e += change
                # 動き 2: センサーごとに付け替えを 1 回試す
                for g in groups:
                    on = g[x[g] == 1]
                    if len(on) != 1:
                        continue
                    i = on[0]
                    j = int(rng.choice(g))
                    if j == i:
                        continue
                    change = -field[i] + field[j] - W[i, j]   # i を 0 に、j を 1 にしたときの変化
                    if change <= 0 or rng.random() < np.exp(-change / temperature):
                        x[i], x[j] = 0, 1
                        field += W[j] - W[i]
                        e += change
                if e < best_e - 1e-12:          # 途中でいちばん低かった状態を覚えておく
                    best_x, best_e = x.copy(), e
        return best_x.astype(int), dict(solver=self.name, energy=float(best_e + prob.const),
                                        seconds=round(time.time() - t_start, 3), sweeps=self.sweeps,
                                        restarts=self.restarts)


def make_amplify_client(token, time_limit_ms=1000):
    """Amplify AE のクライアントを作る. token は .env（環境変数）から読んだものを渡す."""
    import amplify
    client = amplify.AmplifyAEClient()
    client.token = token
    client.parameters.time_limit_ms = timedelta(milliseconds=time_limit_ms)
    return client
