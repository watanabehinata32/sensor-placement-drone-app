# Amplify AE の代わりに手元で解く「カスタムクライアント」
# トークンがないときやネットにつながらないときでも、Amplify のモデル（VariableGenerator で作ったもの）を
# そのまま amplify.solve に渡して動かせるようにするための部品です。
# 一段階目のページの「手元のテスト用ソルバー」と、二段階目の確認用テストで使います。
import time
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
from amplify import AcceptableDegrees, Model

from .solvers import LocalAnnealer, QuboProblem, TabuSearch


class LocalResult:
    def __init__(self, values, seconds):
        self._solutions = [(values, timedelta(seconds=seconds))]
        self._response_time = timedelta(seconds=seconds)
        self._execution_time = timedelta(seconds=seconds)


class LocalSAClient:
    """AmplifyAEClient の代わりに手元で解く（確認用）"""

    def __init__(self, sweeps=300, method='sa'):
        self.parameters = SimpleNamespace(time_limit_ms=timedelta(seconds=1))
        self.token = ''
        self.version = 'local-sa'
        self.acceptable_degrees = AcceptableDegrees(objective={'Binary': 'Quadratic'})
        self.sweeps = sweeps
        self.method = method  # 'sa' なら焼きなまし法、'tabu' ならタブーサーチで解く

    def solve(self, objective, constraints, dry_run=False):
        if dry_run:
            return None
        poly = Model(objective, constraints).to_unconstrained_poly() if constraints else objective
        n = max(v.id for v in poly.variables) + 1
        h = np.zeros(n)
        pairs = {}
        const = 0.0
        for key, c in poly.as_dict().items():
            if len(key) == 0:
                const += c
            elif len(key) == 1:
                h[key[0]] += c
            else:
                i, j = sorted(key)
                pairs[(i, j)] = pairs.get((i, j), 0.0) + c
        prob = QuboProblem(n=n, linear=h, quadratic=pairs, const=const, meta=dict(A=float(np.abs(h).max() if n else 1.0)))
        t = time.perf_counter()
        if self.method == 'tabu':
            x, _ = TabuSearch(iters_per_var=40, restarts=4).solve(prob)
        else:
            x, _ = LocalAnnealer(sweeps=self.sweeps, restarts=2).solve(prob)
        return LocalResult(x.astype(float).tolist(), time.perf_counter() - t)
