"""二段階目をサンプルの地震で 1 回だけ Amplify で解き、未回収と決まり違反の数を出す（Amplify の確認用）.

sensor_app フォルダで:  python3 tools/check_phase2.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient  # noqa: E402

from server.main import app  # noqa: E402

client = TestClient(app)
sample = client.get('/api/map').json()
placed = {p['id'] for p in client.get('/api/phase1/exact').json()['placed']}
job = client.post('/api/phase2/solve', json=dict(closed=sample['sample']['closed'], bases=sample['sample']['bases'],
                                                 down=[s for s in sample['sample']['down'] if s in placed])).json()
while (answer := client.get(f"/api/phase2/jobs/{job['job_id']}").json())['status'] == 'running':
    time.sleep(2)
if answer['status'] != 'done':
    print('エラー:', answer['error'])
else:
    r = answer['result']
    print(f"ソルバー {r['solver']}・最後の回復 {r['makespan_recovery']} 分・回復 {len(r['recover_time'])}/{len(r['targets'])} 基・"
          f"未回収 {len(r['missing'])} 基・決まり違反 {len(r['violations'])} 件・計算 {r['seconds']} 秒")
    for text in r['violations']:
        print('  違反:', text)
