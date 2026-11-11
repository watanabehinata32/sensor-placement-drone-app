// センサー配置と災害時ドローン巡回: 画面の動き
//
// しくみ:
//   - いまの状態（配置・通行止め・通信断・結果）は、このファイルの state にだけ持つ（サーバーは覚えない）。
//   - 計算が必要なときは、サーバーの窓口（/api/...）に条件を全部送って答えをもらう。
//   - 地図は 1 つで、① 配置 / ② 被害 / ③ 結果 のどれを見ているか（state.view）で描くものを変える。
//   - 座標は県の資料と同じメートル（x: 東向き, y: 北向き）。Leaflet には [y, x] の順で渡す。

// ============================================================
// 状態
// ============================================================
const state = {
  data: null,              // /api/map の答え（道路・市町村・候補地点・色など）
  placement: null,         // いま使っている配置（/api/phase1/... の答え）
  closed: new Set(),       // 通行止めの区間 ID
  down: new Set(),         // 通信断のセンサー ID
  unreachable: new Set(),  // 通信断のうち、ドローンで届かない（対応不能）センサー
  history: [],             // 「1つ戻す」用に、変える前の {closed, down} を積む
  epicenter: null,         // 震源 [x, y]
  damageSeed: 0,           // 震源から被害をつくるときの乱数の種（押すたびに変える）
  result: null,            // 二段階目の結果
  resultKey: '',           // 結果を出したときの条件（被害を変えたら「計算し直して」と出すため）
  view: 'p1',              // いま見ている段: 'p1' 配置 / 'p2' 被害 / 'p3' 結果
};

// 車ごとの色（結果の地図と時刻表）
const VEHICLE_COLORS = ['#e8590c', '#1971c2', '#2f9e44', '#9c36b5', '#f08c00', '#0c8599', '#c2255c', '#5c940d',
                        '#495057', '#364fc7'];
// 一段階目のセンサーの種類ごとの色（種類の並び順）
const TYPE_COLORS = ['#e8590c', '#1971c2', '#2f9e44'];

const $ = (id) => document.getElementById(id);
const toLatLng = (x, y) => [y, x];      // メートル座標 → Leaflet の [縦, 横]

// サーバーの窓口を呼ぶ（body を渡すと POST、なければ GET）
async function api(path, body) {
  const options = body === undefined ? {} :
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const response = await fetch(path, options);
  const json = await response.json();
  if (!response.ok) throw new Error(json.detail || `エラー（${response.status}）`);
  return json;
}

// ============================================================
// 地図の準備
// ============================================================
// CRS.Simple: 緯度経度ではなく、座標をそのまま平面として描く（架空の県なので実地図は敷かない）
const map = L.map('map', { crs: L.CRS.Simple, minZoom: -10, maxZoom: 0, zoomSnap: 0.25,
                           renderer: L.canvas({ tolerance: 6 }) });   // tolerance: 細い道路もクリックしやすくする
const layers = {
  area: L.layerGroup().addTo(map),      // 県と市町村
  roads: L.layerGroup().addTo(map),     // 道路
  closed: L.layerGroup().addTo(map),    // 通行止めの印
  result: L.layerGroup().addTo(map),    // 結果のルート
  sensors: L.layerGroup().addTo(map),   // センサー
  bases: L.layerGroup().addTo(map),     // ドローン拠点
  epicenter: L.layerGroup().addTo(map), // 震源
};
const roadById = {};                    // 区間 ID → 道路のデータ
const roadLines = [];                   // 描いた道路の線と、その本来の色・太さ（③ で灰色にして戻すため）

function drawArea() {
  const areas = state.data.style.areas;
  const outline = state.data.boundary.map(([x, y]) => toLatLng(x, y));
  L.polygon(outline, { color: areas.prefecture_outline.color, weight: areas.prefecture_outline.width_px,
                       fillColor: areas.prefecture_fill.color, fillOpacity: 1, interactive: false }).addTo(layers.area);
  for (const muni of state.data.municipalities) {
    const polygon = L.polygon(muni.polygon.map(([x, y]) => toLatLng(x, y)),
      { color: areas.municipality_border.color, weight: areas.municipality_border.width_px,
        dashArray: areas.municipality_border.dash.join(' '), fill: false, interactive: false }).addTo(layers.area);
    // 市町村の名前を真ん中あたりに出す
    L.marker(polygon.getBounds().getCenter(), { interactive: false,
      icon: L.divIcon({ className: 'mark', html: `<div style="color:#8a958f;font-size:12px;white-space:nowrap;
        transform:translate(-50%,-50%)">${muni.name}</div>` }) }).addTo(layers.area);
  }
  map.fitBounds(L.polygon(outline).getBounds(), { padding: [10, 10] });
}

function drawRoads() {
  const roadStyle = state.data.style.roads;
  // 資料と同じく、細い道から先に描いて太い道（高速など）を上に重ねる
  const roads = [...state.data.roads].sort((a, b) =>
    (roadStyle[a.cls]?.draw_order ?? 0) - (roadStyle[b.cls]?.draw_order ?? 0));
  for (const road of roads) {
    roadById[road.id] = road;
    const look = roadStyle[road.cls] || { color: '#999', width_px: 1.5 };
    const line = L.polyline(road.coords.map(([x, y]) => toLatLng(x, y)),
                            { color: look.color, weight: look.width_px, opacity: 0.9 });
    line.bindTooltip(`${road.id} ${road.name || '名前なし'}（${road.cls_ja}）`, { sticky: true, className: 'road-name' });
    line.on('click', () => onRoadClick(road.id));
    line.addTo(layers.roads);
    roadLines.push({ line, look });
  }
}

function drawBases() {
  // ドローン拠点: 紫の丸に白い建物の形（拠点）を入れた印を、拠点のちょうど真上に置く。
  // 名前は文字の周りを白くふちどって地図になじませる（箱で囲まない）。どの段でも、いちばん上に重ねる
  layers.bases.clearLayers();
  const look = state.data.style.markers.drone_base;
  const building = `<svg width="14" height="14" viewBox="0 0 14 14" style="display:block;margin:5px auto 0">
    <path d="M7 1.5 L12.5 6 V12.5 H8.6 V9 H5.4 V12.5 H1.5 V6 Z" fill="#fff"/></svg>`;
  for (const f of state.data.facilities.filter((f) => f.kind === 'drone_base')) {
    const input = $(`base-${f.facility_id}`);
    const count = input ? +input.value : f.dock_vehicles;
    L.marker(toLatLng(f.x, f.y), { interactive: false, zIndexOffset: 5000, icon: L.divIcon({ className: 'mark',
      iconSize: [24, 24], iconAnchor: [12, 12],
      html: `<div style="position:relative;width:24px;height:24px">
        <div style="width:24px;height:24px;border-radius:50%;background:${look.color};border:2px solid #fff;
          box-sizing:border-box;box-shadow:0 1px 4px rgba(0,0,0,0.45)">${building}</div>
        <div class="base-label" style="color:${look.text_color}">${f.name}<span>${count}台</span></div></div>` }) })
      .addTo(layers.bases);
  }
}

// 三角の印（センサー）. 色と大きさを変えて、通信正常・通信断・回復済みを描き分ける
function triangleIcon(color, size, ring, label) {
  const ringHtml = ring ? `<circle cx="12" cy="12" r="11" fill="none" stroke="${ring}" stroke-width="2.5"/>` : '';
  const labelHtml = label ? `<div style="position:absolute;left:14px;top:-16px;font-size:11px;white-space:nowrap;
    background:rgba(255,255,255,0.85);padding:0 2px">${label}</div>` : '';
  return L.divIcon({ className: 'mark', iconSize: [24, 24], iconAnchor: [12, 12],
    html: `<div style="position:relative"><svg width="24" height="24">${ringHtml}
      <polygon points="12,${12 - size * 0.6} ${12 - size * 0.55},${12 + size * 0.45} ${12 + size * 0.55},${12 + size * 0.45}"
      fill="${color}" stroke="#fff" stroke-width="1"/></svg>${labelHtml}</div>` });
}

// ============================================================
// センサーと通行止めを描く（見ている段で描き方を変える）
// ============================================================
function placedTypeOf() {
  // 地点 ID → 置いたセンサーの種類の名前
  const types = {};
  for (const p of state.placement?.placed || []) types[p.id] = p.type_name;
  return types;
}

function drawSensors() {
  layers.sensors.clearLayers();
  anim.sensorMarkers = {};
  const placed = placedTypeOf();
  const markers = state.data.style.markers;
  const typeNames = state.data.sensor_types.map((t) => t.name);
  for (const c of state.data.candidates) {
    const at = toLatLng(c.x, c.y);
    const typeName = placed[c.id];
    if (!typeName) {
      // 置いていない候補地点は小さな灰色の点
      L.circleMarker(at, { radius: 2, color: '#bbb', fillOpacity: 1, interactive: false }).addTo(layers.sensors);
      continue;
    }
    if (state.view === 'p1') {
      // ① 配置: 種類ごとの色の丸と、カバーする範囲の円
      const k = typeNames.indexOf(typeName);
      const color = TYPE_COLORS[k % TYPE_COLORS.length];
      L.circle(at, { radius: +$(`st-radius-${k}`).value * 1000, color, weight: 1, fillOpacity: 0.08,
                     interactive: false }).addTo(layers.sensors);
      L.circleMarker(at, { radius: 5, color: '#fff', weight: 1, fillColor: color, fillOpacity: 1 })
        .bindTooltip(`${c.id} ${typeName}`).addTo(layers.sensors);
      continue;
    }
    // ② 被害 ・ ③ 結果: 三角の印
    const isDown = state.down.has(c.id);
    const unreachable = state.unreachable.has(c.id);
    let color = markers.sensor_normal.color, size = markers.sensor_normal.size_px, label = '';
    if (isDown) {
      color = markers.sensor_down.color;
      size = markers.sensor_down.size_px;
      const t = state.view === 'p3' && state.result ? state.result.recover_time[c.id] : undefined;
      if (t !== undefined) { color = '#2b8a3e'; label = `${c.id} ${t.toFixed(0)}分`; }
      else if (unreachable) label = `${c.id} 届かない`;
    }
    if (state.view === 'p3' && isDown && !unreachable && state.result) {
      // ③ で回復の対象のセンサーは、アニメーションの時刻に合わせて赤→緑に変える（setTime で描き直す）
      const marker = L.marker(at, { icon: triangleIcon(markers.sensor_down.color, size, null, '') });
      marker.bindTooltip(`${c.id} ${typeName}（通信断）`);
      marker.addTo(layers.sensors);
      anim.sensorMarkers[c.id] = { marker, recovered: null };
      continue;
    }
    const marker = L.marker(at, { icon: triangleIcon(color, size, unreachable ? markers.unreachable.color : null, label) });
    marker.bindTooltip(`${c.id} ${typeName}${isDown ? '（通信断' + (unreachable ? '・ドローンで届かない' : '') + '）' : ''}`);
    marker.on('click', () => onSensorClick(c.id));
    marker.addTo(layers.sensors);
  }
}

function drawClosed() {
  layers.closed.clearLayers();
  if (state.view === 'p1') return;
  const look = state.data.style.markers.closed_segment;
  for (const id of state.closed) {
    const road = roadById[id];
    if (!road) continue;
    const coords = road.coords.map(([x, y]) => toLatLng(x, y));
    L.polyline(coords, { color: look.color, weight: 4, dashArray: '4 4', interactive: false }).addTo(layers.closed);
    const middle = coords[Math.floor(coords.length / 2)];
    L.marker(middle, { interactive: false, icon: L.divIcon({ className: 'mark', iconSize: [0, 0],
      html: `<div style="transform:translate(-50%,-55%);font-size:${look.size_px + 6}px;font-weight:bold;
        color:${look.color};text-shadow:0 0 2px #fff">×</div>` }) }).addTo(layers.closed);
  }
}

function drawEpicenter() {
  layers.epicenter.clearLayers();
  if (state.view !== 'p2' || !state.epicenter) return;
  L.marker(toLatLng(...state.epicenter), { interactive: false, icon: L.divIcon({ className: 'mark', iconSize: [0, 0],
    html: '<div style="transform:translate(-50%,-50%);font-size:20px">💥</div>' }) }).addTo(layers.epicenter);
}

function drawLegend() {
  const style = state.data.style;
  const rows = [];
  for (const cls of state.view === 'p3' ? [] : ['motorway', 'trunk', 'primary', 'secondary']) {
    const r = style.roads[cls];
    rows.push(`<div><span class="legend-line" style="border-top:${Math.max(r.width_px, 2)}px solid ${r.color}"></span>${r.label}</div>`);
  }
  rows.push(`<div><span class="legend-mark" style="color:${style.markers.drone_base.color}">●</span>ドローン拠点</div>`);
  if (state.view === 'p1') {
    state.data.sensor_types.forEach((t, k) =>
      rows.push(`<div><span class="legend-mark" style="color:${TYPE_COLORS[k]}">●</span>${t.name}</div>`));
  } else {
    rows.push(`<div><span class="legend-mark" style="color:${style.markers.sensor_down.color}">▲</span>通信断センサー</div>`);
    rows.push(`<div><span class="legend-mark" style="color:${style.markers.sensor_normal.color}">▲</span>通信正常</div>`);
    rows.push(`<div><span class="legend-mark" style="color:${style.markers.unreachable.color}">◯</span>対応不能（届かない）</div>`);
    rows.push(`<div><span class="legend-mark"><b>×</b></span>通行止め区間</div>`);
    if (state.view === 'p3') rows.push('<div><span class="legend-mark" style="color:#2b8a3e">▲</span>回復（分）</div>');
  }
  $('legend').innerHTML = rows.join('');
}

// ③ 結果では、車のルートとドローンの飛行が目立つよう、道路を種類によらず薄い灰色の細い線にする
function styleRoads() {
  const plain = state.view === 'p3';
  for (const { line, look } of roadLines) {
    line.setStyle(plain ? { color: '#d0d4d2', weight: 1.2, opacity: 1 }
                        : { color: look.color, weight: look.width_px, opacity: 0.9 });
  }
}

function redraw() {
  styleRoads();
  drawBases();
  drawSensors();
  drawClosed();
  drawEpicenter();
  drawResultMap();
  drawLegend();
}

// ============================================================
// 段の切り替え
// ============================================================
function setView(view) {
  state.view = view;
  if (view !== 'p3') pause();
  const showPlayer = view === 'p3' && !!state.result;
  $('player').hidden = !showPlayer;
  $('map-area').classList.toggle('with-player', showPlayer);   // 下端に操作パネルを出す分、地図を縮める
  map.invalidateSize();
  for (const [id, v] of [['step1', 'p1'], ['step2', 'p2'], ['step3', 'p3']]) {
    $(id).classList.toggle('active', v === view);
    if (v === view) $(id).scrollIntoView({ behavior: 'smooth', block: 'start' });   // 見ている段を左の欄の上に出す
  }
  redraw();
}
document.querySelectorAll('.step h2').forEach((h) => h.addEventListener('click', () => {
  if (h.dataset.view !== 'p1' && !state.placement) return;     // 配置がないうちは ② ③ に進めない
  setView(h.dataset.view);
}));

// ============================================================
// ① 配置
// ============================================================
function updateBanner() {
  const p = state.placement;
  $('placement-banner').textContent = p
    ? `使っている配置: ${p.source}（${p.count} 台・総コスト ${p.total_cost.toLocaleString()}）`
    : '配置: まだ選んでいません';
}

// 配置を新しくしたとき. 設置していないセンサーが通信断に入っていたら外して知らせる（5-5）
function setPlacement(summary) {
  const first = !state.placement;
  state.placement = summary;
  if (first) setSampleDamage();      // 初めて配置が決まったら、サンプルの地震の被害から始める
  const placedIds = new Set(summary.placed.map((p) => p.id));
  const dropped = [...state.down].filter((id) => !placedIds.has(id));
  dropped.forEach((id) => state.down.delete(id));
  $('p2-note').textContent = dropped.length
    ? `配置が変わったので、設置していないセンサー ${dropped.join('、')} を通信断から外しました。` : '';
  updateBanner();
  renderPhase1Result();
  $('btn-to-p2').hidden = false;
  onDamageChanged(false);
}

function renderPhase1Result() {
  const p = state.placement;
  const rows = p.rows.map((r) => `<tr><td>${r.name}</td><td>${r.count}</td>
    <td>${r.alpha.toFixed(0)} / ${r.need_alpha.toFixed(0)}</td><td>${r.beta.toFixed(0)} / ${r.need_beta.toFixed(0)}</td>
    <td>${(Math.floor(r.coverage * 1000 + 1e-6) / 10).toFixed(1)}% / ${(r.need_coverage * 100).toFixed(0)}%</td>
    <td class="${r.ok ? 'ok' : 'ng'}">${r.ok ? '✓' : '✗'}</td></tr>`).join('');
  $('p1-result').innerHTML = `
    <div class="metrics">
      <div class="metric"><div class="value">${p.total_cost.toLocaleString()}</div><div class="label">総コスト</div></div>
      <div class="metric"><div class="value">${p.count} 台</div><div class="label">設置台数</div></div>
      <div class="metric"><div class="value">${p.rows.filter((r) => r.ok).length} / ${p.rows.length}</div>
        <div class="label">要求を満たした市町村</div></div>
    </div>
    ${p.multi ? `<p class="note">1つの地点に2台以上置いた所が ${p.multi} か所あります（λ_one を大きくしてみてください）</p>` : ''}
    <table><tr><th>市町村</th><th>台数</th><th>α</th><th>β</th><th>カバー率</th><th>判定</th></tr>${rows}</table>`;
}

// 「詳しい設定」のセンサーの性能表と市町村の必要値表（数字を書き換えると次の計算から使う）
function numberCell(id, value, step) {
  return `<td><input type="number" id="${id}" value="${value}" step="${step}"></td>`;
}
function fillPhase1Tables() {
  // 性能表は「行 = コスト・半径・α・β、列 = センサーの種類」にして横に広がらないようにする。
  // 列の見出しは名前の前半（例: 広域型）だけにし、正式名はマウスを乗せると出る
  const types = state.data.sensor_types;
  const shortName = (name) => (name.includes('型') ? name.split('型')[0] + '型' : name.slice(0, 3) + '型');   // 高精度局所モニター → 高精度型
  const rowsSpec = [['コスト', 'cost', 10], ['半径(km)', 'radius_km', 0.5], ['α値', 'alpha', 1], ['β値', 'beta', 1]];
  const ids = { cost: 'st-cost', radius_km: 'st-radius', alpha: 'st-alpha', beta: 'st-beta' };
  $('p1-sensor-table').innerHTML = `<tr><th></th>${types.map((t) => `<th title="${t.name}">${shortName(t.name)}</th>`).join('')}</tr>` +
    rowsSpec.map(([label, key, step]) => `<tr><td>${label}</td>${types.map((t, k) =>
      numberCell(`${ids[key]}-${k}`, t[key], step)).join('')}</tr>`).join('');
  $('p1-muni-table').innerHTML = '<tr><th>市町村</th><th>候補</th><th>α必要量</th><th>β必要量</th><th>カバー率下限</th></tr>' +
    state.data.muni_needs.map((m, i) => `<tr><td>${m.name}</td><td>${m.n_candidates}</td>
      ${numberCell(`mn-alpha-${i}`, m.need_alpha, 1)}${numberCell(`mn-beta-${i}`, m.need_beta, 1)}
      ${numberCell(`mn-cov-${i}`, m.need_coverage, 0.05)}</tr>`).join('');
}
function sensorTable() {
  return state.data.sensor_types.map((t, k) => ({ cost: +$(`st-cost-${k}`).value, radius_km: +$(`st-radius-${k}`).value,
    alpha: +$(`st-alpha-${k}`).value, beta: +$(`st-beta-${k}`).value }));
}
function muniTable() {
  return state.data.muni_needs.map((m, i) => ({ need_alpha: +$(`mn-alpha-${i}`).value,
    need_beta: +$(`mn-beta-${i}`).value, need_coverage: +$(`mn-cov-${i}`).value }));
}
$('btn-p1-reset').onclick = () => fillPhase1Tables();

function phase1Settings() {
  return { grid: +$('p1-grid').value, lam_one: +$('p1-lam-one').value, lam_ab: +$('p1-lam-ab').value,
           lam_cov: +$('p1-lam-cov').value, cross: $('p1-cross').checked, time_limit_ms: +$('p1-time').value,
           sensors: sensorTable(), munis: muniTable() };
}

async function runWithButton(button, statusId, message, work) {
  // ボタンを押せなくして、終わるまで「計算中」を出す
  button.disabled = true;
  $(statusId).className = 'status';
  $(statusId).textContent = message;
  try {
    await work();
    $(statusId).textContent = '';
  } catch (error) {
    $(statusId).className = 'status error';
    $(statusId).textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

$('btn-p1-solve').onclick = () => runWithButton($('btn-p1-solve'), 'p1-status', 'QUBO を解いています…', async () => {
  const summary = await api('/api/phase1/solve', phase1Settings());
  setPlacement(summary);
  $('p1-status').textContent = `計算おわり（${summary.seconds} 秒）`;
});
// 厳密解の配置を読み、いまの表の値でコストと達成状況を数え直す
async function loadExact() {
  const exact = await api('/api/phase1/exact');
  const s = phase1Settings();
  const summary = await api('/api/phase1/check', { placed: exact.placed, grid: s.grid, cross: s.cross,
                                                   sensors: s.sensors, munis: s.munis });
  summary.source = exact.source;
  setPlacement(summary);
}
$('btn-p1-exact').onclick = () => runWithButton($('btn-p1-exact'), 'p1-status', '読み込んでいます…', loadExact);

// 配置をファイルに保存（ブラウザのダウンロード）
$('btn-p1-save').onclick = () => {
  if (!state.placement) return;
  const p = state.placement;
  const text = JSON.stringify({ source: p.source, total_cost: p.total_cost, placed: p.placed }, null, 1);
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  link.download = 'placement.json';
  link.click();
};

// 保存した配置を読む（① の欄と、最初の入口の両方から使う）
async function loadPlacementFile(file) {
  const saved = JSON.parse(await file.text());
  const s = phase1Settings();
  const summary = await api('/api/phase1/check', { placed: saved.placed, grid: s.grid, cross: s.cross,
                                                   sensors: s.sensors, munis: s.munis });
  summary.source = `保存した配置（${file.name}）`;
  setPlacement(summary);
}
$('file-p1-load').onchange = (e) => runWithButton($('btn-p1-solve'), 'p1-status', '読み込んでいます…',
  () => loadPlacementFile(e.target.files[0]));
$('btn-to-p2').onclick = () => setView('p2');

// ============================================================
// ② 被害
// ============================================================
const clickMode = () => document.querySelector('input[name=click-mode]:checked').value;

function remember() {
  state.history.push({ closed: new Set(state.closed), down: new Set(state.down) });
  if (state.history.length > 30) state.history.shift();
}

function onRoadClick(id) {
  if (state.view !== 'p2' || clickMode() !== 'road') return;
  remember();
  state.closed.has(id) ? state.closed.delete(id) : state.closed.add(id);
  onDamageChanged();
}

function onSensorClick(id) {
  if (state.view !== 'p2' || clickMode() !== 'sensor') return;
  remember();
  state.down.has(id) ? state.down.delete(id) : state.down.add(id);
  onDamageChanged();
}

// 震源を置いたら、震源に近いものほど被害を受けやすいように、まとめて被害をつくる
map.on('click', async (e) => {
  if (state.view !== 'p2' || clickMode() !== 'epicenter') return;
  state.epicenter = [e.latlng.lng, e.latlng.lat];
  state.damageSeed += 1;
  const answer = await api('/api/damage', {
    placed_ids: state.placement.placed.map((p) => p.id), epicenter: state.epicenter, seed: state.damageSeed,
    sensor_radius_km: +$('d-s-radius').value, sensor_strength: +$('d-s-strength').value,
    road_radius_km: +$('d-r-radius').value, road_strength: +$('d-r-strength').value });
  remember();
  state.closed = new Set(answer.closed);
  state.down = new Set(answer.down);
  $('p2-note').textContent = `震源から被害をつくりました（もう一度クリックすると別の被害になります）`;
  onDamageChanged();
});

$('btn-sample').onclick = () => { remember(); setSampleDamage(); onDamageChanged(); };
$('btn-clear').onclick = () => { remember(); state.closed.clear(); state.down.clear(); onDamageChanged(); };
$('btn-undo').onclick = () => {
  const last = state.history.pop();
  if (!last) return;
  state.closed = last.closed;
  state.down = last.down;
  onDamageChanged();
};

function setSampleDamage() {
  // サンプルの地震（資料の図から読み取ったもの）. 通信断は設置したセンサーの中だけにする
  const placedIds = new Set((state.placement?.placed || []).map((p) => p.id));
  state.closed = new Set(state.data.sample.closed);
  state.down = new Set(state.data.sample.down.filter((id) => placedIds.has(id)));
  const e = state.data.sample.epicenter;
  state.epicenter = e ? [e.x, e.y] : null;
}

function bases() {
  // 拠点ごとの車の台数（詳しい設定の入力欄から）
  return state.data.sample.bases.map((b) => ({ ...b, vehicles: +$(`base-${b.base_id}`).value }));
}

function damageKey() {
  return JSON.stringify([[...state.closed].sort(), [...state.down].sort(), bases()]);
}

// 被害が変わったら: 対応不能の判定をサーバーに頼み、地図と数を描き直す
let preprocessTimer = null;
function onDamageChanged(draw = true) {
  if (draw) redraw();
  updateCounts();
  clearTimeout(preprocessTimer);
  preprocessTimer = setTimeout(async () => {     // 続けてクリックしたときは最後の 1 回だけ頼む
    if (!state.placement) return;
    const answer = await api('/api/preprocess', { closed: [...state.closed], down: [...state.down], bases: bases() });
    state.unreachable = new Set(answer.unreachable);
    updateCounts();
    redraw();
  }, 250);
  renderStaleNote();
}

function updateCounts() {
  const unreachable = [...state.down].filter((id) => state.unreachable.has(id)).length;
  $('damage-counts').textContent =
    `通信断 ${state.down.size} 基（うちドローンで届かない ${unreachable} 基）・通行止め ${state.closed.size} 区間`;
}

function phase2Request() {
  return {
    closed: [...state.closed], down: [...state.down], bases: bases(),
    end_mode: document.querySelector('input[name=end-mode]:checked').value,
    charge_min: +$('f-charge').value, flights_builtin: +$('f-builtin').value,
    penalty: +$('q-penalty').value, time_limit_ms: +$('q-time').value,
  };
}

// 二段階目を解く: 受付番号をもらい、1 秒ごとに進み具合を問い合わせる
$('btn-p2-solve').onclick = () => runWithButton($('btn-p2-solve'), 'p2-status', 'QUBO を解いています…', async () => {
  const request = phase2Request();
  const key = damageKey();
  const { job_id } = await api('/api/phase2/solve', request);
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const job = await api(`/api/phase2/jobs/${job_id}`);
    $('p2-status').textContent = `計算しています… ${job.stage}`;
    if (job.status === 'error') throw new Error(job.error);
    if (job.status === 'done') {
      state.result = job.result;
      state.result.endMode = request.end_mode;
      state.resultKey = key;
      anim.time = 0;
      renderPhase3();
      setView('p3');
      break;
    }
  }
});

// ============================================================
// ③ 結果
// ============================================================
// ============================================================
// ③ のアニメーション（時刻スライダーで、車・ドローン・センサーの様子を再生する）
// ============================================================
// 時刻 t（分）のとき: 車は時刻表の「走行」の間だけ道路に沿って動き、ドローンは離陸点 → センサー（3 分滞空）→ … →
// 離陸点と直線で飛ぶ。どちらも時刻表（events）の時刻から、その間の位置を割合で求める（線形補間）。
const anim = {
  time: 0, end: 1, playing: false, lastFrame: null,
  vehicles: [],          // 車ごとの動きの予定（buildPlans で作る）
  sensorMarkers: {},     // 回復の対象のセンサーの印
  layer: L.layerGroup().addTo(map),
};

// 折れ線の各点に「出発から何秒で着くか」を持たせる（サーバーが道路の速さから求めた値 = 3 つ目の数）。
// 割合はこの秒数で測るので、高速道路では速く、一般道では遅く動く（同じ道なら同じ速さになる）
function withDistances(points) {
  const coords = points.map((p) => [p[0], p[1]]);
  const along = points.map((p) => p[2] ?? 0);
  return { coords, along, length: along[along.length - 1] || 0 };
}

// 折れ線の始めから割合 r（0〜1）の位置と、そこまでの点の列
function pointAlong(line, r) {
  if (line.coords.length === 0) return { at: null, done: [] };
  const target = Math.min(Math.max(r, 0), 1) * line.length;
  let i = 1;
  while (i < line.coords.length - 1 && line.along[i] < target) i++;
  const piece = line.along[i] - line.along[i - 1] || 1;
  const u = Math.min(Math.max((target - line.along[i - 1]) / piece, 0), 1);
  const [x1, y1] = line.coords[i - 1], [x2, y2] = line.coords[Math.min(i, line.coords.length - 1)];
  const at = [x1 + (x2 - x1) * u, y1 + (y2 - y1) * u];
  return { at, done: [...line.coords.slice(0, i), at] };
}

// 時刻表から、車ごとに「走行の区切り」と「飛行の区切り」を時刻つきで作る
function buildPlans(result) {
  const sensorXY = {};
  for (const c of state.data.candidates) sensorXY[c.id] = [c.x, c.y];
  const hover = result.hover_min;
  return result.vehicles.map((v, k) => {
    const drives = [], flights = [];
    let driveStart = null, legIndex = 0, flight = null;
    for (const e of v.events) {
      if (e.kind === '出発' || e.kind === '着陸') driveStart = e.t;
      if ((e.kind === '到着' || e.kind === '帰着') && driveStart !== null) {
        drives.push({ t0: driveStart, t1: e.t, line: withDistances(v.legs[legIndex] || []) });
        legIndex += 1;
        driveStart = null;
      }
      if (e.kind === '離陸') {
        const launch = v.flights[flights.length].launch;
        flight = { t0: e.t, points: [{ t: e.t, xy: launch }], sensors: [] };
      }
      if (e.kind === '回復' && flight) {
        // センサーの上に着くのは回復の 3 分前（そこから 3 分滞空して回復）
        flight.points.push({ t: e.t - hover, xy: sensorXY[e.where] }, { t: e.t, xy: sensorXY[e.where] });
        flight.sensors.push({ id: e.where, t0: e.t - hover, t1: e.t });
      }
      if (e.kind === '着陸' && flight) {
        flight.points.push({ t: e.t, xy: flight.points[0].xy });
        flight.t1 = e.t;
        flights.push(flight);
        flight = null;
      }
    }
    return { vehicle: v, color: VEHICLE_COLORS[k % VEHICLE_COLORS.length], drives, flights };
  }).filter((plan) => plan.vehicle.events.length);
}

// 時刻 t の車の位置・走った道・状態
function vehicleAt(plan, t) {
  const v = plan.vehicle;
  let at = plan.drives.length ? plan.drives[0].line.coords[0] : null;
  const done = [];
  let status = '出発前';
  for (const d of plan.drives) {
    if (t >= d.t1) {                       // 走り終えた区切り
      done.push(...d.line.coords);
      at = d.line.coords[d.line.coords.length - 1] || at;
      status = '待機';
    } else if (t >= d.t0) {                // いま走っている区切り
      const r = (t - d.t0) / Math.max(d.t1 - d.t0, 1e-9);
      const p = pointAlong(d.line, r);
      done.push(...p.done);
      at = p.at || at;
      status = '走行中';
      break;
    } else break;
  }
  const flying = plan.flights.find((f) => t >= f.t0 && t < f.t1);
  if (flying) status = 'ドローン飛行中';
  if (t >= v.end_return) status = '帰着';
  return { at, done, status, flying };
}

// 飛行中のドローンの位置と、滞空中のセンサー
function droneAt(flight, t) {
  const pts = flight.points;
  for (let i = 1; i < pts.length; i++) {
    if (t <= pts[i].t) {
      const span = Math.max(pts[i].t - pts[i - 1].t, 1e-9);
      const u = Math.min(Math.max((t - pts[i - 1].t) / span, 0), 1);
      const [x1, y1] = pts[i - 1].xy, [x2, y2] = pts[i].xy;
      const hovering = flight.sensors.find((s) => t >= s.t0 && t < s.t1);
      return { xy: [x1 + (x2 - x1) * u, y1 + (y2 - y1) * u], hovering: hovering?.id };
    }
  }
  return { xy: pts[pts.length - 1].xy, hovering: null };
}

const STATUS_COLOR = { '走行中': '#1971c2', 'ドローン飛行中': '#2b8a3e', '待機': '#868e96', '帰着': '#adb5bd',
                       '出発前': '#adb5bd' };

function setupAnimation() {
  anim.vehicles = buildPlans(state.result);
  anim.end = Math.max(...anim.vehicles.map((p) => p.vehicle.end_return), 1);
  $('player-slider').max = anim.end;
  $('player-slider').step = 0.1;
  setTime(Math.min(anim.time, anim.end));
}

// 時刻 t の様子を地図・状況表示・時刻表に描く
function setTime(t) {
  anim.time = t;
  anim.layer.clearLayers();
  const r = state.result;
  if (state.view !== 'p3' || !r) return;
  const markers = state.data.style.markers;
  const rows = [];
  for (const plan of anim.vehicles) {
    const now = vehicleAt(plan, t);
    if (now.done.length > 1) {
      L.polyline(now.done.map(([x, y]) => toLatLng(x, y)), { color: plan.color, weight: 5, opacity: 0.95,
                                                              interactive: false }).addTo(anim.layer);
    }
    if (now.at) {
      L.marker(toLatLng(...now.at), { interactive: false, zIndexOffset: 1000, icon: L.divIcon({ className: 'mark',
        iconSize: [16, 16], iconAnchor: [8, 8], html: `<div style="width:16px;height:16px;border-radius:50%;
        background:${plan.color};border:2px solid #fff;box-shadow:0 0 3px #000"></div>` }) }).addTo(anim.layer);
    }
    if (now.flying) {
      const drone = droneAt(now.flying, t);
      L.polyline([now.flying.points[0].xy, drone.xy].map(([x, y]) => toLatLng(x, y)),
                 { color: plan.color, weight: 2, dashArray: '3 4', interactive: false }).addTo(anim.layer);
      L.marker(toLatLng(...drone.xy), { interactive: false, zIndexOffset: 1100, icon: L.divIcon({ className: 'mark',
        iconSize: [0, 0], html: '<div style="transform:translate(-50%,-50%);font-size:16px">🛸</div>' }) })
        .addTo(anim.layer);
      if (drone.hovering) {      // 滞空中のセンサーの周りを光らせる
        L.marker(toLatLng(...drone.xy), { interactive: false, icon: L.divIcon({ className: 'mark', iconSize: [0, 0],
          html: '<div class="hover-glow"></div>' }) }).addTo(anim.layer);
      }
    }
    rows.push(`<div><span style="color:${plan.color}">●</span> ${plan.vehicle.name}：
      <span style="color:${STATUS_COLOR[now.status]}">${now.status}</span></div>`);
  }
  // センサーの色: 回復した時刻を過ぎたら緑、それまでは赤
  let recovered = 0;
  for (const [id, item] of Object.entries(anim.sensorMarkers)) {
    const done = r.recover_time[id] !== undefined && t >= r.recover_time[id];
    if (done) recovered += 1;
    if (item.recovered !== done) {
      item.marker.setIcon(triangleIcon(done ? '#2b8a3e' : markers.sensor_down.color, markers.sensor_down.size_px, null,
                                       done ? `${id} ${r.recover_time[id].toFixed(0)}分` : (r.missing.includes(id) ? `${id} ⚠ 未回収` : '')));
      item.recovered = done;
    }
  }
  $('player-status').innerHTML = `<b>${t.toFixed(1)} 分：回復 ${recovered} / ${r.targets.length} 基</b>${rows.join('')}`;
  $('player-slider').value = t;
  $('player-time').textContent = `${t.toFixed(1)} / ${anim.end.toFixed(0)} 分`;
  document.querySelectorAll('.gantt-cursor').forEach((c) => { c.style.left = `${(100 * t / anim.end).toFixed(2)}%`; });
}

// 再生: 1 秒ごとに「速さ」分だけ時刻を進める
function frame(now) {
  if (!anim.playing) return;
  if (anim.lastFrame !== null) {
    const speed = +$('player-speed').value;            // 1 秒あたり何分進めるか
    const next = anim.time + (now - anim.lastFrame) / 1000 * speed;
    if (next >= anim.end) { setTime(anim.end); pause(); return; }
    setTime(next);
  }
  anim.lastFrame = now;
  requestAnimationFrame(frame);
}
function play() {
  if (anim.time >= anim.end) anim.time = 0;            // 最後まで行っていたら最初から
  anim.playing = true;
  anim.lastFrame = null;
  $('player-play').textContent = '⏸';
  requestAnimationFrame(frame);
}
function pause() {
  anim.playing = false;
  $('player-play').textContent = '▶';
}
$('player-play').onclick = () => (anim.playing ? pause() : play());
$('player-slider').oninput = (e) => { pause(); setTime(+e.target.value); };

// 時刻表の横棒をクリックしたら、その時刻へ飛ぶ
$('p3-result').addEventListener('click', (e) => {
  const track = e.target.closest('.gantt-track');
  if (!track) return;
  const box = track.getBoundingClientRect();
  pause();
  setTime(Math.min(Math.max((e.clientX - box.left) / box.width, 0), 1) * anim.end);
});

function drawResultMap() {
  anim.layer.clearLayers();
  layers.result.clearLayers();
  if (state.view !== 'p3' || !state.result) return;
  const sensorXY = {};
  for (const c of state.data.candidates) sensorXY[c.id] = [c.x, c.y];
  state.result.vehicles.forEach((v, k) => {
    if (!v.flights.length) return;
    const color = VEHICLE_COLORS[k % VEHICLE_COLORS.length];
    // 車の走行ルート（道路に沿った線）. これから走る道は薄く描き、走り終えた道はアニメーションで濃く重ねる
    L.polyline(v.path.map(([x, y]) => toLatLng(x, y)), { color, weight: 4, opacity: 0.25, interactive: false })
      .addTo(layers.result);
    for (const f of v.flights) {
      // ドローンの飛行（離陸点 → センサー → … → 離陸点）は点線
      const points = [f.launch, ...f.sensors.map((s) => sensorXY[s]), f.launch].map(([x, y]) => toLatLng(x, y));
      L.polyline(points, { color, weight: 2, dashArray: '4 5', interactive: false }).addTo(layers.result);
      // 離陸点に「何回目の飛行か」の数字
      L.marker(toLatLng(...f.launch), { icon: L.divIcon({ className: 'mark', iconSize: [18, 18], iconAnchor: [9, 9],
        html: `<div style="width:18px;height:18px;border-radius:50%;background:${color};color:#fff;font-size:11px;
          text-align:center;line-height:18px;border:1px solid #fff">${f.order}</div>` }) })
        .bindTooltip(`${v.name} の ${f.order} 回目: ${f.sensors.join('・')}`).addTo(layers.result);
    }
  });
  setupAnimation();
}

// 時刻表のできごとの並び（前, 次）→ その間の帯の種類と色
const BAR_KIND = {
  '出発→到着': ['走行', '#adb5bd'], '着陸→到着': ['走行', '#adb5bd'], '着陸→帰着': ['走行（帰り）', '#dee2e6'],
  '準備→離陸': ['離陸の準備', '#a5d8ff'], '準備→充電待ち': ['離陸の準備', '#a5d8ff'],
  '充電待ち→離陸': ['充電待ち', '#ffd43b'], '離陸→回復': ['飛行（行き・滞空）', '#2b8a3e'],
  '回復→着陸': ['飛行（帰り）', '#8ce99a'], '回復→回復': ['飛行（次のセンサーへ）', '#69db7c'],
};

function ganttHtml(result) {
  const used = result.vehicles.filter((v) => v.events.length);
  const end = Math.max(...used.map((v) => v.end_return), 1);   // 横軸の長さ = いちばん遅い帰着
  const pct = (t) => `${(100 * t / end).toFixed(2)}%`;
  const rows = used.map((v) => {
    const k = result.vehicles.indexOf(v);
    const bars = [];
    for (let i = 0; i + 1 < v.events.length; i++) {
      const a = v.events[i], b = v.events[i + 1];
      const kind = BAR_KIND[`${a.kind}→${b.kind}`];
      if (!kind || b.t - a.t < 1e-6) continue;
      bars.push(`<div class="gantt-bar" title="${kind[0]} ${a.where}→${b.where} ${a.t.toFixed(1)}〜${b.t.toFixed(1)} 分"
        style="left:${pct(a.t)};width:${pct(b.t - a.t)};background:${kind[1]}"></div>`);
    }
    for (const e of v.events.filter((e) => e.kind === '回復')) {
      bars.push(`<div class="gantt-star" title="${e.where} 回復 ${e.t.toFixed(1)} 分" style="left:${pct(e.t)}">★</div>`);
    }
    return `<div class="gantt-row"><div class="gantt-name" style="color:${VEHICLE_COLORS[k % VEHICLE_COLORS.length]}">
      ${v.name}</div><div class="gantt-track">${bars.join('')}
      <div class="gantt-end" style="left:${pct(result.makespan_recovery)}"></div>
      <div class="gantt-cursor"></div></div></div>`;
  });
  const legend = [...new Map(Object.values(BAR_KIND).map(([n, c]) => [n, c])).entries()]
    .map(([n, c]) => `<span style="display:inline-block;width:10px;height:10px;background:${c}"></span> ${n}`).join('　');
  return `${rows.join('')}<p class="hint" style="font-size:11px">${legend}　<span style="color:#c92a2a">★</span> 回復
    　<span style="color:#c92a2a">┆</span> 最後の回復　（横軸 0〜${end.toFixed(0)} 分）</p>`;
}

function timetableHtml(result) {
  return result.vehicles.filter((v) => v.events.length).map((v) => `
    <details><summary>${v.name}（飛行 ${v.flights.length} 回・最後の回復 ${v.end_recovery.toFixed(1)} 分・帰着
      ${v.end_return.toFixed(1)} 分）</summary>
      <table><tr><th>時刻（分）</th><th>できごと</th><th>場所</th></tr>
      ${v.events.map((e) => `<tr><td>${e.t.toFixed(1)}</td><td>${e.kind}</td>
        <td>${e.where}${e.note ? '（' + e.note + '）' : ''}</td></tr>`).join('')}</table></details>`).join('');
}

function renderPhase3() {
  const r = state.result;
  const notes = [`ソルバー: ${r.solver}・計算 ${r.seconds} 秒`];
  const step2 = r.steps.step2;
  if (step2.used) {
    notes.push(`② 飛行の候補 ${step2.candidates} 本 → 選んだ飛行 ${step2.chosen} 本（離陸地点 ${step2.launch_points} か所）`);
  } else {
    // ② を飛ばした版：①の離陸地点を全部 ③ に渡している
    notes.push(`② は飛ばしました（飛行の候補 ${step2.candidates} 本・離陸地点 ${step2.launch_points} か所を全部 ③ で使用）`);
  }
  notes.push(`④ ルートの候補 ${r.steps.step4.n_routes} 本（変数 ${r.steps.step4.n_variables} 個）`);
  // 回収できなかったセンサー（QUBO の答えのまま。あとから付け足すことはしない）
  const missingWarning = r.missing.length ? `<p class="warning">⚠ 未回収 ${r.missing.length} 基：${r.missing.join('、')}</p>` : '';
  // 決まりを守れていない所（罰則が効いていない印）
  const violationWarning = r.violations.length ? `<p class="warning">⚠ 決まりを守れていない所があります：<br>
    ${r.violations.join('<br>')}<br>詳しい設定で「罰則の強さ」や「Amplify の制限時間」を上げて、もう一度計算してください。</p>` : '';
  const main = r.endMode === 'return' ? '全車帰着' : '最後の回復';
  $('p3-result').innerHTML = `
    <div id="stale-note" class="note"></div>
    ${missingWarning}${violationWarning}
    <div class="metrics">
      <div class="metric"><div class="value">${r.makespan_recovery.toFixed(1)} 分</div>
        <div class="label">最後のセンサーの回復${main === '最後の回復' ? '（目的）' : ''}</div></div>
      <div class="metric"><div class="value">${r.makespan_return.toFixed(1)} 分</div>
        <div class="label">全車が拠点に帰着${main === '全車帰着' ? '（目的）' : ''}</div></div>
      <div class="metric"><div class="value">${Object.keys(r.recover_time).length} / ${r.targets.length}</div>
        <div class="label">回復したセンサー</div></div>
    </div>
    ${r.unreachable.length ? `<p class="note">ドローンで届かない（対応不能）: ${r.unreachable.join('、')}</p>` : ''}
    <p class="hint">${notes.join('<br>')}</p>
    <h3>車ごとの時刻表</h3>${ganttHtml(r)}${timetableHtml(r)}`;
  $('btn-back-p2').hidden = false;
  renderStaleNote();
}

// 結果を出したあとに被害を変えたら、結果が古いことを知らせる
function renderStaleNote() {
  const note = $('stale-note');
  if (note) note.textContent = state.result && damageKey() !== state.resultKey
    ? '※ 結果を出したあとに被害か車の台数が変わりました。② でもう一度計算してください。' : '';
}
$('btn-back-p2').onclick = () => setView('p2');

// ============================================================
// はじめに: データを読み、入力欄に初期値を入れ、入口を出す
// ============================================================
async function start() {
  state.data = await api('/api/map');
  const d = state.data.defaults;
  $('solver-label').textContent = `ソルバー: ${state.data.solver}`;
  $('p1-grid').value = d.phase1.grid; $('p1-lam-one').value = d.phase1.lam_one;
  $('p1-lam-ab').value = d.phase1.lam_ab; $('p1-lam-cov').value = d.phase1.lam_cov;
  $('p1-cross').checked = d.phase1.cross; $('p1-time').value = d.phase1.time_limit_ms;
  $('f-charge').value = d.fleet.charge_min; $('f-builtin').value = d.fleet.flights_builtin;
  $('d-s-radius').value = d.damage.sensor_radius_km; $('d-s-strength').value = d.damage.sensor_strength;
  $('d-r-radius').value = d.damage.road_radius_km; $('d-r-strength').value = d.damage.road_strength;
  fillPhase1Tables();
  $('base-inputs').innerHTML = state.data.sample.bases.map((b) =>
    `<label>${b.name.split('（')[0]}（${b.base_id}）<input type="number" id="base-${b.base_id}" min="0" max="10" value="${b.vehicles}"></label>`)
    .join('');
  state.data.sample.bases.forEach((b) => { $(`base-${b.base_id}`).onchange = () => onDamageChanged(); });

  drawArea();
  drawRoads();
  drawBases();
  setSampleDamage();
  setView('p1');
  $('entry-status').textContent = '';
}

// 入口のボタン
function closeEntry() { $('entry').hidden = true; }
$('entry-p1').onclick = () => { closeEntry(); setView('p1'); };
$('entry-exact').onclick = async () => {
  $('entry-status').textContent = '読み込んでいます…';
  await loadExact();
  closeEntry();
  setView('p2');
};
$('entry-file').onchange = async (e) => {
  try {
    await loadPlacementFile(e.target.files[0]);
    closeEntry();
    setView('p2');
  } catch (error) {
    $('entry-status').textContent = `読み込めませんでした: ${error.message}`;
  }
};

start().catch((error) => { $('entry-status').textContent = `データを読み込めませんでした: ${error.message}`; });
