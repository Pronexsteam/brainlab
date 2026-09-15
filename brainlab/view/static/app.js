const $ = id => document.getElementById(id);
const canvas = $("c"), ctx = canvas.getContext("2d");
const stateCanvas = $("state"), sctx = stateCanvas.getContext("2d");
let graph = null, run = null, frame = 0, timer = null, selected = null, runIndex = null;
let datasetsList = [], currentDs = null, currentDsInfo = null, currentPop = null;
let graphLoading = null, graphDs = null;  // graphLoading — промис текущей загрузки графа (см. loadRun)

function reportError(e) {
  console.error(e);
  $("status").textContent = "ошибка: " + (e && e.message ? e.message : String(e));
}
const COLORS = { neuron: "#6cf", end_organ: "#886", GABA: "#f66", "": "#6cf" };
const STATE_COLORS = ["#6cf", "#f66", "#6f6", "#fc6", "#c6f", "#fff", "#f90", "#0ff"];

function resize() { canvas.width = innerWidth; canvas.height = innerHeight - 40; draw(); }
addEventListener("resize", resize);

async function api(path) { const r = await fetch(path); if (!r.ok) throw new Error(path); return r.json(); }

function toScreen(n) {
  const pad = 30;
  return [pad + (n.x + 1) / 2 * (canvas.width - 2 * pad), pad + (n.y + 1) / 2 * (canvas.height - 2 * pad)];
}

function draw() {
  if (!graph) { ctx.fillStyle = "#111"; ctx.fillRect(0, 0, canvas.width, canvas.height); return; }
  ctx.fillStyle = "#111"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  const pos = {}; graph.nodes.forEach(n => pos[n.name] = toScreen(n));
  const mode = $("color").value;
  ctx.lineWidth = 1;
  for (const e of graph.edges) {
    if (selected && e.pre !== selected && e.post !== selected) continue;
    ctx.strokeStyle = selected ? (e.kind === "electrical" ? "#fc6" : (e.sign < 0 ? "#f66" : "#6f6")) : "#333";
    ctx.beginPath(); ctx.moveTo(...pos[e.pre]); ctx.lineTo(...pos[e.post]); ctx.stroke();
  }
  const rates = run ? run.rates[Math.min(frame, run.rates.length - 1)] : null;
  graph.nodes.forEach((n, i) => {
    const [x, y] = pos[n.name];
    let col = COLORS[mode === "class" ? n.class : n.transmitter] || "#6cf";
    let r = n.focus ? 5 : 3;
    if (rates) { const k = runIndex ? runIndex.get(n.name) : undefined; const a = k >= 0 ? rates[k] : 0; r = (n.focus ? 5 : 3) + 8 * a; col = `rgb(${Math.round(80 + 175 * a)},${Math.round(80 + 60 * a)},80)`; }
    ctx.fillStyle = n.name === selected ? "#fff" : col;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    if (n.focus) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.stroke(); }
  });
}

function drawState() {
  sctx.fillStyle = "#111"; sctx.fillRect(0, 0, stateCanvas.width, stateCanvas.height);
  const st = run && run.extra && run.extra.state;
  if (!st || !st.values) return;
  const groups = st.groups || Object.keys(st.values);
  if (!groups.length) return;
  const nW = st.windows || (groups[0] && st.values[groups[0]] ? st.values[groups[0]].length : 0);
  if (!nW) return;
  const labelW = 56;
  const plotW = stateCanvas.width - labelW - 4;
  const rowH = stateCanvas.height / groups.length;
  sctx.font = "9px system-ui";
  groups.forEach((g, gi) => {
    const vals = st.values[g] || [];
    const y0 = gi * rowH;
    const max = Math.max(1e-6, ...vals);
    sctx.fillStyle = STATE_COLORS[gi % STATE_COLORS.length];
    sctx.fillText(g.slice(0, 10), 2, y0 + rowH - 3);
    sctx.strokeStyle = STATE_COLORS[gi % STATE_COLORS.length];
    sctx.beginPath();
    vals.forEach((v, i) => {
      const x = labelW + (i / Math.max(1, nW - 1)) * plotW;
      const y = y0 + rowH - 2 - (v / max) * (rowH - 4);
      if (i === 0) sctx.moveTo(x, y); else sctx.lineTo(x, y);
    });
    sctx.stroke();
  });
  const fx = labelW + (Math.min(frame, nW - 1) / Math.max(1, nW - 1)) * plotW;
  sctx.strokeStyle = "#fff9";
  sctx.beginPath(); sctx.moveTo(fx, 0); sctx.lineTo(fx, stateCanvas.height); sctx.stroke();
}

canvas.addEventListener("click", ev => {
  if (!graph) return;
  let best = null, bd = 100;
  for (const n of graph.nodes) { const [x, y] = toScreen(n); const d = (x - ev.offsetX) ** 2 + (y - ev.offsetY) ** 2; if (d < bd) { bd = d; best = n; } }
  selected = best ? best.name : null;
  const deg = graph.edges.filter(e => e.pre === selected || e.post === selected);
  $("info").textContent = best ? `${best.name}\nкласс: ${best.class}\nмедиатор: ${best.transmitter || "—"}\nсвязей: ${deg.length}` : "";
  draw();
});

async function loadPopulations(id) {
  try {
    currentPop = await api(`/api/populations/${id}`);
  } catch (e) {
    currentPop = null;
  }
  const groups = currentPop ? Object.keys(currentPop.groups) : [];
  $("group").innerHTML = groups.map(g => `<option value="${g}">${g} (${currentPop.groups[g]})</option>`).join("");
}

async function loadGraph() {
  const id = currentDs;
  // Загрузку графа держим отдельным промисом (graphLoading), чтобы loadRun(),
  // вызванный параллельно (смена #run во время await ниже), мог дождаться
  // актуального graph вместо того, чтобы уйти в /api/run без names на большом наборе.
  const fetchGraph = (async () => {
    if (currentDsInfo && currentDsInfo.big) {
      const focus = $("group").value;
      if (!focus) { $("status").textContent = `${id}: выберите группу`; graph = null; graphDs = id; draw(); return; }
      $("status").textContent = "подграф " + id + "…";
      graph = await api(`/api/graph/${id}?focus=${encodeURIComponent(focus)}&top=300&min_count=5`);
      graphDs = id;
      $("status").textContent = `${id}: подграф ${focus}, ${graph.nodes.length} клеток, ${graph.edges.length} связей`;
    } else {
      $("status").textContent = "граф " + id + "…";
      graph = await api(`/api/graph/${id}?min_count=2`);
      graphDs = id;
      $("status").textContent = `${id}: ${graph.nodes.length} клеток, ${graph.edges.length} связей`;
    }
    draw();
  })();
  graphLoading = fetchGraph.finally(() => { if (graphLoading === fetchGraph) graphLoading = null; });
  await graphLoading;
  if (run) await loadRun($("run").value);
}

async function loadDataset(id) {
  if (timer) { clearInterval(timer); timer = null; $("play").textContent = "▶"; }
  currentDs = id;
  currentDsInfo = datasetsList.find(d => d.id === id) || null;
  run = null; runIndex = null; $("time").max = 0; $("run").value = "";
  await loadPopulations(id);
  await loadGraph();
  drawState();
}

async function loadRun(id) {
  if (!id) {
    if (timer) { clearInterval(timer); timer = null; $("play").textContent = "▶"; }
    run = null; runIndex = null; $("time").max = 0; draw(); drawState(); return;
  }
  if (currentDsInfo && currentDsInfo.big) {
    if (graphLoading) await graphLoading;  // дождаться подграфа, если он ещё грузится
    if (!graph || graphDs !== currentDs) {
      $("status").textContent = `${currentDs}: сначала выберите группу`;
      return;
    }
  }
  let url = `/api/run/${id}`;
  if (currentDsInfo && currentDsInfo.big) {
    url += `?names=${encodeURIComponent(graph.nodes.map(n => n.name).join(","))}`;
  }
  run = await api(url);
  runIndex = new Map(run.names.map((name, i) => [name, i]));
  frame = 0; $("time").max = Math.max(0, run.rates.length - 1); $("time").value = 0;
  $("status").textContent = `прогон ${id}: ${run.rates.length} окон по ${run.window_ms} мс, оценка ${run.valence}`;
  draw();
  drawState();
}

$("play").addEventListener("click", () => {
  if (timer) { clearInterval(timer); timer = null; $("play").textContent = "▶"; return; }
  if (!run) return;
  $("play").textContent = "⏸";
  timer = setInterval(() => { frame = (frame + 1) % run.rates.length; $("time").value = frame; draw(); drawState(); }, 80);
});
$("time").addEventListener("input", () => { frame = +$("time").value; draw(); drawState(); });
$("color").addEventListener("change", draw);
$("dataset").addEventListener("change", () => loadDataset($("dataset").value).catch(reportError));
$("group").addEventListener("change", () => loadGraph().catch(reportError));
$("run").addEventListener("change", () => loadRun($("run").value).catch(reportError));

(async () => {
  resize();
  datasetsList = await api("/api/datasets");
  $("dataset").innerHTML = datasetsList.map(d => `<option value="${d.id}">${d.id} (${d.neurons})</option>`).join("");
  const runs = await api("/api/runs");
  $("run").innerHTML = `<option value="">—</option>` + runs.map(r =>
    `<option value="${r.id}" data-ds="${r.dataset}">${r.id} ${r.model} ${r.valence >= 0 ? "+" : ""}${r.valence}</option>`).join("");
  if (datasetsList.length) await loadDataset(datasetsList[0].id);
})().catch(reportError);
