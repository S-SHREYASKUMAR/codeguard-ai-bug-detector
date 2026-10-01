/* global CodeGuard UI */
let editor;
let lastResult = null;
let tokenChart;
const HISTORY_KEY = "codeguard-history-v1";
const THEME_KEY = "codeguard-theme";

function toast(msg, ok = true) {
  const wrap = document.getElementById("toasts");
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = msg;
  el.style.borderColor = ok ? "rgba(16,185,129,.4)" : "rgba(239,68,68,.4)";
  wrap.appendChild(el);
  setTimeout(() => el.remove(), 3200);
}

function setTab(name) {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    const on = btn.dataset.tab === name;
    btn.classList.toggle("is-active", on);
    btn.setAttribute("aria-selected", String(on));
  });
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.add("hidden"));
  document.getElementById(`tab-${name}`).classList.remove("hidden");
  if (name === "insights") loadInsights();
  if (name === "history") renderHistory();
}

async function refreshHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const chip = document.getElementById("model-chip");
    chip.textContent = data.model_status;
    chip.classList.toggle("green", data.model_ready);
    chip.classList.toggle("amber", !data.model_ready);
  } catch {
    /* ignore */
  }
}

function monacoTheme() {
  const light = document.documentElement.classList.contains("light");
  return light ? "vs" : "vs-dark";
}

function bootEditor() {
  require.config({
    paths: { vs: "https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs" },
  });
  window.MonacoEnvironment = {
    getWorkerUrl() {
      const src = `self.MonacoEnvironment={baseUrl:'https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/'};importScripts('https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs/base/worker/workerMain.js');`;
      return URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
    },
  };
  require(["vs/editor/editor.main"], () => {
    editor = monaco.editor.create(document.getElementById("editor"), {
      value: "",
      language: "c",
      theme: monacoTheme(),
      fontFamily: "JetBrains Mono, monospace",
      fontLigatures: true,
      minimap: { enabled: false },
      lineNumbers: "on",
      automaticLayout: true,
      scrollBeyondLastLine: false,
      glyphMargin: true,
    });
    editor.onDidChangeModelContent(updateCounts);
    updateCounts();
    const frame = document.getElementById("editor");
    frame.addEventListener("dragover", (e) => e.preventDefault());
    frame.addEventListener("drop", async (e) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) editor.setValue(await file.text());
    });
  });
}

function updateCounts() {
  if (!editor) return;
  const v = editor.getValue();
  const lines = v ? v.split("\n").length : 0;
  document.getElementById("char-count").textContent = `${v.length} chars · ${lines} lines`;
}

async function loadSamples() {
  const sel = document.getElementById("sample-select");
  const res = await fetch("/api/samples");
  const data = await res.json();
  data.forEach((s) => {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.textContent = s.title;
    opt.dataset.code = s.code;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", () => {
    const opt = sel.selectedOptions[0];
    if (opt && opt.dataset.code) editor.setValue(opt.dataset.code);
  });
}

function severityRank(s) {
  return { critical: 0, high: 1, medium: 2, low: 3 }[s] ?? 9;
}

function cweLink(cwe) {
  const id = String(cwe).replace(/CWE-/i, "");
  return `https://cwe.mitre.org/data/definitions/${id}.html`;
}

function setGauge(score) {
  const arc = document.getElementById("gauge-arc");
  const label = document.getElementById("gauge-value");
  const max = 157;
  const offset = max - (max * Math.max(0, Math.min(100, score))) / 100;
  arc.style.strokeDashoffset = String(offset);
  const color = score < 30 ? "#10b981" : score < 55 ? "#eab308" : "#ef4444";
  arc.style.stroke = color;
  label.textContent = Math.round(score);
}

function renderIssues(result) {
  const box = document.getElementById("issues");
  box.innerHTML = "";
  const findings = [...(result.findings || [])].sort(
    (a, b) => severityRank(a.severity) - severityRank(b.severity)
  );
  if (!findings.length) {
    box.innerHTML = `<p class="text-sm text-slate-400">No named static findings.</p>`;
    return;
  }
  findings.forEach((f, idx) => {
    const el = document.createElement("article");
    el.className = "issue-card";
    el.tabIndex = 0;
    el.innerHTML = `
      <div class="flex flex-wrap items-center gap-2">
        <span class="badge ${f.severity}">${f.severity}</span>
        <strong>${f.bug_type}</strong>
        <a class="text-xs text-indigo-400" href="${cweLink(f.cwe)}" target="_blank" rel="noreferrer">${f.cwe}</a>
        <span class="text-xs text-slate-400">${f.function} · Line ${f.line}</span>
      </div>
      <p class="mt-2 text-sm">${f.why_it_is_a_bug}</p>
      <p class="mt-1 text-xs text-slate-400">${f.message}</p>
      <div class="confidence mt-2" aria-label="confidence"><span style="width:${Math.round(f.confidence * 100)}%"></span></div>
      <details class="mt-2">
        <summary class="text-xs">Suggested fix</summary>
        <div class="diff mt-1">
          <div class="del">- ${f.message}</div>
          <pre class="add">+ ${f.fixed_code_snippet}</pre>
        </div>
        <button type="button" class="btn-secondary mt-2 copy-fix">Copy fix</button>
      </details>`;
    el.addEventListener("click", (ev) => {
      if (ev.target.closest("a, button, summary, details")) return;
      jumpTo(f.line);
    });
    el.querySelector(".copy-fix").addEventListener("click", async (ev) => {
      ev.stopPropagation();
      await navigator.clipboard.writeText(f.fixed_code_snippet);
      toast("Fix copied");
    });
    box.appendChild(el);
  });
}

let decos = [];
function jumpTo(line) {
  if (!editor) return;
  editor.revealLineInCenter(line);
  editor.setPosition({ lineNumber: line, column: 1 });
  editor.focus();
  const ids = editor.deltaDecorations(
    [],
    [
      {
        range: new monaco.Range(line, 1, line, 1),
        options: { isWholeLine: true, className: "flash-line" },
      },
    ]
  );
  setTimeout(() => editor.deltaDecorations(ids, []), 600);
}

function applyDecorations(result) {
  if (!editor) return;
  const marks = [];
  (result.findings || []).forEach((f) => {
    const color =
      f.severity === "critical" ? "squiggly-crit" : f.severity === "high" ? "squiggly-high" : "squiggly-med";
    marks.push({
      range: new monaco.Range(f.line, 1, f.end_line || f.line, 1),
      options: {
        isWholeLine: true,
        className: color,
        glyphMarginClassName: "glyph-bug",
        glyphMarginHoverMessage: { value: f.bug_type },
        minimap: { color: "#ef4444", position: 1 },
      },
    });
  });
  if (document.getElementById("ml-highlights").checked) {
    (result.ml?.top_tokens || []).forEach((t) => {
      if (t.weight <= 0) return;
      (t.positions || []).forEach((p) => {
        marks.push({
          range: new monaco.Range(p.line, (p.start_col || 0) + 1, p.line, (p.end_col || 0) + 1),
          options: { inlineClassName: "ml-token" },
        });
      });
    });
  }
  decos = editor.deltaDecorations(decos, marks);
}

function renderFunctions(result) {
  const tb = document.getElementById("fn-table");
  tb.innerHTML = "";
  (result.functions || []).forEach((fn) => {
    const tr = document.createElement("tr");
    tr.className = "cursor-pointer border-t border-white/5";
    tr.innerHTML = `<td class="py-2">${fn.name}</td><td>${fn.start_line}–${fn.end_line}</td>
      <td><div class="confidence"><span style="width:${fn.risk_score}%"></span></div></td>
      <td>${fn.verdict}</td>`;
    tr.addEventListener("click", () => jumpTo(fn.start_line));
    tb.appendChild(tr);
  });
}

function renderTokens(result) {
  const ctx = document.getElementById("token-chart");
  const tokens = result.ml?.top_tokens || [];
  const labels = tokens.map((t) => t.token);
  const data = tokens.map((t) => t.weight);
  const colors = data.map((w) => (w >= 0 ? "#ef4444" : "#10b981"));
  if (tokenChart) tokenChart.destroy();
  tokenChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{ data, backgroundColor: colors }],
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: { x: { ticks: { color: "#94a3b8" } }, y: { ticks: { color: "#94a3b8" } } },
    },
  });
}

function saveHistory(code, result) {
  const items = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  items.unshift({
    ts: Date.now(),
    filename: result.filename || code.trim().split("\n")[0].slice(0, 80),
    verdict: result.verdict,
    score: result.risk_score,
    code,
  });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 25)));
}

function renderHistory() {
  const list = document.getElementById("history-list");
  const items = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  if (!items.length) {
    list.innerHTML = `<li class="text-slate-400">No history yet.</li>`;
    return;
  }
  list.innerHTML = "";
  items.forEach((it, i) => {
    const li = document.createElement("li");
    li.className = "card p-3 flex justify-between gap-3";
    li.innerHTML = `<button class="text-left flex-1" type="button">
        <div class="text-sm font-medium">${it.verdict} · ${Math.round(it.score)}</div>
        <div class="text-xs text-slate-400">${new Date(it.ts).toLocaleString()} · ${it.filename}</div>
      </button>
      <button class="btn-secondary del" type="button">Delete</button>`;
    li.querySelector("button").addEventListener("click", () => {
      setTab("analyzer");
      editor.setValue(it.code);
    });
    li.querySelector(".del").addEventListener("click", () => {
      items.splice(i, 1);
      localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
      renderHistory();
    });
    list.appendChild(li);
  });
}

async function analyze() {
  if (!editor) return;
  const code = editor.getValue();
  document.getElementById("empty-state").classList.add("hidden");
  document.getElementById("result-body").classList.add("hidden");
  document.getElementById("skeleton").classList.remove("hidden");
  const btn = document.getElementById("btn-analyze");
  btn.disabled = true;
  btn.innerHTML = `<span class="inline-block animate-spin">◌</span> Analyzing`;
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code, filename: "editor.c" }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Analyze failed");
    }
    const result = await res.json();
    lastResult = result;
    document.getElementById("skeleton").classList.add("hidden");
    document.getElementById("result-body").classList.remove("hidden");
    const warnEl = document.getElementById("language-warning");
    const warnText = document.getElementById("language-warning-text");
    if (result.language?.warning && warnEl && warnText) {
      warnText.textContent = result.language.warning;
      warnEl.classList.remove("hidden");
    } else if (warnEl) {
      warnEl.classList.add("hidden");
    }
    document.getElementById("verdict-label").textContent = result.verdict;
    document.getElementById("verdict-summary").textContent = result.summary;
    document.getElementById("verdict-chips").innerHTML = `
      <span class="badge low">${result.stats.functions} functions</span>
      <span class="badge medium">${result.stats.issues} issues</span>
      <span class="badge low">${result.stats.analysis_ms} ms</span>`;
    setGauge(result.risk_score);
    renderIssues(result);
    renderFunctions(result);
    const p = result.ml?.probability;
    document.getElementById("ml-prob").textContent = result.ml?.available
      ? `Model P(defective) = ${p?.toFixed(3)} (${result.ml.label ? "defective" : "clean"})`
      : result.ml?.message || "Model not trained";
    renderTokens(result);
    applyDecorations(result);
    saveHistory(code, result);
    toast("Analysis complete");
  } catch (err) {
    document.getElementById("skeleton").classList.add("hidden");
    document.getElementById("empty-state").classList.remove("hidden");
    toast(String(err.message || err), false);
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze Code";
  }
}

function countUp(el, target) {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduced) {
    el.textContent = target.toFixed(3);
    return;
  }
  const t0 = performance.now();
  const tick = (now) => {
    const p = Math.min(1, (now - t0) / 700);
    el.textContent = (target * p).toFixed(3);
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

let insightCharts = [];
async function loadInsights() {
  insightCharts.forEach((c) => c.destroy());
  insightCharts = [];
  const res = await fetch("/api/metrics");
  const kpis = document.getElementById("kpi-row");
  if (!res.ok) {
    kpis.innerHTML = `<p class="text-amber-300">Train the model to see metrics. python -m backend.ml.train</p>`;
    return;
  }
  const m = await res.json();
  const t = m.test || {};
  const names = [
    ["Accuracy", t.accuracy],
    ["Precision", t.precision],
    ["Recall", t.recall],
    ["F1", t.f1],
    ["ROC-AUC", t.roc_auc],
  ];
  kpis.innerHTML = names
    .map(
      ([n]) =>
        `<div class="kpi"><div class="text-xs text-slate-400">${n}</div><div class="text-2xl font-semibold kpi-val">0</div></div>`
    )
    .join("");
  [...kpis.querySelectorAll(".kpi-val")].forEach((el, i) => countUp(el, names[i][1] || 0));
  const cm = t.confusion_matrix || [
    [0, 0],
    [0, 0],
  ];
  insightCharts.push(
    new Chart(document.getElementById("cm-chart"), {
      type: "bar",
      data: {
        labels: ["Pred 0", "Pred 1"],
        datasets: [
          { label: "True 0", data: cm[0], backgroundColor: "#3b82f6" },
          { label: "True 1", data: cm[1], backgroundColor: "#ef4444" },
        ],
      },
      options: { plugins: { title: { display: true, text: "Confusion matrix", color: "#e5e7eb" } } },
    })
  );
  insightCharts.push(
    new Chart(document.getElementById("roc-chart"), {
      type: "line",
      data: {
        labels: (t.roc?.fpr || []).map((x) => x.toFixed(2)),
        datasets: [{ label: "ROC", data: t.roc?.tpr || [], borderColor: "#8b5cf6", tension: 0.2 }],
      },
      options: { plugins: { title: { display: true, text: "ROC curve", color: "#e5e7eb" } } },
    })
  );
  const ds = m.dataset || {};
  insightCharts.push(
    new Chart(document.getElementById("balance-chart"), {
      type: "doughnut",
      data: {
        labels: ["Train neg", "Train pos"],
        datasets: [
          {
            data: [ds.train?.negative || 0, ds.train?.positive || 0],
            backgroundColor: ["#10b981", "#ef4444"],
          },
        ],
      },
      options: { plugins: { title: { display: true, text: "Train class balance", color: "#e5e7eb" } } },
    })
  );
  const ours = t.accuracy || 0;
  const bases = (m.published_baselines || []).map((b) => b.accuracy);
  insightCharts.push(
    new Chart(document.getElementById("baseline-chart"), {
      type: "bar",
      data: {
        labels: ["Ours", ...(m.published_baselines || []).map((b) => b.name)],
        datasets: [{ data: [ours, ...bases], backgroundColor: "#6366f1" }],
      },
      options: { plugins: { title: { display: true, text: "Accuracy vs published references", color: "#e5e7eb" } } },
    })
  );
  const pos = m.top_tokens?.positive || [];
  insightCharts.push(
    new Chart(document.getElementById("risk-tokens-chart"), {
      type: "bar",
      data: {
        labels: pos.map((x) => x.token),
        datasets: [{ label: "LR weight", data: pos.map((x) => x.weight), backgroundColor: "#f97316" }],
      },
      options: { plugins: { title: { display: true, text: "Top risk tokens", color: "#e5e7eb" } } },
    })
  );
  document.getElementById("dataset-card").innerHTML = `
    <strong>Dataset</strong> ${ds.source || "CodeXGLUE"} ·
    train ${ds.train?.n || "?"} · val ${ds.validation?.n || "?"} · test ${ds.test?.n || "?"}
    · trained in ${(m.training_time_sec || 0).toFixed(1)}s · model ${m.selected_model || ""}`;
}

function wire() {
  document.querySelectorAll(".tab-btn").forEach((b) =>
    b.addEventListener("click", () => setTab(b.dataset.tab))
  );
  document.getElementById("btn-analyze").addEventListener("click", analyze);
  document.getElementById("btn-clear").addEventListener("click", () => editor && editor.setValue(""));
  document.getElementById("btn-copy").addEventListener("click", async () => {
    await navigator.clipboard.writeText(editor.getValue());
    toast("Copied");
  });
  document.getElementById("file-input").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (f) editor.setValue(await f.text());
  });
  document.getElementById("theme-toggle").addEventListener("click", () => {
    document.documentElement.classList.toggle("light");
    localStorage.setItem(THEME_KEY, document.documentElement.classList.contains("light") ? "light" : "dark");
    if (window.monaco && editor) monaco.editor.setTheme(monacoTheme());
  });
  document.getElementById("btn-json").addEventListener("click", () => {
    if (!lastResult) return;
    const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "codeguard-report.json";
    a.click();
  });
  document.getElementById("btn-print").addEventListener("click", () => window.print());
  document.getElementById("btn-clear-history").addEventListener("click", () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
  });
  document.getElementById("ml-highlights").addEventListener("change", () => lastResult && applyDecorations(lastResult));
  window.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      analyze();
    }
  });
  if (localStorage.getItem(THEME_KEY) === "light") document.documentElement.classList.add("light");
  const style = document.createElement("style");
  style.textContent = `
    .squiggly-crit { text-decoration: underline wavy #ef4444; background: rgba(239,68,68,.08); }
    .squiggly-high { text-decoration: underline wavy #f97316; background: rgba(249,115,22,.08); }
    .squiggly-med { text-decoration: underline wavy #eab308; }
    .ml-token { background: rgba(139,92,246,.25); }
    .glyph-bug { background: #ef4444; border-radius: 99px; }
    .flash-line { background: rgba(99,102,241,.25) !important; }
  `;
  document.head.appendChild(style);
}

wire();
bootEditor();
loadSamples().catch(() => toast("Could not load samples", false));
refreshHealth();
