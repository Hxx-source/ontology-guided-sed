"use strict";

const $ = (selector) => document.querySelector(selector);
const state = {
  manifest: null,
  example: null,
  scores: null,
  events: [],
  byClass: new Map(),
  threshold: 0.5,
  lang: "zh",
  view: "timeline",
  scope: "detected",
  selectedClass: null,
  requestId: 0,
};

const copy = {
  zh: {
    brandContext: "论文方法与声音事件检测", classes: "类", listenAndInspect: "LISTEN & INSPECT",
    demoList: "演示音频", sidebarIntro: "真实推理 · 开放授权音频", recordings: "原始录音",
    frozenModel: "冻结权重 · 无服务端推理", loading: "正在加载示例…", export: "导出 TSV",
    recording: "音频", spectrogram: "梅尔声谱图", modelOutput: "MODEL OUTPUT",
    detections: "事件检测", threshold: "阈值", reset: "重置", timeline: "时间轴",
    eventList: "事件列表", detected: "已检出", allClasses: "全部类别", searchClass: "搜索类别",
    classLabel: "类别", time: "时间", frameProbability: "帧概率 · CAMF 后",
    start: "开始", end: "结束", readingResult: "HOW TO READ THIS CASE",
    caseNotes: "案例说明", audioProvenance: "AUDIO PROVENANCE", sources: "音频来源",
    footerNote: "冻结模型的预计算推理；示例经过挑选，不代表总体性能。GitHub Pages 不运行 PyTorch 模型。",
    inferenceInstructions: "仓库与推理命令", eventCase: "单段录音", challengeCase: "对照实验",
    challengeGroup: "混合挑战", events: "个事件", classCount: "个类别",
    showing: "显示", of: "/", searchToNarrow: "；可搜索其余类别",
    emptyTracks: "当前筛选下没有可显示的类别。", emptyEvents: "当前阈值下没有事件。",
    noData: "无法加载演示数据。请通过 HTTP 服务或 GitHub Pages 打开页面。",
    inserted: "黄色区域为人为插入的咳嗽（4.00–4.66 秒），是构造样本的已知信息，不是模型预测。",
    sourceCut: "音频最多取原文件前 10 秒，转为 32 kHz 单声道 WAV。",
    sourceMix: "由下列两段录音合成；局部 RMS 信噪比见标题。",
    selectedClass: "类别帧概率", missed: "漏检", detectedNote: "检出",
  },
  en: {
    brandContext: "Research method & sound detection", classes: "classes", listenAndInspect: "LISTEN & INSPECT",
    demoList: "Demo audio", sidebarIntro: "Actual inference · openly licensed audio", recordings: "Recordings",
    frozenModel: "Frozen checkpoint · no server inference", loading: "Loading example…", export: "Export TSV",
    recording: "Audio", spectrogram: "Mel spectrogram", modelOutput: "MODEL OUTPUT",
    detections: "Detections", threshold: "Threshold", reset: "Reset", timeline: "Timeline",
    eventList: "Event list", detected: "Detected", allClasses: "All classes", searchClass: "Search classes",
    classLabel: "Class", time: "Time", frameProbability: "Frame probability · after CAMF",
    start: "Start", end: "End", readingResult: "HOW TO READ THIS CASE",
    caseNotes: "Case notes", audioProvenance: "AUDIO PROVENANCE", sources: "Audio sources",
    footerNote: "Precomputed frozen-model inference on curated examples, not a performance benchmark. GitHub Pages does not run PyTorch.",
    inferenceInstructions: "Repository & inference", eventCase: "Recording", challengeCase: "Controlled comparison",
    challengeGroup: "Mixture challenge", events: "events", classCount: "classes",
    showing: "Showing", of: "of", searchToNarrow: "; search to see other classes",
    emptyTracks: "No classes match the current filter.", emptyEvents: "No events at this threshold.",
    noData: "Demo assets could not be loaded. Open through an HTTP server or GitHub Pages.",
    inserted: "The gold band is the manually inserted cough (4.00–4.66 s). It is known from construction, not a model prediction.",
    sourceCut: "Up to the first 10 seconds of the original, converted to 32 kHz mono WAV.",
    sourceMix: "Mixed from the two recordings below; the title gives local RMS SNR.",
    selectedClass: "Class frame probability", missed: "missed", detectedNote: "detected",
  },
};

function t(key) { return copy[state.lang][key]; }
function n(value) { return Number(value).toFixed(2); }
function escapeTsv(value) { return String(value).replace(/[\t\r\n]/g, " "); }

function setLanguage(lang) {
  state.lang = lang;
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = t(element.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((element) => {
    element.placeholder = t(element.dataset.i18nPlaceholder);
  });
  document.querySelectorAll("[data-lang]").forEach((button) => {
    const active = button.dataset.lang === lang;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  if (state.manifest) renderSidebar();
  if (state.example) { renderCase(); renderResults(); }
}

function renderSidebar() {
  const list = $("#sample-list");
  const previousScroll = list.scrollLeft;
  list.replaceChildren();
  state.manifest.examples.forEach((example, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sample-item";
    if (example.group === "challenge") button.classList.add("challenge");
    if (index === 5) button.classList.add("challenge-start");
    if (state.example && example.id === state.example.id) button.classList.add("active");
    button.setAttribute("aria-current", example.id === state.example?.id ? "true" : "false");
    const number = document.createElement("span");
    number.className = "sample-number";
    number.textContent = String(index + 1).padStart(2, "0");
    const text = document.createElement("span");
    const title = document.createElement("span");
    title.className = "sample-title";
    title.textContent = example.title[state.lang];
    const meta = document.createElement("span");
    meta.className = "sample-meta";
    meta.textContent = `${n(example.duration)} s · ${example.group === "challenge" ? t("challengeCase") : t("eventCase")}`;
    text.append(title, meta);
    button.append(number, text);
    button.addEventListener("click", () => selectExample(example.id));
    list.append(button);
  });
  list.scrollLeft = previousScroll;
  const active = list.querySelector(".sample-item.active");
  if (active && list.scrollWidth > list.clientWidth) {
    const itemRect = active.getBoundingClientRect();
    const listRect = list.getBoundingClientRect();
    if (itemRect.left < listRect.left || itemRect.right > listRect.right) {
      list.scrollLeft += itemRect.left - listRect.left - 12;
    }
  }
}

async function selectExample(id) {
  const example = state.manifest.examples.find((entry) => entry.id === id);
  if (!example) return;
  const requestId = ++state.requestId;
  $("#audio-player").pause();
  $("#loading").hidden = false;
  $("#content").hidden = true;
  $("#error").hidden = true;
  try {
    const response = await fetch(example.scores);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const buffer = await response.arrayBuffer();
    if (requestId !== state.requestId) return;
    if (buffer.byteLength !== state.manifest.labels.length * example.frames * 4) {
      throw new Error("Score file has an unexpected size");
    }
    state.example = example;
    state.scores = new Float32Array(buffer);
    state.selectedClass = state.manifest.labels.indexOf(example.focus[0]);
    state.threshold = 0.5;
    $("#threshold").value = "0.50";
    $("#class-search").value = "";
    const audio = $("#audio-player");
    audio.src = example.audio;
    audio.load();
    renderSidebar();
    renderCase();
    renderResults();
    $("#loading").hidden = true;
    $("#content").hidden = false;
    updatePlayhead();
    requestAnimationFrame(() => { drawWaveform(); drawScore(); });
    if (window.history.replaceState) {
      const url = new URL(window.location.href);
      url.searchParams.set("demo", id);
      window.history.replaceState(null, "", url);
    }
  } catch (error) {
    if (requestId !== state.requestId) return;
    $("#loading").hidden = true;
    $("#error").hidden = false;
    $("#error").textContent = `${t("noData")} ${error.message}`;
  }
}

function renderCase() {
  const example = state.example;
  const index = state.manifest.examples.findIndex((entry) => entry.id === example.id);
  $("#case-index").textContent = `${String(index + 1).padStart(2, "0")} / ${String(state.manifest.examples.length).padStart(2, "0")}`;
  const type = $("#case-type");
  type.classList.toggle("challenge", example.group === "challenge");
  type.textContent = example.group === "challenge" ? t("challengeCase") : t("eventCase");
  $("#case-title").textContent = example.title[state.lang];
  $("#case-description").textContent = example.description[state.lang];
  $("#duration-pill").textContent = `${n(example.duration)} s`;
  $("#spectrogram").src = example.spectrogram;
  $("#half-time").textContent = `${n(example.duration / 2)} s`;
  $("#full-time").textContent = `${n(example.duration)} s`;
  $("#score-half-time").textContent = `${n(example.duration / 2)} s`;
  $("#score-full-time").textContent = `${n(example.duration)} s`;
  const ref = $("#reference-note");
  ref.hidden = !example.inserted;
  ref.textContent = example.inserted ? t("inserted") : "";
  document.querySelectorAll(".seek-surface .reference-band").forEach((band) => {
    band.hidden = !example.inserted;
    if (example.inserted) {
      band.style.left = `${example.inserted[0] / example.duration * 100}%`;
      band.style.width = `${(example.inserted[1] - example.inserted[0]) / example.duration * 100}%`;
    }
  });
  const sources = $("#sources-list");
  sources.replaceChildren();
  example.sources.forEach((source) => {
    const entry = document.createElement("div");
    const link = document.createElement("a");
    link.href = source.page;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = (source.page.split("File:")[1] || source.page).replaceAll("_", " ");
    const credit = document.createElement("small");
    credit.textContent = source.credit;
    entry.append(link, document.createElement("br"), credit);
    sources.append(entry);
  });
  const sourceCut = document.createElement("small");
  sourceCut.textContent = example.inserted ? t("sourceMix") : t("sourceCut");
  sources.append(sourceCut);
}

function computeEvents() {
  const events = [];
  const byClass = new Map();
  const frames = state.example.frames;
  const duration = state.example.duration;
  const frameSeconds = state.manifest.frame_seconds;
  for (let index = 0; index < state.manifest.labels.length; index++) {
    let start = -1;
    for (let frame = 0; frame <= frames; frame++) {
      const active = frame < frames && state.scores[index * frames + frame] > state.threshold;
      if (active && start < 0) start = frame;
      if (!active && start >= 0) {
        const event = { index, onset: start * frameSeconds, offset: Math.min(frame * frameSeconds, duration) };
        if (event.onset < event.offset) {
          events.push(event);
          if (!byClass.has(index)) byClass.set(index, []);
          byClass.get(index).push(event);
        }
        start = -1;
      }
    }
  }
  events.sort((a, b) => a.onset - b.onset || a.index - b.index);
  state.events = events;
  state.byClass = byClass;
}

function visibleClasses() {
  const focus = state.example.focus.map((label) => state.manifest.labels.indexOf(label)).filter((index) => index >= 0);
  let indices = state.scope === "detected"
    ? Array.from(new Set([...focus, ...state.byClass.keys()]))
    : state.manifest.labels.map((_, index) => index);
  const search = $("#class-search").value.trim().toLowerCase();
  if (search) indices = indices.filter((index) => state.manifest.labels[index].toLowerCase().includes(search));
  indices.sort((a, b) => {
    const fa = focus.indexOf(a), fb = focus.indexOf(b);
    if (fa >= 0 || fb >= 0) return (fa < 0 ? 1000 : fa) - (fb < 0 ? 1000 : fb);
    const da = (state.byClass.get(a) || []).reduce((sum, event) => sum + event.offset - event.onset, 0);
    const db = (state.byClass.get(b) || []).reduce((sum, event) => sum + event.offset - event.onset, 0);
    return db - da || state.manifest.labels[a].localeCompare(state.manifest.labels[b]);
  });
  const total = indices.length;
  if (state.scope === "all" && !search) indices = indices.slice(0, 70);
  $("#visible-count").textContent = `${t("showing")} ${indices.length} ${t("of")} ${total}${total > indices.length ? t("searchToNarrow") : ""}`;
  return indices;
}

function renderResults() {
  if (!state.example || !state.scores) return;
  computeEvents();
  $("#threshold-value").textContent = n(state.threshold);
  $("#result-summary").textContent = `${state.events.length} ${t("events")} · ${state.byClass.size} ${t("classCount")}`;
  const visible = visibleClasses();
  if (visible.length && !visible.includes(state.selectedClass)) state.selectedClass = visible[0];
  renderTracks(visible);
  renderEventsTable();
  renderCaseNote();
  drawScore();
  updatePlayhead();
}

function renderTracks(indices) {
  const tracks = $("#tracks");
  tracks.replaceChildren();
  $("#track-empty").hidden = indices.length > 0;
  $("#track-empty").textContent = t("emptyTracks");
  const focus = new Set(state.example.focus);
  indices.forEach((index) => {
    const label = state.manifest.labels[index];
    const row = document.createElement("div");
    row.className = "track-row";
    if (index === state.selectedClass) row.classList.add("selected");
    if (focus.has(label)) row.classList.add("focus-row");
    if (label === "Cough" && state.example.inserted) row.classList.add("challenge-focus");
    const title = document.createElement("button");
    title.type = "button";
    title.className = "track-label";
    if (focus.has(label)) title.classList.add("focus");
    title.textContent = label;
    title.title = `${t("selectedClass")}: ${label}`;
    title.addEventListener("click", () => { state.selectedClass = index; renderTracks(indices); drawScore(); });
    const rail = document.createElement("div");
    rail.className = "track-rail";
    rail.addEventListener("click", (event) => seekFromPointer(event, rail));
    if (state.example.inserted) {
      const band = document.createElement("div");
      band.className = "reference-band";
      band.style.left = `${state.example.inserted[0] / state.example.duration * 100}%`;
      band.style.width = `${(state.example.inserted[1] - state.example.inserted[0]) / state.example.duration * 100}%`;
      rail.append(band);
    }
    (state.byClass.get(index) || []).forEach((event) => {
      const segment = document.createElement("button");
      segment.type = "button";
      segment.className = "event-segment";
      segment.style.left = `${event.onset / state.example.duration * 100}%`;
      segment.style.width = `${(event.offset - event.onset) / state.example.duration * 100}%`;
      segment.title = `${label}: ${n(event.onset)}–${n(event.offset)} s`;
      segment.setAttribute("aria-label", segment.title);
      segment.addEventListener("click", (pointerEvent) => {
        pointerEvent.stopPropagation();
        $("#audio-player").currentTime = event.onset;
        $("#audio-player").play().catch(() => {});
        state.selectedClass = index;
        renderTracks(indices);
        drawScore();
      });
      rail.append(segment);
    });
    const playhead = document.createElement("div");
    playhead.className = "media-playhead";
    playhead.setAttribute("aria-hidden", "true");
    rail.append(playhead);
    row.append(title, rail);
    tracks.append(row);
  });
}

function renderEventsTable() {
  const body = $("#events-body");
  body.replaceChildren();
  const search = $("#class-search").value.trim().toLowerCase();
  const events = state.events.filter((event) => state.manifest.labels[event.index].toLowerCase().includes(search));
  $("#events-empty").hidden = events.length > 0;
  $("#events-empty").textContent = t("emptyEvents");
  events.forEach((event) => {
    const row = document.createElement("tr");
    const onset = document.createElement("td"), offset = document.createElement("td"), label = document.createElement("td");
    onset.textContent = `${n(event.onset)} s`;
    offset.textContent = `${n(event.offset)} s`;
    label.textContent = state.manifest.labels[event.index];
    row.append(onset, offset, label);
    row.addEventListener("click", () => { $("#audio-player").currentTime = event.onset; $("#audio-player").play().catch(() => {}); });
    body.append(row);
  });
}

function renderCaseNote() {
  const example = state.example;
  const labels = state.manifest.labels;
  const focus = example.focus.map((name) => {
    const index = labels.indexOf(name);
    return `${name}: ${(state.byClass.get(index) || []).length}`;
  });
  let note = `${focus.join(" · ")}. `;
  if (example.inserted) {
    const cough = state.byClass.get(labels.indexOf("Cough")) || [];
    note += state.lang === "zh"
      ? `在当前阈值下，插入区间的 Cough ${cough.some((event) => event.onset < 4.66 && event.offset > 4.0) ? "被检出" : "未检出"}。这是单个构造案例，不是召回率估计。`
      : `At this threshold, the inserted Cough is ${cough.some((event) => event.onset < 4.66 && event.offset > 4.0) ? "detected" : "missed"}. This single mixture is not a recall estimate.`;
  } else {
    note += state.lang === "zh"
      ? "父类与子类可在同一时段并存；列表中的类别不是互斥选项。"
      : "Parent and child categories can co-occur; the listed labels are not mutually exclusive.";
  }
  $("#case-note").textContent = note;
}

function canvasContext(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(canvas.clientWidth));
  const height = Math.max(1, Math.floor(canvas.clientHeight));
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function drawWaveform() {
  if (!state.example || $("#content").hidden) return;
  const { ctx, width, height } = canvasContext($("#waveform"));
  const values = state.example.waveform;
  const max = Math.max(0.1, ...values);
  const barWidth = width / values.length;
  ctx.fillStyle = "#178479";
  values.forEach((value, index) => {
    const size = Math.max(1, value / max * (height - 14));
    const x = index * barWidth;
    ctx.fillRect(x, (height - size) / 2, Math.max(1, barWidth * 0.72), size);
  });
}

function drawScore() {
  if (!state.example || !state.scores || $("#content").hidden) return;
  const index = state.selectedClass;
  if (index < 0 || index >= state.manifest.labels.length) return;
  $("#score-title").textContent = `${t("selectedClass")}: ${state.manifest.labels[index]}`;
  const { ctx, width, height } = canvasContext($("#score-canvas"));
  const pad = 10;
  const graphHeight = height - pad * 2;
  const frames = state.example.frames;
  ctx.strokeStyle = "#dce5e1";
  ctx.lineWidth = 1;
  for (const level of [0, 0.5, 1]) {
    const y = pad + (1 - level) * graphHeight;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }
  ctx.setLineDash([5, 4]);
  ctx.strokeStyle = "#d55f4d";
  const thresholdY = pad + (1 - state.threshold) * graphHeight;
  ctx.beginPath(); ctx.moveTo(0, thresholdY); ctx.lineTo(width, thresholdY); ctx.stroke();
  ctx.setLineDash([]);
  ctx.lineWidth = 2;
  ctx.strokeStyle = index === state.manifest.labels.indexOf("Cough") && state.example.inserted ? "#cb6652" : "#087a73";
  ctx.beginPath();
  for (let frame = 0; frame < frames; frame++) {
    const x = frames === 1 ? 0 : frame / (frames - 1) * width;
    const y = pad + (1 - state.scores[index * frames + frame]) * graphHeight;
    if (frame === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function seekFromPointer(event, surface) {
  const rect = surface.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  $("#audio-player").currentTime = ratio * state.example.duration;
  updatePlayhead();
}

function updatePlayhead() {
  if (!state.example) return;
  const audio = $("#audio-player");
  const time = Math.min(audio.currentTime || 0, state.example.duration);
  const percent = time / state.example.duration * 100;
  $("#time-label").textContent = `${n(time)} / ${n(state.example.duration)} s`;
  document.querySelectorAll(".media-playhead").forEach((line) => { line.style.left = `${percent}%`; });
}

function downloadTsv() {
  if (!state.example) return;
  const filename = state.example.audio.split("/").pop();
  const rows = ["filename\tonset\toffset\tevent_label"];
  state.events.forEach((event) => {
    rows.push(`${escapeTsv(filename)}\t${event.onset.toFixed(3)}\t${event.offset.toFixed(3)}\t${escapeTsv(state.manifest.labels[event.index])}`);
  });
  const blob = new Blob([rows.join("\n") + "\n"], { type: "text/tab-separated-values;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${state.example.id}_threshold_${n(state.threshold)}.tsv`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function setPressed(group, key, value) {
  document.querySelectorAll(`[data-${group}]`).forEach((button) => {
    const active = button.dataset[group] === value;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function setupControls() {
  document.querySelectorAll("[data-lang]").forEach((button) => button.addEventListener("click", () => setLanguage(button.dataset.lang)));
  document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => {
    state.view = button.dataset.view;
    setPressed("view", "view", state.view);
    $("#timeline-view").hidden = state.view !== "timeline";
    $("#events-view").hidden = state.view !== "events";
    if (state.view === "timeline") requestAnimationFrame(drawScore);
  }));
  document.querySelectorAll("[data-scope]").forEach((button) => button.addEventListener("click", () => {
    state.scope = button.dataset.scope;
    setPressed("scope", "scope", state.scope);
    renderResults();
  }));
  $("#threshold").addEventListener("input", (event) => {
    state.threshold = Number(event.target.value);
    renderResults();
  });
  $("#reset-threshold").addEventListener("click", () => {
    state.threshold = 0.5;
    $("#threshold").value = "0.50";
    renderResults();
  });
  $("#class-search").addEventListener("input", renderResults);
  $("#export-button").addEventListener("click", downloadTsv);
  for (const selector of ["#waveform-wrap", "#spectrogram-wrap"]) {
    const surface = $(selector);
    surface.addEventListener("click", (event) => seekFromPointer(event, surface));
    surface.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        const player = $("#audio-player");
        if (player.paused) player.play().catch(() => {}); else player.pause();
      }
    });
  }
  $("#audio-player").addEventListener("timeupdate", updatePlayhead);
  $("#audio-player").addEventListener("seeked", updatePlayhead);
  if (window.ResizeObserver) {
    const observer = new ResizeObserver(() => { drawWaveform(); drawScore(); });
    observer.observe($("#waveform"));
    observer.observe($("#score-canvas"));
  } else {
    window.addEventListener("resize", () => { drawWaveform(); drawScore(); });
  }
  const host = window.location.hostname;
  if (host.endsWith(".github.io")) {
    const owner = host.slice(0, -".github.io".length);
    const repo = window.location.pathname.split("/").filter(Boolean)[0];
    if (owner && repo) $("#readme-link").href = `https://github.com/${owner}/${repo}`;
  }
}

async function main() {
  setupControls();
  setLanguage("zh");
  try {
    const response = await fetch("data/manifest.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.manifest = await response.json();
    const preferred = new URLSearchParams(window.location.search).get("demo");
    await selectExample(state.manifest.examples.some((entry) => entry.id === preferred) ? preferred : state.manifest.examples[0].id);
    if (window.lucide) window.lucide.createIcons();
  } catch (error) {
    $("#loading").hidden = true;
    $("#error").hidden = false;
    $("#error").textContent = `${t("noData")} ${error.message}`;
  }
}

main();
