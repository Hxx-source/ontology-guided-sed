"use strict";

(() => {
  const copy = {
    zh: {
      kicker: "04 / PERFORMANCE EVIDENCE",
      title: "同一条真值，逐步看见性能提升",
      intro: "三条案例来自 AudioSet-Strong 验证集，并由四个阶段各自保存的最佳输出自动筛选。播放原音，比较 Ground truth 与 A0–A3；绿色边框标出本案例对应创新后的输出。",
      truth: "Ground truth · 强标注",
      classPsds: "该类验证集 PSDS1",
      clipF1: "本片段时间 F1",
      operatingPoint: "固定操作点",
      window: "CAMF 窗口",
      missed: "漏检",
      events: "段",
      exact: "本步恢复",
      source: "验证音频",
      note: "逐类 PSDS 来自完整验证集；片段 F1 仅用于解释这条音频的时间重合度。单个案例不是总体性能证明，上方完整消融结果才是总体证据。",
      loading: "正在加载性能证据…",
      error: "性能证据加载失败：",
    },
    en: {
      kicker: "04 / PERFORMANCE EVIDENCE",
      title: "One reference, visible gains at each stage",
      intro: "These AudioSet-Strong validation cases were selected automatically from each stage's saved best output. Play the source, then compare Ground truth with A0–A3; the green outline marks the output after the featured innovation.",
      truth: "Ground truth · strong label",
      classPsds: "Validation PSDS1 for this class",
      clipF1: "Clip temporal F1",
      operatingPoint: "Fixed operating point",
      window: "CAMF window",
      missed: "missed",
      events: "events",
      exact: "recovered here",
      source: "Validation audio",
      note: "Class PSDS is measured on the full validation set. Clip F1 only describes temporal overlap in this recording. A single case is explanatory evidence; the full ablation above is the aggregate evidence.",
      loading: "Loading performance evidence…",
      error: "Could not load performance evidence: ",
    },
  };
  const stageNames = {
    a0: "A0 · DASM",
    a1: "A1 · + OCL",
    a2: "A2 · + CAW",
    a3: "A3 · + CAMF",
  };
  const state = { data: null, selected: 0, lang: "zh" };
  const el = (id) => document.getElementById(id);
  const make = (tag, className, text) => {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined) item.textContent = text;
    return item;
  };
  const t = (key) => copy[state.lang][key];
  const fmt = (value, digits = 2) => Number(value).toFixed(digits);

  function buildShell() {
    const old = el("audio-compare");
    if (!old) return;
    old.hidden = true;
    const section = make("section", "paper-band evidence-band");
    section.id = "performance-evidence";
    const container = make("div", "paper-container");
    container.innerHTML = `
      <div class="paper-section-top evidence-heading">
        <div><span class="paper-index" id="evidence-kicker"></span><h2 id="evidence-title"></h2><p id="evidence-intro"></p></div>
        <div id="evidence-tabs" class="evidence-tabs" role="tablist" aria-label="OCL CAW CAMF evidence"></div>
      </div>
      <div id="evidence-body" class="evidence-body"><p class="evidence-loading"></p></div>`;
    section.append(container);
    old.before(section);
    const nav = document.querySelector('a[href="#audio-compare"]');
    if (nav) nav.href = "#performance-evidence";
  }

  function updateStaticCopy() {
    el("evidence-kicker").textContent = t("kicker");
    el("evidence-title").textContent = t("title");
    el("evidence-intro").textContent = t("intro");
    const loading = document.querySelector(".evidence-loading");
    if (loading) loading.textContent = t("loading");
  }

  function segment(onset, offset, duration, reference = false) {
    const item = make("button", reference ? "evidence-segment reference" : "evidence-segment");
    item.type = "button";
    item.style.left = `${onset / duration * 100}%`;
    item.style.width = `${Math.max(.18, (offset - onset) / duration * 100)}%`;
    item.title = `${fmt(onset)}–${fmt(offset)} s`;
    item.setAttribute("aria-label", item.title);
    item.addEventListener("click", (event) => {
      event.stopPropagation();
      const player = el("evidence-player");
      player.currentTime = onset;
      player.play().catch(() => {});
    });
    return item;
  }

  function timelineRow(label, events, duration, options = {}) {
    const row = make("div", `evidence-row${options.before ? " is-before" : ""}${options.after ? " is-after" : ""}${options.reference ? " is-reference" : ""}`);
    const heading = make("div", "evidence-row-label");
    heading.append(make("strong", "", label));
    if (options.detail) heading.append(make("small", "", options.detail));
    const rail = make("div", "evidence-rail");
    events.forEach(([onset, offset]) => rail.append(segment(onset, offset, duration, options.reference)));
    rail.addEventListener("click", (event) => {
      const bounds = rail.getBoundingClientRect();
      el("evidence-player").currentTime = (event.clientX - bounds.left) / bounds.width * duration;
    });
    let status = events.length ? `${events.length} ${t("events")}` : t("missed");
    if (options.after) status = `${status} · ${t("exact")}`;
    const badge = make("div", "evidence-row-status", status);
    row.append(heading, rail, badge);
    return row;
  }

  function metric(label, value, note) {
    const item = make("div", "evidence-metric");
    item.append(make("span", "", label), make("strong", "", value));
    if (note) item.append(make("small", "", note));
    return item;
  }

  function renderTabs() {
    const root = el("evidence-tabs");
    root.replaceChildren(...state.data.cases.map((item, index) => {
      const button = make("button", index === state.selected ? "active" : "", item.method);
      button.type = "button";
      button.role = "tab";
      button.setAttribute("aria-selected", String(index === state.selected));
      button.addEventListener("click", () => { state.selected = index; render(); });
      return button;
    }));
  }

  function render() {
    if (!state.data) return;
    renderTabs();
    const item = state.data.cases[state.selected];
    const before = item.temporal_metrics[item.before].f1;
    const after = item.temporal_metrics[item.after].f1;
    const body = el("evidence-body");
    body.innerHTML = `
      <div class="evidence-summary">
        <div class="evidence-summary-copy"><span class="evidence-method">${item.method}</span><h3></h3><p></p><small></small></div>
        <div class="evidence-metrics"></div>
      </div>
      <audio id="evidence-player" controls preload="metadata"></audio>
      <div id="evidence-spectrogram" class="evidence-spectrogram" role="button" tabindex="0" aria-label="Seek in spectrogram">
        <img alt="Spectrogram of validation evidence"><div class="media-playhead"></div>
      </div>
      <div class="paper-ruler"><span>0 s</span><span>${fmt(item.duration / 2)} s</span><span>${fmt(item.duration)} s</span></div>
      <div id="evidence-rows" class="evidence-rows"></div>
      <p class="evidence-note"></p>`;
    body.querySelector("h3").textContent = item.title[state.lang];
    body.querySelector(".evidence-summary-copy p").textContent = item.description[state.lang];
    body.querySelector(".evidence-summary-copy small").textContent = `${t("source")}: AudioSet-Strong · ${item.filename}`;
    const metrics = body.querySelector(".evidence-metrics");
    metrics.append(
      metric(t("classPsds"), `${fmt(item.class_psds.before)} → ${fmt(item.class_psds.after)}`, `+${fmt(item.class_psds.delta)} pp`),
      metric(t("clipF1"), `${fmt(before)} → ${fmt(after)}`, `+${fmt(after - before)}`),
      metric(t("operatingPoint"), `τ = ${fmt(state.data.threshold)}`, item.method === "CAMF" ? `${t("window")}: ${item.adaptive_window_frames}f` : "uniform MF · 5f")
    );
    const player = el("evidence-player");
    player.src = item.audio;
    const wrap = el("evidence-spectrogram");
    wrap.querySelector("img").src = item.spectrogram;
    wrap.addEventListener("click", (event) => {
      const bounds = wrap.getBoundingClientRect();
      player.currentTime = (event.clientX - bounds.left) / bounds.width * item.duration;
    });
    player.addEventListener("timeupdate", () => {
      wrap.querySelector(".media-playhead").style.left = `${Math.min(100, player.currentTime / item.duration * 100)}%`;
    });
    const rows = el("evidence-rows");
    rows.append(timelineRow(t("truth"), item.reference, item.duration, { reference: true, detail: item.label }));
    ["a0", "a1", "a2", "a3"].forEach((stage) => {
      const isBefore = stage === item.before;
      const isAfter = stage === item.after;
      const detail = `F1 ${fmt(item.temporal_metrics[stage].f1)}${stage === "a3" ? ` · ${item.adaptive_window_frames}f` : " · 5f"}`;
      rows.append(timelineRow(stageNames[stage], item.events[stage], item.duration, { before: isBefore, after: isAfter, detail }));
    });
    body.querySelector(".evidence-note").textContent = t("note");
  }

  async function init() {
    buildShell();
    updateStaticCopy();
    try {
      const response = await fetch("data/evidence.json");
      if (!response.ok) throw Error(`HTTP ${response.status}`);
      state.data = await response.json();
      render();
    } catch (error) {
      el("evidence-body").textContent = t("error") + String(error);
    }
  }

  document.querySelectorAll("[data-lang]").forEach((button) => button.addEventListener("click", () => {
    state.lang = button.dataset.lang;
    updateStaticCopy();
    render();
  }));
  init();
})();
