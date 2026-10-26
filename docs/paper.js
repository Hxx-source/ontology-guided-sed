"use strict";

(() => {
  const el = (id) => document.getElementById(id);
  const node = (tag, className, value) => {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (value !== undefined) item.textContent = value;
    return item;
  };
  const language = {
    zh: {
      navOverview: "研究概览", navEvidence: "验证结果", navMethod: "方法", navCompare: "音频对照", navExplore: "完整试听",
      subtitle: "面向开放词汇声音事件检测的层级知识迁移与类别自适应后处理",
      lede: "以 DASM 为基础，让数据稀缺的子类借助可靠父类改善查询表示，并让每个类别按自身事件时长选择平滑窗口。下方把论文结果、训练关系和真实音频预测分开呈现。",
      draft: "研究稿件演示", overallMetric: "整体 PSDS1 (%)", rareMetric: "稀缺类别 PSDS1 (%)",
      pairMetric: "可靠父子关系 → 受约束子类", pairMetricNote: "支持一个子类对应多个父类",
      benchmarkTitle: "每一步带来了什么", overall: "整体", common: "常见类", rare: "稀缺类",
      benchmarkCaption: "AudioSet-Strong 验证集，PSDS1 设置；仅 381 个类别有有效逐类标注。百分数来自论文表 I，不是本页精选音频的得分。",
      methodTitle: "两条互补的改进路径", oclTitle: "父类作为语义锚点",
      oclBody: "从 AudioSet 本体中筛选可靠父子关系，只在训练时用层级一致性约束查询表示。父类与子类可以同时检出，不是互斥分类。",
      cawTitle: "逐关系调整迁移强度",
      cawBody: "依据子/父训练时长比、父类支持量及查询相似度，为每条关系设置不同权重；一个子类有多个父类时分别计算。",
      camfTitle: "逐类别选择时间尺度",
      camfBody: "从训练事件时长分布估计每类中值滤波窗口：短促事件保留瞬态，持续事件抑制碎片化。无需新增推理参数。",
      pairExplorerTitle: "可靠父子关系", pairExplorerIntro: "选一个子类，查看代码实际使用的父类、训练时长和关系权重。",
      childLabel: "子类", formulaCaution: "实现核对：论文草稿的 CAW 公式仅写时长比；当前发布配置还乘了父类支持量与相似度区间权重。这里显示的是代码中的实际权重，正式发表前应统一论文与代码表述。",
      windowTitle: "一类一个时间窗口", windowClass: "查看类别",
      windowCaption: "窗口取自训练事件时长统计。这里的 17 帧固定窗口用于同口径音频演示；论文 PSDS 数字来自独立的验证流程，不由这些示例计算。",
      compareTitle: "同一段声音，三个处理阶段", recordingLabel: "录音", compareClass: "类别", compareThreshold: "阈值",
      compareCaption: "A0：原 DASM + 固定窗口；A2：OCL+CAW 权重 + 同一固定窗口；A3：同一 OCL+CAW 权重 + CAMF。均使用同一音频、查询和阈值。精选案例不是因果证明或基准评测。静态页面只支持预计算的 407 个音频查询。",
      demosTitle: "完整试听与 407 类检视", demosIntro: "继续查看波形、全部类别时间轴，并导出当前阈值下的事件 TSV。",
      child: "子类", parent: "父类", trainHours: "训练时长", durationRatio: "时长比", similarity: "查询相似度", weight: "关系权重",
      duration: "事件时长", count: "训练事件数", method: "估计方法", fixed: "固定窗口", adaptive: "CAMF 窗口",
      events: "个事件", noEvents: "无事件", rareBreakdown: "此稀缺类的逐阶段 PSDS1", rareUnavailable: "此子类不在稀缺类逐类表中。",
      reference: "人工插入的咳嗽", curve: "逐帧概率", preview: "精选示例，仅供观察；请试听并核对预测事件。"
    },
    en: {
      navOverview: "Overview", navEvidence: "Validation", navMethod: "Method", navCompare: "Audio comparison", navExplore: "Full inspection",
      subtitle: "Ontology-guided knowledge transfer and class-adaptive post-processing for open-vocabulary SED",
      lede: "Building on DASM, reliable parents help data-scarce child queries during training while each class receives a temporal window derived from its own event durations. Paper results, training relationships, and real audio predictions are shown separately.",
      draft: "Research manuscript demo", overallMetric: "Overall PSDS1 (%)", rareMetric: "Rare-class PSDS1 (%)",
      pairMetric: "Reliable relations → constrained children", pairMetricNote: "One child may have multiple parents",
      benchmarkTitle: "Contribution by stage", overall: "Overall", common: "Common", rare: "Rare",
      benchmarkCaption: "AudioSet-Strong validation, PSDS1 protocol. Only 381 classes have valid per-class annotations. Percentages are from manuscript Table I, not these curated audio clips.",
      methodTitle: "Two complementary improvements", oclTitle: "Parents as semantic anchors",
      oclBody: "Reliable AudioSet ontology edges constrain query representations during training. A parent and its child may co-occur; this is not mutually exclusive classification.",
      cawTitle: "Weight each relationship",
      cawBody: "Duration ratio, parent support, and query similarity set a separate transfer weight for each edge. A child with multiple parents receives separate weights.",
      camfTitle: "Choose a scale per class",
      camfBody: "Training-event durations determine each class's median-filter window. Short transients remain visible while sustained events become less fragmented. No new inference parameters.",
      pairExplorerTitle: "Reliable ontology edges", pairExplorerIntro: "Select a child to inspect the actual training edges, durations, and implementation weights.",
      childLabel: "Child", formulaCaution: "Implementation check: the draft CAW formula only states the duration ratio; the released configuration also multiplies parent support and cosine-band weights. This page reports the code weights. Manuscript and implementation need harmonizing before publication.",
      windowTitle: "One temporal window per class", windowClass: "Class",
      windowCaption: "Windows come from training-event duration statistics. The 17-frame fixed comparator is only for matched audio demos; paper PSDS values come from separate validation.",
      compareTitle: "One recording, three stages", recordingLabel: "Recording", compareClass: "Class", compareThreshold: "Threshold",
      compareCaption: "A0: original DASM + fixed window; A2: OCL+CAW checkpoint + same fixed window; A3: same checkpoint + CAMF. Audio, query, and threshold are matched. Curated clips are neither causal proof nor a benchmark. This static page only supports 407 precomputed audio queries.",
      demosTitle: "Listen and inspect all 407 classes", demosIntro: "Inspect waveforms, every class timeline, and export detected events as TSV.",
      child: "Child", parent: "Parent", trainHours: "Training hours", durationRatio: "Duration ratio", similarity: "Query cosine", weight: "Edge weight",
      duration: "Event duration", count: "Training events", method: "Estimator", fixed: "Fixed window", adaptive: "CAMF window",
      events: "events", noEvents: "No events", rareBreakdown: "Per-stage PSDS1 for this rare class", rareUnavailable: "This child is not in the rare-class table.",
      reference: "Inserted cough (known)", curve: "Frame probability", preview: "Curated case for inspection; listen and verify the detections."
    }
  };
  const stageNames = ["A0 · DASM", "A1 · + OCL", "A2 · + OCL + CAW", "A3 · + CAMF"];
  const scoreStages = [
    { id: "a0_fixed", name: "A0 · DASM", window: "17f" },
    { id: "a2_fixed", name: "A2 · OCL + CAW", window: "17f" },
    { id: "a3", name: "A3 · + CAMF", window: null }
  ];
  const paperState = {
    lang: "zh", paper: null, manifest: null, metric: "overall",
    example: null, classLabel: null, scores: null, threshold: 0.5,
    curveStage: "a3", loadId: 0
  };
  const t = (key) => language[paperState.lang][key];
  const fmt = (value, digits = 2) => Number(value).toFixed(digits);
  const signed = (value) => (value >= 0 ? "+" : "") + fmt(value);
  const select = (element, value) => { element.value = value; };
  const lookupWindow = (name) => paperState.paper.windows.find((row) => row.event_label === name);

  function applyLanguage() {
    document.querySelectorAll("[data-paper-key]").forEach((item) => {
      item.textContent = t(item.dataset.paperKey);
    });
    renderAblation();
    renderPair();
    renderWindow();
    renderAudio();
  }

  function renderAblation() {
    const root = el("ablation-chart");
    if (!paperState.paper) return;
    const base = paperState.paper.stages[0][paperState.metric];
    root.replaceChildren(...paperState.paper.stages.map((stage, index) => {
      const row = node("div", "ablation-row");
      const label = node("div", "ablation-label", stageNames[index]);
      const rail = node("div", "ablation-rail");
      const bar = node("div", "ablation-bar");
      bar.style.width = `${Math.min(100, stage[paperState.metric] / 60 * 100)}%`;
      rail.append(bar);
      const value = node("div", "ablation-value", fmt(stage[paperState.metric]));
      value.append(node("small", "", index ? `${signed(stage[paperState.metric] - base)} pp` : "baseline"));
      row.append(label, rail, value);
      return row;
    }));
    document.querySelectorAll("[data-metric]").forEach((button) => {
      const active = button.dataset.metric === paperState.metric;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }

  function renderPair() {
    if (!paperState.paper) return;
    const child = el("paper-child-select").value;
    const pairs = paperState.paper.pairs.filter((row) => row.child_label === child);
    const root = el("pair-explorer");
    root.replaceChildren(...pairs.map((pair) => {
      const row = node("div", "pair-row");
      const childNode = node("div", "pair-node");
      childNode.append(node("span", "", t("child")), node("strong", "", pair.child_label),
        node("small", "", `${fmt(pair.child_train_duration_hrs, 3)} h · ${pair.child_train_events} ${t("events")}`));
      const parentNode = node("div", "pair-node");
      parentNode.append(node("span", "", t("parent")), node("strong", "", pair.parent_label),
        node("small", "", `${fmt(pair.parent_train_duration_hrs, 3)} h · ${pair.parent_train_events} ${t("events")}`));
      const details = node("div", "pair-details");
      [
        [t("durationRatio"), fmt(pair.child_parent_duration_ratio, 4)],
        [t("similarity"), fmt(pair.pair_cosine, 3)],
        [t("weight"), fmt(pair.pair_weight, 3)],
        ["CAW", `${fmt(pair.duration_weight, 2)} × ${fmt(pair.parent_support_weight, 2)} × ${fmt(pair.cosine_weight, 2)}`]
      ].forEach(([key, value]) => {
        const cell = node("div");
        cell.append(node("span", "", key), node("strong", "", value));
        details.append(cell);
      });
      row.append(childNode, node("div", "pair-arrow", "→"), parentNode, details);
      return row;
    }));
    const rare = paperState.paper.rare_classes.find((entry) => entry.event_label === child);
    const rareRoot = el("rare-ablation");
    rareRoot.replaceChildren();
    if (!rare) {
      rareRoot.append(node("p", "", t("rareUnavailable")));
      return;
    }
    rareRoot.append(node("strong", "", t("rareBreakdown")));
    const bars = node("div", "rare-bars");
    ["A0 Baseline", "A1 HCL", "A2 HCL+APW", "A4 Full"].forEach((key, index) => {
      const cell = node("div");
      cell.append(node("b", "", fmt(rare[key])), node("span", "", `${["A0", "A1", "A2", "A3"][index]} · ${index ? stageNames[index].split("· ")[1] : "DASM"}`));
      bars.append(cell);
    });
    rareRoot.append(bars);
  }

  function renderWindow() {
    if (!paperState.paper) return;
    const row = lookupWindow(el("paper-window-select").value);
    if (!row) return;
    const frames = Number(row.median_window_frames);
    [["window-fixed", t("fixed"), 17], ["window-adaptive", t("adaptive"), frames]].forEach(([id, label, count]) => {
      const root = el(id);
      const box = node("div", "window-bar-row");
      const rail = node("div", "window-bar-rail");
      const fill = node("div", "window-bar-fill");
      fill.style.width = `${Math.min(100, count / 101 * 100)}%`;
      rail.append(fill);
      box.append(node("span", "", label), rail, node("strong", "", `${count}f · ${fmt(count * 0.02)}s`));
      root.replaceChildren(box);
    });
    const detail = el("window-detail");
    detail.replaceChildren();
    [[t("duration"), `${fmt(row.duration_seconds, 3)} s`],
      [t("count"), String(row.event_count)],
      [t("method"), row.duration_method.replaceAll("_", " ")]].forEach(([label, value]) => {
      const box = node("div");
      box.append(node("span", "", label), node("strong", "", value));
      detail.append(box);
    });
  }

  function readEvents(array, classIndex, frames, threshold, duration) {
    const events = [];
    let start = -1;
    for (let frame = 0; frame <= frames; frame++) {
      const positive = frame < frames && array[classIndex * frames + frame] >= threshold;
      if (positive && start < 0) start = frame;
      if (!positive && start >= 0) {
        events.push({ start: start * 0.02, end: Math.min(duration, frame * 0.02) });
        start = -1;
      }
    }
    return events;
  }

  function renderAudio() {
    const example = paperState.example;
    if (!example || !paperState.scores) return;
    const index = paperState.manifest.labels.indexOf(paperState.classLabel);
    if (index < 0) return;
    const root = el("paper-stage-rows");
    root.replaceChildren(...scoreStages.map((stage) => {
      const events = readEvents(paperState.scores[stage.id], index, example.frames, paperState.threshold, example.duration);
      const row = node("div", "paper-stage-row");
      const label = node("div", "paper-stage-label", stage.name);
      const win = stage.window || `${lookupWindow(paperState.classLabel)?.median_window_frames || "?"}f`;
      label.append(node("small", "", win));
      const rail = node("div", "paper-stage-rail");
      events.forEach((event) => {
        const span = node("div", "paper-stage-segment");
        span.style.left = `${event.start / example.duration * 100}%`;
        span.style.width = `${Math.max(0.15, (event.end - event.start) / example.duration * 100)}%`;
        span.title = `${fmt(event.start)}–${fmt(event.end)} s`;
        span.addEventListener("click", (e) => {
          e.stopPropagation();
          el("paper-audio-player").currentTime = event.start;
        });
        rail.append(span);
      });
      rail.addEventListener("click", (e) => {
        const bounds = rail.getBoundingClientRect();
        el("paper-audio-player").currentTime = Math.max(0, Math.min(example.duration, (e.clientX - bounds.left) / bounds.width * example.duration));
      });
      row.append(label, rail, node("div", "paper-stage-count", events.length ? `${events.length} ${t("events")}` : t("noEvents")));
      return row;
    }));
    el("paper-curve-title").textContent = `${paperState.classLabel} · ${t("curve")}`;
    const counts = scoreStages.map((stage) =>
      readEvents(paperState.scores[stage.id], index, example.frames, paperState.threshold, example.duration).length);
    el("paper-case-note").textContent = `${t("preview")} ${scoreStages.map((stage, i) => `${stage.name.split(" · ")[0]}: ${counts[i]}`).join("  ·  ")} ${t("events")}`;
    drawCurve();
  }

  function drawCurve() {
    const canvas = el("paper-score-canvas");
    const example = paperState.example;
    if (!example || !paperState.scores) return;
    const index = paperState.manifest.labels.indexOf(paperState.classLabel);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || index < 0) return;
    const scale = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const ctx = canvas.getContext("2d");
    ctx.scale(scale, scale);
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "#d3dfd9";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const thresholdY = (1 - paperState.threshold) * (height - 16) + 8;
    ctx.setLineDash([4, 4]);
    ctx.moveTo(0, thresholdY);
    ctx.lineTo(width, thresholdY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#6c7b76";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(fmt(paperState.threshold), 6, Math.max(12, thresholdY - 4));
    ctx.strokeStyle = paperState.curveStage === "a3" ? "#d45e50" : paperState.curveStage === "a2_fixed" ? "#177e78" : "#8aa9a0";
    ctx.lineWidth = 2;
    ctx.beginPath();
    const scores = paperState.scores[paperState.curveStage];
    for (let frame = 0; frame < example.frames; frame++) {
      const x = frame / (example.frames - 1) * width;
      const y = (1 - scores[index * example.frames + frame]) * (height - 16) + 8;
      if (frame === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  async function selectExample(id) {
    const example = paperState.manifest.examples.find((item) => item.id === id);
    if (!example) return;
    const loadId = ++paperState.loadId;
    paperState.example = example;
    paperState.scores = null;
    el("paper-stage-rows").textContent = paperState.lang === "zh" ? "正在加载对照分数…" : "Loading matched scores…";
    select(el("paper-audio-select"), id);
    el("paper-audio-player").src = example.audio;
    el("paper-spectrogram").src = example.spectrogram;
    el("paper-half-time").textContent = `${fmt(example.duration / 2)} s`;
    el("paper-full-time").textContent = `${fmt(example.duration)} s`;
    const band = el("paper-reference-band");
    const inserted = id.startsWith("rain_cough_");
    band.hidden = !inserted;
    if (inserted) {
      band.style.left = `${4 / example.duration * 100}%`;
      band.style.width = `${0.66 / example.duration * 100}%`;
      band.title = t("reference");
    }
    el("paper-audio-class").replaceChildren();
    const ordered = [...example.focus, ...paperState.manifest.labels.filter((label) => !example.focus.includes(label))];
    ordered.forEach((label) => {
      const option = node("option", "", label);
      option.value = label;
      el("paper-audio-class").append(option);
    });
    paperState.classLabel = example.focus[0];
    select(el("paper-audio-class"), paperState.classLabel);
    try {
      const paths = [example.paper_scores.a0_fixed, example.paper_scores.a2_fixed, example.scores];
      const buffers = await Promise.all(paths.map(async (path) => {
        const response = await fetch(path);
        if (!response.ok) throw Error(`HTTP ${response.status}: ${path}`);
        return response.arrayBuffer();
      }));
      if (loadId !== paperState.loadId) return;
      const expected = paperState.manifest.labels.length * example.frames;
      const arrays = buffers.map((buffer) => new Float32Array(buffer));
      if (arrays.some((array) => array.length !== expected)) throw Error("Score dimensions do not match the manifest.");
      paperState.scores = { a0_fixed: arrays[0], a2_fixed: arrays[1], a3: arrays[2] };
      renderAudio();
    } catch (error) {
      if (loadId !== paperState.loadId) return;
      el("paper-stage-rows").textContent = String(error);
    }
  }

  function updatePlayhead() {
    const player = el("paper-audio-player");
    const duration = paperState.example?.duration || 1;
    el("paper-playhead").style.left = `${Math.min(100, player.currentTime / duration * 100)}%`;
  }

  async function init() {
    try {
      const [paperResponse, manifestResponse] = await Promise.all([fetch("data/paper.json"), fetch("data/manifest.json")]);
      if (!paperResponse.ok || !manifestResponse.ok) throw Error("Paper demo data could not be loaded.");
      paperState.paper = await paperResponse.json();
      paperState.manifest = await manifestResponse.json();
      const children = [...new Set(paperState.paper.pairs.map((row) => row.child_label))].sort((a, b) => a.localeCompare(b));
      children.forEach((name) => {
        const option = node("option", "", name);
        option.value = name;
        el("paper-child-select").append(option);
      });
      select(el("paper-child-select"), "Squish");
      paperState.paper.windows.forEach((row) => {
        const option = node("option", "", row.event_label);
        option.value = row.event_label;
        el("paper-window-select").append(option);
      });
      select(el("paper-window-select"), "Tick-tock");
      paperState.manifest.examples.forEach((example) => {
        const option = node("option", "", example.title[paperState.lang]);
        option.value = example.id;
        el("paper-audio-select").append(option);
      });
      applyLanguage();
      await selectExample("clock");
    } catch (error) {
      el("ablation-chart").textContent = String(error);
    }
  }

  document.querySelectorAll("[data-metric]").forEach((button) => button.addEventListener("click", () => {
    paperState.metric = button.dataset.metric;
    renderAblation();
  }));
  document.querySelectorAll("[data-paper-stage]").forEach((button) => button.addEventListener("click", () => {
    paperState.curveStage = button.dataset.paperStage;
    document.querySelectorAll("[data-paper-stage]").forEach((item) => {
      const active = item === button;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", String(active));
    });
    drawCurve();
  }));
  document.querySelectorAll("[data-lang]").forEach((button) => button.addEventListener("click", () => {
    paperState.lang = button.dataset.lang;
    if (paperState.manifest) {
      paperState.manifest.examples.forEach((example, index) => {
        el("paper-audio-select").options[index].textContent = example.title[paperState.lang];
      });
    }
    applyLanguage();
  }));
  el("paper-child-select").addEventListener("change", renderPair);
  el("paper-window-select").addEventListener("change", renderWindow);
  el("paper-audio-select").addEventListener("change", (event) => selectExample(event.target.value));
  el("paper-audio-class").addEventListener("change", (event) => {
    paperState.classLabel = event.target.value;
    renderAudio();
  });
  el("paper-threshold").addEventListener("input", (event) => {
    paperState.threshold = Number(event.target.value);
    el("paper-threshold-value").textContent = fmt(paperState.threshold);
    renderAudio();
  });
  el("paper-spectrogram-wrap").addEventListener("click", (event) => {
    if (!paperState.example) return;
    const rect = event.currentTarget.getBoundingClientRect();
    el("paper-audio-player").currentTime = Math.max(0, Math.min(paperState.example.duration, (event.clientX - rect.left) / rect.width * paperState.example.duration));
  });
  el("paper-audio-player").addEventListener("timeupdate", updatePlayhead);
  window.addEventListener("resize", drawCurve);
  init();
})();
