const stageNames = {a0: 'DASM baseline', a1: 'DASM + OCL', a2: '+ CAW', a3: '+ CAMF'};
const stageShort = {a0: 'A0', a1: 'A1', a2: 'A2', a3: 'A3'};
const caseNames = {OCL: 'Ontology transfer', CAW: 'Adaptive weighting', CAMF: 'Class-specific timing'};
let cases = [];
let selected = 0;

const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const signed = value => `${value > 0 ? '+' : ''}${value.toFixed(2)}`;

function renderChart(stages) {
  const chart = document.getElementById('stage-chart');
  chart.innerHTML = stages.map((stage, index) => {
    const previous = stages[index - 1];
    const gain = previous ? stage.overall - previous.overall : 0;
    const rareGain = previous ? stage.rare - previous.rare : 0;
    const title = ['DASM', 'OCL', 'CAW', 'CAMF'][index];
    const subtitle = ['Baseline', 'Ontology-guided consistency', 'Class-wise adaptive weighting', 'Class-adaptive median filtering'][index];
    return `<div class="stage-row ${index === 3 ? 'stage-final' : ''}">
      <div class="stage-identity"><span class="stage-index">A${index}</span><div><strong>${title}</strong><small>${subtitle}</small></div></div>
      <div class="stage-measure"><div class="measure-header"><span>OVERALL PSDS</span><strong>${stage.overall.toFixed(2)}%</strong></div><div class="measure-rail"><i style="width:${stage.overall / 55 * 100}%"></i></div><small>${previous ? `${signed(gain)} pp from A${index - 1}` : 'Reference'}</small></div>
      <div class="stage-measure rare-measure"><div class="measure-header"><span>RARE-CLASS PSDS</span><strong>${stage.rare.toFixed(2)}%</strong></div><div class="measure-rail"><i style="width:${stage.rare / 45 * 100}%"></i></div><small>${previous ? `${signed(rareGain)} pp from A${index - 1}` : 'Reference'}</small></div>
    </div>`;
  }).join('');
}

function bars(intervals, duration, kind) {
  return intervals.map(([start, end]) => `<span class="event-bar ${kind}" style="left:${Math.max(0, start / duration * 100)}%;width:${Math.max(0.6, (end - start) / duration * 100)}%" title="${start.toFixed(2)}–${end.toFixed(2)} s"></span>`).join('');
}

function drawWaveform(canvas, values) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext('2d');
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  context.fillStyle = '#0b7c75';
  const count = Math.min(values.length, width);
  for (let column = 0; column < count; column++) {
    const a = Math.floor(column * values.length / count);
    const b = Math.max(a + 1, Math.floor((column + 1) * values.length / count));
    const peak = Math.max(...values.slice(a, b));
    const size = Math.max(1, Math.min(height * .46, peak * height * .46));
    context.fillRect(column * width / count, height / 2 - size, Math.max(1, width / count - .7), size * 2);
  }
}

function renderCase(index) {
  selected = index;
  const item = cases[index];
  document.querySelectorAll('.case-tab').forEach((button, i) => {
    button.classList.toggle('selected', i === index);
    button.setAttribute('aria-selected', String(i === index));
  });
  const beforeF1 = item.temporal_metrics[item.before].f1;
  const afterF1 = item.temporal_metrics[item.after].f1;
  const classPsds = item.class_psds ? `<div class="case-stat"><small>CLASS PSDS · VALIDATION</small><strong>${item.class_psds.before.toFixed(2)} → ${item.class_psds.after.toFixed(2)}%</strong><span class="gain">${signed(item.class_psds.delta)} pp</span></div>` : '';
  const windowStat = item.method === 'CAMF' ? `<div class="case-stat"><small>MEDIAN WINDOW · THIS CLASS</small><strong>5 → ${item.adaptive_window_frames} frames</strong><span>0.10 → ${(item.adaptive_window_frames * .02).toFixed(2)} s</span></div>` : '';
  document.getElementById('evidence-card').innerHTML = `
    <div class="case-intro"><div><span class="case-kicker">CASE 0${index + 1} <i>/</i> ${item.method}</span><h3>${escapeHtml(item.title.en)}</h3><p>${escapeHtml(item.description.en)}</p></div><span class="class-badge">TARGET CLASS <strong>${escapeHtml(item.label)}</strong></span></div>
    <div class="case-stats"><div class="case-stat"><small>CLIP OVERLAP F1 · ${stageShort[item.before]} → ${stageShort[item.after]}</small><strong>${beforeF1.toFixed(3)} → ${afterF1.toFixed(3)}</strong><span class="gain">${signed(afterF1 - beforeF1)}</span></div>${classPsds}${windowStat}</div>
    <div class="audio-visual"><div class="audio-heading"><span>10-SECOND VALIDATION CLIP</span><span id="clock">0.00 / ${item.duration.toFixed(2)} s</span></div><div class="spectrogram"><img src="${encodeURI(item.spectrogram)}" alt="Spectrogram of ${escapeHtml(item.label)} example"><span class="media-cursor" id="spectrogram-cursor"></span></div><canvas id="waveform" aria-label="Audio waveform"></canvas><audio id="case-audio" controls preload="metadata" src="${encodeURI(item.audio)}"></audio></div>
    <div class="timeline"><div class="timeline-heading"><strong>Strong labels vs saved model outputs</strong><span>0.50 event threshold</span></div>
      <div class="track-row truth-row"><div class="track-label"><strong>GROUND TRUTH</strong><small>Strong label</small></div><div class="track">${bars(item.reference, item.duration, 'truth')}</div><div class="track-score">REFERENCE</div></div>
      ${['a0','a1','a2','a3'].map(stage => `<div class="track-row ${stage === item.after ? 'after-row' : ''} ${stage === item.before ? 'before-row' : ''}"><div class="track-label"><strong>${stageShort[stage]} <span>${stageNames[stage]}</span></strong><small>${stage === item.before ? 'Before this step' : stage === item.after ? 'After this step' : 'Saved output'}</small></div><div class="track">${bars(item.events[stage], item.duration, 'prediction')}</div><div class="track-score">F1 ${item.temporal_metrics[stage].f1.toFixed(3)}</div></div>`).join('')}
      <div class="track-axis"><span>0 s</span><span>${(item.duration / 2).toFixed(0)} s</span><span>${item.duration.toFixed(0)} s</span></div></div>
    <p class="case-source">Recording: <code>${escapeHtml(item.filename)}</code> · AudioSet-Strong validation · 20 ms output frame · saved event TSVs</p>`;
  const audio = document.getElementById('case-audio');
  const cursor = document.getElementById('spectrogram-cursor');
  audio.addEventListener('timeupdate', () => {
    const proportion = Math.min(1, audio.currentTime / item.duration);
    cursor.style.left = `${proportion * 100}%`;
    document.getElementById('clock').textContent = `${audio.currentTime.toFixed(2)} / ${item.duration.toFixed(2)} s`;
  });
  document.querySelector('.spectrogram').addEventListener('click', event => {
    const bounds = event.currentTarget.getBoundingClientRect();
    audio.currentTime = Math.min(item.duration, Math.max(0, (event.clientX - bounds.left) / bounds.width * item.duration));
  });
  drawWaveform(document.getElementById('waveform'), item.waveform);
}

async function start() {
  try {
    const [paperResponse, evidenceResponse] = await Promise.all([fetch('data/paper.json'), fetch('data/evidence.json')]);
    if (!paperResponse.ok || !evidenceResponse.ok) throw new Error('Could not load saved research data.');
    const [paper, evidence] = await Promise.all([paperResponse.json(), evidenceResponse.json()]);
    renderChart(paper.stages);
    cases = evidence.cases;
    document.getElementById('case-tabs').innerHTML = cases.map((item, index) => `<button class="case-tab" type="button" role="tab" aria-selected="false" data-index="${index}"><span>0${index + 1} / ${item.method}</span><strong>${escapeHtml(item.label)}</strong><small>${caseNames[item.method]}</small></button>`).join('');
    document.querySelectorAll('.case-tab').forEach(button => button.addEventListener('click', () => renderCase(Number(button.dataset.index))));
    renderCase(0);
    window.addEventListener('resize', () => drawWaveform(document.getElementById('waveform'), cases[selected].waveform));
  } catch (error) {
    document.getElementById('evidence-card').innerHTML = `<p class="load-state">${escapeHtml(error.message)} Open this page through a local web server (for example, <code>python -m http.server</code>), not as a file URL.</p>`;
  }
}

start();
