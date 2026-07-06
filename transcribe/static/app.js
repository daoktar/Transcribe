'use strict';

/* Transcribe frontend — setup → working → reading state machine.
   Talks to the FastAPI backend; native pywebview hooks for file
   picking and saving (WKWebView can't handle Content-Disposition). */

const $ = (id) => document.getElementById(id);

const state = {
  config: {
    supported_extensions: [],
    model_choices: [],
    engine_choices: [],
    qwen_available: false,
    hf_token_set: false,
  },
  files: [],      // browser mode: File objects
  paths: [],      // native mode: absolute paths from pick_files()
  jobId: null,
  filenames: [],
  statuses: [],
  errors: {},     // file index (string) -> message
  results: {},    // file index (number) -> result payload
  current: 0,
  view: 'plain',
  es: null,
  running: false,
};

const isNative = () => !!(window.pywebview && window.pywebview.api);

// ---------------------------------------------------------------- boot

document.addEventListener('DOMContentLoaded', init);
// pywebview injects its API after load — reveal native-only UI then
window.addEventListener('pywebviewready', () => {
  $('save-alongside-wrap').hidden = false;
});

async function init() {
  buildWave();
  wireDropzone();
  wireSettings();
  wireButtons();
  if (isNative()) $('save-alongside-wrap').hidden = false;

  try {
    const res = await fetch('/api/config');
    if (res.ok) state.config = await res.json();
  } catch {
    notice('Could not reach the backend.');
  }
  applyConfig();
}

function applyConfig() {
  const c = state.config;

  const model = $('model');
  model.textContent = '';
  for (const m of c.model_choices) {
    model.append(new Option(m, m, m === 'large-v3', m === 'large-v3'));
  }

  const engine = $('engine');
  engine.textContent = '';
  for (const e of c.engine_choices) {
    const label = e === 'qwen' ? 'qwen3-asr' : 'whisper.cpp';
    const opt = new Option(label, e, e === 'whisper', e === 'whisper');
    if (e === 'qwen' && !c.qwen_available) {
      opt.disabled = true;
      opt.text += ' — unavailable';
    }
    engine.append(opt);
  }

  const exts = c.supported_extensions.map((e) => e.slice(1).toUpperCase());
  $('ext-list').textContent = exts.join(' ');
  $('file-input').accept = c.supported_extensions.join(',');
}

function buildWave() {
  const wave = $('wave');
  for (let i = 0; i < 28; i++) wave.append(document.createElement('span'));
}

// ---------------------------------------------------------------- files

function wireDropzone() {
  const dz = $('dropzone');
  const input = $('file-input');

  dz.addEventListener('click', pickFiles);
  dz.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pickFiles(); }
  });
  input.addEventListener('change', () => {
    addBrowserFiles([...input.files]);
    input.value = '';
  });

  for (const t of ['dragenter', 'dragover']) {
    dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add('drag'); });
  }
  for (const t of ['dragleave', 'drop']) {
    dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove('drag'); });
  }
  dz.addEventListener('drop', (e) => addBrowserFiles([...e.dataTransfer.files]));
}

async function pickFiles() {
  if (isNative() && typeof window.pywebview.api.pick_files === 'function') {
    try {
      const paths = await window.pywebview.api.pick_files();
      if (paths && paths.length) {
        state.files = []; // modes are exclusive
        for (const p of paths) {
          if (!state.paths.includes(p)) state.paths.push(p);
        }
        renderManifest();
      }
    } catch {
      notice('File picker failed.');
    }
    return;
  }
  $('file-input').click();
}

function addBrowserFiles(files) {
  const ok = state.config.supported_extensions;
  let rejected = 0;
  let accepted = 0;
  for (const f of files) {
    const ext = '.' + f.name.split('.').pop().toLowerCase();
    if (!ok.includes(ext)) { rejected++; continue; }
    if (!state.files.some((x) => x.name === f.name && x.size === f.size)) {
      state.files.push(f);
      accepted++;
    }
  }
  if (accepted) state.paths = []; // modes are exclusive
  if (rejected) notice(`Skipped ${rejected} unsupported file${rejected > 1 ? 's' : ''}.`);
  renderManifest();
}

function fileEntries() {
  return state.paths.length
    ? state.paths.map((p) => ({ name: p.split('/').pop(), size: null }))
    : state.files.map((f) => ({ name: f.name, size: f.size }));
}

function removeFile(i) {
  if (state.paths.length) state.paths.splice(i, 1);
  else state.files.splice(i, 1);
  renderManifest();
}

function renderManifest() {
  const list = $('manifest');
  const entries = fileEntries();
  list.textContent = '';
  list.hidden = entries.length === 0;
  $('dropzone').classList.toggle('compact', entries.length > 0);

  entries.forEach((f, i) => {
    const li = document.createElement('li');
    li.append(
      el('span', 'm-idx', String(i + 1).padStart(2, '0')),
      el('span', 'm-name', f.name),
      el('span', 'm-size', f.size != null ? fmtSize(f.size) : 'local file'),
    );
    const rm = el('button', 'm-remove', '×');
    rm.setAttribute('aria-label', `Remove ${f.name}`);
    rm.addEventListener('click', () => removeFile(i));
    li.append(rm);
    list.append(li);
  });

  $('start-btn').disabled = entries.length === 0;
}

function fmtSize(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
  if (bytes < 1024 * 1024 * 1024) return (bytes / 1048576).toFixed(1) + ' MB';
  return (bytes / 1073741824).toFixed(2) + ' GB';
}

function el(tag, cls, text) {
  const n = document.createElement(tag);
  n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

// ---------------------------------------------------------------- settings

// clamp to the backend's 1–20 range; empty/garbage -> null (auto)
function numSpeakers() {
  const n = parseInt($('num-speakers').value, 10);
  return Number.isNaN(n) ? null : Math.min(20, Math.max(1, n));
}

function wireSettings() {
  $('diarize').addEventListener('change', () => {
    const on = $('diarize').checked;
    $('num-speakers').disabled = !on;
    $('token-wrap').hidden = !on || state.config.hf_token_set;
    $('token-note').hidden = !on || !state.config.hf_token_set;
  });
}

// ---------------------------------------------------------------- run

function wireButtons() {
  $('start-btn').addEventListener('click', start);
  $('cancel-btn').addEventListener('click', cancelJob);
  $('work-back-btn').addEventListener('click', () => showStage('setup'));
  $('reset-btn').addEventListener('click', reset);
  $('copy-btn').addEventListener('click', copyTranscript);
  $('download-btn').addEventListener('click', download);
  $('retry-btn').addEventListener('click', retryDiarize);
  for (const btn of $('view-toggle').querySelectorAll('.toggle-btn')) {
    btn.addEventListener('click', () => setView(btn.dataset.view));
  }
}

async function start() {
  const btn = $('start-btn');
  btn.disabled = true;
  btn.textContent = 'Uploading…';

  try {
    const fd = new FormData();
    if (state.paths.length) for (const p of state.paths) fd.append('paths', p);
    else for (const f of state.files) fd.append('files', f, f.name);

    const up = await fetch('/api/upload', { method: 'POST', body: fd });
    const upData = await up.json().catch(() => ({}));
    if (!up.ok) throw new Error(upData.detail || `Upload failed (${up.status})`);

    state.jobId = upData.job_id;
    state.filenames = upData.filenames;

    const body = {
      model: $('model').value,
      engine: $('engine').value,
      language: $('language').value || null,
      diarize: $('diarize').checked,
      hf_token: $('hf-token').value.trim() || null,
      num_speakers: $('diarize').checked ? numSpeakers() : null,
      save_alongside: $('save-alongside').checked,
      original_paths: state.paths.length ? state.paths : null,
    };
    const tr = await fetch(`/api/transcribe/${state.jobId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const trData = await tr.json().catch(() => ({}));
    if (!tr.ok) throw new Error(trData.detail || `Could not start (${tr.status})`);

    state.statuses = state.filenames.map(() => 'pending');
    state.errors = {};
    state.results = {};
    state.running = true;

    const cancel = $('cancel-btn');
    cancel.hidden = false;
    cancel.disabled = false;
    cancel.textContent = 'Cancel';
    $('work-back-btn').hidden = true;

    showStage('work');
    renderWorkManifest();
    setProgress(0, 'Starting…');
    openStream(onRunComplete);
  } catch (err) {
    notice(err.message);
  } finally {
    btn.disabled = fileEntries().length === 0;
    btn.textContent = 'Transcribe →';
  }
}

function openStream(onDone) {
  closeStream();
  const es = new EventSource(`/api/progress/${state.jobId}`);
  state.es = es;
  es.onmessage = (e) => {
    let ev;
    try { ev = JSON.parse(e.data); } catch { return; }

    if (ev.type === 'progress') {
      if (ev.statuses) state.statuses = ev.statuses;
      if (ev.errors) state.errors = ev.errors;
      renderWorkManifest();
      setProgress(ev.fraction, ev.message);
    } else if (ev.type === 'complete' || ev.type === 'error') {
      closeStream();
      onDone(ev);
    }
  };
  // network hiccups: EventSource reconnects on its own; nothing to do
}

function closeStream() {
  if (state.es) { state.es.close(); state.es = null; }
}

async function onRunComplete(ev) {
  state.running = false;
  document.title = 'Transcribe';
  if (ev.statuses) state.statuses = ev.statuses;
  if (ev.errors) state.errors = ev.errors;
  renderWorkManifest();

  if (ev.done_count > 0) {
    state.current = ev.first_result_index ?? 0;
    await loadResult(state.current);
    showStage('read');
    if (ev.cancelled) notice('Cancelled — showing what finished.');
    else if (ev.error_count > 0) notice(`${ev.error_count} file(s) failed — see the list.`);
  } else {
    setProgress(1, ev.cancelled ? 'Cancelled.' : 'Finished with errors.');
    $('cancel-btn').hidden = true;
    $('work-back-btn').hidden = false;
  }
}

function renderWorkManifest() {
  const list = $('work-manifest');
  list.textContent = '';
  const LABELS = {
    pending: 'queued',
    processing: 'working',
    done: 'done ✓',
    error: 'error ✕',
    cancelled: 'cancelled',
  };
  state.filenames.forEach((name, i) => {
    const s = state.statuses[i] || 'pending';
    const li = document.createElement('li');
    li.append(
      el('span', 'm-idx', String(i + 1).padStart(2, '0')),
      el('span', 'm-name', name),
      el('span', `m-status is-${s}`, LABELS[s] || s),
    );
    const err = state.errors[String(i)];
    if (err) li.append(el('p', 'm-err', err));
    list.append(li);
  });
}

function setProgress(fraction, message) {
  const pct = Math.round((fraction || 0) * 100);
  $('progress-fill').style.width = pct + '%';
  $('progress-pct').textContent = pct + '%';
  if (message) $('progress-msg').textContent = message;
  if (state.running) document.title = `${pct}% · Transcribe`;
}

async function cancelJob() {
  if (!state.jobId) return;
  if (!confirm('Stop transcription? Files already finished are kept.')) return;
  const btn = $('cancel-btn');
  btn.disabled = true;
  btn.textContent = 'Cancelling…';
  try {
    await fetch(`/api/cancel/${state.jobId}`, { method: 'POST' });
  } catch {
    notice('Could not reach the backend to cancel.');
    btn.disabled = false;
    btn.textContent = 'Cancel';
  }
}

// ---------------------------------------------------------------- reading

function renderTabs() {
  const nav = $('file-tabs');
  nav.textContent = '';
  nav.hidden = state.filenames.length < 2;
  if (nav.hidden) return;
  state.filenames.forEach((name, i) => {
    const b = el('button', 'tab' + (i === state.current ? ' active' : ''),
      `${String(i + 1).padStart(2, '0')} ${name}`);
    b.disabled = state.statuses[i] !== 'done';
    b.addEventListener('click', () => loadResult(i));
    nav.append(b);
  });
}

async function loadResult(i) {
  state.current = i;
  let r = state.results[i];
  if (!r) {
    try {
      const res = await fetch(`/api/result/${state.jobId}/${i}`);
      if (!res.ok) throw new Error();
      r = await res.json();
      state.results[i] = r;
    } catch {
      notice('Could not load this result.');
      return;
    }
  }
  renderResult(r);
  renderTabs();
}

function renderResult(r) {
  $('read-title').textContent = r.filename;

  const meta = [];
  if (r.language) meta.push(r.language);
  if (r.engine) meta.push(r.engine);
  if (r.fallback) meta.push('fell back to qwen');
  $('read-meta').textContent = meta.join(' · ');

  $('view-toggle').hidden = !r.has_speakers;
  state.view = r.has_speakers ? 'speakers' : 'plain';
  paintToggle();
  renderTranscript(r);

  const failed = r.diarize_requested && r.diarize_error;
  $('diarize-note').hidden = !failed;
  if (failed) {
    $('diarize-error-msg').textContent = 'Speaker detection failed: ' + r.diarize_error;
    $('retry-token').hidden = state.config.hf_token_set;
  }
}

function setView(view) {
  state.view = view;
  paintToggle();
  renderTranscript(state.results[state.current]);
}

function paintToggle() {
  for (const btn of $('view-toggle').querySelectorAll('.toggle-btn')) {
    btn.classList.toggle('active', btn.dataset.view === state.view);
  }
}

function renderTranscript(r) {
  const box = $('transcript');
  box.textContent = '';
  box.classList.remove('empty', 'plain');

  if (state.view === 'speakers' && r.speaker_text) {
    // merge consecutive lines from the same speaker into one block
    let spk = null;
    let buf = [];
    const flush = () => {
      if (!buf.length) return;
      const p = el('p', 'line');
      p.append(el('span', 'spk', spk || '?'), document.createTextNode(buf.join(' ')));
      box.append(p);
      buf = [];
    };
    for (const line of r.speaker_text.split('\n')) {
      const i = line.indexOf(':');
      const who = i > 0 ? line.slice(0, i).trim() : '?';
      const text = i > 0 ? line.slice(i + 1).trim() : line.trim();
      if (!text) continue;
      if (who !== spk) { flush(); spk = who; }
      buf.push(text);
    }
    flush();
    return;
  }

  const text = (r.text || '').trim();
  if (!text) {
    box.classList.add('empty');
    box.textContent = '— no speech detected —';
    return;
  }
  box.classList.add('plain');
  for (const par of text.split(/\n{2,}/)) box.append(el('p', '', par.trim()));
}

async function copyTranscript() {
  const r = state.results[state.current];
  if (!r) return;
  const text = state.view === 'speakers' && r.speaker_text ? r.speaker_text : r.text;
  const btn = $('copy-btn');
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = 'Copied ✓';
  } catch {
    btn.textContent = 'Copy failed';
  }
  setTimeout(() => { btn.textContent = 'Copy'; }, 1800);
}

async function download() {
  if (!state.jobId) return;
  // Native: WKWebView renders attachments inline, replacing the app window —
  // must go through the pywebview save dialog instead.
  if (isNative()) {
    const api = window.pywebview.api || {};
    if (typeof api.save_transcript !== 'function') {
      notice('Native save is unavailable — try reopening the app.');
      return;
    }
    try {
      const result = await api.save_transcript(`/api/download/${state.jobId}`);
      if (typeof result === 'string' && result.startsWith('error:')) {
        notice('Save failed: ' + result.slice(6).trim());
      }
    } catch {
      notice('Save failed.');
    }
    return;
  }
  window.location.href = `/api/download/${state.jobId}`;
}

async function retryDiarize() {
  const btn = $('retry-btn');
  const token = $('retry-token').value.trim() || $('hf-token').value.trim() || null;
  const body = {
    hf_token: token,
    num_speakers: numSpeakers(),
    file_index: state.current,
  };
  try {
    const res = await fetch(`/api/retry-diarize/${state.jobId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'Could not start retry.');
  } catch (err) {
    notice(err.message);
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Working…';
  openStream(async (ev) => {
    btn.disabled = false;
    btn.textContent = 'Retry speaker detection';
    if (ev.type === 'error') {
      notice(ev.error || 'Speaker detection failed again.');
      return;
    }
    delete state.results[state.current];
    await loadResult(state.current);
  });
}

// ---------------------------------------------------------------- stages

function showStage(name) {
  for (const s of ['setup', 'work', 'read']) {
    $(`stage-${s}`).hidden = s !== name;
  }
  window.scrollTo(0, 0);
}

function reset() {
  closeStream();
  state.files = [];
  state.paths = [];
  state.jobId = null;
  state.filenames = [];
  state.statuses = [];
  state.errors = {};
  state.results = {};
  state.current = 0;
  state.running = false;
  document.title = 'Transcribe';
  renderManifest();
  showStage('setup');
}

// ---------------------------------------------------------------- misc

let noticeTimer = null;

function notice(msg) {
  const n = $('notice');
  n.textContent = msg;
  n.hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { n.hidden = true; }, 7000);
}
