// Local Vision — frontend.
//
// Responsibilities:
//   1. Tab toggle between Prepared / Upload modes
//   2. POST /start with the right form data
//   3. Open SSE to /events; update preview + status + poring + log
//   4. POST /stop on user request

const $ = (sel) => document.querySelector(sel);

// ---------- tab toggle ----------
const tabs = document.querySelectorAll('.tab');
const forms = {
  prepared: $('#prepared-form'),
  upload:   $('#upload-form'),
};
tabs.forEach(t => t.addEventListener('click', () => {
  tabs.forEach(o => o.classList.toggle('active', o === t));
  Object.entries(forms).forEach(([mode, form]) => {
    form.classList.toggle('active', mode === t.dataset.mode);
  });
}));

// ---------- start handlers ----------
$('#prepared-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData();
  fd.set('mode', 'prepared');
  await postStart(fd);
});

$('#upload-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  fd.set('mode', 'upload');
  await postStart(fd);
});

$('#stop-btn').addEventListener('click', async () => {
  await fetch('/stop', { method: 'POST' });
  setActive(false);
});

async function postStart(fd) {
  setStartButtonsDisabled(true);
  try {
    const r = await fetch('/start', { method: 'POST', body: fd });
    if (!r.ok) {
      const text = await r.text();
      alert(`Start failed: ${text}`);
      setStartButtonsDisabled(false);
      return;
    }
    const data = await r.json();
    setActive(true, data);
    resetEventLog();
    openEventStream();
  } catch (err) {
    alert(`Start error: ${err}`);
    setStartButtonsDisabled(false);
  }
}

function setStartButtonsDisabled(disabled) {
  document.querySelectorAll('.mode-form button.primary').forEach(b => b.disabled = disabled);
}

function setActive(running, runData) {
  const status = $('#active-status');
  const text = $('#active-text');
  if (running) {
    status.classList.remove('hidden');
    text.textContent = `Running (${runData.mode}) — "${runData.observation}"`;
  } else {
    status.classList.add('hidden');
    text.textContent = '';
    setStartButtonsDisabled(false);
  }
}

// ---------- SSE ----------
let eventSrc = null;
let firedCount = 0;

function openEventStream() {
  if (eventSrc) { eventSrc.close(); }
  eventSrc = new EventSource('/events');
  eventSrc.addEventListener('hello', () => { /* connected */ });
  eventSrc.addEventListener('end',   () => {
    setActive(false);
    eventSrc.close(); eventSrc = null;
    $('#status-line').textContent = 'done';
  });
  eventSrc.onmessage = (ev) => {
    const u = JSON.parse(ev.data);
    handleUpdate(u);
  };
  eventSrc.onerror = () => {
    // Browser will reconnect automatically; nothing to do
  };
}

// Progressive unit formatter for elapsed time.
//   < 60s     → "12.4s"  (decimal seconds)
//   < 1h      → "9m 24s"
//   >= 1h     → "1h 12m" (seconds drop off once we're in hours)
function formatElapsed(seconds) {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  if (seconds < 3600) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}m ${s}s`;
  }
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

function handleUpdate(u) {
  // 1. Update preview
  if (u.preview_b64) {
    const img = $('#preview-img');
    img.src = `data:image/jpeg;base64,${u.preview_b64}`;
    img.classList.add('loaded');
    $('#preview-placeholder').classList.add('hidden');
  }

  // 2. Status line
  if (u.decision) {
    const verdict = u.decision.condition ? 'TRUE' : 'false';
    $('#status-line').textContent = `t=${formatElapsed(u.ts_virtual_sec)} · ${verdict}`;
    let progress = '';
    if (u.decision.reason) progress = u.decision.reason;
    if (u.true_episode_duration_sec != null) {
      progress = `held for ${u.true_episode_duration_sec.toFixed(1)}s · ${progress}`;
    }
    $('#progress-line').textContent = progress;
  } else {
    $('#status-line').textContent = 'inference error';
  }

  // 3. Negative state: keep the .sad class in sync with the backend
  //    is_negative_state flag. Robust to page reloads — backend is the
  //    single source of truth.
  const poring = $('#poring');
  if (u.is_negative_state) {
    poring.classList.add('sad');
  } else {
    poring.classList.remove('sad');
  }

  // 4. Trigger fire (positive)
  if (u.fired) {
    firedCount += 1;
    $('#fire-count').textContent = firedCount;
    // Always orange when pulsing — backend already flipped is_negative_state
    // to false in this same update so .sad is gone by here.
    pulsePoring();
    addEventRow(firedCount, u.fired);
  }
}

function pulsePoring() {
  const p = $('#poring');
  p.classList.remove('fired');
  // Force reflow so the animation re-fires
  void p.offsetWidth;
  p.classList.add('fired');
  // Remove after the pulse finishes so it can re-trigger cleanly
  setTimeout(() => p.classList.remove('fired'), 2000);
}

function resetEventLog() {
  firedCount = 0;
  $('#fire-count').textContent = '0';
  $('#events-tbody').innerHTML = '<tr class="empty"><td colspan="3" class="muted">(no events yet)</td></tr>';
}

function addEventRow(n, fired) {
  const tbody = $('#events-tbody');
  // Remove the "(no events yet)" row on first add
  const empty = tbody.querySelector('tr.empty');
  if (empty) empty.remove();

  const t = formatElapsed(fired.fired_at_sec);
  const row = document.createElement('tr');
  row.innerHTML = `<td>${n}</td><td>${t}</td><td>${escapeHtml(fired.reason)}</td>`;
  tbody.prepend(row);
}

function escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ---------- on-load: check if a run is already happening (page refresh case) ----------
fetch('/status').then(r => r.json()).then(s => {
  if (s.running) {
    setActive(true, { mode: s.mode, observation: s.observation });
    openEventStream();
  }
});
