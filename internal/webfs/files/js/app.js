// NeuroStudio SPA — vanilla ES modules, no build step, no external deps.
// Everything runs locally against the Go API on the same origin.

const $app = document.getElementById('main');
const $nav = document.getElementById('nav');

const S = {
  token: localStorage.getItem('ns-token') || '',
  info: null,
  project: null,
  tab: 'datasets',
  dataset: null,
  job: null,
  es: null, // EventSource handle
};

// ---------- api ----------
class ApiError extends Error {
  constructor(status, code, message) { super(message); this.status = status; this.code = code; }
}
async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (S.token) headers['Authorization'] = 'Bearer ' + S.token;
  if (opts.json !== undefined) {
    headers['Content-Type'] = 'application/json';
    opts = Object.assign({}, opts, { body: JSON.stringify(opts.json) });
    delete opts.json;
  }
  const res = await fetch('/api/v1' + path, Object.assign({}, opts, { headers }));
  if (res.status === 401) { askToken(true); throw new ApiError(401, 'unauthorized', 'API token required (remote access mode)'); }
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    throw new ApiError(res.status, data?.code || 'error', data?.message || res.statusText);
  }
  return data;
}

// ---------- dom helpers ----------
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') el.className = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return el;
}
function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }
function toast(msg, kind = '') {
  const t = h('div', { class: 'toast ' + kind }, msg);
  document.getElementById('toast-root').append(t);
  setTimeout(() => t.remove(), 4200);
}
function fmtBytes(n) {
  if (n == null || isNaN(n)) return '—';
  const u = 1024; if (n < u) return n + ' B';
  let d = u, e = 0; for (let m = n / u; m >= u; m /= u) { d *= u; e++; }
  return (n / d).toFixed(1) + ' KMGTP'[e] + 'B';
}
function fmtSec(s) {
  if (s == null || !isFinite(s)) return '—';
  s = Math.round(s);
  if (s < 60) return s + 's';
  return Math.floor(s / 60) + 'm ' + (s % 60) + 's';
}
function fmtPct(v) { return (v * 100).toFixed(1) + '%'; }
function badge(status) { return h('span', { class: 'badge ' + status }, status); }
function esc(s) { const d = h('div'); d.textContent = s == null ? '' : String(s); return d.innerHTML; }

// ---------- modal ----------
function modal(title, bodyEl) {
  const mask = h('div', { class: 'modal-mask', onclick: (e) => { if (e.target === mask) mask.remove(); } });
  const box = h('div', { class: 'modal' }, h('h3', {}, title), bodyEl);
  mask.append(box);
  document.body.append(mask);
  return mask;
}

// ---------- token ----------
function askToken(isError) {
  const form = h('div', {},
    h('p', { class: 'muted' }, 'This server runs in remote-access mode. Enter the API token printed in the server console.'),
    isError ? h('div', { class: 'err-box' }, 'Request rejected: missing or wrong token.') : null,
  );
  const input = h('input', { type: 'password', placeholder: 'API token' });
  form.append(input);
  const mask = modal('API token', form);
  const save = () => {
    S.token = input.value.trim();
    localStorage.setItem('ns-token', S.token);
    mask.remove(); route();
  };
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') save(); });
  form.append(h('div', { class: 'actions' }, h('button', { class: 'btn btn-primary', onclick: save }, 'Save')));
  input.focus();
}
document.getElementById('token-btn').addEventListener('click', () => askToken(false));

// ---------- navigation ----------
function setNav(items) {
  clear($nav);
  for (const it of items) {
    if (it.sep) { $nav.append(h('div', { class: 'nav-sep' }, it.sep)); continue; }
    $nav.append(h('button', {
      class: 'nav-item' + (it.active ? ' active' : ''),
      onclick: it.onclick,
    }, h('span', { class: 'ico' }, it.ico), it.label));
  }
}

async function boot() {
  try { S.info = await api('/system/info'); } catch { S.info = null; }
  if (S.info) {
    document.getElementById('version-line').textContent = 'v' + S.info.version + ' · ' + S.info.os + '/' + S.info.arch;
    const up = () => { document.getElementById('uptime-line').textContent = 'up ' + fmtSec((Date.now() - new Date(S.info.startedAt).getTime()) / 1000); };
    up(); setInterval(up, 1000);
  }
  window.addEventListener('hashchange', route);
  route();
}

// ---------- router ----------
function route() {
  if (S.es) { S.es.close(); S.es = null; }
  const hash = location.hash.replace(/^#\/?/, '') || 'projects';
  const parts = hash.split('/');
  if (parts[0] === 'projects') return viewProjects();
  if (parts[0] === 'project' && parts[1]) {
    const tab = parts[2] || 'datasets';
    const sel = parts[3] || null;
    return viewProject(parts[1], tab, sel);
  }
  if (parts[0] === 'system') return viewSystem();
  return viewProjects();
}
function go(hash) { location.hash = '#/' + hash.replace(/^\//, ''); }

// ---------- projects ----------
async function viewProjects() {
  setNav([
    { ico: '◱', label: 'Projects', active: true, onclick: () => go('projects') },
    { sep: 'Settings' },
    { ico: '♡', label: 'System', onclick: () => go('system') },
  ]);
  clear($app).append(h('h1', { class: 'page-title' }, 'Projects'),
    h('p', { class: 'page-sub' }, 'Every project keeps its own datasets, experiments, models and history on your machine.'));
  const wrap = h('div');
  $app.append(wrap);
  let projects = [];
  try { projects = await api('/projects'); } catch (e) { wrap.append(h('div', { class: 'err-box' }, e.message)); return; }

  const bar = h('div', { class: 'row', style: 'margin-bottom:16px' },
    h('button', { class: 'btn btn-primary', onclick: () => newProjectDialog() }, '+ New project'),
    h('span', { class: 'faint' }, 'Local-first: no account, no cloud, no telemetry.'));
  wrap.append(bar);

  if (!projects.length) {
    wrap.append(h('div', { class: 'empty' },
      h('div', { style: 'font-size:30px;margin-bottom:8px' }, '◆'),
      'No projects yet. Create one, then generate a sample shapes dataset to try the full workflow.'));
    return;
  }
  const grid = h('div', { class: 'grid-3' });
  for (const p of projects) {
    const counts = p.counts || {};
    grid.append(h('div', { class: 'card', style: 'cursor:pointer', onclick: () => go(`project/${p.id}`) },
      h('h3', {}, p.name),
      h('div', { class: 'faint', style: 'margin-bottom:10px' }, p.taskType || 'image_classification'),
      p.description ? h('div', { class: 'muted', style: 'margin-bottom:10px;font-size:12.5px' }, p.description) : null,
      h('div', { class: 'row', style: 'gap:16px;font-size:12px;color:var(--text-faint)' },
        h('span', {}, '⛁ ' + (counts.datasets ?? '·') + ' datasets'),
        h('span', {}, '⚙ ' + (counts.experiments ?? '·') + ' runs'),
        h('span', {}, '▣ ' + (counts.models ?? '·') + ' models')),
    ));
  }
  wrap.append(grid);
}

function newProjectDialog() {
  const name = h('input', { placeholder: 'Project name' });
  const desc = h('input', { placeholder: 'What are you building? (optional)' });
  const task = h('select', {},
    h('option', { value: 'image_classification' }, 'Image classification (supported now)'),
    h('option', { value: 'object_detection', disabled: true }, 'Object detection (coming later)'),
    h('option', { value: 'audio_classification', disabled: true }, 'Audio classification (coming later)'),
    h('option', { value: 'text_classification', disabled: true }, 'Text classification (coming later)'));
  const form = h('div', {},
    h('div', { class: 'form-row' }, h('label', { class: 'field' }, h('span', {}, 'Name'), name)),
    h('div', { class: 'form-row' }, h('label', { class: 'field' }, h('span', {}, 'Description'), desc)),
    h('div', { class: 'form-row' }, h('label', { class: 'field' }, h('span', {}, 'Task type'), task)),
    h('div', { class: 'hint' }, 'NeuroStudio only advertises tasks it can genuinely train. Others unlock as backends land.'));
  const mask = modal('New project', form);
  const create = async () => {
    try {
      const p = await api('/projects', { method: 'POST', json: { name: name.value.trim(), description: desc.value.trim(), taskType: task.value } });
      mask.remove(); toast('Project created', 'ok'); go('project/' + p.id);
    } catch (e) { toast(e.message, 'err'); }
  };
  name.addEventListener('keydown', (e) => { if (e.key === 'Enter') create(); });
  form.append(h('div', { class: 'actions' },
    h('button', { class: 'btn', onclick: () => mask.remove() }, 'Cancel'),
    h('button', { class: 'btn btn-primary', onclick: create }, 'Create')));
  name.focus();
}

// ---------- project workspace ----------
const TABS = [
  ['datasets', '⛁ Datasets'],
  ['train', '✦ Train'],
  ['jobs', '⏱ Jobs'],
  ['experiments', '≡ Experiments'],
  ['models', '▣ Models'],
  ['playground', '◎ Playground'],
];

async function viewProject(id, tab, sel) {
  setNav([
    { ico: '◱', label: 'Projects', onclick: () => go('projects') },
    { sep: 'Project' },
    { ico: '◈', label: S.project?.name || '…', active: true, onclick: () => {} },
    { sep: 'Settings' },
    { ico: '♡', label: 'System', onclick: () => go('system') },
  ]);
  clear($app);
  try { S.project = await api(`/projects/${id}`); } catch (e) {
    $app.append(h('div', { class: 'err-box' }, e.message)); return;
  }
  setNav([
    { ico: '◱', label: 'Projects', onclick: () => go('projects') },
    { sep: S.project.name },
    { ico: '◈', label: 'Overview', active: true, onclick: () => {} },
    { sep: 'Settings' },
    { ico: '♡', label: 'System', onclick: () => go('system') },
  ]);
  S.tab = tab;
  $app.append(
    h('div', { class: 'crumb' }, h('a', { onclick: () => go('projects') }, 'Projects'), ' / ', S.project.name),
    h('div', { class: 'row spread' },
      h('h1', { class: 'page-title' }, S.project.name),
      h('span', { class: 'faint' }, 'task: ' + S.project.taskType)),
    h('p', { class: 'page-sub' }, S.project.description || ''),
  );
  const tabs = h('div', { class: 'tabs' });
  for (const [key, label] of TABS) {
    tabs.append(h('button', { class: 'tab' + (tab === key ? ' active' : ''), onclick: () => go(`project/${id}/${key}`) }, label));
  }
  $app.append(tabs);
  const body = h('div'); $app.append(body);
  if (tab === 'datasets') await tabDatasets(body, sel);
  else if (tab === 'train') await tabTrain(body);
  else if (tab === 'jobs') await tabJobs(body, sel);
  else if (tab === 'experiments') await tabExperiments(body);
  else if (tab === 'models') await tabModels(body);
  else if (tab === 'playground') await tabPlayground(body);
}

// ---------- dataset studio ----------
async function tabDatasets(root, sel) {
  const pid = S.project.id;
  let datasets = [];
  try { datasets = await api(`/projects/${pid}/datasets`); } catch (e) { root.append(h('div', { class: 'err-box' }, e.message)); return; }

  const head = h('div', { class: 'row', style: 'margin-bottom:14px' },
    h('button', { class: 'btn btn-primary btn-sm', onclick: () => newDatasetDialog() }, '+ New dataset'),
    h('button', { class: 'btn btn-sm', onclick: () => sampleDatasetDialog() }, '✨ Generate sample shapes'),
    h('span', { class: 'faint' }, 'Tip: a ZIP with one folder per class imports with labels automatically.'));
  root.append(head);

  if (!datasets.length) { root.append(h('div', { class: 'empty' }, 'No datasets yet — create one or generate the sample shapes dataset.')); return; }

  if (sel == null) {
    sel = datasets[0].id;
  }
  const exists = datasets.some((d) => d.id === sel);
  if (!exists) sel = datasets[0].id;

  const layout = h('div', { class: 'grid-2', style: 'grid-template-columns:280px 1fr' });
  root.append(layout);
  const listCol = h('div');
  const detailCol = h('div');
  layout.append(listCol, detailCol);

  for (const d of datasets) {
    const card = h('div', { class: 'card', style: 'cursor:pointer;padding:12px 14px;' + (d.id === sel ? 'border-color:var(--accent)' : ''),
      onclick: () => go(`project/${pid}/datasets/${d.id}`) },
      h('div', { class: 'row spread' }, h('strong', {}, d.name), h('span', { class: 'faint' }, d.dataType)),
      h('div', { class: 'faint', style: 'margin-top:4px' }, `${d.sampleCount} samples · ${d.classCount} classes`));
    listCol.append(card);
  }

  const ds = datasets.find((d) => d.id === sel);
  S.dataset = ds;
  let stats = null;
  try { stats = await api(`/datasets/${ds.id}/stats`); } catch { /* ignore */ }

  const panel = h('div', { class: 'card' });
  detailCol.append(panel);
  panel.append(
    h('div', { class: 'row spread' },
      h('h3', {}, ds.name),
      h('button', { class: 'btn btn-danger btn-sm', onclick: () => deleteDataset(ds) }, 'Delete dataset')),
    ds.notes ? h('div', { class: 'faint', style: 'margin-bottom:8px' }, ds.notes) : null,
    stats ? h('div', { class: 'row', style: 'gap:18px;margin-bottom:8px' },
      h('span', { class: 'faint' }, `samples: ${stats.totalSamples}`),
      h('span', { class: 'faint' }, `size: ${fmtBytes(stats.totalBytes)}`),
      h('span', { class: 'faint' }, `corrupt: ${stats.byStatus?.corrupt || 0}`),
      h('span', { class: 'faint' }, `duplicates: ${stats.byStatus?.duplicate || 0}`)) : null,
  );

  // classes row
  const classRow = h('div', { class: 'row', style: 'flex-wrap:wrap;gap:8px;margin:10px 0 16px' });
  for (const c of ds.classes || []) {
    classRow.append(h('span', { class: 'badge none', style: `border:1px solid ${c.color || 'var(--border)'};color:${c.color || 'var(--text-dim)'}` },
      c.name + ' (' + (c.samples || '·') + ')'));
  }
  const newCls = h('input', { placeholder: 'new class name', style: 'width:150px' });
  newCls.addEventListener('keydown', async (e) => {
    if (e.key === 'Enter' && newCls.value.trim()) {
      try { await api(`/datasets/${ds.id}/classes`, { method: 'POST', json: { name: newCls.value.trim() } }); refresh(); } catch (err) { toast(err.message, 'err'); }
    }
  });
  classRow.append(newCls, h('button', { class: 'btn btn-sm', onclick: async () => {
    if (!newCls.value.trim()) return;
    try { await api(`/datasets/${ds.id}/classes`, { method: 'POST', json: { name: newCls.value.trim() } }); refresh(); } catch (err) { toast(err.message, 'err'); }
  } }, '+ add class'));
  panel.append(classRow);

  // import row
  const classSel = h('select', { style: 'width:auto' }, h('option', { value: '' }, '— no class —'),
    ...(ds.classes || []).map((c) => h('option', { value: c.name }, c.name)));
  const fileInput = h('input', { type: 'file', multiple: true, accept: 'image/*', style: 'display:none',
    onchange: () => importFiles(fileInput.files, classSel.value) });
  const zipInput = h('input', { type: 'file', accept: '.zip', style: 'display:none', onchange: () => importZip(zipInput.files[0]) });
  panel.append(h('div', { class: 'row', style: 'margin-bottom:16px;flex-wrap:wrap' },
    h('button', { class: 'btn btn-sm btn-primary', onclick: () => fileInput.click() }, '⬆ Import images'),
    h('button', { class: 'btn btn-sm', onclick: () => zipInput.click() }, '⬆ Import ZIP'),
    classSel, h('span', { class: 'faint' }, 'class applied to imported images'),
    fileInput, zipInput));

  // samples grid
  const gridWrap = h('div');
  panel.append(gridWrap);
  const loadSamples = async () => {
    clear(gridWrap).append(h('div', { class: 'faint' }, 'loading…'));
    try {
      const page = await api(`/datasets/${ds.id}/samples?page=0&pageSize=60`);
      clear(gridWrap);
      if (!page.samples.length) { gridWrap.append(h('div', { class: 'empty' }, 'No samples imported yet.')); return; }
      const grid = h('div', { class: 'samples' });
      for (const sm of page.samples) {
        grid.append(h('div', { class: 'sample' },
          h('img', { src: `/api/v1/samples/${sm.id}/file?size=128`, loading: 'lazy', title: sm.filename }),
          h('div', { class: 'cls' }, sm.class || '(no class)', sm.status !== 'ok' ? ' · ' + sm.status : ''),
          h('button', { class: 'x', title: 'delete sample', onclick: async (e) => {
            e.stopPropagation();
            await api('/samples/delete', { method: 'POST', json: { ids: [sm.id] } }); refresh();
          } }, '×'),
        ));
      }
      gridWrap.append(grid);
      if (page.total > page.samples.length) {
        gridWrap.append(h('div', { class: 'faint', style: 'margin-top:8px' }, `showing first ${page.samples.length} of ${page.total}`));
      }
    } catch (e) { clear(gridWrap).append(h('div', { class: 'err-box' }, e.message)); }
  };
  await loadSamples();

  // versions
  const verCard = h('div', { class: 'card', style: 'margin-top:16px' });
  detailCol.append(verCard);
  verCard.append(h('h3', {}, 'Dataset versions (reproducible splits)'));
  const tr = versionInput('0.7'), va = versionInput('0.15'), te = versionInput('0.15'), sd = h('input', { type: 'number', value: '42', style: 'width:90px' });
  verCard.append(h('div', { class: 'row', style: 'flex-wrap:wrap;gap:10px;align-items:flex-end' },
    h('label', { class: 'field' }, h('span', {}, 'train'), tr),
    h('label', { class: 'field' }, h('span', {}, 'val'), va),
    h('label', { class: 'field' }, h('span', {}, 'test'), te),
    h('label', { class: 'field' }, h('span', {}, 'seed'), sd),
    h('button', { class: 'btn btn-primary btn-sm', onclick: async () => {
      try {
        await api(`/datasets/${ds.id}/versions`, { method: 'POST', json: { train: +tr.value, val: +va.value, test: +te.value, seed: +sd.value } });
        toast('Version created', 'ok'); refresh();
      } catch (e) { toast(e.message, 'err'); }
    } }, 'Create version')));
  const verList = h('div', { style: 'margin-top:12px' });
  verCard.append(verList);
  try {
    const versions = await api(`/datasets/${ds.id}/versions`);
    if (versions.length) {
      const tbl = h('table', {},
        h('thead', {}, h('tr', {}, h('th', {}, 'v'), h('th', {}, 'train'), h('th', {}, 'val'), h('th', {}, 'test'), h('th', {}, 'seed'), h('th', {}, 'warnings'))),
        h('tbody', {}, ...versions.map((v) => h('tr', {},
          h('td', {}, 'v' + v.version),
          h('td', {}, v.stats?.counts?.train ?? '·'),
          h('td', {}, v.stats?.counts?.val ?? '·'),
          h('td', {}, v.stats?.counts?.test ?? '·'),
          h('td', {}, String(v.seed)),
          h('td', { class: 'faint' }, (v.stats?.warnings || []).join('; ') || '—')))));
      verList.append(tbl);
    } else {
      verList.append(h('div', { class: 'faint' }, 'No versions yet — create one to enable training (a version pins a seeded train/val/test split).'));
    }
  } catch (e) { verList.append(h('div', { class: 'err-box' }, e.message)); }

  function refresh() { go(`project/${pid}/datasets/${sel}`); }
}

function versionInput(v) { return h('input', { type: 'number', step: '0.05', min: '0', max: '1', value: v, style: 'width:90px' }); }

function newDatasetDialog() {
  const name = h('input', { placeholder: 'e.g. product-photos' });
  const mask = modal('New dataset', h('div', {},
    h('div', { class: 'form-row' }, h('label', { class: 'field' }, h('span', {}, 'Name'), name)),
    h('div', { class: 'hint' }, 'Image datasets support PNG / JPEG / GIF / WebP. Audio and other types arrive with their backends.'),
    h('div', { class: 'actions' },
      h('button', { class: 'btn', onclick: () => mask.remove() }, 'Cancel'),
      h('button', { class: 'btn btn-primary', onclick: async () => {
        try {
          const d = await api(`/projects/${S.project.id}/datasets`, { method: 'POST', json: { name: name.value.trim(), dataType: 'image' } });
          mask.remove(); go(`project/${S.project.id}/datasets/${d.id}`);
        } catch (e) { toast(e.message, 'err'); }
      } }, 'Create'))));
  name.focus();
}

function sampleDatasetDialog() {
  const name = h('input', { value: 'Sample Shapes' });
  const classes = h('input', { type: 'number', value: '3', min: '2', max: '6' });
  const per = h('input', { type: 'number', value: '30', min: '10', max: '200' });
  const seed = h('input', { type: 'number', value: '42' });
  const mask = modal('Generate sample shapes dataset', h('div', {},
    h('p', { class: 'muted' }, 'Creates a synthetic dataset of noisy shape images (circle, square, triangle…) — perfect for verifying the whole training loop on a laptop CPU. Generated locally, no downloads.'),
    h('div', { class: 'form-row' },
      h('label', { class: 'field' }, h('span', {}, 'Name'), name),
      h('label', { class: 'field' }, h('span', {}, 'Classes (2–6)'), classes),
      h('label', { class: 'field' }, h('span', {}, 'Per class (10–200)'), per),
      h('label', { class: 'field' }, h('span', {}, 'Seed'), seed)),
    h('div', { class: 'actions' },
      h('button', { class: 'btn', onclick: () => mask.remove() }, 'Cancel'),
      h('button', { class: 'btn btn-primary', onclick: async () => {
        try {
          const res = await api(`/projects/${S.project.id}/sample-dataset`, { method: 'POST', json: { name: name.value.trim(), classes: +classes.value, perClass: +per.value, seed: +seed.value } });
          mask.remove(); toast(`Generated ${res.result.added} samples in ${res.dataset.name}`, 'ok');
          go(`project/${S.project.id}/datasets/${res.dataset.id}`);
        } catch (e) { toast(e.message, 'err'); }
      } }, 'Generate'))));
}

async function importFiles(files, className) {
  if (!files || !files.length) return;
  const fd = new FormData();
  if (className) fd.append('class', className);
  for (const f of files) fd.append('files.' + (className || ''), f, f.name);
  toast(`Importing ${files.length} file(s)…`);
  try {
    const res = await api(`/datasets/${S.dataset.id}/samples/import`, { method: 'POST', body: fd });
    toast(`Imported: ${res.added} added, ${res.duplicates} duplicates, ${res.corrupt} corrupt`, res.added ? 'ok' : 'err');
    if (res.errors?.length) toast(res.errors.slice(0, 3).join(' · '), 'err');
    refreshView();
  } catch (e) { toast(e.message, 'err'); }
}

async function importZip(file) {
  if (!file) return;
  toast('Importing ZIP…');
  try {
    const res = await api(`/datasets/${S.dataset.id}/samples/import-zip`, { method: 'POST', body: file, headers: { 'Content-Type': 'application/zip' } });
    toast(`ZIP: ${res.added} added, ${res.duplicates} duplicates, ${res.corrupt} corrupt`, 'ok');
    refreshView();
  } catch (e) { toast(e.message, 'err'); }
}

async function deleteDataset(ds) {
  const name = h('input', { placeholder: `type "${ds.name}" to confirm` });
  const mask = modal('Delete dataset', h('div', {},
    h('p', { class: 'muted' }, `This permanently removes dataset "${esc(ds.name)}", its samples and versions from disk.`),
    name,
    h('div', { class: 'actions' },
      h('button', { class: 'btn', onclick: () => mask.remove() }, 'Cancel'),
      h('button', { class: 'btn btn-danger', onclick: async () => {
        if (name.value !== ds.name) { toast('Name does not match', 'err'); return; }
        try { await api(`/datasets/${ds.id}?confirm=true`, { method: 'DELETE' }); mask.remove(); toast('Dataset deleted', 'ok'); go(`project/${S.project.id}/datasets`); } catch (e) { toast(e.message, 'err'); }
      } }, 'Delete forever'))));
  name.focus();
}

function refreshView() { route(); }

// ---------- train tab ----------
async function tabTrain(root) {
  const pid = S.project.id;
  let datasets = [];
  try { datasets = await api(`/projects/${pid}/datasets`); } catch (e) { root.append(h('div', { class: 'err-box' }, e.message)); return; }
  if (!datasets.length) {
    root.append(h('div', { class: 'empty' }, 'Create a dataset with at least 2 classes first, then come back to train.'));
    return;
  }
  let backends = [];
  try { backends = (await api('/training/backends')).backends; } catch (e) { /* leave empty */ }

  const dsSel = h('select', {}, ...datasets.map((d) => h('option', { value: d.id }, `${d.name} (${d.sampleCount})`)));
  const verSel = h('select', {}, h('option', { value: '' }, '— create a version first —'));
  const jobName = h('input', { placeholder: 'auto name', style: 'width:220px' });

  const card = h('div', { class: 'card' });
  root.append(card);
  card.append(h('div', { class: 'form-row' },
    h('label', { class: 'field' }, h('span', {}, 'Dataset'), dsSel),
    h('label', { class: 'field' }, h('span', {}, 'Dataset version (the seeded split)'), verSel),
    h('label', { class: 'field' }, h('span', {}, 'Run name'), jobName)));

  // backends (honest capability display)
  const beWrap = h('div', { class: 'form-row' });
  card.append(beWrap);
  let pickedBackend = null;
  const paintBackends = () => {
    clear(beWrap).append(h('label', { class: 'field' }, h('span', {}, 'Training backend')));
    const col = h('div');
    for (const b of backends) {
      const compat = b.tasks?.image_classification;
      const ok = b.status === 'available' && compat?.supported;
      col.append(h('div', { class: 'card', style: 'padding:10px 14px;margin-bottom:8px;cursor:' + (ok ? 'pointer' : 'not-allowed') + ';border-color:' + (ok ? 'var(--border)' : '#ef444433'),
        onclick: ok ? () => { pickedBackend = b.id; [...col.children].forEach((c) => (c.style.borderColor = 'var(--border)')); card.style.borderColor = 'var(--accent)'; } : null },
        h('div', { class: 'row spread' },
          h('strong', {}, b.name),
          ok ? h('span', { class: 'badge ok' }, 'available') : h('span', { class: 'badge failed' }, 'unavailable')),
        h('div', { class: 'faint', style: 'margin-top:4px' },
          ok ? 'image classification · ' + (b.accelerators || []).join('/') : (compat?.reason || b.installHint || 'cannot run image classification')),
      ));
    }
    beWrap.append(col);
  };
  paintBackends();

  // config
  const cfg = {
    epochs: input(30, 1, 500), batchSize: input(16, 1, 128), lr: h('input', { type: 'number', value: '0.003', step: '0.0005', style: 'width:100%' }),
    optimizer: h('select', {}, h('option', { value: 'adam' }, 'adam'), h('option', { value: 'sgd' }, 'sgd (momentum)')),
    size: input(48, 16, 96), gray: h('select', {}, h('option', { value: 'rgb' }, 'RGB (3 channels)'), h('option', { value: 'gray' }, 'Grayscale (1 channel)')),
    seed: input(42), workers: input(2, 1, 16),
    schedule: h('select', {}, h('option', { value: 'none' }, 'constant'), h('option', { value: 'cosine' }, 'cosine decay'), h('option', { value: 'step' }, 'step decay')),
    dropout: h('input', { type: 'number', value: '0.1', step: '0.05', min: '0', max: '0.8', style: 'width:100%' }),
    wd: h('input', { type: 'number', value: '0.0001', step: '0.0001', min: '0', style: 'width:100%' }),
    ckpt: input(5, 0, 100),
  };
  function input(v, min, max) { return h('input', { type: 'number', value: String(v), min, max, style: 'width:100%' }); }
  const aug = { flip: checkbox('Horizontal flip', true), rot: checkbox('Rotate ±10°', false), bright: checkbox('Brightness', true),
    contrast: checkbox('Contrast', false), noise: checkbox('Noise', false) };
  function checkbox(label, on) {
    const c = h('input', { type: 'checkbox' }); c.checked = on;
    return { input: c, label };
  }
  const es = h('input', { type: 'checkbox' }); es.checked = true;

  card.append(h('h3', {}, 'Training configuration'));
  card.append(h('div', { class: 'form-row' },
    h('label', { class: 'field' }, h('span', {}, 'Epochs'), cfg.epochs),
    h('label', { class: 'field' }, h('span', {}, 'Batch size'), cfg.batchSize),
    h('label', { class: 'field' }, h('span', {}, 'Learning rate'), cfg.lr),
    h('label', { class: 'field' }, h('span', {}, 'Optimizer'), cfg.optimizer),
    h('label', { class: 'field' }, h('span', {}, 'Input size (px, 16–96)'), cfg.size),
    h('label', { class: 'field' }, h('span', {}, 'Channels'), cfg.gray),
    h('label', { class: 'field' }, h('span', {}, 'LR schedule'), cfg.schedule),
    h('label', { class: 'field' }, h('span', {}, 'Dropout'), cfg.dropout),
    h('label', { class: 'field' }, h('span', {}, 'Weight decay'), cfg.wd),
    h('label', { class: 'field' }, h('span', {}, 'Seed'), cfg.seed),
    h('label', { class: 'field' }, h('span', {}, 'CPU workers'), cfg.workers),
    h('label', { class: 'field' }, h('span', {}, 'Checkpoint every N epochs'), cfg.ckpt)));
  card.append(h('div', { class: 'form-row' }, h('label', { class: 'field' }, h('span', {}, 'Augmentation (train split only)'),
    h('div', { class: 'checks' }, ...Object.values(aug).map((a) => h('label', {}, a.input, a.label))))));
  card.append(h('div', { class: 'form-row' }, h('label', { class: 'field' }, h('span', {}, 'Early stopping (patience 5, min-Δ 1e-4)'),
    h('label', { style: 'display:flex;gap:8px;align-items:center' }, es, 'enabled'))));

  const warn = h('div');
  card.append(warn, h('div', { class: 'row' },
    h('button', { class: 'btn btn-primary', onclick: start }, '▶ Start training'),
    h('span', { class: 'faint' }, 'Runs locally on your CPU. You can cancel safely at any time — checkpoints are kept.')));

  const loadVersions = async () => {
    clear(verSel).append(h('option', { value: '' }, '— create a version first —'));
    try {
      const versions = await api(`/datasets/${dsSel.value}/versions`);
      for (const v of [...versions].reverse()) {
        verSel.append(h('option', { value: v.id }, `v${v.version} — train ${v.stats?.counts?.train ?? '?'} / val ${v.stats?.counts?.val ?? '?'} / test ${v.stats?.counts?.test ?? '?'}`));
      }
    } catch { /* ignore */ }
  };
  dsSel.addEventListener('change', loadVersions);
  await loadVersions();

  async function start() {
    warn.innerHTML = '';
    const errs = [];
    if (!pickedBackend) errs.push('pick a training backend');
    if (!verSel.value) errs.push('create and pick a dataset version (seeded split)');
    const n = parseFloat(cfg.size.value);
    if (n < 16 || n > 96) errs.push('input size must be 16–96');
    if (errs.length) { warn.append(h('div', { class: 'warn-box' }, errs.join(' · '))); return; }
    const body = {
      name: jobName.value.trim(),
      datasetId: dsSel.value,
      versionId: verSel.value,
      backendId: pickedBackend,
      config: {
        taskType: 'image_classification', architecture: 'ns-cnn-small',
        input: { size: Math.round(n), grayscale: cfg.gray.value === 'gray' },
        epochs: Math.round(+cfg.epochs.value), batchSize: Math.round(+cfg.batchSize.value),
        learningRate: +cfg.lr.value, optimizer: cfg.optimizer.value,
        weightDecay: +cfg.wd.value, lrSchedule: cfg.schedule.value,
        seed: Math.round(+cfg.seed.value), workers: Math.round(+cfg.workers.value),
        dropout: +cfg.dropout.value,
        earlyStopping: { enabled: es.checked, patience: 5, minDelta: 0.0001 },
        augmentation: {
          hFlip: aug.flip.input.checked, rotate: aug.rot.input.checked, maxAngle: 10,
          brightness: aug.bright.input.checked, contrast: aug.contrast.input.checked,
          noise: aug.noise.input.checked, cropPad: 0,
        },
        checkpointEveryEpochs: Math.round(+cfg.ckpt.value),
      },
    };
    try {
      const job = await api(`/projects/${pid}/jobs`, { method: 'POST', json: body });
      toast('Training started — job ' + job.name, 'ok');
      go(`project/${pid}/jobs/${job.id}`);
    } catch (e) { toast(e.message, 'err'); }
  }
}

// ---------- jobs tab ----------
async function tabJobs(root, sel) {
  const pid = S.project.id;
  let jobs = [];
  try { jobs = await api(`/projects/${pid}/jobs`); } catch (e) { root.append(h('div', { class: 'err-box' }, e.message)); return; }

  const head = h('div', { class: 'row', style: 'margin-bottom:14px' },
    h('button', { class: 'btn btn-sm', onclick: () => refreshView() }, '⟳ Refresh'),
    h('span', { class: 'faint' }, 'One training runs at a time (the native engine is CPU-bound); others queue.'));
  root.append(head);

  if (!jobs.length) {
    root.append(h('div', { class: 'empty' }, 'No training jobs yet — open the Train tab to start one.'));
    return;
  }
  if (sel == null) sel = jobs[0].id;
  const layout = h('div', { class: 'grid-2', style: 'grid-template-columns:320px 1fr;align-items:start' });
  root.append(layout);
  const listCol = h('div'), detailCol = h('div');
  layout.append(listCol, detailCol);

  const tbl = h('table', {},
    h('thead', {}, h('tr', {}, h('th', {}, 'Job'), h('th', {}, 'Status'), h('th', {}, 'Best val'))),
    h('tbody', {}, ...jobs.map((j) => h('tr', { class: 'clickable', style: j.id === sel ? 'background:#6366f10f' : '',
      onclick: () => go(`project/${pid}/jobs/${j.id}`) },
      h('td', {}, h('div', {}, h('strong', {}, j.name)), h('div', { class: 'faint' }, new Date(j.createdAt).toLocaleString())),
      h('td', {}, badge(j.status)),
      h('td', {}, j.progress?.bestValLoss ? j.progress.bestValLoss.toFixed(3) : '—')))));
  listCol.append(tbl);

  const job = jobs.find((j) => j.id === sel) || jobs[0];
  await renderJobDetail(detailCol, job);
}

async function renderJobDetail(root, job) {
  const card = h('div', { class: 'card' });
  root.append(card);

  const title = h('h3', {}, job.name, ' ', badge(job.status));
  const phaseLine = h('div', { class: 'faint', style: 'margin-bottom:10px' }, '');
  const epLabel = h('div', { class: 'row spread', style: 'font-size:12.5px;margin-bottom:4px' }, h('span', {}, 'epoch —'), h('span', {}, ''));
  const epBar = h('div', { class: 'progress' }, h('div', { style: 'width:0%' }));
  const stepLabel = h('div', { class: 'row spread', style: 'font-size:12px;color:var(--text-faint);margin:6px 0 10px' }, h('span', {}, ''), h('span', {}, ''));
  const statsRow = h('div', { class: 'grid-3', style: 'margin-bottom:12px' });
  const chartWrap = h('div', { class: 'chart-wrap' }, h('canvas', { width: '760', height: '220', style: 'width:100%' }));
  const legend = h('div', { class: 'legend' },
    h('span', {}, h('span', { class: 'dot', style: 'background:#818cf8' }), 'train loss'),
    h('span', {}, h('span', { class: 'dot', style: 'background:#10b981' }), 'val loss'),
    h('span', {}, h('span', { class: 'dot', style: 'background:#f59e0b' }), 'val acc ×10'));
  const logPanel = h('div', { class: 'log-panel' });
  const ckptBox = h('div');
  const evalBox = h('div');

  card.append(title, phaseLine, epLabel, epBar, stepLabel, statsRow, chartWrap, legend,
    h('h3', { style: 'margin-top:16px' }, 'Event log'), logPanel,
    h('h3', { style: 'margin-top:16px' }, 'Checkpoints'), ckptBox,
    h('div', { style: 'margin-top:16px' }, evalBox));

  // cancel button
  const cancelBtn = h('button', { class: 'btn btn-danger btn-sm', onclick: async () => {
    try { await api(`/jobs/${job.id}/cancel`, { method: 'POST' }); toast('Cancellation requested', 'ok'); } catch (e) { toast(e.message, 'err'); }
  } }, '■ Cancel training');
  card.prepend(h('div', { class: 'row spread' }, h('div', { class: 'row' }, title), cancelBtn));

  let curves = job.progress?.curves || [];
  const drawChart = () => drawCurves(chartWrap.querySelector('canvas'), curves);
  const paintStats = (p) => {
    clear(statsRow);
    const st = [
      ['train loss', p?.trainLoss?.toFixed(4)],
      ['val loss', p?.valLoss?.toFixed(4)],
      ['val accuracy', p?.valAccuracy != null ? fmtPct(p.valAccuracy) : null],
      ['best val loss', p?.bestValLoss?.toFixed(4)],
      ['elapsed', p?.elapsedMs ? fmtSec(p.elapsedMs / 1000) : null],
      ['eta', p?.etaSeconds ? fmtSec(p.etaSeconds) : null],
    ];
    for (const [k, v] of st) statsRow.append(h('div', { class: 'stat' }, h('span', { class: 'v' }, v ?? '—'), h('span', { class: 'k' }, k)));
  };

  const paintProgress = (p) => {
    epLabel.children[0].textContent = `epoch ${p?.epoch ?? 0} / ${p?.epochs ?? job.config?.epochs ?? '?'}`;
    const frac = p?.epochs ? (p.epoch / p.epochs) : 0;
    epBar.children[0].style.width = (frac * 100).toFixed(1) + '%';
    stepLabel.children[0].textContent = p?.phase ? 'phase: ' + p.phase : '';
    stepLabel.children[1].textContent = p?.stepsPerEpoch ? `step ${p.step ?? 0}/${p.stepsPerEpoch}` : '';
    paintStats(p);
  };
  paintProgress(job.progress);
  drawChart();

  const loadCheckpoints = async () => {
    try {
      const cks = await api(`/jobs/${job.id}/checkpoints`);
      clear(ckptBox);
      if (!cks.length) { ckptBox.append(h('div', { class: 'faint' }, 'No checkpoints saved yet.')); return; }
      ckptBox.append(h('table', {}, h('tbody', {}, ...cks.map((c) => h('tr', {},
        h('td', {}, 'epoch ' + c.epoch),
        h('td', {}, 'val loss ' + (c.metrics?.valLoss?.toFixed?.(4) ?? '—')),
        h('td', {}, 'val acc ' + (c.metrics?.valAccuracy != null ? fmtPct(c.metrics.valAccuracy) : '—')),
        h('td', { class: 'faint' }, new Date(c.createdAt).toLocaleTimeString()))))));
    } catch (e) { ckptBox.textContent = e.message; }
  };
  await loadCheckpoints();

  const loadEvaluation = async () => {
    try {
      const ev = await api(`/jobs/${job.id}/evaluation`);
      renderEvaluation(evalBox, ev);
      return true;
    } catch { return false; }
  };

  // SSE live stream (replays persisted events first).
  const q = S.token ? `?token=${encodeURIComponent(S.token)}` : '';
  const es = new EventSource(`/api/v1/jobs/${job.id}/events${q}`);
  S.es = es;
  const terminal = ['completed', 'failed', 'cancelled'];
  let sawDone = false;
  es.addEventListener('metrics', (e) => {
    const d = JSON.parse(e.data).payload;
    if (!d || !d.epoch) return;
    curves = curves.filter((c) => c.epoch !== d.epoch);
    curves.push({ epoch: d.epoch, trainLoss: d.trainLoss, valLoss: d.valLoss, valAcc: d.valAcc, lr: d.lr });
    curves.sort((a, b) => a.epoch - b.epoch);
    paintProgress({ phase: 'train', epoch: d.epoch, epochs: d.epochs, step: d.step, stepsPerEpoch: d.steps,
      trainLoss: d.trainLoss, valLoss: d.valLoss, valAccuracy: d.valAcc });
    drawChart();
  });
  es.addEventListener('tick', (e) => {
    const d = JSON.parse(e.data).payload;
    stepLabel.children[1].textContent = `step ${d.step}/${d.steps} · loss ${d.trainLoss?.toFixed?.(4)}`;
  });
  es.addEventListener('phase', (e) => {
    const d = JSON.parse(e.data).payload;
    phaseLine.textContent = d.message || d.phase || '';
  });
  es.addEventListener('log', (e) => {
    const d = JSON.parse(e.data);
    logPanel.append(h('div', {}, `[${new Date(d.time).toLocaleTimeString()}] ${esc(d.payload?.message || '')}`));
    logPanel.scrollTop = logPanel.scrollHeight;
  });
  es.addEventListener('error', (e) => {
    if (e.data) {
      const d = JSON.parse(e.data);
      logPanel.append(h('div', { style: 'color:var(--red)' }, `[error] ${esc(d.payload?.error || '')}`));
      logPanel.scrollTop = logPanel.scrollHeight;
    }
  });
  es.addEventListener('status', (e) => {
    const d = JSON.parse(e.data);
    const st = d.payload?.status;
    if (st) {
      title.children[1].replaceWith(badge(st));
      if (terminal.includes(st)) { es.close(); afterTerminal(); }
    }
  });
  es.addEventListener('done', () => { if (!sawDone) { sawDone = true; es.close(); afterTerminal(); } });
  es.onerror = () => { es.close(); };

  async function afterTerminal() {
    await loadCheckpoints();
    await loadEvaluation();
    refreshView();
  }
}

function renderEvaluation(box, ev) {
  const m = ev.metrics || {};
  clear(box);
  if (m.warnings?.length) {
    for (const w of m.warnings) box.append(h('div', { class: 'warn-box' }, '⚠ ' + esc(w)));
  }
  box.append(h('div', { class: 'ok-box' },
    `Held-out evaluation (${m.evalSplit || 'test'} split, ${m.evalSamples ?? (m.testSamples ?? '·')} samples): accuracy ${fmtPct(m.evalAccuracy ?? 0)} · macro-F1 ${(m.evalMacroF1 ?? 0).toFixed(3)} · latency p50 ${m.latencyP50Ms?.toFixed?.(2) ?? '—'} ms`));
  const cols = m.classes || [];
  const cm = m.confusion || [];
  if (cols.length && cm.length) {
    const head = h('tr', {}, h('th', {}, 'truth ↓ / pred →'), ...cols.map((c) => h('th', {}, c)));
    const body = cm.map((row, i) => h('tr', {}, h('th', {}, cols[i]),
      ...row.map((v, j) => h('td', { class: i === j ? 'diag' : '' }, v))));
    const per = m.perClass || [];
    box.append(h('h3', {}, 'Confusion matrix'), h('table', { class: 'cm-table' }, h('thead', {}, head), h('tbody', {}, ...body)));
    if (per.length) {
      box.append(h('h3', {}, 'Per-class metrics'), h('table', {},
        h('thead', {}, h('tr', {}, h('th', {}, 'class'), h('th', {}, 'precision'), h('th', {}, 'recall'), h('th', {}, 'f1'), h('th', {}, 'support'))),
        h('tbody', {}, ...per.map((pc) => h('tr', {},
          h('td', {}, pc.class), h('td', {}, pc.precision?.toFixed?.(3)), h('td', {}, pc.recall?.toFixed?.(3)),
          h('td', {}, pc.f1?.toFixed?.(3)), h('td', {}, pc.support))))));
    }
  }
  const totalSec = m.trainTimeSeconds ? fmtSec(m.trainTimeSeconds) : '—';
  box.append(h('div', { class: 'faint', style: 'margin-top:10px' },
    `trained in ${totalSec} · best epoch ${m.bestEpoch ?? '—'} · architecture ${m.architecture || 'ns-cnn-small'} · the model is registered under the Models tab`));
}

function drawCurves(canvas, curves) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height, pad = 34;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = '#262c3d'; ctx.fillStyle = '#6b7488'; ctx.font = '10px system-ui';
  if (!curves.length) { ctx.fillText('waiting for epoch 1…', pad, H / 2); return; }
  let maxV = 0.0001;
  for (const c of curves) maxV = Math.max(maxV, c.trainLoss || 0, c.valLoss || 0);
  maxV = Math.ceil(maxV * 10) / 10;
  const n = curves.length;
  const x = (i) => pad + (i / Math.max(1, n - 1)) * (W - pad - 12);
  const y = (v) => H - pad - (v / maxV) * (H - pad - 14);
  // gridlines
  for (let g = 0; g <= 4; g++) {
    const gy = y((maxV / 4) * g);
    ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(W - 12, gy); ctx.stroke();
    ctx.fillText(((maxV / 4) * g).toFixed(1), 4, gy + 3);
  }
  const line = (vals, color) => {
    ctx.beginPath();
    vals.forEach((v, i) => i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v)));
    ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.stroke();
  };
  line(curves.map((c) => c.trainLoss || 0), '#818cf8');
  line(curves.map((c) => c.valLoss || 0), '#10b981');
  line(curves.map((c) => (c.valAcc || 0) * 10), '#f59e0b');
  ctx.fillStyle = '#9aa3b8';
  ctx.fillText('epoch 1', pad, H - 8);
  ctx.fillText(String(n), W - 22, H - 8);
}

// ---------- experiments ----------
async function tabExperiments(root) {
  const pid = S.project.id;
  let exps = [];
  try { exps = await api(`/projects/${pid}/experiments`); } catch (e) { root.append(h('div', { class: 'err-box' }, e.message)); return; }
  root.append(h('p', { class: 'page-sub' }, 'Every completed run is recorded with its exact configuration and honest metrics — compare and pick.'));
  if (!exps.length) { root.append(h('div', { class: 'empty' }, 'No completed training runs yet.')); return; }
  const card = h('div', { class: 'card' });
  root.append(card);
  card.append(h('table', {},
    h('thead', {}, h('tr', {}, h('th', {}, 'Run'), h('th', {}, 'Backend'), h('th', {}, 'Epochs'), h('th', {}, 'Best val loss'), h('th', {}, 'Eval accuracy'), h('th', {}, 'Time'), h('th', {}, ''))),
    h('tbody', {}, ...exps.map((e) => {
      const m = e.metrics || {};
      const cfgE = e.config || {};
      return h('tr', {},
        h('td', {}, h('strong', {}, e.name), h('div', { class: 'faint' }, new Date(e.createdAt).toLocaleString())),
        h('td', { class: 'faint' }, e.backendId),
        h('td', {}, String(cfgE.epochs ?? '—')),
        h('td', {}, m.bestValLoss?.toFixed?.(4) ?? '—'),
        h('td', {}, m.evalAccuracy != null ? fmtPct(m.evalAccuracy) : '—'),
        h('td', { class: 'faint' }, m.trainTimeSeconds ? fmtSec(m.trainTimeSeconds) : '—'),
        h('td', {}, h('button', { class: 'btn btn-ghost btn-sm', onclick: () => expDetail(e) }, 'details')));
    }))));
}

function expDetail(e) {
  const canvas = h('canvas', { width: '460', height: '180', style: 'width:100%' });
  const m = e.metrics || {};
  const body = h('div', {},
    h('div', { class: 'grid-2' },
      h('div', {}, h('h3', {}, 'Curves'), canvas),
      h('div', {},
        h('h3', {}, 'Key metrics'),
        h('table', {}, h('tbody', {},
          row('eval accuracy', m.evalAccuracy != null ? fmtPct(m.evalAccuracy) : '—'),
          row('macro-F1', m.evalMacroF1?.toFixed?.(4)),
          row('best val loss', m.bestValLoss?.toFixed?.(4)),
          row('best epoch', m.bestEpoch),
          row('split used', m.evalSplit),
          row('seed', e.seed),
          row('backend', e.backendId + ' ' + (e.backendVersion || '')),
          row('architecture', e.architecture))))),
    m.warnings?.length ? h('div', { class: 'warn-box', style: 'margin-top:12px' }, m.warnings.map(esc).join(' · ')) : null);
  const mask = modal('Experiment: ' + e.name, body);
  drawCurves(canvas, e.curves || []);
  function row(k, v) { return h('tr', {}, h('td', { class: 'faint' }, k), h('td', {}, String(v ?? '—'))); }
  return mask;
}

// ---------- models ----------
async function tabModels(root) {
  const pid = S.project.id;
  let models = [];
  try { models = await api(`/projects/${pid}/models`); } catch (e) { root.append(h('div', { class: 'err-box' }, e.message)); return; }
  root.append(h('p', { class: 'page-sub' }, 'Trained models with versioned weights, checksums and metrics. Native models run in this process and in the playground.'));
  if (!models.length) { root.append(h('div', { class: 'empty' }, 'No models yet — complete a training run.')); return; }
  for (const mo of models) {
    const card = h('div', { class: 'card' });
    root.append(card);
    card.append(h('div', { class: 'row spread' },
      h('div', {}, h('h3', {}, mo.name), h('div', { class: 'faint' }, `${mo.taskType} · ${mo.architecture} · ${mo.notes || ''}`)),
      h('div', { class: 'row' },
        h('button', { class: 'btn btn-sm', onclick: () => renameModel(mo) }, 'rename'),
        h('button', { class: 'btn btn-danger btn-sm', onclick: async () => {
          if (!confirm(`Delete model "${mo.name}" and its files?`)) return;
          try { await api(`/models/${mo.id}?confirm=true`, { method: 'DELETE' }); toast('Model deleted', 'ok'); refreshView(); } catch (e) { toast(e.message, 'err'); }
        } }, 'delete'))));
    const tbl = h('table', {},
      h('thead', {}, h('tr', {}, h('th', {}, 'v'), h('th', {}, 'format'), h('th', {}, 'size'), h('th', {}, 'sha256'), h('th', {}, 'eval acc'), h('th', {}, 'created'), h('th', {}, ''))),
      h('tbody', {}, ...(mo.versions || []).map((v) => h('tr', {},
        h('td', {}, 'v' + v.version),
        h('td', {}, v.format),
        h('td', {}, fmtBytes(v.fileSize)),
        h('td', { class: 'mono faint' }, (v.sha256 || '').slice(0, 12) + '…'),
        h('td', {}, v.metrics?.evalAccuracy != null ? fmtPct(v.metrics.evalAccuracy) : '—'),
        h('td', { class: 'faint' }, new Date(v.createdAt).toLocaleString()),
        h('td', {}, h('a', { href: `/api/v1/models/${mo.id}/versions/${v.id}/download`, class: 'btn btn-sm' }, '⬇ download'))))));
    card.append(tbl);
  }
}

function renameModel(mo) {
  const name = h('input', { value: mo.name });
  const notes = h('textarea', { style: 'width:100%' }, mo.notes || '');
  const mask = modal('Rename / annotate model', h('div', {},
    h('div', { class: 'form-row' }, h('label', { class: 'field' }, h('span', {}, 'Name'), name)),
    h('div', { class: 'form-row' }, h('label', { class: 'field' }, h('span', {}, 'Notes'), notes)),
    h('div', { class: 'actions' },
      h('button', { class: 'btn', onclick: () => mask.remove() }, 'Cancel'),
      h('button', { class: 'btn btn-primary', onclick: async () => {
        try { await api(`/models/${mo.id}`, { method: 'PATCH', json: { name: name.value.trim(), notes: notes.value } }); mask.remove(); refreshView(); } catch (e) { toast(e.message, 'err'); }
      } }, 'Save'))));
}

// ---------- playground ----------
async function tabPlayground(root) {
  const pid = S.project.id;
  let models = [];
  try { models = await api(`/projects/${pid}/models`); } catch (e) { root.append(h('div', { class: 'err-box' }, e.message)); return; }
  if (!models.length) { root.append(h('div', { class: 'empty' }, 'Train a model first, then test it here with new images.')); return; }
  const sel = h('select', {}, ...models.flatMap((mo) => (mo.versions || []).map((v) =>
    h('option', { value: `${mo.id}/${v.id}` }, `${mo.name} v${v.version} (acc ${v.metrics?.evalAccuracy != null ? fmtPct(v.metrics.evalAccuracy) : '—'})`))));
  const preview = h('div', { style: 'margin:14px 0' });
  const result = h('div');
  const dz = h('div', { class: 'dropzone' }, 'Drop an image here, or click to choose — inference runs locally in this process.');
  const fileInput = h('input', { type: 'file', accept: 'image/*', style: 'display:none', onchange: () => fileInput.files[0] && predict(fileInput.files[0]) });
  dz.addEventListener('click', () => fileInput.click());
  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('drag'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
  dz.addEventListener('drop', (e) => {
    e.preventDefault(); dz.classList.remove('drag');
    const f = e.dataTransfer.files[0]; if (f) predict(f);
  });

  root.append(h('div', { class: 'card' },
    h('div', { class: 'row' }, h('span', { class: 'muted' }, 'Model:'), sel),
    h('div', { style: 'margin-top:12px' }, dz, fileInput), preview, result));

  async function predict(file) {
    if (!sel.value) { toast('pick a model', 'err'); return; }
    const [mo, v] = sel.value.split('/');
    preview.innerHTML = '';
    const url = URL.createObjectURL(file);
    preview.append(h('img', { src: url, style: 'max-width:180px;border-radius:8px;border:1px solid var(--border)' }));
    const fd = new FormData();
    fd.append('image', file, file.name);
    result.innerHTML = '<div class="faint">running inference…</div>';
    try {
      const res = await api(`/models/${mo}/versions/${v}/predict`, { method: 'POST', body: fd });
      clear(result);
      result.append(h('div', { class: 'faint', style: 'margin-bottom:10px' },
        `runtime: ${res.runtime} · latency ${res.latencyMs.toFixed(2)} ms`));
      const preds = (res.predictions || []).slice().sort((a, b) => b.score - a.score);
      for (const p of preds) {
        result.append(h('div', { class: 'pred-row' },
          h('strong', {}, p.label),
          h('div', { class: 'pred-bar' }, h('div', { style: `width:${(p.score * 100).toFixed(1)}%` })),
          h('span', { class: 'mono' }, fmtPct(p.score))));
      }
    } catch (e) { clear(result).append(h('div', { class: 'err-box' }, e.message)); }
  }
}

// ---------- system ----------
async function viewSystem() {
  setNav([
    { ico: '◱', label: 'Projects', onclick: () => go('projects') },
    { sep: 'Settings' },
    { ico: '♡', label: 'System', active: true, onclick: () => {} },
  ]);
  clear($app).append(h('h1', { class: 'page-title' }, 'System'), h('p', { class: 'page-sub' }, 'Local runtime health, storage and backups.'));
  const info = await api('/system/info').catch(() => null);
  if (info) {
    $app.append(h('div', { class: 'card' }, h('h3', {}, 'Runtime'),
      h('div', { class: 'grid-3' },
        stat('version', info.version), stat('go', info.goVersion), stat('platform', `${info.os}/${info.arch}`),
        stat('data dir', info.dataDir), stat('telemetry', info.telemetry), stat('api', 'v' + info.apiVersion))));
  }
  const diagCard = h('div', { class: 'card' }, h('h3', {}, 'Diagnostics'));
  $app.append(diagCard);
  try {
    const d = await api('/system/diagnostics');
    diagCard.append(h('table', {}, h('tbody', {}, ...d.checks.map((c) => h('tr', {},
      h('td', {}, h('strong', {}, c.name)),
      h('td', {}, h('span', { class: 'badge ' + (c.status === 'ok' ? 'ok' : c.status === 'fail' ? 'failed' : 'none') }, c.status)),
      h('td', { class: 'faint mono' }, c.detail))))));
  } catch (e) { diagCard.append(h('div', { class: 'err-box' }, e.message)); }

  const stCard = h('div', { class: 'card' }, h('h3', {}, 'Storage'));
  $app.append(stCard);
  try {
    const st = await api('/system/storage');
    stCard.append(h('div', { class: 'grid-3' },
      stat('database', fmtBytes(st.database)), stat('datasets', fmtBytes(st.datasets)), stat('models', fmtBytes(st.models)),
      stat('jobs', fmtBytes(st.jobs)), stat('exports', fmtBytes(st.exports)), stat('disk free', st.diskFreeGb?.toFixed?.(1) + ' GB')));
  } catch (e) { stCard.append(h('div', { class: 'err-box' }, e.message)); }

  const bkCard = h('div', { class: 'card' }, h('h3', {}, 'Backup & restore'));
  $app.append(bkCard);
  bkCard.append(h('p', { class: 'muted' },
    'Backups snapshot the SQLite metadata database. Keep a copy of the whole data directory to preserve images and model files too. Restores are staged and applied on next start while no jobs run.'));
  const bkList = h('div', { class: 'faint mono', style: 'margin:10px 0' });
  bkCard.append(h('button', { class: 'btn btn-primary btn-sm', onclick: async () => {
    try {
      const res = await api('/system/backup', { method: 'POST', json: {} });
      toast('Backup written: ' + res.file, 'ok'); listBackups();
    } catch (e) { toast(e.message, 'err'); }
  } }, 'Create backup now'), bkList);
  const restoreName = h('input', { placeholder: 'backup file name (e.g. neurostudio-backup-….db)', style: 'margin-right:8px' });
  bkCard.append(h('div', { class: 'row', style: 'margin-top:12px' }, restoreName,
    h('button', { class: 'btn btn-sm', onclick: async () => {
      if (!restoreName.value.trim()) { toast('enter a backup file name', 'err'); return; }
      try {
        const res = await api('/system/restore', { method: 'POST', json: { file: restoreName.value.trim() } });
        toast(res.note, 'ok');
      } catch (e) { toast(e.message, 'err'); }
    } }, 'Stage restore')));
  await listBackups();
  async function listBackups() {
    clear(bkList);
    try {
      const st = await api('/system/storage');
      bkList.textContent = 'Backups live in the data directory under backups/.';
    } catch { /* ignore */ }
  }
  function stat(k, v) { return h('div', { class: 'stat' }, h('span', { class: 'v', style: 'font-size:15px;font-family:var(--mono);overflow-wrap:anywhere' }, String(v ?? '—')), h('span', { class: 'k' }, k)); }
}

boot();

