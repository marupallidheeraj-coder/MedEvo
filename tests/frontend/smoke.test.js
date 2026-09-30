// End-to-end smoke test: runs the page's whole script against a minimal fake DOM/canvas, waits for the
// in-browser GA + models to finish, and checks the rendered report. Run: node --test tests/frontend/
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../../frontend/index.html'), 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);

function makeEnv() {
  const els = {};
  const el = id => (els[id] ||= { id, style: {}, textContent: '', innerHTML: '', value: '', hidden: false, className: '',
    disabled: false, attrs: {}, setAttribute(k, v) { this.attrs[k] = String(v); }, focus() { this.focused = true; },
    scrollIntoView() { this.scrolled = true; }, click() {} });
  const ctx2d = new Proxy({}, {
    get: (t, k) => {
      if (k in t) return t[k];
      if (k === 'createLinearGradient' || k === 'createRadialGradient') return () => ({ addColorStop() {} });
      if (k === 'createImageData') return (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) });
      if (k === 'getImageData') return (x, y, w, h) => { const d = new Uint8ClampedArray(w * h * 4); for (let i = 0; i < d.length; i++) d[i] = (i * 37) % 200; return { data: d }; };
      return () => {};
    },
    set: (t, k, v) => { t[k] = v; return true; },
    has: () => false,
  });
  const canvas = () => ({ width: 0, height: 0, getContext: () => ctx2d, toDataURL: () => 'data:image/png;base64,AAAA' });
  const document = { getElementById: el, createElement: t => (t === 'canvas' ? canvas() : { click() {}, setAttribute() {} }) };
  class ImageStub { set src(v) { this._s = v; setTimeout(() => this.onload && this.onload(), 0); } get src() { return this._s; } get width() { return 256; } get height() { return 256; } }
  const alerts = [];
  const context = vm.createContext({
    document, Image: ImageStub, setTimeout, clearTimeout, console, alert: m => alerts.push(m),
    window: { matchMedia: () => ({ matches: true }), print() {} }, matchMedia: () => ({ matches: true }),
    URL: { createObjectURL: () => 'blob:x', revokeObjectURL() {} }, Blob: class {}, FormData: class { append() {} },
    fetch: () => Promise.reject(new Error('offline')),
  });
  return { els, el, context, alerts, canvas };
}

async function boot() {
  const env = makeEnv();
  vm.runInContext(scripts.join('\n'), env.context);
  const out = env.el('out');
  for (let i = 0; i < 600 && !out.innerHTML; i++) await new Promise(r => setTimeout(r, 50));
  assert.ok(out.innerHTML, 'the dataset analysis never rendered');
  return env;
}

test('page boots, finishes the GA and renders a complete report', async () => {
  const { el } = await boot();
  const out = el('out').innerHTML;
  for (const heading of ['3. Clinical features selected by the GA', '4. Prediction and risk score', '5. X-ray explanation',
    '6. Clinical explanation', '7. What if? Explore changes']) assert.ok(out.includes(heading), heading);
  assert.ok(out.includes('Suggested action:'), 'triage banner missing');
  assert.ok(!/undefined|NaN|\[object/.test(out), 'report contains a broken value');
  assert.match(el('glog').textContent, /done/);
  assert.equal(el('pw').attrs['aria-valuenow'], '100');
  assert.equal(el('pick').disabled, false);
  assert.match(el('status').textContent, /Analysis complete/);
});

test('the automatic first analysis does not steal focus or scroll the page', async () => {
  const { el } = await boot();
  assert.ok(!el('res').focused && !el('res').scrolled);
});

test('random pick moves focus to the result and announces it', async () => {
  const { el } = await boot();
  el('pick').onclick();
  assert.ok(el('res').focused, 'focus should move to the result');
  assert.match(el('status').textContent, /risk, \d+ percent/);
});

const numericSelected = ctx => ctx.sel.map(j => ctx.F[j]).filter(f => f.t === 'n');

test('what-if explorer recomputes the risk and rejects invalid values', async () => {
  const { el, context } = await boot();
  const nums = numericSelected(context);
  assert.ok(nums.length > 0, 'the GA should select at least one numeric feature');
  for (const f of nums) el('w_' + f.k).value = String(Math.round(f.m[0]));   // valid baseline for every input
  const f = nums[0];
  el('w_' + f.k).value = String(f.b); el('w_' + f.k).oninput();
  assert.match(el('wres').textContent, /combined risk would be \d+%/);
  el('w_' + f.k).value = String(f.b * 10 + 1000); el('w_' + f.k).oninput();
  assert.match(el('wres').textContent, /must be between/);
});

test('a red-flag vital sign forces an Urgent action even when the AI score is low', async () => {
  const env = await boot();
  const { el, context } = env;
  const cv = env.canvas(); cv.width = cv.height = 320;
  const x = context.F.map(f => (f.t === 'n' ? f.m[0] : 0));
  x[context.F.findIndex(f => f.k === 'spo2')] = 82;
  context.analyse({ x, title: 'Crafted patient' }, cv, false, 0);           // sample X-ray 0 is a normal film
  const out = el('out').innerHTML;
  assert.match(out, /Suggested action: Urgent/);
  assert.match(out, /Very low oxygen saturation/);
});

test('warns when the X-ray model and the clinical model disagree', async () => {
  const env = await boot();
  const { el, context } = env;
  const cv = env.canvas(); cv.width = cv.height = 320;
  const x = context.F.map(f => (f.t === 'n' ? f.m[0] : 0));                  // a clearly normal clinical picture
  context.analyse({ x, title: 'Crafted patient' }, cv, false, 4);           // sample X-ray 4: preset pneumonia (86%)
  assert.match(el('out').innerHTML, /point in different directions/);
});

test('upload mode: missing image and out-of-range values are reported, not analysed', async () => {
  const env = await boot();
  const { el, context } = env;
  el('go').onclick();
  assert.match(el('formerr').textContent, /valid X-ray image/);
  context.upImg = env.canvas(); context.upImg.width = context.upImg.height = 320;
  el('go').onclick();                                                         // inputs are still empty
  assert.match(el('formerr').textContent, /needs a number/);
});

test('upload mode: optional vitals feed the red-flag check; invalid ones are rejected', async () => {
  const env = await boot();
  const { el, context } = env;
  context.upImg = env.canvas(); context.upImg.width = context.upImg.height = 320;
  for (const f of context.F) env.el('i_' + f.k).value = f.t === 'n' ? String(Math.round(f.m[0])) : '0';
  const optional = context.optIdx().map(j => context.F[j].k);
  assert.ok(optional.length > 0, 'some safety-net vitals should be optional when the GA leaves them out');
  assert.ok(el('form').innerHTML.includes('optional'), 'the form should offer the optional inputs');
  for (const k of optional) env.el('i_' + k).value = '';                     // "not measured"
  el('out').innerHTML = ''; el('go').onclick();
  assert.equal(el('formerr').textContent, '');
  assert.match(el('out').innerHTML, /Uploaded data/);
  assert.doesNotMatch(el('out').innerHTML, /Suggested action: Urgent/);
  if (optional.includes('spo2')) {
    env.el('i_spo2').value = '999'; el('go').onclick();
    assert.match(el('formerr').textContent, /between/);
    env.el('i_spo2').value = '84'; el('out').innerHTML = ''; el('go').onclick();
    assert.match(el('out').innerHTML, /Suggested action: Urgent/);
  }
});
