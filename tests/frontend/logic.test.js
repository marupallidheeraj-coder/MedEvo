// Unit tests for the pure logic in frontend/index.html (no browser needed): node --test tests/frontend/
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../../frontend/index.html'), 'utf8');
const MARK = '/* ---------- X-ray generator';
const main = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).find(s => s.includes(MARK));
assert.ok(main, 'main script not found');
const pure = main.slice(0, main.indexOf(MARK));       // everything before the canvas/DOM code

function load() { const ctx = vm.createContext({}); vm.runInContext(pure, ctx); return ctx; }
const A = load();
const feat = k => A.F.find(f => f.k === k);

test('sigmoid is bounded, centred and monotonic', () => {
  assert.equal(A.sig(0), 0.5);
  assert.ok(A.sig(-50) >= 0 && A.sig(50) <= 1);
  assert.ok(A.sig(1) > A.sig(0) && A.sig(0) > A.sig(-1));
});

test('AUC: perfect, reversed and chance-level rankings', () => {
  assert.equal(A.aucOf([.1, .2, .8, .9], [0, 0, 1, 1]), 1);
  assert.equal(A.aucOf([.9, .8, .2, .1], [0, 0, 1, 1]), 0);
  const r = A.mk(1), sc = [], y = [];
  for (let i = 0; i < 2000; i++) { sc.push(r()); y.push(r() < .5 ? 1 : 0); }
  assert.ok(Math.abs(A.aucOf(sc, y) - 0.5) < 0.05);
});

test('logistic regression learns a separable problem', () => {
  const r = A.mk(3), X = [], y = [];
  for (let i = 0; i < 200; i++) { const c = i % 2; X.push([(c ? 1.5 : -1.5) + (r() - .5), r() - .5]); y.push(c); }
  const m = A.lrTrain(X, y, 0, 200, 300, .01);
  assert.ok(A.aucOf(X.map(row => A.lrZ(m, row)), y) > 0.97);
  assert.ok(Math.abs(m.w[0]) > Math.abs(m.w[1]));      // uses the informative feature
});

test('seeded generator is deterministic', () => {
  const B = load();
  assert.equal(JSON.stringify(A.PT), JSON.stringify(B.PT));
});

test('50-patient demo cohort is well formed', () => {
  assert.equal(A.PT.length, 50);
  assert.equal(new Set(A.PT.map(p => p.id)).size, 50);
  assert.equal(A.PT[0].id, 'P1001');
  const pos = A.PT.filter(p => p.y).length;
  assert.ok(pos > 10 && pos < 40, 'both classes present');
  for (const p of A.PT) {
    assert.equal(p.x.length, A.F.length);
    A.F.forEach((f, j) => {
      if (f.t === 'b') assert.ok(p.x[j] === 0 || p.x[j] === 1, f.k);
      else assert.ok(p.x[j] >= f.a && p.x[j] <= f.b, f.k);
    });
  }
});

test('training data show the expected clinical direction', () => {
  const mean = (k, d) => { const j = A.F.findIndex(f => f.k === k); const v = A.X.filter((_, i) => A.Y[i] === d).map(r => r[j]); return v.reduce((a, b) => a + b) / v.length; };
  assert.ok(mean('spo2', 1) < mean('spo2', 0));
  for (const k of ['temp', 'crp', 'wbc', 'rr', 'fever']) assert.ok(mean(k, 1) > mean(k, 0), k);
  assert.ok(Math.abs(A.Y.reduce((a, b) => a + b) / A.Y.length - 0.4) < 0.07);
});

test('fusion stays inside (0,1), is monotonic and survives extreme image scores', () => {
  assert.ok(Math.abs(A.fuse(0, .5) - .5) < 1e-12);
  assert.ok(A.fuse(1, .5) > A.fuse(0, .5) && A.fuse(0, .9) > A.fuse(0, .1));
  for (const pi of [0, 1, -3, 7]) { const v = A.fuse(0, pi); assert.ok(v > 0 && v < 1 && Number.isFinite(v)); }
  assert.ok(A.fuse(-4, .95) < 0.5, 'a strong clinical signal can outweigh the image score');
});

test('red flags fire exactly at the documented thresholds', () => {
  assert.equal(A.redFlags({ spo2: 89 }).length, 1);
  assert.equal(A.redFlags({ spo2: 90 }).length, 0);
  assert.equal(A.redFlags({ rr: 30 }).length, 1);
  assert.equal(A.redFlags({ rr: 29 }).length, 0);
  assert.equal(A.redFlags({ sbp: 89 }).length, 1);
  assert.equal(A.redFlags({ sbp: 90 }).length, 0);
  assert.equal(A.redFlags({ spo2: 85, rr: 35, sbp: 80 }).length, 3);
  assert.equal(A.redFlags({ temp: 41 }).length, 0);
  assert.equal(A.redFlags({}).length, 0, 'unknown values never raise a flag');
});

test('knownVals only reports the requested features', () => {
  const x = A.F.map((_, j) => j);
  const o = A.knownVals(x, [A.F.findIndex(f => f.k === 'spo2')]);
  assert.deepEqual(Object.keys(o), ['spo2']);
  assert.equal(Object.keys(A.knownVals(x)).length, A.F.length);
});

test('triage: red flags override the score, then 30/70 boundaries', () => {
  assert.equal(A.triageLevel(5, ['x']).label, 'Urgent');
  assert.equal(A.triageLevel(71, []).label, 'Evaluate soon');
  assert.equal(A.triageLevel(70, []).label, 'Clinical review');
  assert.equal(A.triageLevel(31, []).label, 'Clinical review');
  assert.equal(A.triageLevel(30, []).label, 'Routine');
  assert.equal(A.triageLevel(0, undefined).label, 'Routine');
});

test('agreement check flags only strong, opposite-sided disagreement', () => {
  assert.equal(A.agreement(.9, .1).agree, false);
  assert.equal(A.agreement(.1, .9).agree, false);
  assert.equal(A.agreement(.6, .4).agree, true, 'opposite sides but close');
  assert.equal(A.agreement(.9, .6).agree, true, 'same side');
  assert.match(A.agreement(.9, .1).msg, /90%.*10%/);
});

test('numeric input validation', () => {
  const t = feat('temp');
  assert.equal(A.validateInput(t, '37.5'), '');
  assert.equal(A.validateInput(t, 35), '');
  assert.equal(A.validateInput(t, 41.5), '');
  for (const bad of ['', 'abc', null, undefined, '34.9', '42']) assert.notEqual(A.validateInput(t, bad), '', String(bad));
  assert.match(A.validateInput(t, '99'), /Temperature must be between 35 and 41.5/);
});

test('file validation', () => {
  const MB = 1024 * 1024;
  assert.equal(A.validateFile({ type: 'image/png', size: 100 }), '');
  assert.equal(A.validateFile({ type: 'image/jpeg', size: 8 * MB }), '');
  assert.notEqual(A.validateFile({ type: 'image/gif', size: 100 }), '');
  assert.notEqual(A.validateFile({ type: 'application/pdf', size: 100 }), '');
  assert.notEqual(A.validateFile({ type: 'image/png', size: 0 }), '');
  assert.notEqual(A.validateFile({ type: 'image/png', size: 8 * MB + 1 }), '');
  assert.notEqual(A.validateFile(null), '');
});

test('server response validation rejects malformed payloads', () => {
  const con = Array.from({ length: 64 }, () => 0.5);
  assert.equal(A.validApi({ p: .8, con }), true);
  for (const bad of [null, {}, { p: 2, con }, { p: -1, con }, { p: '0.5', con }, { p: .5, con: con.slice(1) },
    { p: .5, con: con.map(() => NaN) }, { p: .5, con: con.map(() => 'x') }]) assert.equal(A.validApi(bad), false);
});
