// Sanity checks for the ML library against hand-computed values.
const ml = require('./ml');

let pass = 0, fail = 0;
function eq(name, got, exp, tol = 1e-9) {
  const ok = Math.abs(got - exp) <= tol;
  console.log((ok ? 'PASS ' : 'FAIL ') + name + `  got=${got} exp=${exp}`);
  ok ? pass++ : fail++;
}

// --- metrics: a known 2x2 confusion ---
// classes ['NEG','POS']; TP(POS)=tp etc. Build y arrays.
// Construct: 50 POS, 50 NEG. predict: POS->40 TP,10 FN ; NEG->45 TN,5 FP
const yT = [], yP = [];
for (let i = 0; i < 40; i++) { yT.push('POS'); yP.push('POS'); }     // TP
for (let i = 0; i < 10; i++) { yT.push('POS'); yP.push('NEG'); }     // FN
for (let i = 0; i < 5; i++) { yT.push('NEG'); yP.push('POS'); }      // FP
for (let i = 0; i < 45; i++) { yT.push('NEG'); yP.push('NEG'); }     // TN
const m = ml.metricsFrom(yT, yP, ['NEG', 'POS']);
eq('accuracy', m.acc, 85 / 100);
const posP = 40 / 45, posR = 40 / 50, posF = 2 * posP * posR / (posP + posR);
eq('POS precision', m.per[1].prec, posP);
eq('POS recall', m.per[1].rec, posR);
eq('POS f1', m.per[1].f1, posF);
// MCC = (TP*TN - FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) for binary
const TP = 40, TN = 45, FP = 5, FN = 10;
const mccExp = (TP * TN - FP * FN) / Math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
eq('MCC binary', m.mcc, mccExp, 1e-9);

// --- splits: stratified k-fold preserves all indices, no overlap ---
const labels = Array.from({ length: 100 }, (_, i) => (i < 30 ? 'A' : 'B'));
const folds = ml.stratifiedKFold(labels, 5, 42);
const allIdx = folds.flat().sort((a, b) => a - b);
eq('stratified covers all', allIdx.length, 100);
eq('stratified distinct', new Set(allIdx).size, 100);
// each fold ~ same class ratio
const ratios = folds.map(f => f.filter(i => labels[i] === 'A').length);
console.log('  A-count per fold:', ratios, '(want ~6 each)');

// --- group k-fold: no group spans two folds ---
const groups = Array.from({ length: 100 }, (_, i) => 'g' + (i % 10));
const gfolds = ml.groupKFold(groups, 5, 7);
const groupToFold = {};
let leak = 0;
gfolds.forEach((f, fi) => f.forEach(i => {
  if (groupToFold[groups[i]] !== undefined && groupToFold[groups[i]] !== fi) leak++;
  groupToFold[groups[i]] = fi;
}));
eq('group k-fold no leakage', leak, 0);

// --- models learn a trivially separable toy task ---
const texts = [], ys = [];
for (let i = 0; i < 40; i++) { texts.push('good happy great nice'); ys.push('POS'); }
for (let i = 0; i < 40; i++) { texts.push('bad awful terrible sad'); ys.push('NEG'); }
const nb = new ml.MultinomialNB({ word: [1, 1] }).fit(texts, ys);
eq('NB learns POS', nb.predict('happy nice') === 'POS' ? 1 : 0, 1);
eq('NB learns NEG', nb.predict('awful sad') === 'NEG' ? 1 : 0, 1);
const lr = new ml.LogisticRegression(new ml.Vectorizer({ word: [1, 1] }), { epochs: 50 }).fit(texts, ys);
eq('LR learns POS', lr.predict('great good') === 'POS' ? 1 : 0, 1);
eq('LR learns NEG', lr.predict('terrible bad') === 'NEG' ? 1 : 0, 1);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
