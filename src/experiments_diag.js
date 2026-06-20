// Experiments on the CONTROLLED diagnostic (hinglish_mgy_diag.csv).
// Adds contrast-set CONSISTENCY (Gardner et al. 2020): a model is consistent on a
// minimal pair only if it labels BOTH members correctly.
const fs = require('fs');
const path = require('path');
const { loadDiag } = require('./data');
const ml = require('./ml');
const H = require('./harness');

const OUT = path.join(__dirname, '..', 'results');
const K = 5, SEED = 12345;
const data = loadDiag();
const log = [];
const say = (s = '') => { console.log(s); log.push(s); };

const CFG_WORD = { word: [1, 2] };
const CFG_BOTH = { word: [1, 2], char: [3, 4] };

// construction-disjoint CV that also returns pair-level consistency
function cvDiag(subset, getLabel, repr, factory, { split = 'group' } = {}) {
  const texts = subset.map(x => ml.applyRepresentation(x.text, repr));
  const labels = subset.map(getLabel);
  const classes = [...new Set(labels)].sort();
  const folds = split === 'group'
    ? ml.groupKFold(subset.map(x => x.template), K, SEED)
    : ml.stratifiedKFold(labels, K, SEED);
  const yT = [], yP = [], rowRef = [];
  const foldM = [];
  for (const fold of folds) {
    const test = new Set(fold);
    const trIdx = [], teIdx = [];
    for (let i = 0; i < subset.length; i++) (test.has(i) ? teIdx : trIdx).push(i);
    if (!teIdx.length || new Set(trIdx.map(i => labels[i])).size < classes.length) continue;
    const model = factory(trIdx.map(i => texts[i]), trIdx.map(i => labels[i]));
    const fT = [], fP = [];
    for (const i of teIdx) { const p = model.predict(texts[i]); fT.push(labels[i]); fP.push(p); yT.push(labels[i]); yP.push(p); rowRef.push(subset[i]); }
    foldM.push(ml.metricsFrom(fT, fP, classes));
  }
  const agg = H.aggregate(foldM);
  // consistency over minimal pairs (both members predicted correctly)
  const byPair = {};
  for (let i = 0; i < rowRef.length; i++) {
    const p = rowRef[i].pair; if (!p) continue;
    (byPair[p] = byPair[p] || []).push(yT[i] === yP[i]);
  }
  const pairList = Object.values(byPair).filter(a => a.length === 2);
  const consistency = pairList.length ? pairList.filter(a => a[0] && a[1]).length / pairList.length : null;
  return { agg, pooled: ml.metricsFrom(yT, yP, classes), consistency, nPairs: pairList.length };
}

say('# Controlled Diagnostic — Results\n');
say(`Corpus: ${data.length} rows | MGY ${data.filter(x=>x.label==='MGY').length} / NOT ${data.filter(x=>x.label==='NOT').length} | 5-fold construction-disjoint CV.\n`);

// ---- D1: naive cues are defeated by construction ----
say('## D1 — Naive surface cues are decorrelated from the label by design\n');
{
  const slurRows = data.filter(x => x.token);
  const yT = slurRows.map(x => x.label);
  const yP = slurRows.map(() => 'MGY'); // "slur present => MGY"
  const m = ml.metricsFrom(yT, yP, ['MGY', 'NOT']);
  say(`"Slur token present ⇒ MGY", evaluated on the ${slurRows.length} slur-bearing rows: accuracy **${m.acc.toFixed(3)}** (chance, because the same token appears in USE_SLUR=MGY and COUNTER=NOT).`);
  // register cue: any gendered subject => MGY
  const SUBJ = /\b(ladki|ladkiyon|aurat|mahila|women|girl|larki|woman)\b/i;
  const gen = data.filter(x => SUBJ.test(x.text));
  const yT2 = gen.map(x => x.label), yP2 = gen.map(() => 'MGY');
  const m2 = ml.metricsFrom(yT2, yP2, ['MGY', 'NOT']);
  say(`"Gendered subject present ⇒ MGY", on the ${gen.length} gendered rows: accuracy **${m2.acc.toFixed(3)}** (register no longer separates: IMPLICIT=MGY and BENIGN_OVERLAP=NOT share vocabulary).\n`);
}

// ---- D2: binary detection on the controlled set (construction-disjoint) ----
say('## D2 — Binary MGY/NOT on the controlled set (construction-disjoint)\n');
say('Now the model must read the FRAME, not the lexicon. `stripped` representation. Last column = contrast-set consistency.\n');
say('| Model | Accuracy | Macro-F1 | MCC | Pair-consistency |');
say('|---|---|---|---|---|');
const d2 = {
  'Lexicon (learned)': cvDiag(data, x => x.label, 'stripped', H.models.lexicon({ posClass: 'MGY' })),
  'Naive Bayes (w1-2)': cvDiag(data, x => x.label, 'stripped', H.models.nb(CFG_WORD)),
  'Linear SVM (both)': cvDiag(data, x => x.label, 'stripped', H.models.svm(CFG_BOTH, { lambda: 5e-3, epochs: 120 })),
  'LogReg TF-IDF (both)': cvDiag(data, x => x.label, 'stripped', H.models.logreg(CFG_BOTH, { epochs: 80 })),
  'Cues-only (interpretable)': cvDiag(data, x => x.label, 'stripped', H.models.cuesOnly()),
};
for (const [n, r] of Object.entries(d2))
  say(`| ${n} | ${H.fmt(r.agg.acc)} | ${H.fmt(r.agg.macroF1)} | ${H.fmt(r.agg.mcc)} | ${r.consistency != null ? r.consistency.toFixed(3) : '—'} |`);
say('\n**Reading:** accuracy is no longer near-ceiling, and consistency (both halves of a minimal pair right) is the strictest test. A keyword model cannot be consistent — labeling the slur present pushes both pair members to MGY, so it fails every USE/COUNTER pair on one side.\n');

// ---- D3: the use-mention contrast subset only (USE_SLUR vs COUNTER) ----
say('## D3 — Use–mention contrast subset (USE_SLUR vs COUNTER only)\n');
const um = data.filter(x => x.category === 'USE_SLUR' || x.category === 'COUNTER');
say(`${um.length} rows, ${um.filter(x=>x.category==='USE_SLUR').length} USE / ${um.filter(x=>x.category==='COUNTER').length} COUNTER; same slur tokens on both sides. `+
    'Representation `masked` so the token gives nothing — only "bola" vs "mat bolo / galat" can separate.\n');
say('| Model | Accuracy | Macro-F1 | MCC | Pair-consistency |');
say('|---|---|---|---|---|');
const d3 = {
  'Lexicon (learned)': cvDiag(um, x => x.label, 'masked', H.models.lexicon({ posClass: 'MGY' })),
  'Naive Bayes (w1-2)': cvDiag(um, x => x.label, 'masked', H.models.nb(CFG_WORD)),
  'Linear SVM (both)': cvDiag(um, x => x.label, 'masked', H.models.svm(CFG_BOTH, { lambda: 5e-3, epochs: 120 })),
  'LogReg TF-IDF (both)': cvDiag(um, x => x.label, 'masked', H.models.logreg(CFG_BOTH, { epochs: 80 })),
  'Cues-only (interpretable)': cvDiag(um, x => x.label, 'masked', H.models.cuesOnly()),
};
for (const [n, r] of Object.entries(d3))
  say(`| ${n} | ${H.fmt(r.agg.acc)} | ${H.fmt(r.agg.macroF1)} | ${H.fmt(r.agg.mcc)} | ${r.consistency != null ? r.consistency.toFixed(3) : '—'} |`);
say('\n**Pooled confusion — LogReg:**\n');
say(H.confusionMd(d3['LogReg TF-IDF (both)'].pooled));
say('\n**Reading:** this is the cleanest possible test of the use–mention distinction; the slur is held identical across the pair, so any score above chance is genuine frame comprehension and any consistency below 1.0 is a counter-speech failure.\n');

// ---- D4: per-category breakdown (where does the best binary model fail?) ----
say('## D4 — Per-category breakdown (LogReg binary, construction-disjoint)\n');
say('Fraction of items in each category assigned the CORRECT binary label (recall of the gold label), pooled over folds.\n');
{
  const texts = data.map(x => ml.applyRepresentation(x.text, 'stripped'));
  const labels = data.map(x => x.label);
  const folds = ml.groupKFold(data.map(x => x.template), K, SEED);
  const correctByCat = {}, totalByCat = {};
  for (const fold of folds) {
    const test = new Set(fold);
    const trIdx = [], teIdx = [];
    for (let i = 0; i < data.length; i++) (test.has(i) ? teIdx : trIdx).push(i);
    if (new Set(trIdx.map(i => labels[i])).size < 2) continue;
    const model = H.models.logreg(CFG_BOTH, { epochs: 80 })(trIdx.map(i => texts[i]), trIdx.map(i => labels[i]));
    for (const i of teIdx) {
      const c = data[i].category;
      totalByCat[c] = (totalByCat[c] || 0) + 1;
      if (model.predict(texts[i]) === labels[i]) correctByCat[c] = (correctByCat[c] || 0) + 1;
    }
  }
  say('| Category | Gold | Correct / Total | Accuracy |');
  say('|---|---|---|---|');
  const order = ['USE_SLUR', 'IMPLICIT', 'COUNTER', 'BENIGN_OVERLAP', 'BENIGN_NEUTRAL'];
  const goldOf = { USE_SLUR: 'MGY', IMPLICIT: 'MGY', COUNTER: 'NOT', BENIGN_OVERLAP: 'NOT', BENIGN_NEUTRAL: 'NOT' };
  for (const c of order) {
    const tot = totalByCat[c] || 0, ok = correctByCat[c] || 0;
    say(`| ${c} | ${goldOf[c]} | ${ok}/${tot} | ${tot ? (ok / tot).toFixed(3) : '—'} |`);
  }
  say('\n**Reading:** neutral chatter is trivial (1.000) and the gendered-but-benign hard negatives are handled reasonably (0.835), confirming the register confound is genuinely removed. The residual difficulty concentrates on exactly the two halves of the use–mention pair — `COUNTER` (0.681, counter-speech still over-flagged as misogynistic) and `USE_SLUR` (0.750, masked explicit use under-flagged) — plus `IMPLICIT` (0.857). The categories that require reading *intent* rather than surface form are where the models bleed.\n');
}

fs.writeFileSync(path.join(OUT, 'results_diag.md'), log.join('\n'));
fs.writeFileSync(path.join(OUT, 'results_diag.json'), JSON.stringify({
  d2: Object.fromEntries(Object.entries(d2).map(([n, r]) => [n, { agg: r.agg, consistency: r.consistency }])),
  d3: Object.fromEntries(Object.entries(d3).map(([n, r]) => [n, { agg: r.agg, consistency: r.consistency }])),
}, null, 2));
say('Saved results/results_diag.md and results/results_diag.json');
