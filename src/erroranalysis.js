// Qualitative error analysis for the hard task (STRONG vs CONTEXT) under
// template-disjoint CV: collect every misclassified example with its predicted
// label, for the LogReg and Cues-only models.
const fs = require('fs');
const path = require('path');
const { load } = require('./data');
const ml = require('./ml');
const H = require('./harness');

const K = 5, SEED = 12345;
const mgy = load().filter(x => x.label === 'MGY');
const texts = mgy.map(x => ml.applyRepresentation(x.text, 'masked'));
const labels = mgy.map(x => x.bucket);
const folds = ml.groupKFold(mgy.map(x => x.template), K, SEED);

function collectErrors(factory) {
  const errs = [];
  for (let f = 0; f < folds.length; f++) {
    const test = new Set(folds[f]);
    const trIdx = [], teIdx = [];
    for (let i = 0; i < mgy.length; i++) (test.has(i) ? teIdx : trIdx).push(i);
    const model = factory(trIdx.map(i => texts[i]), trIdx.map(i => labels[i]));
    for (const i of teIdx) {
      const pred = model.predict(texts[i]);
      if (pred !== labels[i]) errs.push({ text: mgy[i].text, gold: labels[i], pred });
    }
  }
  return errs;
}

const out = [];
function say(s = '') { out.push(s); }

say('# Error Analysis — STRONG vs CONTEXT (masked, template-disjoint CV)\n');

const lrErr = collectErrors(H.models.logreg({ word: [1, 2], char: [3, 4] }, { epochs: 80 }));
say(`## LogReg TF-IDF — ${lrErr.length} errors / ${mgy.length}\n`);
say('| gold | predicted | text |');
say('|---|---|---|');
for (const e of lrErr) say(`| ${e.gold} | ${e.pred} | ${e.text} |`);

const cuesErr = collectErrors(H.models.cuesOnly());
say(`\n## Cues-only — ${cuesErr.length} errors / ${mgy.length}\n`);
say('| gold | predicted | text |');
say('|---|---|---|');
for (const e of cuesErr) say(`| ${e.gold} | ${e.pred} | ${e.text} |`);

// summarize error direction
function dirSummary(errs) {
  const d = {};
  for (const e of errs) { const k = e.gold + '->' + e.pred; d[k] = (d[k] || 0) + 1; }
  return d;
}
say('\n## Error direction counts');
say('LogReg: ' + JSON.stringify(dirSummary(lrErr)));
say('Cues-only: ' + JSON.stringify(dirSummary(cuesErr)));

fs.writeFileSync(path.join(__dirname, '..', 'results', 'error_analysis.md'), out.join('\n'));
console.log(out.join('\n'));
