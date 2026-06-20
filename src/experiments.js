const fs = require('fs');
const path = require('path');
const { load } = require('./data');
const H = require('./harness');

const OUT = path.join(__dirname, '..', 'results');
const K = 5, SEED = 12345;
const data = load();
const log = [];
function say(s = '') { console.log(s); log.push(s); }

// feature configs
const CFG_WORD = { word: [1, 2] };
const CFG_CHAR = { char: [2, 4] };
const CFG_BOTH = { word: [1, 2], char: [3, 4] };

// =====================================================================
say('# Experimental Results\n');
say(`Dataset: ${data.length} rows | 5-fold CV | seed ${SEED}\n`);

// ---------------------------------------------------------------------
// EXPERIMENT 1 — Two leakage channels from redaction artifacts
// ---------------------------------------------------------------------
say('## Experiment 1 — Two label-leakage channels in redacted data\n');
const exp1 = {};

// Channel A: placeholder PRESENCE leaks the binary label (every MGY row has a
// token, every NOT row has none). Register also separates, so we report both.
say('### 1A — Placeholder *presence* vs the binary MGY/NOT label\n');
say('A degenerate rule "predict MGY iff the row contains any redaction token" is evaluated directly (no training).\n');
{
  const yT = data.map(x => x.label);
  const yP = data.map(x => (/\[(?:FSLUR|CTX)_\d+\]/.test(x.text) ? 'MGY' : 'NOT'));
  const m = require('./ml').metricsFrom(yT, yP, ['MGY', 'NOT']);
  say(`Accuracy = **${m.acc.toFixed(3)}**, Macro-F1 = **${m.macroF1.toFixed(3)}**, MCC = **${m.mcc.toFixed(3)}** — a no-learning rule is perfect.\n`);
  exp1.presenceRule = { acc: m.acc, macroF1: m.macroF1, mcc: m.mcc };
}

// Channel B: placeholder IDENTITY/TYPE leaks the SUBTYPE (FSLUR=>STRONG, CTX=>CONTEXT).
// Register is held constant (both classes are MGY), so this isolates the artifact.
say('### 1B — Placeholder *identity* vs the STRONG/CONTEXT subtype (register held constant)\n');
say('LogReg on the 125 misogynistic rows under four representations, stratified 5-fold. Because both classes are misogynistic, topical register is constant; any signal above chance under a leaking representation is the artifact alone.\n');
say(H.tableHeader());
const mgySub = data.filter(x => x.label === 'MGY');
for (const repr of ['raw', 'mask-distinct', 'masked', 'stripped']) {
  const r = H.crossVal(mgySub, x => x.bucket, repr, 'strat', K, SEED, H.models.logreg(CFG_BOTH, { epochs: 80 }));
  const note = { 'raw': 'token index kept (leaks)', 'mask-distinct': '<FSLUR>/<CTX> kept (leaks type)', 'masked': '<RDW> only (neutralized)', 'stripped': 'token removed (neutralized)' }[repr];
  say(H.rowMd(`LogReg — \`${repr}\` (${note})`, r));
  exp1[repr] = r.agg;
}
say('\n**Reading:** representations that preserve the redaction token (`raw`, `mask-distinct`) recover the subtype almost perfectly *with register held constant* — proof the signal is the artifact, not meaning. Neutralizing the token (`masked`, `stripped`) forces the model onto the linguistic frame and scores fall to their honest level. **Anonymization that encodes class in the placeholder silently inflates results.**\n');

// ---------------------------------------------------------------------
// EXPERIMENT 2 — Evaluation protocol: random vs template-disjoint (Task A, stripped)
// ---------------------------------------------------------------------
say('\n## Experiment 2 — Random vs template-disjoint evaluation (binary, `stripped`)\n');
say('On templated data, random k-fold leaks near-duplicate frames across train/test. A template-disjoint (grouped) split holds out whole frames, measuring generalization to unseen phrasings.\n');
const exp2 = {};
for (const split of ['strat', 'group']) {
  say(`\n### Split: ${split === 'strat' ? 'random stratified 5-fold' : 'template-disjoint 5-fold (unseen frames)'}`);
  say('');
  say(H.tableHeader());
  const getLabel = x => x.label;
  const rows = {
    'Lexicon (learned)': H.crossVal(data, getLabel, 'stripped', split, K, SEED, H.models.lexicon({ posClass: 'MGY' })),
    'Naive Bayes (w1-2)': H.crossVal(data, getLabel, 'stripped', split, K, SEED, H.models.nb(CFG_WORD)),
    'Naive Bayes (char2-4)': H.crossVal(data, getLabel, 'stripped', split, K, SEED, H.models.nb(CFG_CHAR)),
    'LogReg TF-IDF (both)': H.crossVal(data, getLabel, 'stripped', split, K, SEED, H.models.logreg(CFG_BOTH, { epochs: 60 })),
  };
  for (const [n, r] of Object.entries(rows)) say(H.rowMd(n, r));
  exp2[split] = Object.fromEntries(Object.entries(rows).map(([n, r]) => [n, r.agg]));
}
say('\n**Reading:** binary misogyny detection stays easy even across unseen frames — the misogynistic and benign *registers* are lexically disjoint. This is exactly why a headline binary accuracy on this kind of corpus is a weak claim, and why we turn to the harder subtype axis below.\n');

// ---------------------------------------------------------------------
// EXPERIMENT 3 — Hard task: explicit (STRONG) vs context-dependent (CONTEXT) misogyny
// ---------------------------------------------------------------------
say('\n## Experiment 3 — Explicit vs context-dependent misogyny (STRONG vs CONTEXT)\n');
say('Restricted to the 125 misogynistic rows. Representation `masked` (both buckets carry `<RDW>`, so the redaction token cannot leak the bucket — the model must read the surrounding frame). This is the analogue of the false-positive failure mode reported by Yadav et al. (statements misclassified because a charged word appears).\n');
const mgy = data.filter(x => x.label === 'MGY');
say(`Subset size: ${mgy.length} (STRONG=${mgy.filter(x=>x.bucket==='STRONG').length}, CONTEXT=${mgy.filter(x=>x.bucket==='CONTEXT').length})\n`);
const exp3 = {};
for (const split of ['strat', 'group']) {
  say(`\n### Split: ${split === 'strat' ? 'random stratified 5-fold' : 'template-disjoint 5-fold'}`);
  say('');
  say(H.tableHeader());
  const getLabel = x => x.bucket;
  const rows = {
    'Majority': H.crossVal(mgy, getLabel, 'masked', split, K, SEED, H.models.majority()),
    'Lexicon (learned)': H.crossVal(mgy, getLabel, 'masked', split, K, SEED, H.models.lexicon({ posClass: 'CONTEXT', minPrecision: 0.8 })),
    'Naive Bayes (w1-2)': H.crossVal(mgy, getLabel, 'masked', split, K, SEED, H.models.nb(CFG_WORD)),
    'LogReg TF-IDF (both)': H.crossVal(mgy, getLabel, 'masked', split, K, SEED, H.models.logreg(CFG_BOTH, { epochs: 80 })),
    'NB + engineered cues': H.crossVal(mgy, getLabel, 'masked', split, K, SEED, H.models.nbEng(CFG_WORD)),
    'Cues ONLY (interpretable)': H.crossVal(mgy, getLabel, 'masked', split, K, SEED, H.models.cuesOnly()),
  };
  for (const [n, r] of Object.entries(rows)) say(H.rowMd(n, r));
  exp3[split] = Object.fromEntries(Object.entries(rows).map(([n, r]) => [n, { agg: r.agg, pooled: { M: r.pooled.M, classes: r.pooled.classes } }]));
}
// pooled confusion for best model under group split
say('\n**Pooled confusion — LogReg, template-disjoint split:**\n');
const cmRun = H.crossVal(mgy, x => x.bucket, 'masked', 'group', K, SEED, H.models.logreg(CFG_BOTH, { epochs: 80 }));
say(H.confusionMd(cmRun.pooled));
say('\n**Reading:** the explicit/implicit distinction is the genuinely non-trivial axis. Under template-disjoint evaluation the redaction token is uninformative and the model must transfer condemnation/reporting cues to unseen frames — where performance drops markedly relative to the random split.\n');

// ---------------------------------------------------------------------
// EXPERIMENT 4 — 3-way (STRONG/CONTEXT/NOT), stripped, template-disjoint
// ---------------------------------------------------------------------
say('\n## Experiment 4 — Three-way classification (STRONG / CONTEXT / NOT)\n');
say('Representation `stripped`, template-disjoint 5-fold. Full pipeline view.\n');
say(H.tableHeader());
const get3 = x => x.bucket;
const r3 = {
  'Naive Bayes (w1-2)': H.crossVal(data, get3, 'stripped', 'group', K, SEED, H.models.nb(CFG_WORD)),
  'LogReg TF-IDF (both)': H.crossVal(data, get3, 'stripped', 'group', K, SEED, H.models.logreg(CFG_BOTH, { epochs: 80 })),
  'NB + engineered cues': H.crossVal(data, get3, 'stripped', 'group', K, SEED, H.models.nbEng(CFG_WORD)),
};
for (const [n, r] of Object.entries(r3)) say(H.rowMd(n, r));
say('\n**Pooled confusion — NB+engineered, 3-way template-disjoint:**\n');
const cm3 = H.crossVal(data, get3, 'stripped', 'group', K, SEED, H.models.nbEng(CFG_WORD));
say(H.confusionMd(cm3.pooled));

// ---------------------------------------------------------------------
// EXPERIMENT 5 — Interpretability: which frame cues drive STRONG vs CONTEXT
// ---------------------------------------------------------------------
say('\n## Experiment 5 — What the model keys on (LogReg weights, STRONG vs CONTEXT, `masked`)\n');
const tf = H.topFeatures(mgy, x => x.bucket, 'masked', { word: [1, 2] }, { epochs: 150, l2: 1e-3 }, 12);
say(`Positive class = **${tf.posClass}**, negative = **${tf.negClass}**.\n`);
say(`Top features → \`${tf.posClass}\`:`);
say('```');
tf.top.forEach(([f, w]) => say(`  ${w >= 0 ? '+' : ''}${w.toFixed(3)}  ${f}`));
say('```');
say(`Top features → \`${tf.negClass}\`:`);
say('```');
tf.bottom.forEach(([f, w]) => say(`  ${w.toFixed(3)}  ${f}`));
say('```');
say('\n**Reading:** the model recovers the intended contrast — condemnation/meta cues (`mat`, `galat`, `type`, `lafz`, `wajah`) pull toward CONTEXT; direct-address reporting frames (`bol diya`, `likh diya`, `kehke`) pull toward STRONG. The distinction is linguistically real, but it lives in a handful of cues that do not all transfer to unseen frames — hence the template-disjoint drop.\n');

// ---------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------
fs.writeFileSync(path.join(OUT, 'results.md'), log.join('\n'));
fs.writeFileSync(path.join(OUT, 'results.json'), JSON.stringify({ exp1, exp2, exp3, meta: { K, SEED, n: data.length } }, null, 2));
say('\n---\nSaved results/results.md and results/results.json');
