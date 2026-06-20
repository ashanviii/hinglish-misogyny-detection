// Cross-validation harness, learned-lexicon baseline, engineered features,
// metric aggregation and markdown formatting. Built on ml.js.
const ml = require('./ml');

// ---------- Learned lexicon baseline ----------
// Pick word tokens strongly associated with the positive class in TRAIN only.
function buildLexicon(texts, labels, posClass, { minCount = 2, minPrecision = 0.9 } = {}) {
  const posCount = new Map(), totCount = new Map();
  for (let i = 0; i < texts.length; i++) {
    const toks = new Set(ml.wordTokens(texts[i]));
    for (const t of toks) {
      totCount.set(t, (totCount.get(t) || 0) + 1);
      if (labels[i] === posClass) posCount.set(t, (posCount.get(t) || 0) + 1);
    }
  }
  const lex = new Set();
  for (const [t, tot] of totCount) {
    if (tot < minCount) continue;
    const prec = (posCount.get(t) || 0) / tot;
    if (prec >= minPrecision) lex.add(t);
  }
  return lex;
}

// ---------- Engineered linguistic features (for explicit-vs-implicit) ----------
// Condemnation / counter-speech cues vs direct-address slur frames.
const CONDEMN = ['mat', 'mt', 'nahi', 'nai', 'galat', 'band', 'बंद', 'personal', 'attack', 'sahi', 'jaisi', 'jaise', 'type', 'lafz', 'wajah'];
const REPORT = ['bol', 'diya', 'likh', 'kehke', 'kehna', 'btate', 'comment', 'likha', 'dekh'];
function engineeredFeatures(text) {
  const toks = ml.wordTokens(text);
  const set = new Set(toks);
  const f = [];
  for (const w of CONDEMN) if (set.has(w)) f.push('CND:' + w);
  for (const w of REPORT) if (set.has(w)) f.push('REP:' + w);
  if (CONDEMN.some(w => set.has(w))) f.push('HAS_CONDEMN');
  if (REPORT.some(w => set.has(w))) f.push('HAS_REPORT');
  return f;
}

// ---------- Generic CV evaluation ----------
// data: [{text,label,bucket,template}], labelKey: 'label'|'bucket3'|'strongctx'
// repr: representation mode; split: 'strat'|'group'; modelFactory: (trainTexts,trainLabels)=>{predict(text)}
function crossVal(data, getLabel, repr, splitType, k, seed, modelFactory) {
  const texts = data.map(x => ml.applyRepresentation(x.text, repr));
  const labels = data.map(getLabel);
  const classes = [...new Set(labels)].sort();
  let folds;
  if (splitType === 'strat') folds = ml.stratifiedKFold(labels, k, seed);
  else folds = ml.groupKFold(data.map(x => x.template), k, seed);

  const foldMetrics = [];
  const pooledTrue = [], pooledPred = [];
  for (let f = 0; f < folds.length; f++) {
    const testIdx = new Set(folds[f]);
    const trIdx = []; const teIdx = [];
    for (let i = 0; i < data.length; i++) (testIdx.has(i) ? teIdx : trIdx).push(i);
    if (teIdx.length === 0) continue;
    const model = modelFactory(trIdx.map(i => texts[i]), trIdx.map(i => labels[i]));
    const yT = [], yP = [];
    for (const i of teIdx) { yT.push(labels[i]); yP.push(model.predict(texts[i])); }
    foldMetrics.push(ml.metricsFrom(yT, yP, classes));
    pooledTrue.push(...yT); pooledPred.push(...yP);
  }
  const agg = aggregate(foldMetrics);
  const pooled = ml.metricsFrom(pooledTrue, pooledPred, classes);
  return { agg, pooled, classes, nFolds: foldMetrics.length };
}

function mean(a) { return a.reduce((x, y) => x + y, 0) / a.length; }
function std(a) { const m = mean(a); return Math.sqrt(mean(a.map(x => (x - m) ** 2))); }
function aggregate(fms) {
  return {
    acc: [mean(fms.map(m => m.acc)), std(fms.map(m => m.acc))],
    macroF1: [mean(fms.map(m => m.macroF1)), std(fms.map(m => m.macroF1))],
    macroP: [mean(fms.map(m => m.macroP)), std(fms.map(m => m.macroP))],
    macroR: [mean(fms.map(m => m.macroR)), std(fms.map(m => m.macroR))],
    mcc: [mean(fms.map(m => m.mcc)), std(fms.map(m => m.mcc))],
  };
}

// ---------- Model factories ----------
const models = {
  majority: () => (trT, trL) => {
    const c = {}; trL.forEach(l => c[l] = (c[l] || 0) + 1);
    const maj = Object.entries(c).sort((a, b) => b[1] - a[1])[0][0];
    return { predict: () => maj };
  },
  lexicon: (opts = {}) => (trT, trL) => {
    const classes = [...new Set(trL)].sort();
    // pos = the minority/"interesting" class: choose the class with FEWER samples? we pass explicit posClass via opts
    const pos = opts.posClass || classes[classes.length - 1];
    const neg = classes.find(c => c !== pos);
    const lex = buildLexicon(trT, trL, pos, opts);
    return {
      predict: (text) => {
        for (const t of ml.wordTokens(text)) if (lex.has(t)) return pos;
        return neg;
      }, lex,
    };
  },
  nb: (cfg, opts = {}) => (trT, trL) => new ml.MultinomialNB(cfg, opts).fit(trT, trL),
  logreg: (cfg, opts = {}) => (trT, trL) =>
    new ml.LogisticRegression(new ml.Vectorizer(cfg, opts), opts).fit(trT, trL),
  svm: (cfg, opts = {}) => (trT, trL) =>
    new ml.LinearSVM(new ml.Vectorizer(cfg, opts), opts).fit(trT, trL),
  // NB with engineered features appended
  nbEng: (cfg, opts = {}) => (trT, trL) => {
    const base = new ml.MultinomialNB(cfg, opts);
    const aug = t => t + ' ' + engineeredFeatures(t).join(' ');
    base.fit(trT.map(aug), trL);
    return { predict: t => base.predict(aug(t)) };
  },
  // Interpretable classifier using ONLY the hand-built condemnation/reporting cues
  cuesOnly: (opts = {}) => (trT, trL) => {
    const base = new ml.MultinomialNB({ word: [1, 1] }, { alpha: 0.5 });
    const aug = t => (engineeredFeatures(t).join(' ') || 'NONE');
    base.fit(trT.map(aug), trL);
    return { predict: t => base.predict(aug(t)) };
  },
};

// ---------- LogReg feature importance (train on full subset) ----------
function topFeatures(data, getLabel, repr, cfg, opts, topN = 15) {
  const texts = data.map(x => ml.applyRepresentation(x.text, repr));
  const labels = data.map(getLabel);
  const vec = new ml.Vectorizer(cfg, opts);
  const model = new ml.LogisticRegression(vec, Object.assign({ epochs: 120 }, opts)).fit(texts, labels);
  const inv = new Map(); for (const [f, j] of vec.vocab) inv.set(j, f);
  const weighted = [];
  for (let j = 0; j < model.w.length; j++) weighted.push([inv.get(j), model.w[j]]);
  weighted.sort((a, b) => b[1] - a[1]);
  return {
    posClass: model.classes[1], negClass: model.classes[0],
    top: weighted.slice(0, topN),
    bottom: weighted.slice(-topN).reverse(),
  };
}

// ---------- Formatting ----------
function fmt([m, s]) { return `${m.toFixed(3)}±${s.toFixed(3)}`; }
function rowMd(name, r) {
  return `| ${name} | ${fmt(r.agg.acc)} | ${fmt(r.agg.macroF1)} | ${fmt(r.agg.macroP)} | ${fmt(r.agg.macroR)} | ${fmt(r.agg.mcc)} |`;
}
function tableHeader() {
  return `| Model | Accuracy | Macro-F1 | Macro-P | Macro-R | MCC |\n|---|---|---|---|---|---|`;
}
function confusionMd(metrics) {
  const { M, classes } = metrics;
  let out = '| true\\pred | ' + classes.join(' | ') + ' |\n';
  out += '|---|' + classes.map(() => '---').join('|') + '|\n';
  M.forEach((row, i) => { out += `| **${classes[i]}** | ` + row.join(' | ') + ' |\n'; });
  return out;
}

module.exports = {
  buildLexicon, engineeredFeatures, crossVal, models, aggregate, topFeatures,
  fmt, rowMd, tableHeader, confusionMd, mean, std, CONDEMN, REPORT,
};
