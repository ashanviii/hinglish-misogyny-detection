// ============================================================================
// Pure-Node ML library (no external deps): text normalization, n-gram features,
// TF-IDF, Multinomial Naive Bayes, L2 Logistic Regression (SGD), metrics, and
// reproducible stratified / group (template-disjoint) k-fold splits.
// ============================================================================

// ---------- Reproducible RNG (mulberry32) ----------
function rng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function shuffle(arr, rand) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ---------- Text representations ----------
// mode: 'raw' keeps [FSLUR_xx]/[CTX_xx]; 'masked' -> single <RDW>; 'stripped' removes them.
function applyRepresentation(text, mode) {
  if (mode === 'raw') return text;
  if (mode === 'masked') return text.replace(/\[(?:FSLUR|CTX)_\d+\]/g, '<RDW>');
  if (mode === 'stripped') return text.replace(/\[(?:FSLUR|CTX)_\d+\]\s*/g, ' ');
  if (mode === 'mask-distinct') // keep FSLUR vs CTX distinction but drop the index
    return text.replace(/\[FSLUR_\d+\]/g, '<FSLUR>').replace(/\[CTX_\d+\]/g, '<CTX>');
  throw new Error('unknown mode ' + mode);
}

function normalize(text) {
  // lowercase, keep <RDW>/<FSLUR>/<CTX> placeholders, keep Devanagari, strip other punct
  return text
    .toLowerCase()
    .replace(/[।.,!?;:"'()\[\]{}]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

// ---------- Tokenizers / featurizers ----------
function wordTokens(text) {
  return normalize(text).split(' ').filter(Boolean);
}
function wordNGrams(text, nMin, nMax) {
  const toks = wordTokens(text);
  const feats = [];
  for (let n = nMin; n <= nMax; n++)
    for (let i = 0; i + n <= toks.length; i++)
      feats.push('w' + n + ':' + toks.slice(i, i + n).join('_'));
  return feats;
}
function charNGrams(text, nMin, nMax) {
  const s = ' ' + normalize(text).replace(/ /g, '_') + ' ';
  const feats = [];
  for (let n = nMin; n <= nMax; n++)
    for (let i = 0; i + n <= s.length; i++)
      feats.push('c' + n + ':' + s.slice(i, i + n));
  return feats;
}
// configurable: {word:[1,2], char:[3,4]}
function featurize(text, cfg) {
  let f = [];
  if (cfg.word) f = f.concat(wordNGrams(text, cfg.word[0], cfg.word[1]));
  if (cfg.char) f = f.concat(charNGrams(text, cfg.char[0], cfg.char[1]));
  return f;
}

// ---------- Vectorizer (vocab + TF-IDF) ----------
class Vectorizer {
  constructor(cfg, { minDF = 1, useIDF = true } = {}) {
    this.cfg = cfg; this.minDF = minDF; this.useIDF = useIDF;
    this.vocab = new Map(); this.idf = [];
  }
  fit(texts) {
    const df = new Map();
    const counted = texts.map(t => {
      const set = new Set(featurize(t, this.cfg));
      for (const f of set) df.set(f, (df.get(f) || 0) + 1);
      return set;
    });
    let idx = 0;
    for (const [f, c] of df) if (c >= this.minDF) this.vocab.set(f, idx++);
    this.idf = new Array(this.vocab.size).fill(0);
    const N = texts.length;
    for (const [f, j] of this.vocab) this.idf[j] = this.useIDF ? Math.log((1 + N) / (1 + df.get(f))) + 1 : 1;
    return this;
  }
  // returns sparse vector: Map(index -> tfidf weight), L2-normalized
  transformOne(text) {
    const counts = new Map();
    for (const f of featurize(text, this.cfg)) {
      const j = this.vocab.get(f);
      if (j !== undefined) counts.set(j, (counts.get(j) || 0) + 1);
    }
    const vec = new Map();
    let norm = 0;
    for (const [j, c] of counts) { const w = c * this.idf[j]; vec.set(j, w); norm += w * w; }
    norm = Math.sqrt(norm) || 1;
    for (const [j, w] of vec) vec.set(j, w / norm);
    return vec;
  }
  transform(texts) { return texts.map(t => this.transformOne(t)); }
}

// ---------- Multinomial Naive Bayes (counts, Laplace smoothing) ----------
class MultinomialNB {
  constructor(cfg, { alpha = 1.0, minDF = 1 } = {}) {
    this.cfg = cfg; this.alpha = alpha; this.minDF = minDF;
  }
  fit(texts, labels) {
    this.classes = [...new Set(labels)].sort();
    const cidx = Object.fromEntries(this.classes.map((c, i) => [c, i]));
    // vocab via DF
    const df = new Map();
    const featLists = texts.map(t => {
      const fl = featurize(t, this.cfg);
      for (const f of new Set(fl)) df.set(f, (df.get(f) || 0) + 1);
      return fl;
    });
    this.vocab = new Map(); let idx = 0;
    for (const [f, c] of df) if (c >= this.minDF) this.vocab.set(f, idx++);
    const V = this.vocab.size, C = this.classes.length;
    this.featCount = Array.from({ length: C }, () => new Float64Array(V));
    this.classTotal = new Float64Array(C);
    this.classDocs = new Float64Array(C);
    for (let i = 0; i < texts.length; i++) {
      const ci = cidx[labels[i]]; this.classDocs[ci]++;
      for (const f of featLists[i]) {
        const j = this.vocab.get(f);
        if (j !== undefined) { this.featCount[ci][j]++; this.classTotal[ci]++; }
      }
    }
    this.logPrior = this.classDocs.map(n => Math.log(n / texts.length));
    this.V = V;
    return this;
  }
  predictScores(text) {
    const fl = featurize(text, this.cfg);
    const scores = this.classes.map((_, ci) => {
      let s = this.logPrior[ci];
      const denom = this.classTotal[ci] + this.alpha * this.V;
      for (const f of fl) {
        const j = this.vocab.get(f);
        const num = (j !== undefined ? this.featCount[ci][j] : 0) + this.alpha;
        s += Math.log(num / denom);
      }
      return s;
    });
    return scores;
  }
  predict(text) {
    const s = this.predictScores(text);
    let best = 0; for (let i = 1; i < s.length; i++) if (s[i] > s[best]) best = i;
    return this.classes[best];
  }
}

// ---------- Binary Logistic Regression (TF-IDF features, SGD + L2) ----------
class LogisticRegression {
  constructor(vectorizer, { lr = 0.5, epochs = 80, l2 = 1e-4, seed = 1 } = {}) {
    this.vec = vectorizer; this.lr = lr; this.epochs = epochs; this.l2 = l2; this.seed = seed;
  }
  fit(texts, labels) {
    this.classes = [...new Set(labels)].sort(); // [neg, pos] => pos is classes[1]
    const pos = this.classes[1];
    this.vec.fit(texts);
    const X = this.vec.transform(texts);
    const y = labels.map(l => (l === pos ? 1 : 0));
    this.w = new Float64Array(this.vec.vocab.size);
    this.b = 0;
    const rand = rng(this.seed);
    const idxs = X.map((_, i) => i);
    for (let e = 0; e < this.epochs; e++) {
      const order = shuffle(idxs, rand);
      for (const i of order) {
        let z = this.b;
        for (const [j, v] of X[i]) z += this.w[j] * v;
        const p = 1 / (1 + Math.exp(-z));
        const g = p - y[i];
        for (const [j, v] of X[i]) this.w[j] -= this.lr * (g * v + this.l2 * this.w[j]);
        this.b -= this.lr * g;
      }
    }
    return this;
  }
  proba(text) {
    const v = this.vec.transformOne(text);
    let z = this.b;
    for (const [j, val] of v) z += this.w[j] * val;
    return 1 / (1 + Math.exp(-z));
  }
  predict(text) { return this.proba(text) >= 0.5 ? this.classes[1] : this.classes[0]; }
}

// ---------- Linear SVM (hinge loss, Pegasos-style SGD) ----------
class LinearSVM {
  constructor(vectorizer, { lambda = 1e-4, epochs = 60, seed = 1 } = {}) {
    this.vec = vectorizer; this.lambda = lambda; this.epochs = epochs; this.seed = seed;
  }
  fit(texts, labels) {
    this.classes = [...new Set(labels)].sort();
    const pos = this.classes[1];
    this.vec.fit(texts);
    const X = this.vec.transform(texts);
    const y = labels.map(l => (l === pos ? 1 : -1));
    this.w = new Float64Array(this.vec.vocab.size); this.b = 0;
    const rand = rng(this.seed);
    const idxs = X.map((_, i) => i);
    let t = 1;
    for (let e = 0; e < this.epochs; e++) {
      for (const i of shuffle(idxs, rand)) {
        const eta = 1 / (this.lambda * t); t++;
        let m = this.b; for (const [j, v] of X[i]) m += this.w[j] * v;
        const wrong = y[i] * m < 1;
        for (let j = 0; j < this.w.length; j++) this.w[j] *= (1 - eta * this.lambda);
        if (wrong) { for (const [j, v] of X[i]) this.w[j] += eta * y[i] * v; this.b += eta * y[i]; }
      }
    }
    return this;
  }
  score(text) { const v = this.vec.transformOne(text); let m = this.b; for (const [j, val] of v) m += this.w[j] * val; return m; }
  predict(text) { return this.score(text) >= 0 ? this.classes[1] : this.classes[0]; }
}

// ---------- Lexicon / keyword baseline ----------
// Predicts positive class if ANY feature in `lexicon` set appears.
class LexiconClassifier {
  constructor(lexicon, posClass, negClass, tokenizer = wordTokens) {
    this.lex = new Set(lexicon); this.pos = posClass; this.neg = negClass; this.tok = tokenizer;
  }
  predict(text) {
    for (const t of this.tok(text)) if (this.lex.has(t)) return this.pos;
    // also catch bracket tokens directly
    if (/\[(?:FSLUR|CTX)_\d+\]/.test(text)) return this.pos;
    return this.neg;
  }
}

// ---------- Metrics ----------
function confusion(yTrue, yPred, classes) {
  const ci = Object.fromEntries(classes.map((c, i) => [c, i]));
  const M = classes.map(() => classes.map(() => 0));
  for (let i = 0; i < yTrue.length; i++) M[ci[yTrue[i]]][ci[yPred[i]]]++;
  return M;
}
function metricsFrom(yTrue, yPred, classes) {
  const M = confusion(yTrue, yPred, classes);
  const n = yTrue.length;
  let correct = 0; for (let i = 0; i < classes.length; i++) correct += M[i][i];
  const acc = correct / n;
  const per = classes.map((c, i) => {
    const tp = M[i][i];
    let fp = 0, fn = 0;
    for (let k = 0; k < classes.length; k++) { if (k !== i) { fp += M[k][i]; fn += M[i][k]; } }
    const prec = tp + fp ? tp / (tp + fp) : 0;
    const rec = tp + fn ? tp / (tp + fn) : 0;
    const f1 = prec + rec ? 2 * prec * rec / (prec + rec) : 0;
    return { c, prec, rec, f1, support: tp + fn };
  });
  const macroF1 = per.reduce((s, p) => s + p.f1, 0) / classes.length;
  const macroP = per.reduce((s, p) => s + p.prec, 0) / classes.length;
  const macroR = per.reduce((s, p) => s + p.rec, 0) / classes.length;
  // MCC (multiclass Gorodkin formula)
  const mcc = multiclassMCC(M);
  return { acc, macroF1, macroP, macroR, per, M, classes, n, mcc };
}
function multiclassMCC(M) {
  const K = M.length;
  const t = new Array(K).fill(0); // true totals per class
  const p = new Array(K).fill(0); // pred totals per class
  let c = 0, s = 0;
  for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) { s += M[i][j]; if (i === j) c += M[i][j]; t[i] += M[i][j]; p[j] += M[i][j]; }
  let num = c * s;
  for (let k = 0; k < K; k++) num -= t[k] * p[k];
  let sp = s * s, st = 0, ssp = 0;
  for (let k = 0; k < K; k++) { st += t[k] * t[k]; ssp += p[k] * p[k]; }
  const den = Math.sqrt((sp - ssp) * (sp - st));
  return den === 0 ? 0 : num / den;
}

// ---------- Splits ----------
// stratified k-fold over indices given labels
function stratifiedKFold(labels, k, seed) {
  const rand = rng(seed);
  const byClass = {};
  labels.forEach((l, i) => { (byClass[l] = byClass[l] || []).push(i); });
  const folds = Array.from({ length: k }, () => []);
  for (const c of Object.keys(byClass)) {
    const shuffled = shuffle(byClass[c], rand);
    shuffled.forEach((idx, i) => folds[i % k].push(idx));
  }
  return folds;
}
// group k-fold: whole groups (e.g. templates) stay together; no group spans train/test
function groupKFold(groups, k, seed) {
  const rand = rng(seed);
  const uniq = shuffle([...new Set(groups)], rand);
  const groupFold = new Map();
  uniq.forEach((g, i) => groupFold.set(g, i % k));
  const folds = Array.from({ length: k }, () => []);
  groups.forEach((g, i) => folds[groupFold.get(g)].push(i));
  return folds;
}

module.exports = {
  rng, shuffle, applyRepresentation, normalize, wordTokens, wordNGrams, charNGrams,
  featurize, Vectorizer, MultinomialNB, LogisticRegression, LinearSVM, LexiconClassifier,
  confusion, metricsFrom, multiclassMCC, stratifiedKFold, groupKFold,
};
