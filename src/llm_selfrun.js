// ============================================================================
// LLM zero-shot baseline, executed by the AUTHOR-MODEL (Claude Opus 4.8) acting
// as the classifier in lieu of an external API. Predictions below were produced
// by reading each frame's text ONLY (labels were withheld at classification
// time), applying the same use-mention-aware policy as src/llm_baseline.js.
//
// HONEST CAVEATS (also stated in the paper, §7):
//   • The author-model has prior exposure to the diagnostic's design in-context,
//     so this is an OPTIMISTIC CEILING, not a naive zero-shot estimate.
//   • Items are clean and templated. A production prompt may omit the explicit
//     use-mention instruction; weaker/older models are expected to score lower
//     (run src/llm_baseline.js against such a model to measure that).
//   • Granularity is per-FRAME (40 frames): the model's decision is constant
//     across subject/slur surface variants of a frame. This is disclosed.
//
// 'conf' marks frames the model judged genuinely borderline at decision time.
// ============================================================================
const fs = require('fs');
const path = require('path');
const { loadDiag } = require('./data');
const ml = require('./ml');

// frame_id -> { pred, conf, note }   (pred chosen blind to gold)
const PRED = {
  // benign office/casual chatter -> NOT (clear)
  'BENIGN_NEUTRAL:0': { pred: 'NOT', conf: 'high' }, 'BENIGN_NEUTRAL:1': { pred: 'NOT', conf: 'high' },
  'BENIGN_NEUTRAL:2': { pred: 'NOT', conf: 'high' }, 'BENIGN_NEUTRAL:3': { pred: 'NOT', conf: 'high' },
  'BENIGN_NEUTRAL:4': { pred: 'NOT', conf: 'high' }, 'BENIGN_NEUTRAL:5': { pred: 'NOT', conf: 'high' },
  'BENIGN_NEUTRAL:6': { pred: 'NOT', conf: 'high' }, 'BENIGN_NEUTRAL:7': { pred: 'NOT', conf: 'high' },
  'BENIGN_NEUTRAL:8': { pred: 'NOT', conf: 'high' }, 'BENIGN_NEUTRAL:9': { pred: 'NOT', conf: 'high' },
  // benign but gendered (positive/neutral statements about women) -> NOT
  'BENIGN_OVERLAP:bahar': { pred: 'NOT', conf: 'high' }, 'BENIGN_OVERLAP:driving': { pred: 'NOT', conf: 'high' },
  'BENIGN_OVERLAP:ghar': { pred: 'NOT', conf: 'high' }, 'BENIGN_OVERLAP:kaam': { pred: 'NOT', conf: 'high' },
  'BENIGN_OVERLAP:kapde': { pred: 'NOT', conf: 'high' }, 'BENIGN_OVERLAP:leadership': { pred: 'NOT', conf: 'high' },
  'BENIGN_OVERLAP:padhai': { pred: 'NOT', conf: 'high' },
  // counter-speech / condemnation: slur MENTIONED, not used -> NOT
  'COUNTER:0': { pred: 'NOT', conf: 'high', note: 'bolna galat hai = saying it is wrong' },
  'COUNTER:1': { pred: 'NOT', conf: 'high', note: 'mat bolo = do not say' },
  'COUNTER:2': { pred: 'NOT', conf: 'high', note: 'band karo = stop' },
  'COUNTER:3': { pred: 'NOT', conf: 'high', note: 'use mat karo = do not use' },
  'COUNTER:4': { pred: 'NOT', conf: 'high', note: 'report karo = report it' },
  'COUNTER:5': { pred: 'NOT', conf: 'high', note: 'sahi nahi = not right' },
  'COUNTER:6': { pred: 'NOT', conf: 'high', note: 'mat kaho, ye galat hai' },
  'COUNTER:7': { pred: 'NOT', conf: 'high', note: 'kehne walon ko block karo' },
  // implicit misogyny without slur: stereotype / moral-policing / objectification -> MGY
  'IMPLICIT:bahar': { pred: 'MGY', conf: 'borderline', note: 'restricting womens movement vs safety-concern reading' },
  'IMPLICIT:driving': { pred: 'MGY', conf: 'high' }, 'IMPLICIT:ghar': { pred: 'MGY', conf: 'high' },
  'IMPLICIT:kaam': { pred: 'MGY', conf: 'high' }, 'IMPLICIT:kapde': { pred: 'MGY', conf: 'high' },
  'IMPLICIT:leadership': { pred: 'MGY', conf: 'high' }, 'IMPLICIT:padhai': { pred: 'MGY', conf: 'high' },
  // explicit slur used against a woman -> MGY
  'USE_SLUR:0': { pred: 'MGY', conf: 'high' }, 'USE_SLUR:1': { pred: 'MGY', conf: 'high' },
  'USE_SLUR:2': { pred: 'MGY', conf: 'high' }, 'USE_SLUR:3': { pred: 'MGY', conf: 'high' },
  'USE_SLUR:4': { pred: 'MGY', conf: 'high' },
  'USE_SLUR:5': { pred: 'MGY', conf: 'borderline', note: 'reports slur as reaction to clothing; no condemnation -> use' },
  'USE_SLUR:6': { pred: 'MGY', conf: 'borderline', note: 'passive report of slur; closest to a bare mention' },
  'USE_SLUR:7': { pred: 'MGY', conf: 'high', note: 'chhed rahe the = were harassing' },
};

const data = loadDiag();
const yT = [], yP = [], rows = [];
for (const x of data) {
  const p = PRED[x.frame_id];
  if (!p) { console.error('NO PREDICTION for', x.frame_id); process.exit(1); }
  yT.push(x.label); yP.push(p.pred); rows.push(x);
}
const m = ml.metricsFrom(yT, yP, ['MGY', 'NOT']);

// contrast-set consistency
const byPair = {};
rows.forEach((x, i) => { if (x.pair) (byPair[x.pair] = byPair[x.pair] || []).push(yT[i] === yP[i]); });
const pairs = Object.values(byPair).filter(a => a.length === 2);
const consistency = pairs.filter(a => a[0] && a[1]).length / pairs.length;

const borderline = Object.entries(PRED).filter(([, v]) => v.conf === 'borderline').map(([k, v]) => `${k} (${v.note})`);
const report = {
  model: 'claude-opus-4-8 (author-model, in-context, author-exposed)',
  protocol: 'zero-shot, use-mention-aware prompt, per-frame decision, blind to gold at decision time',
  n: yT.length, accuracy: m.acc, macroF1: m.macroF1, mcc: m.mcc, consistency, nPairs: pairs.length,
  borderlineFramesFlagged: borderline,
  caveat: 'OPTIMISTIC CEILING, not a naive zero-shot estimate; author-model had design exposure; clean templated items.',
};
fs.writeFileSync(path.join(__dirname, '..', 'results', 'results_llm.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
