// Generate self-contained SVG figures from recomputed experiment numbers.
const fs = require('fs');
const path = require('path');
const { load } = require('./data');
const ml = require('./ml');
const H = require('./harness');

const FIG = path.join(__dirname, '..', 'figures');
const K = 5, SEED = 12345;
const data = load();
const mgy = data.filter(x => x.label === 'MGY');
const CFG_BOTH = { word: [1, 2], char: [3, 4] };
const CFG_WORD = { word: [1, 2] };

function mccOf(subset, getLabel, repr, split, factory) {
  return H.crossVal(subset, getLabel, repr, split, K, SEED, factory).agg;
}

// ---------- tiny SVG grouped bar chart ----------
function barChart({ title, groups, series, ylabel = 'score', ymax = 1, colors }) {
  // groups: [label,...]; series: [{name, values:[...]}], aligned to groups
  const W = 720, H = 420, padL = 60, padR = 20, padT = 56, padB = 90;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const nG = groups.length, nS = series.length;
  const gW = plotW / nG, bW = (gW * 0.7) / nS;
  const cols = colors || ['#2c5f8a', '#e07a3f', '#4a9d5b', '#9b59b6'];
  let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="Segoe UI, Arial, sans-serif">`;
  s += `<rect width="${W}" height="${H}" fill="white"/>`;
  s += `<text x="${W / 2}" y="28" text-anchor="middle" font-size="18" font-weight="bold">${title}</text>`;
  // y axis grid + labels
  for (let t = 0; t <= 5; t++) {
    const yv = ymax * t / 5;
    const y = padT + plotH - (yv / ymax) * plotH;
    s += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#e5e5e5"/>`;
    s += `<text x="${padL - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#555">${yv.toFixed(1)}</text>`;
  }
  s += `<text x="16" y="${padT + plotH / 2}" transform="rotate(-90 16 ${padT + plotH / 2})" text-anchor="middle" font-size="12" fill="#333">${ylabel}</text>`;
  // bars
  groups.forEach((g, gi) => {
    const gx = padL + gi * gW + gW * 0.15;
    series.forEach((ser, si) => {
      const v = ser.values[gi];
      const h = Math.max(0, (v / ymax) * plotH);
      const x = gx + si * bW;
      const y = padT + plotH - h;
      s += `<rect x="${x}" y="${y}" width="${bW - 3}" height="${h}" fill="${cols[si % cols.length]}"/>`;
      s += `<text x="${x + (bW - 3) / 2}" y="${y - 4}" text-anchor="middle" font-size="9" fill="#333">${v.toFixed(2)}</text>`;
    });
    s += `<text x="${padL + gi * gW + gW / 2}" y="${padT + plotH + 16}" text-anchor="middle" font-size="11" fill="#333">${g}</text>`;
  });
  // legend
  series.forEach((ser, si) => {
    const lx = padL + si * 180, ly = H - 28;
    s += `<rect x="${lx}" y="${ly - 10}" width="12" height="12" fill="${cols[si % cols.length]}"/>`;
    s += `<text x="${lx + 18}" y="${ly}" font-size="12" fill="#333">${ser.name}</text>`;
  });
  s += `</svg>`;
  return s;
}

// ---------- confusion heatmap ----------
function heatmap({ title, M, classes }) {
  const W = 360, Hh = 360, padL = 90, padT = 70, cell = 75;
  const total = M.flat().reduce((a, b) => a + b, 0);
  const max = Math.max(...M.flat());
  let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${Hh}" font-family="Segoe UI, Arial, sans-serif">`;
  s += `<rect width="${W}" height="${Hh}" fill="white"/>`;
  s += `<text x="${W / 2}" y="26" text-anchor="middle" font-size="15" font-weight="bold">${title}</text>`;
  s += `<text x="${padL + classes.length * cell / 2}" y="50" text-anchor="middle" font-size="11" fill="#555">predicted</text>`;
  classes.forEach((c, j) => s += `<text x="${padL + j * cell + cell / 2}" y="66" text-anchor="middle" font-size="11">${c}</text>`);
  M.forEach((row, i) => {
    s += `<text x="${padL - 8}" y="${padT + i * cell + cell / 2 + 4}" text-anchor="end" font-size="11">${classes[i]}</text>`;
    row.forEach((v, j) => {
      const inten = max ? v / max : 0;
      const r = Math.round(44 + (1 - inten) * 180), g = Math.round(95 + (1 - inten) * 130), b = Math.round(138 + (1 - inten) * 100);
      const fill = `rgb(${r},${g},${b})`;
      const x = padL + j * cell, y = padT + i * cell;
      s += `<rect x="${x}" y="${y}" width="${cell - 2}" height="${cell - 2}" fill="${fill}" stroke="#fff"/>`;
      s += `<text x="${x + cell / 2}" y="${y + cell / 2 + 4}" text-anchor="middle" font-size="14" fill="${inten > 0.5 ? '#fff' : '#222'}" font-weight="bold">${v}</text>`;
    });
  });
  s += `<text x="${W / 2}" y="${Hh - 8}" text-anchor="middle" font-size="10" fill="#777">n=${total}</text>`;
  s += `</svg>`;
  return s;
}

// === Figure 1: leakage channels (Exp 1B) ===
const reprs = ['raw', 'mask-distinct', 'masked', 'stripped'];
const leakMCC = reprs.map(r => mccOf(mgy, x => x.bucket, r, 'strat', H.models.logreg(CFG_BOTH, { epochs: 80 })).mcc[0]);
fs.writeFileSync(path.join(FIG, 'fig1_leakage.svg'), barChart({
  title: 'Fig 1. Redaction-token leakage (STRONG vs CONTEXT)',
  groups: ['raw', 'mask-distinct', 'masked', 'stripped'],
  series: [{ name: 'LogReg MCC (stratified CV)', values: leakMCC }],
  ylabel: 'MCC', ymax: 1,
}));

// === Figure 2: generalization gap (Exp 2, binary) ===
const modelsBin = {
  'Lexicon': H.models.lexicon({ posClass: 'MGY' }),
  'NaiveBayes': H.models.nb(CFG_WORD),
  'NB char': H.models.nb({ char: [2, 4] }),
  'LogReg': H.models.logreg(CFG_BOTH, { epochs: 60 }),
};
const randF1 = Object.values(modelsBin).map(f => mccOf(data, x => x.label, 'stripped', 'strat', f).macroF1[0]);
const grpF1 = Object.values(modelsBin).map(f => mccOf(data, x => x.label, 'stripped', 'group', f).macroF1[0]);
fs.writeFileSync(path.join(FIG, 'fig2_generalization.svg'), barChart({
  title: 'Fig 2. Random vs template-disjoint (binary MGY/NOT)',
  groups: Object.keys(modelsBin),
  series: [
    { name: 'random 5-fold', values: randF1 },
    { name: 'template-disjoint', values: grpF1 },
  ],
  ylabel: 'macro-F1', ymax: 1,
}));

// === Figure 3: hard task MCC, random vs template-disjoint ===
const modelsHard = {
  'Lexicon': H.models.lexicon({ posClass: 'CONTEXT', minPrecision: 0.8 }),
  'NaiveBayes': H.models.nb(CFG_WORD),
  'LogReg': H.models.logreg(CFG_BOTH, { epochs: 80 }),
  'Cues-only': H.models.cuesOnly(),
};
const hardRand = Object.values(modelsHard).map(f => mccOf(mgy, x => x.bucket, 'masked', 'strat', f).mcc[0]);
const hardGrp = Object.values(modelsHard).map(f => mccOf(mgy, x => x.bucket, 'masked', 'group', f).mcc[0]);
fs.writeFileSync(path.join(FIG, 'fig3_hardtask.svg'), barChart({
  title: 'Fig 3. Explicit vs context-dependent misogyny (MCC)',
  groups: Object.keys(modelsHard),
  series: [
    { name: 'random 5-fold', values: hardRand },
    { name: 'template-disjoint', values: hardGrp },
  ],
  ylabel: 'MCC', ymax: 1,
}));

// === Figure 4: confusion heatmap, LogReg STRONG/CONTEXT template-disjoint ===
const cm = H.crossVal(mgy, x => x.bucket, 'masked', 'group', K, SEED, H.models.logreg(CFG_BOTH, { epochs: 80 }));
fs.writeFileSync(path.join(FIG, 'fig4_confusion_hard.svg'), heatmap({
  title: 'Fig 4. STRONG vs CONTEXT (LogReg, template-disjoint)',
  M: cm.pooled.M, classes: cm.pooled.classes,
}));

// === Figure 5: 3-way confusion (NB+eng) ===
const cm3 = H.crossVal(data, x => x.bucket, 'stripped', 'group', K, SEED, H.models.nbEng(CFG_WORD));
fs.writeFileSync(path.join(FIG, 'fig5_confusion_3way.svg'), heatmap({
  title: 'Fig 5. Three-way (NB+cues, template-disjoint)',
  M: cm3.pooled.M, classes: cm3.pooled.classes,
}));

console.log('Wrote 5 SVG figures to figures/');
console.log('Fig1 leak MCC:', leakMCC.map(x => x.toFixed(3)));
console.log('Fig2 random F1:', randF1.map(x => x.toFixed(3)), '| group F1:', grpF1.map(x => x.toFixed(3)));
console.log('Fig3 hard random MCC:', hardRand.map(x => x.toFixed(3)), '| group MCC:', hardGrp.map(x => x.toFixed(3)));
