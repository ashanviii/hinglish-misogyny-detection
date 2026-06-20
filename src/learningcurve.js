// Learning curve under TEMPLATE-DISJOINT protocol: hold out a fixed pool of test
// templates, train on a growing prefix of the remaining templates, evaluate.
// Averaged over many seeds. Answers: "is the dataset big enough, or still climbing?"
const fs = require('fs');
const path = require('path');
const { load } = require('./data');
const ml = require('./ml');
const H = require('./harness');

const data = load();
const FRACTIONS = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0];
const REPEATS = 25;
const TEST_FRAC = 0.30; // fraction of templates held out for testing each repeat

// task: {name, subset, getLabel, repr, factory, metric}
function runCurve(task) {
  const subset = task.subset;
  const templates = [...new Set(subset.map(x => x.template))];
  // per-fraction accumulators across repeats
  const acc = FRACTIONS.map(() => []);
  for (let rep = 0; rep < REPEATS; rep++) {
    const shuffled = ml.shuffle(templates, ml.rng(1000 + rep));
    const nTest = Math.max(1, Math.round(templates.length * TEST_FRAC));
    const testT = new Set(shuffled.slice(0, nTest));
    const poolT = shuffled.slice(nTest); // training template pool (fixed order per rep)
    const testRows = subset.filter(x => testT.has(x.template));
    const yT = testRows.map(task.getLabel);
    const classes = [...new Set(subset.map(task.getLabel))].sort();
    FRACTIONS.forEach((frac, fi) => {
      const nTrainT = Math.max(2, Math.round(poolT.length * frac));
      const trainT = new Set(poolT.slice(0, nTrainT));
      const trainRows = subset.filter(x => trainT.has(x.template));
      const trTexts = trainRows.map(x => ml.applyRepresentation(x.text, task.repr));
      const trLabels = trainRows.map(task.getLabel);
      // guard: need both classes in train
      if (new Set(trLabels).size < classes.length) { acc[fi].push(null); return; }
      const model = task.factory(trTexts, trLabels);
      const yP = testRows.map(x => model.predict(ml.applyRepresentation(x.text, task.repr)));
      const m = ml.metricsFrom(yT, yP, classes);
      acc[fi].push(task.metric(m));
    });
  }
  // mean train-template count per fraction (approx, using full pool size)
  const poolSize = templates.length - Math.round(templates.length * TEST_FRAC);
  return FRACTIONS.map((frac, fi) => {
    const vals = acc[fi].filter(v => v !== null);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
    return { frac, nTemplates: Math.max(2, Math.round(poolSize * frac)), mean, sd };
  });
}

const tasks = {
  'Binary MGY/NOT (LogReg, macro-F1)': {
    subset: data, getLabel: x => x.label, repr: 'stripped',
    factory: H.models.logreg({ word: [1, 2], char: [3, 4] }, { epochs: 60 }),
    metric: m => m.macroF1,
  },
  'STRONG vs CONTEXT (Cues-only, MCC)': {
    subset: data.filter(x => x.label === 'MGY'), getLabel: x => x.bucket, repr: 'masked',
    factory: H.models.cuesOnly(),
    metric: m => m.mcc,
  },
  'STRONG vs CONTEXT (LogReg, MCC)': {
    subset: data.filter(x => x.label === 'MGY'), getLabel: x => x.bucket, repr: 'masked',
    factory: H.models.logreg({ word: [1, 2], char: [3, 4] }, { epochs: 80 }),
    metric: m => m.mcc,
  },
};

const results = {};
const out = ['# Learning Curve (template-disjoint, ' + REPEATS + ' repeats, 30% templates held out)\n'];
for (const [name, task] of Object.entries(tasks)) {
  const curve = runCurve(task);
  results[name] = curve;
  out.push(`## ${name}\n`);
  out.push('| train templates | frac | score (mean±SD) |');
  out.push('|---|---|---|');
  for (const p of curve) out.push(`| ${p.nTemplates} | ${(p.frac * 100).toFixed(0)}% | ${p.mean.toFixed(3)} ± ${p.sd.toFixed(3)} |`);
  // marginal gain last step
  const last = curve[curve.length - 1].mean, prev = curve[curve.length - 2].mean;
  out.push(`\n**Marginal gain from 90%→100% of templates:** ${(last - prev >= 0 ? '+' : '')}${(last - prev).toFixed(3)} ` +
    `(${last - prev > 0.02 ? 'still climbing — more data would help' : 'roughly saturated'}).\n`);
}

fs.writeFileSync(path.join(__dirname, '..', 'results', 'learning_curve.md'), out.join('\n'));
fs.writeFileSync(path.join(__dirname, '..', 'results', 'learning_curve.json'), JSON.stringify(results, null, 2));

// ---- SVG line chart ----
function lineChart(series, { title, ylabel, ymax = 1, xmax }) {
  const W = 760, Hh = 440, padL = 64, padR = 24, padT = 56, padB = 84;
  const plotW = W - padL - padR, plotH = Hh - padT - padB;
  const cols = ['#2c5f8a', '#e07a3f', '#4a9d5b'];
  const X = v => padL + (v / xmax) * plotW;
  const Y = v => padT + plotH - (v / ymax) * plotH;
  let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${Hh}" font-family="Segoe UI, Arial, sans-serif">`;
  s += `<rect width="${W}" height="${Hh}" fill="white"/>`;
  s += `<text x="${W / 2}" y="30" text-anchor="middle" font-size="17" font-weight="bold">${title}</text>`;
  for (let t = 0; t <= 5; t++) {
    const yv = ymax * t / 5, y = Y(yv);
    s += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#ececec"/>`;
    s += `<text x="${padL - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#555">${yv.toFixed(1)}</text>`;
  }
  // x ticks
  series[0].pts.forEach(p => {
    s += `<text x="${X(p.x)}" y="${padT + plotH + 18}" text-anchor="middle" font-size="10" fill="#555">${p.x}</text>`;
  });
  s += `<text x="${padL + plotW / 2}" y="${Hh - 40}" text-anchor="middle" font-size="12" fill="#333"># training templates (independent frames)</text>`;
  s += `<text x="16" y="${padT + plotH / 2}" transform="rotate(-90 16 ${padT + plotH / 2})" text-anchor="middle" font-size="12" fill="#333">${ylabel}</text>`;
  series.forEach((ser, si) => {
    const c = cols[si % cols.length];
    // band
    let up = '', dn = '';
    ser.pts.forEach((p, i) => { up += `${i ? 'L' : 'M'}${X(p.x)},${Y(Math.min(ymax, p.y + p.sd))} `; });
    for (let i = ser.pts.length - 1; i >= 0; i--) { const p = ser.pts[i]; up += `L${X(p.x)},${Y(Math.max(0, p.y - p.sd))} `; }
    s += `<path d="${up}Z" fill="${c}" opacity="0.12"/>`;
    // line
    let d = '';
    ser.pts.forEach((p, i) => { d += `${i ? 'L' : 'M'}${X(p.x)},${Y(p.y)} `; });
    s += `<path d="${d}" fill="none" stroke="${c}" stroke-width="2.5"/>`;
    ser.pts.forEach(p => { s += `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="3.5" fill="${c}"/>`; });
    // legend
    const lx = padL + si * 250, ly = Hh - 16;
    s += `<rect x="${lx}" y="${ly - 10}" width="14" height="4" fill="${c}"/>`;
    s += `<text x="${lx + 20}" y="${ly - 4}" font-size="11" fill="#333">${ser.name}</text>`;
  });
  s += `</svg>`;
  return s;
}

const xmax = Math.max(...Object.values(results).flat().map(p => p.nTemplates));
const svgSeries = Object.entries(results).map(([name, curve]) => ({
  name: name.replace(/\(.*\)/, '').trim(),
  pts: curve.map(p => ({ x: p.nTemplates, y: p.mean, sd: p.sd })),
}));
// split into two charts: binary (F1) and hard (MCC) share 0..1 scale, plot together
fs.writeFileSync(path.join(__dirname, '..', 'figures', 'fig6_learning_curve.svg'),
  lineChart(svgSeries, { title: 'Fig 6. Learning curve (template-disjoint, 25 repeats)', ylabel: 'score (F1 / MCC)', ymax: 1, xmax }));

console.log(out.join('\n'));
console.log('\nWrote results/learning_curve.{md,json} and figures/fig6_learning_curve.svg');
