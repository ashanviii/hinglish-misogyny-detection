// Replace fig9 with a meaningful figure: use–mention CONSISTENCY across the
// capability gradient — lightweight/classical models (which fail) vs frontier
// LLMs (which saturate). Pulls numbers from the actual results files.
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');

const diag = JSON.parse(fs.readFileSync(path.join(ROOT, 'results', 'results_diag.json'), 'utf8'));
const prompts = JSON.parse(fs.readFileSync(path.join(ROOT, 'results', 'results_prompts.json'), 'utf8'));

// classical consistency on the use–mention subset (D3)
const d3 = diag.d3;
const classical = [
  ['Lexicon', d3['Lexicon (learned)'].consistency],
  ['Cues-only', d3['Cues-only (interpretable)'].consistency],
  ['Naive Bayes', d3['Naive Bayes (w1-2)'].consistency],
  ['LogReg', d3['LogReg TF-IDF (both)'].consistency],
  ['Linear SVM', d3['Linear SVM (both)'].consistency],
];
// LLMs: every model/prompt scored 1.000 consistency; take the min as a floor
const llmVals = [];
for (const bp of Object.values(prompts)) for (const r of Object.values(bp)) if (r.consistency != null) llmVals.push(r.consistency);
const llmMin = llmVals.length ? Math.min(...llmVals) : 1;
const nModels = Object.keys(prompts).length;

const bars = classical.concat([[`${nModels} LLMs\nall 8 prompts`, llmMin]]);

const W = 800, H = 450, padL = 70, padR = 86, padT = 64, padB = 96;
const plotW = W - padL - padR, plotH = H - padT - padB;
const bw = plotW / bars.length;
function Y(v) { return padT + plotH - v * plotH; }
let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="Segoe UI, Arial, sans-serif">`;
s += `<rect width="${W}" height="${H}" fill="white"/>`;
s += `<text x="${W / 2}" y="28" text-anchor="middle" font-size="17" font-weight="bold">Fig 9. The use–mention test is a capability gradient</text>`;
s += `<text x="${W / 2}" y="46" text-anchor="middle" font-size="12" fill="#666">contrast-set consistency on USE_SLUR vs COUNTER (the cleanest use–mention split)</text>`;
for (let t = 0; t <= 5; t++) { const v = t / 5, y = Y(v);
  s += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#ececec"/>`;
  s += `<text x="${padL - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#555">${v.toFixed(1)}</text>`; }
s += `<text x="16" y="${padT + plotH / 2}" transform="rotate(-90 16 ${padT + plotH / 2})" text-anchor="middle" font-size="12">pair-consistency</text>`;
bars.forEach((b, i) => {
  const [label, v] = b;
  const isLLM = i === bars.length - 1;
  const x = padL + i * bw + bw * 0.18, w = bw * 0.64, y = Y(v), h = padT + plotH - y;
  s += `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${isLLM ? '#00a05a' : '#2c5f8a'}"/>`;
  s += `<text x="${x + w / 2}" y="${y - 5}" text-anchor="middle" font-size="11" font-weight="bold">${v.toFixed(3)}</text>`;
  label.split('\n').forEach((ln, k) =>
    s += `<text x="${padL + i * bw + bw / 2}" y="${padT + plotH + 16 + k * 12}" text-anchor="middle" font-size="10" fill="#333">${ln}</text>`);
});
s += `<line x1="${padL + 5 * bw}" y1="${padT}" x2="${padL + 5 * bw}" y2="${padT + plotH}" stroke="#ccc" stroke-dasharray="4 3"/>`;
s += `</svg>`;
fs.writeFileSync(path.join(ROOT, 'figures', 'fig9_prompts.svg'), s);
console.log('Wrote figures/fig9_prompts.svg (capability gradient). classical:', classical.map(c=>c[1]).join(','), '| LLM floor:', llmMin, '| models:', nModels);
