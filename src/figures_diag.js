// Figures for the controlled diagnostic: accuracy vs contrast-set consistency.
const fs = require('fs');
const path = require('path');
const R = require(path.join(__dirname, '..', 'results', 'results_diag.json'));
const FIG = path.join(__dirname, '..', 'figures');

function barChart({ title, groups, series, ylabel = 'score', ymax = 1, colors }) {
  const W = 760, H = 430, padL = 60, padR = 20, padT = 56, padB = 96;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const nG = groups.length, nS = series.length, gW = plotW / nG, bW = (gW * 0.72) / nS;
  const cols = colors || ['#2c5f8a', '#e07a3f', '#4a9d5b'];
  let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="Segoe UI, Arial, sans-serif">`;
  s += `<rect width="${W}" height="${H}" fill="white"/>`;
  s += `<text x="${W / 2}" y="28" text-anchor="middle" font-size="17" font-weight="bold">${title}</text>`;
  for (let t = 0; t <= 5; t++) { const yv = ymax * t / 5, y = padT + plotH - (yv / ymax) * plotH;
    s += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#ececec"/>`;
    s += `<text x="${padL - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#555">${yv.toFixed(1)}</text>`; }
  s += `<text x="16" y="${padT + plotH / 2}" transform="rotate(-90 16 ${padT + plotH / 2})" text-anchor="middle" font-size="12">${ylabel}</text>`;
  groups.forEach((g, gi) => {
    const gx = padL + gi * gW + gW * 0.14;
    series.forEach((ser, si) => {
      const v = ser.values[gi], h = Math.max(0, (v / ymax) * plotH), x = gx + si * bW, y = padT + plotH - h;
      s += `<rect x="${x}" y="${y}" width="${bW - 3}" height="${h}" fill="${cols[si % cols.length]}"/>`;
      s += `<text x="${x + (bW - 3) / 2}" y="${y - 4}" text-anchor="middle" font-size="9">${v.toFixed(2)}</text>`;
    });
    const lbl = g.length > 12 ? g.slice(0, 11) + '…' : g;
    s += `<text x="${padL + gi * gW + gW / 2}" y="${padT + plotH + 16}" text-anchor="middle" font-size="10" transform="rotate(0 ${padL + gi * gW + gW / 2} ${padT + plotH + 16})">${lbl}</text>`;
  });
  series.forEach((ser, si) => { const lx = padL + si * 200, ly = H - 30;
    s += `<rect x="${lx}" y="${ly - 10}" width="12" height="12" fill="${cols[si % cols.length]}"/>`;
    s += `<text x="${lx + 18}" y="${ly}" font-size="12">${ser.name}</text>`; });
  s += `<text x="${W/2}" y="${H-8}" text-anchor="middle" font-size="10" fill="#777">dashed expectation: a lexicon/keyword model is structurally incapable of high consistency</text>`;
  s += `</svg>`;
  return s;
}

function draw(block, key, title, file) {
  const groups = Object.keys(block);
  const acc = groups.map(g => block[g].agg.acc[0]);
  const cons = groups.map(g => block[g].consistency);
  fs.writeFileSync(path.join(FIG, file), barChart({
    title, groups, ylabel: 'score', ymax: 1,
    series: [{ name: 'accuracy', values: acc }, { name: 'contrast consistency', values: cons }],
  }));
}

draw(R.d2, 'd2', 'Fig 7. Controlled binary task: accuracy vs consistency', 'fig7_diag_binary.svg');
draw(R.d3, 'd3', 'Fig 8. Use–mention contrast (USE vs COUNTER): accuracy vs consistency', 'fig8_diag_usemention.svg');
console.log('Wrote figures/fig7_diag_binary.svg and figures/fig8_diag_usemention.svg');
