const { loadDataset } = require('./csv');
const path = require('path');

const DATA = path.join(__dirname, '..', 'data', 'hinglish_mgy_binary_redacted_500.csv');
const d = loadDataset(DATA);

function dist(arr, key) {
  const m = {};
  for (const x of arr) m[x[key]] = (m[x[key]] || 0) + 1;
  return m;
}

console.log('Total rows:', d.length);
console.log('Label dist:', dist(d, 'label'));
console.log('Bucket dist:', dist(d, 'bucket'));

// cross-tab label x bucket
const ct = {};
for (const x of d) {
  const k = x.label + ' / ' + x.bucket;
  ct[k] = (ct[k] || 0) + 1;
}
console.log('\nLabel x Bucket crosstab:');
for (const k of Object.keys(ct).sort()) console.log('  ', k, '=', ct[k]);

// FSLUR tokens
const slurRe = /\[FSLUR_\d+\]/g;
const slurCounts = {};
let rowsWithSlur = 0;
for (const x of d) {
  const m = x.text.match(slurRe);
  if (m) { rowsWithSlur++; for (const t of m) slurCounts[t] = (slurCounts[t] || 0) + 1; }
}
console.log('\nRows containing an [FSLUR_xx] token:', rowsWithSlur);
console.log('Distinct FSLUR tokens:', Object.keys(slurCounts).length);

// slur presence vs bucket
const slurByBucket = {};
for (const x of d) {
  const has = slurRe.test(x.text); slurRe.lastIndex = 0;
  const k = x.bucket + (has ? ' +slur' : ' -slur');
  slurByBucket[k] = (slurByBucket[k] || 0) + 1;
}
console.log('\nSlur presence x bucket:');
for (const k of Object.keys(slurByBucket).sort()) console.log('  ', k, '=', slurByBucket[k]);

// slur presence vs label (the KEY confound: does slur => MGY?)
let slurMGY = 0, slurNOT = 0, noslurMGY = 0, noslurNOT = 0;
for (const x of d) {
  const has = slurRe.test(x.text); slurRe.lastIndex = 0;
  if (has && x.label === 'MGY') slurMGY++;
  else if (has && x.label === 'NOT') slurNOT++;
  else if (!has && x.label === 'MGY') noslurMGY++;
  else noslurNOT++;
}
console.log('\nSlur-presence confusion vs LABEL:');
console.log('  slur present & MGY :', slurMGY);
console.log('  slur present & NOT :', slurNOT, '  <-- lexical false positives if we keyword-classify');
console.log('  no slur     & MGY  :', noslurMGY, '  <-- lexical false negatives');
console.log('  no slur     & NOT  :', noslurNOT);

// Devanagari presence
const devRe = /[ऀ-ॿ]/;
let dev = 0;
for (const x of d) if (devRe.test(x.text)) dev++;
console.log('\nRows containing Devanagari script:', dev, `(${(100*dev/d.length).toFixed(1)}%)`);

// duplicates (exact text)
const seen = {};
for (const x of d) seen[x.text] = (seen[x.text] || 0) + 1;
const dups = Object.entries(seen).filter(([, n]) => n > 1);
console.log('\nDistinct texts:', Object.keys(seen).length, '| duplicated texts:', dups.length);
console.log('Total duplicate-instance rows:', dups.reduce((s, [, n]) => s + n, 0));

// avg tokens
const toks = d.map(x => x.text.split(/\s+/).filter(Boolean).length);
console.log('\nAvg tokens/row:', (toks.reduce((a, b) => a + b, 0) / toks.length).toFixed(2),
  '| min', Math.min(...toks), '| max', Math.max(...toks));

// sample CONTEXT examples (slur present but NOT misogynistic)
console.log('\n--- CONTEXT bucket samples (the hard subset) ---');
d.filter(x => x.bucket === 'CONTEXT').slice(0, 15).forEach(x =>
  console.log(`  [${x.label}] ${x.text}`));

console.log('\n--- STRONG bucket samples ---');
d.filter(x => x.bucket === 'STRONG').slice(0, 8).forEach(x =>
  console.log(`  [${x.label}] ${x.text}`));

console.log('\n--- NOT bucket w/ slur token (if any) ---');
d.filter(x => x.bucket === 'NOT' && slurRe.test(x.text)).slice(0, 8).forEach(x => { slurRe.lastIndex = 0;
  console.log(`  [${x.label}] ${x.text}`); });
