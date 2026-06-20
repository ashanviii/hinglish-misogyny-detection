const { loadDataset } = require('./csv');
const path = require('path');
const d = loadDataset(path.join(__dirname, '..', 'data', 'hinglish_mgy_binary_redacted_500.csv'));

// Token families
const fslur = new Set(), ctx = new Set(), other = new Set();
for (const x of d) {
  for (const m of x.text.matchAll(/\[([A-Z]+)_(\d+)\]/g)) {
    if (m[1] === 'FSLUR') fslur.add(m[0]);
    else if (m[1] === 'CTX') ctx.add(m[0]);
    else other.add(m[0]);
  }
}
console.log('FSLUR tokens:', [...fslur].sort());
console.log('CTX tokens:', [...ctx].sort());
console.log('Other bracket tokens:', [...other].sort());

// Build templates: replace bracket tokens with <TOK> and lowercased variable slots.
// Variable subject words observed
const subjects = ['ladkiyon','ladki','ladkion','aurat','aurton','mahila','mahilaon','women','woman','girl','girls','larki','larkiyan'];
function templatize(t) {
  let s = t.replace(/\[(?:FSLUR|CTX)_\d+\]/g, '<TOK>');
  s = s.toLowerCase();
  // collapse subject words to <SUBJ>
  const re = new RegExp('\\b(' + subjects.join('|') + ')\\b', 'g');
  s = s.replace(re, '<SUBJ>');
  s = s.replace(/\s+/g, ' ').trim();
  return s;
}

const tmpl = {};
for (const x of d) {
  const t = templatize(x.text);
  if (!tmpl[t]) tmpl[t] = { count: 0, label: x.label, bucket: new Set() };
  tmpl[t].count++; tmpl[t].bucket.add(x.bucket);
}
const tlist = Object.entries(tmpl).sort((a, b) => b[1].count - a[1].count);
console.log('\nDistinct templates (after <TOK>/<SUBJ> abstraction):', tlist.length);
console.log('\nAll templates [label|buckets|count]:');
for (const [t, info] of tlist) {
  console.log(`  [${info.label}|${[...info.bucket].join(',')}|${info.count}] ${t}`);
}

// distinct raw texts per bucket
const distinctByBucket = {};
for (const b of ['STRONG','CONTEXT','NOT']) {
  distinctByBucket[b] = new Set(d.filter(x=>x.bucket===b).map(x=>x.text)).size;
}
console.log('\nDistinct raw texts per bucket:', distinctByBucket);
