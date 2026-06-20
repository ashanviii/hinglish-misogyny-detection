// ============================================================================
// CONTROLLED CONTRAST-SET GENERATOR for code-mixed Hinglish misogyny diagnostics.
//
// Design goal: build a *diagnostic* corpus in which the naive surface cues that
// inflate scores on found data are DECORRELATED from the label by construction,
// so that performance reflects use of the linguistic FRAME, not the lexicon.
//
// Five categories (binary label in brackets):
//   USE_SLUR        [MGY]  explicit slur USED against a woman
//   IMPLICIT        [MGY]  misogyny with NO slur (stereotype / moral-policing / objectification)
//   COUNTER         [NOT]  the SAME slur token, but condemned / mentioned (counter-speech)
//   BENIGN_OVERLAP  [NOT]  benign, but shares gendered vocabulary (register hard-negative)
//   BENIGN_NEUTRAL  [NOT]  benign office / casual chatter (easy negative)
//
// Two decorrelations this buys us:
//   (1) Slur-token PRESENCE no longer predicts the label: it appears in BOTH
//       USE_SLUR (MGY) and COUNTER (NOT). A keyword classifier is defeated by design.
//   (2) Gendered REGISTER no longer predicts the label: BENIGN_OVERLAP (NOT) and
//       IMPLICIT (MGY) share the same topical vocabulary (women, clothes, work).
//
// Contrast sets (minimal pairs, linked by pair_id):
//   USE_SLUR  <-> COUNTER         same {subject, slur}, frame flips USE -> MENTION
//   IMPLICIT  <-> BENIGN_OVERLAP  same {subject, topic},  frame flips misogynistic -> respectful
//
// Slurs remain redacted as [FSLUR_xx] throughout (ethics + it is the cue under study).
// Fully deterministic under SEED.
// ============================================================================
const fs = require('fs');
const path = require('path');
const { rng, shuffle } = require('./ml');

const SEED = 20240614;
const rand = rng(SEED);
function pick(a) { return a[Math.floor(rand() * a.length)]; }

const SUBJ = ['ladki', 'ladkiyon', 'aurat', 'mahila', 'women', 'girl', 'larki', 'woman', 'us ladki', 'us aurat'];
const SLUR = Array.from({ length: 20 }, (_, i) => `[FSLUR_${String(i + 1).padStart(2, '0')}]`);
const GARMENT = ['jeans', 'shorts', 'crop top', 'western kapde', 'skirt', 'dress'];

// ---- Frame banks. {S}=subject, {T}=slur token, {G}=garment ----
// USE and COUNTER are paired by index family via a shared "verb stem" so we can
// build true minimal pairs.
const USE = [
  s => `${s.S} ko ${s.T} bol diya.`,
  s => `${s.S} ko ${s.T} keh ke DM kiya.`,
  s => `comment section me ${s.S} ko ${s.T} likh diya.`,
  s => `${s.S} ko ${s.T} bol ke tag kar diya.`,
  s => `online sab ${s.S} ko ${s.T} keh rahe hain.`,
  s => `${s.S} ne ${s.G} pehna to log ${s.T} bolne lage.`,
  s => `${s.S} ko public post me ${s.T} likha gaya.`,
  s => `group me ${s.S} ko ${s.T} keh ke chhed rahe the.`,
];
const COUNTER = [
  s => `${s.S} ko ${s.T} bolna bilkul galat hai.`,
  s => `kisi ${s.S} ko ${s.T} mat bolo.`,
  s => `${s.S} ko ${s.T} kehna turant band karo.`,
  s => `${s.T} jaise shabd ${s.S} ke liye use mat karo.`,
  s => `report karo agar koi ${s.S} ko ${s.T} bole.`,
  s => `${s.S} ko ${s.T} bolna kisi bhi haal me sahi nahi.`,
  s => `please ${s.S} ko ${s.T} mat kaho, ye galat hai.`,
  s => `${s.S} ko ${s.T} kehne walon ko block karo.`,
];
// IMPLICIT and BENIGN_OVERLAP paired by a shared "topic" id.
const TOPICS = ['ghar', 'kapde', 'kaam', 'padhai', 'driving', 'leadership', 'bahar'];
const IMPLICIT = {
  ghar: s => `${s.S} ka kaam sirf ghar sambhalna hai.`,
  kapde: s => `${s.S} ko ${pick(GARMENT)} nahi pehनना chahiye, sanskaar khatam.`,
  kaam: s => `${s.S} ka kaam office me nahi, kitchen me hai.`,
  padhai: s => `${s.S} ko zyada padhne ki zaroorat nahi hoti.`,
  driving: s => `${s.S} dhang se gaadi chala hi nahi sakti.`,
  leadership: s => `${s.S} emotional hoti hai, team lead nahi kar sakti.`,
  bahar: s => `${s.S} ko raat ko bahar nikalna hi nahi chahiye.`,
};
const BENIGN_OVERLAP = {
  ghar: s => `${s.S} ghar aur office dono badhiya sambhal leti hai.`,
  kapde: s => `${s.S} ne aaj ${pick(GARMENT)} liye, look acha tha.`,
  kaam: s => `${s.S} ne aaj ka kaam time se pehle finish kiya.`,
  padhai: s => `${s.S} ne entrance exam top kiya, proud moment.`,
  driving: s => `${s.S} ne lambi highway drive solo complete ki.`,
  leadership: s => `${s.S} ne meeting bahut confidently lead ki.`,
  bahar: s => `${s.S} raat ki shift karke safe ghar pahunch gayi.`,
};
const BENIGN_NEUTRAL = [
  () => `kal standup kitne baje hai?`,
  () => `weekend plans bana liye kya?`,
  () => `PR me comments resolve kar do.`,
  () => `coffee peene chalte ho?`,
  () => `meeting notes share kar dena.`,
  () => `laptop charge rehna chahiye meeting se pehle.`,
  () => `aaj ka match dekhoge?`,
  () => `client call postpone ho gaya.`,
  () => `presentation kal tak ready ho jayegi.`,
  () => `code review kal subah rakho.`,
];

const rows = [];
let uid = 0, pid = 0;
function add(text, label, category, frameId, pairId, subject, token) {
  rows.push({ id: 'D' + String(++uid).padStart(4, '0'), text, label, category, frame_id: category + ':' + frameId, pair_id: pairId, subject: subject || '', token: token || '' });
}

// --- contrast pairs: USE_SLUR <-> COUNTER (same subject + slur, frame index = pair family) ---
const N_USE_COUNTER = 9; // instantiations per frame index
for (let fi = 0; fi < USE.length; fi++) {
  for (let n = 0; n < N_USE_COUNTER; n++) {
    const S = pick(SUBJ), T = pick(SLUR), G = pick(GARMENT);
    const slots = { S, T, G };
    pid++;
    add(USE[fi](slots), 'MGY', 'USE_SLUR', fi, 'P' + pid, S, T);
    add(COUNTER[fi](slots), 'NOT', 'COUNTER', fi, 'P' + pid, S, T);
  }
}

// --- contrast pairs: IMPLICIT <-> BENIGN_OVERLAP (same subject + topic) ---
const N_IMPL = 13;
for (const topic of TOPICS) {
  for (let n = 0; n < N_IMPL; n++) {
    const S = pick(SUBJ);
    pid++;
    add(IMPLICIT[topic]({ S }), 'MGY', 'IMPLICIT', topic, 'P' + pid, S, '');
    add(BENIGN_OVERLAP[topic]({ S }), 'NOT', 'BENIGN_OVERLAP', topic, 'P' + pid, S, '');
  }
}

// --- benign neutral filler (unpaired NOT) to reflect real base-rate of off-topic chatter ---
const N_NEUTRAL = 9;
for (let fi = 0; fi < BENIGN_NEUTRAL.length; fi++)
  for (let n = 0; n < N_NEUTRAL; n++)
    add(BENIGN_NEUTRAL[fi](), 'NOT', 'BENIGN_NEUTRAL', fi, '', '', '');

// shuffle for storage
const shuffled = shuffle(rows, rng(SEED + 1));

// write CSV (quote text)
function esc(v) { return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
const header = ['id', 'text', 'label', 'category', 'frame_id', 'pair_id', 'subject', 'token'];
const csv = [header.join(',')].concat(
  shuffled.map(r => header.map(h => esc(String(r[h]))).join(','))
).join('\n');
const outPath = path.join(__dirname, '..', 'data', 'hinglish_mgy_diag.csv');
fs.writeFileSync(outPath, csv);

// summary
const byCat = {}, byLabel = {};
for (const r of rows) { byCat[r.category] = (byCat[r.category] || 0) + 1; byLabel[r.label] = (byLabel[r.label] || 0) + 1; }
const pairs = new Set(rows.filter(r => r.pair_id).map(r => r.pair_id)).size;
console.log('Wrote', outPath);
console.log('Total rows:', rows.length, '| labels:', byLabel);
console.log('Categories:', byCat);
console.log('Contrast pairs (linked):', pairs);
console.log('Distinct frames:', new Set(rows.map(r => r.frame_id)).size);
console.log('Rows with slur token:', rows.filter(r => r.token).length,
  '| of those MGY:', rows.filter(r => r.token && r.label === 'MGY').length,
  'NOT:', rows.filter(r => r.token && r.label === 'NOT').length, '(should be ~50/50 => leakage defeated)');
