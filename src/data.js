const { loadDataset } = require('./csv');
const path = require('path');

const SUBJECTS = ['ladkiyon','ladkion','ladki','larkiyan','larki','aurton','aurat','mahilaon','mahila','women','woman','girls','girl'];

// Abstract a text into a template id by replacing redaction tokens and subject
// words with placeholders. Used for template-disjoint (group) evaluation.
function templateId(text) {
  let s = text.replace(/\[(?:FSLUR|CTX)_\d+\]/g, '<TOK>').toLowerCase();
  const re = new RegExp('\\b(' + SUBJECTS.join('|') + ')\\b', 'g');
  s = s.replace(re, '<SUBJ>').replace(/[।.,!?]/g, '').replace(/\s+/g, ' ').trim();
  return s;
}

function load() {
  const d = loadDataset(path.join(__dirname, '..', 'data', 'hinglish_mgy_binary_redacted_500.csv'));
  return d.map(x => ({ ...x, template: templateId(x.text) }));
}

// Construction group for the controlled diagnostic: paired frame-families are one
// held-out unit, so (a) minimal pairs are never split across train/test and
// (b) generalization is to an entirely unseen CONSTRUCTION.
function constructionGroup(category, frameId) {
  const idx = frameId.split(':')[1];
  if (category === 'USE_SLUR' || category === 'COUNTER') return 'UC:' + idx;       // use<->mention family
  if (category === 'IMPLICIT' || category === 'BENIGN_OVERLAP') return 'IB:' + idx; // misogynistic<->respectful family
  return 'BN:' + idx;
}

function loadDiag() {
  const { parseCSV } = require('./csv');
  const rows = parseCSV(path.join(__dirname, '..', 'data', 'hinglish_mgy_diag.csv'));
  const header = rows[0];
  const ix = Object.fromEntries(header.map((h, i) => [h.trim(), i]));
  return rows.slice(1).map(r => {
    const category = r[ix.category], frame_id = r[ix.frame_id];
    return {
      id: r[ix.id], text: r[ix.text], label: r[ix.label].trim(), bucket: category,
      category, frame_id, pair: r[ix.pair_id], subject: r[ix.subject], token: r[ix.token],
      template: constructionGroup(category, frame_id), // reuse 'template' as the group key for crossVal
    };
  });
}

module.exports = { load, loadDiag, templateId, constructionGroup, SUBJECTS };
