// ============================================================================
// Score externally-produced predictions (the no-API "agent mode" path, Option B).
//
// If you (an AI agent with model access, e.g. GitHub Copilot) classify the items
// yourself instead of calling an HTTP endpoint, save a predictions file and run
// this to fold it into results/results_prompts.json + results_prompts.md.
//
// Predictions file format (JSON):
//   { "model": "gpt-4o", "prompt": "minimal",
//     "predictions": [ { "id": "D0001", "pred": "MGY" }, { "id": "D0002", "pred": "NOT" }, ... ] }
//
// Usage:  node src/score_predictions.js path/to/predictions.json
// IDs must match data/hinglish_mgy_diag.csv. Classify BLIND (use text only;
// do not read the label/category columns when deciding).
// ============================================================================
const fs = require('fs');
const path = require('path');
const { loadDiag } = require('./data');
const { scorePredictions, renderMarkdown } = require('./prompt_experiment');

const inPath = process.argv[2];
if (!inPath) { console.error('usage: node src/score_predictions.js <predictions.json>'); process.exit(1); }
const pred = JSON.parse(fs.readFileSync(inPath, 'utf8'));
if (!pred.model || !pred.prompt || !Array.isArray(pred.predictions))
  { console.error('predictions file needs {model, prompt, predictions:[{id,pred}]}'); process.exit(1); }

const data = loadDiag();
const byId = new Map(pred.predictions.map(p => [p.id, (p.pred || '').toUpperCase().includes('MGY') ? 'MGY' : 'NOT']));
const missing = data.filter(x => !byId.has(x.id));
if (missing.length) { console.error(`MISSING predictions for ${missing.length} ids, e.g. ${missing.slice(0,3).map(x=>x.id).join(', ')}`); process.exit(1); }

const yT = data.map(x => x.label);
const yP = data.map(x => byId.get(x.id));
const score = scorePredictions(data, yT, yP);

const OUT = path.join(__dirname, '..', 'results');
const jsonPath = path.join(OUT, 'results_prompts.json');
const all = fs.existsSync(jsonPath) ? JSON.parse(fs.readFileSync(jsonPath, 'utf8')) : {};
all[pred.model] = all[pred.model] || {};
all[pred.model][pred.prompt] = score;
fs.writeFileSync(jsonPath, JSON.stringify(all, null, 2));
fs.writeFileSync(path.join(OUT, 'results_prompts.md'), renderMarkdown(all));
console.log(`Scored ${pred.model} / ${pred.prompt}:`, JSON.stringify(score));
console.log('Updated results/results_prompts.{json,md}');
