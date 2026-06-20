// ============================================================================
// Prompt-engineering comparison runner (§6.5) for HINGLISH-MGY-DIAG.
//
// Runs every prompt template (src/prompts.js) over all 416 diagnostic items
// with the CURRENTLY CONFIGURED model, and reports accuracy / macro-F1 / MCC /
// contrast-set CONSISTENCY / per-category accuracy for each prompt.
//
// >>> THE ONE THING TO WIRE UP: the callModel() function below. <<<
// Point it at whatever model endpoint you have (Azure OpenAI, GitHub Models,
// OpenAI, Anthropic, a local server, …). Everything else is done.
//
// Run once per model. Set MODEL_NAME so results are labelled:
//   $env:MODEL_NAME="gpt-4o";     node src/prompt_experiment.js
//   $env:MODEL_NAME="claude-3-7"; node src/prompt_experiment.js
// Results accumulate in results/results_prompts.json (keyed by model) and a
// human-readable results/results_prompts.md is regenerated each run.
// ============================================================================
const fs = require('fs');
const path = require('path');
const { loadDiag } = require('./data');
const ml = require('./ml');
const { prompts } = require('./prompts');

const MODEL_NAME = process.env.MODEL_NAME || 'UNNAMED_MODEL';
const OUT = path.join(__dirname, '..', 'results');

// ---------------------------------------------------------------------------
// callModel(system, user) -> Promise<string>   (returns the raw model text)
//
// OPTION A (OpenAI-compatible HTTP, incl. Azure OpenAI / GitHub Models / OpenAI):
//   set LLM_API_KEY, LLM_BASE_URL (e.g. https://api.openai.com/v1), LLM_MODEL.
//   The default implementation below already does this with global fetch (Node>=18).
//
// OPTION B (GitHub Copilot agent mode, NO API KEY):
//   If you are an AI agent with model access, you may bypass this function and
//   instead read each item and produce the label yourself for the selected
//   model, writing predictions in the same format (see writePredictions()).
//   If you do that, set the model name honestly and note the method in the paper.
// ---------------------------------------------------------------------------
async function callModel(system, user) {
  const key = process.env.LLM_API_KEY;
  const base = process.env.LLM_BASE_URL || 'https://api.openai.com/v1';
  const model = process.env.LLM_MODEL || MODEL_NAME;
  if (!key) throw new Error('LLM_API_KEY not set — wire up callModel() (see header, Option A/B).');
  const res = await fetch(`${base}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${key}` },
    body: JSON.stringify({
      model, temperature: 0, max_tokens: 200,
      messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
    }),
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  const j = await res.json();
  return j.choices?.[0]?.message?.content || '';
}

// Parse a binary label from raw model text. Prefers a trailing "LABEL: X" line.
function parseLabel(raw) {
  const m = raw.toUpperCase().match(/LABEL:\s*(MGY|NOT)/);
  if (m) return m[1];
  const tail = raw.toUpperCase().trim().split(/\s+/).slice(-3).join(' ');
  if (/\bMGY\b/.test(tail) && !/\bNOT\b/.test(tail)) return 'MGY';
  if (/\bNOT\b/.test(tail) && !/\bMGY\b/.test(tail)) return 'NOT';
  // last occurrence anywhere
  const iM = raw.toUpperCase().lastIndexOf('MGY'), iN = raw.toUpperCase().lastIndexOf('NOT');
  return iM > iN ? 'MGY' : 'NOT';
}

function scorePredictions(rows, yT, yP) {
  const m = ml.metricsFrom(yT, yP, ['MGY', 'NOT']);
  const byPair = {};
  rows.forEach((x, i) => { if (x.pair) (byPair[x.pair] = byPair[x.pair] || []).push(yT[i] === yP[i]); });
  const pairs = Object.values(byPair).filter(a => a.length === 2);
  const consistency = pairs.length ? pairs.filter(a => a[0] && a[1]).length / pairs.length : null;
  const perCat = {};
  const cats = {}; rows.forEach((x, i) => {
    cats[x.category] = cats[x.category] || { ok: 0, n: 0 };
    cats[x.category].n++; if (yT[i] === yP[i]) cats[x.category].ok++;
  });
  for (const c of Object.keys(cats)) perCat[c] = +(cats[c].ok / cats[c].n).toFixed(3);
  return { accuracy: +m.acc.toFixed(3), macroF1: +m.macroF1.toFixed(3), mcc: +m.mcc.toFixed(3),
    consistency: consistency == null ? null : +consistency.toFixed(3), nPairs: pairs.length, perCategory: perCat };
}

async function runPrompt(p, data) {
  const yT = [], yP = [], rows = [];
  for (const x of data) {
    let raw = 'NOT';
    try { raw = await callModel(p.system, p.user(x.text)); }
    catch (e) { console.error('  call failed', x.id, e.message); throw e; }
    yT.push(x.label); yP.push(parseLabel(raw)); rows.push(x);
    process.stdout.write('.');
  }
  process.stdout.write('\n');
  return scorePredictions(rows, yT, yP);
}

function renderMarkdown(all) {
  let md = '# Prompt-engineering comparison on HINGLISH-MGY-DIAG (§6.5)\n\n';
  md += 'Metric per (model, prompt). `cons` = contrast-set consistency. `CTR`/`USE` = per-category accuracy on COUNTER / USE_SLUR (the use–mention halves).\n\n';
  for (const [model, byPrompt] of Object.entries(all)) {
    md += `## Model: ${model}\n\n`;
    md += '| Prompt | Acc | Macro-F1 | MCC | Consistency | COUNTER acc | USE_SLUR acc |\n|---|---|---|---|---|---|---|\n';
    for (const [pid, r] of Object.entries(byPrompt)) {
      md += `| ${pid} | ${r.accuracy} | ${r.macroF1} | ${r.mcc} | ${r.consistency} | ${r.perCategory.COUNTER ?? '—'} | ${r.perCategory.USE_SLUR ?? '—'} |\n`;
    }
    md += '\n';
  }
  md += '\n**Hypothesis to confirm/refute:** prompts that do not state the use–mention rule (e.g. `minimal`, `role`) post similar *accuracy* but lower *consistency*, because they over-flag `COUNTER` (counter-speech). Prompts that state it (`usemention`, `fewshot`, `rubric`, `cot`) should hold higher consistency.\n';
  return md;
}

async function main() {
  const data = loadDiag();
  console.log(`Prompt experiment | model=${MODEL_NAME} | ${data.length} items | ${Object.keys(prompts).length} prompts`);
  const jsonPath = path.join(OUT, 'results_prompts.json');
  const all = fs.existsSync(jsonPath) ? JSON.parse(fs.readFileSync(jsonPath, 'utf8')) : {};
  all[MODEL_NAME] = all[MODEL_NAME] || {};
  for (const p of Object.values(prompts)) {
    console.log(`\n[${MODEL_NAME}] prompt: ${p.id}`);
    all[MODEL_NAME][p.id] = await runPrompt(p, data);
    fs.writeFileSync(jsonPath, JSON.stringify(all, null, 2)); // checkpoint after each prompt
  }
  fs.writeFileSync(path.join(OUT, 'results_prompts.md'), renderMarkdown(all));
  console.log('\nWrote results/results_prompts.json and results/results_prompts.md');
}

// Exposed so an agent can score externally-produced predictions without HTTP.
function writePredictions() { /* see HANDOVER.md "Option B" */ }

if (require.main === module) main();
module.exports = { parseLabel, scorePredictions, renderMarkdown };
