// ============================================================================
// LLM / transformer zero-shot baseline harness for the controlled diagnostic.
//
//   *** NOT EXECUTED in the present study: no model API/credentials or GPU were
//   available in the authoring environment. This script is provided so the
//   zero-shot comparison (the analogue of Yadav et al.'s BART/ChatGPT runs) is
//   one command away for anyone with access. It computes the SAME metrics —
//   accuracy, macro-F1, MCC, and contrast-set CONSISTENCY — as the classical
//   baselines, on the SAME construction-disjoint test items. ***
//
// Usage (OpenAI-compatible chat endpoint):
//   set LLM_API_KEY=...               (PowerShell: $env:LLM_API_KEY="...")
//   set LLM_BASE_URL=https://api.openai.com/v1   (or any compatible gateway)
//   set LLM_MODEL=gpt-4o-mini
//   node src/llm_baseline.js
//
// Requires Node >= 18 (global fetch).
// ============================================================================
const fs = require('fs');
const path = require('path');
const { loadDiag } = require('./data');
const ml = require('./ml');

const API_KEY = process.env.LLM_API_KEY;
const BASE = process.env.LLM_BASE_URL || 'https://api.openai.com/v1';
const MODEL = process.env.LLM_MODEL || 'gpt-4o-mini';

const SYSTEM = `You are a careful content-moderation classifier for code-mixed Hinglish (Romanised Hindi + English).
Decide whether a message is MISOGYNISTIC (label MGY) or NOT (label NOT).
CRITICAL: a slur being *mentioned, quoted, or condemned* (counter-speech, e.g. "don't call her X", "calling her X is wrong") is NOT misogynistic.
A slur or sexist claim *directed at / used against* a woman IS misogynistic.
Implicit misogyny (stereotyping, moral-policing, objectification) with no slur is still MGY.
Answer with exactly one token: MGY or NOT.`;

function buildPrompt(text) {
  return `Message: "${text}"\nLabel (MGY or NOT):`;
}

async function classify(text) {
  const res = await fetch(`${BASE}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${API_KEY}` },
    body: JSON.stringify({
      model: MODEL, temperature: 0, max_tokens: 4,
      messages: [{ role: 'system', content: SYSTEM }, { role: 'user', content: buildPrompt(text) }],
    }),
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  const j = await res.json();
  const out = (j.choices?.[0]?.message?.content || '').toUpperCase();
  return out.includes('MGY') ? 'MGY' : 'NOT';
}

async function main() {
  if (!API_KEY) {
    console.error('LLM_API_KEY not set — this harness is a provided-but-unexecuted baseline.');
    console.error('See header for usage. Exiting without calling any API.');
    process.exit(2);
  }
  const data = loadDiag();
  const yT = [], yP = [], rows = [];
  for (const x of data) {
    let pred = 'NOT';
    try { pred = await classify(x.text); } catch (e) { console.error('skip', x.id, e.message); continue; }
    yT.push(x.label); yP.push(pred); rows.push(x);
    process.stdout.write('.');
  }
  const m = ml.metricsFrom(yT, yP, ['MGY', 'NOT']);
  // consistency
  const byPair = {};
  rows.forEach((x, i) => { if (x.pair) (byPair[x.pair] = byPair[x.pair] || []).push(yT[i] === yP[i]); });
  const pairs = Object.values(byPair).filter(a => a.length === 2);
  const consistency = pairs.filter(a => a[0] && a[1]).length / pairs.length;
  const report = { model: MODEL, n: yT.length, accuracy: m.acc, macroF1: m.macroF1, mcc: m.mcc, consistency, nPairs: pairs.length };
  console.log('\n', report);
  fs.writeFileSync(path.join(__dirname, '..', 'results', 'results_llm.json'), JSON.stringify(report, null, 2));
}
main();
