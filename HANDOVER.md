# HANDOVER — finish the LLM / prompt-engineering experiments

**Audience:** an AI coding agent (e.g. GitHub Copilot) with access to one or more LLMs.
**Goal:** run the experiments the original author could not (no API key in their environment), then fold the results into the paper. Everything else in this project is finished, tested, and reproducible.

---

## 0. TL;DR — what you need to do

1. Pick a model in your model picker (or wire an API endpoint).
2. Run the **8 prompt templates** in `src/prompts.js` over the **416 items** in `data/hinglish_mgy_diag.csv`.
3. Score each with the code already written (accuracy, macro-F1, MCC, **contrast-set consistency**, per-category).
4. Repeat for 2–3 different models.
5. Write the results into **§6.5** of `paper/PAPER.md`, add one table + one figure, rebuild `paper/paper.html`.
6. Also do the simplest version first: **one arms-length model, the `usemention` prompt** — that single result closes the paper's biggest gap (no arms-length system evaluation).

**Do not fabricate numbers. Do not read the gold labels while classifying. Keep slurs redacted. Disclose which models/versions you used and the date.**

---

## 1. What this project is (so you understand the point)

The paper studies **misogyny detection in code-mixed Hinglish** and centres on the **use–mention distinction**: a slur *used* against a woman (misogynistic) vs. the same slur *mentioned / condemned* in counter-speech like "don't call her that" (not misogynistic). Lexicon-keyed models can't tell these apart.

The contribution is **HINGLISH-MGY-DIAG**, a controlled contrast-set test built so the easy shortcuts (slur presence, gendered vocabulary) don't predict the label. It has **416 items in 163 minimal pairs** across 5 categories: `USE_SLUR`, `IMPLICIT` (both MGY), `COUNTER`, `BENIGN_OVERLAP`, `BENIGN_NEUTRAL` (all NOT). It is scored with **pair-consistency**: a model gets credit on a pair only if it labels *both* halves correctly.

**Why the prompt experiment matters:** classical baselines already run, and an author-model "ceiling" run exists but is contaminated (the author-model had seen the design). The paper has **no arms-length real-model result** — that's the gap. The novel hook is that the diagnostic lets you compare prompts on **consistency**, not just accuracy: the expected finding is that prompts which don't mention the use–mention rule keep similar accuracy but lose consistency by over-flagging counter-speech (`COUNTER`).

---

## 2. File map (what's already here)

```
data/hinglish_mgy_diag.csv        the 416-item diagnostic (id,text,label,category,frame_id,pair_id,subject,token)
src/prompts.js                    8 prompt templates  <-- the experiment's prompts
src/prompt_experiment.js          runner: loops prompts x items, scores everything.  WIRE UP callModel() (Option A)
src/score_predictions.js          score predictions you produced yourself (Option B, no API)
src/ml.js                         metrics (accuracy, macro-F1, MCC), splits — already unit-tested
src/data.js                       loadDiag() loader
src/llm_baseline.js               single-prompt arms-length harness (older; prompt_experiment.js supersedes it)
src/build_paper.js                rebuilds paper/paper.html from paper/PAPER.md (inlines figures)
paper/PAPER.md                    the paper (markdown source)            <-- add §6.5 here
results/results_diag.md/.json     classical-model results (for comparison)
results/results_llm.json          the author-model ceiling run (caveated)
```

Run `node src/selftest.js` first — it should print `12 passed, 0 failed`. That confirms the metric code is intact.

---

## 3. Do the experiment — two paths

### Option A — you have an API endpoint (preferred; fully automated)

`src/prompt_experiment.js` is ready. The ONLY thing to wire is `callModel(system, user)` at the top — by default it does an OpenAI-compatible POST using env vars. Set them and run, once per model:

```powershell
# OpenAI-compatible (OpenAI, Azure OpenAI, GitHub Models, local servers, …)
$env:LLM_API_KEY = "<key>"
$env:LLM_BASE_URL = "https://api.openai.com/v1"   # or your Azure/GH Models endpoint
$env:LLM_MODEL   = "gpt-4o"
$env:MODEL_NAME  = "gpt-4o"                         # label for the results table
node src/prompt_experiment.js
```

For Azure OpenAI or Anthropic, edit `callModel()` to match that API's request shape (URL, headers, body). Keep `temperature: 0`. Results accumulate in `results/results_prompts.json` (keyed by `MODEL_NAME`) and `results/results_prompts.md` is regenerated each run. Re-run with a different `MODEL_NAME`/model for each model you want to compare.

### Option B — you are an agent with model access but NO API key

Classify the items yourself with the model selected in your picker, then score with the provided script. For **each** (model, prompt) combination:

1. Load the items. Use **text only** — do **not** look at the `label` or `category` columns when deciding (stay blind). You can get a clean list with:
   ```
   node -e "require('./src/data').loadDiag().forEach(x=>console.log(x.id+'\t'+x.text))"
   ```
2. For each item, build the message with the chosen template from `src/prompts.js` (`prompts[<id>].system` and `prompts[<id>].user(text)`), classify it, and record `{ "id": "...", "pred": "MGY" | "NOT" }`.
3. Save a predictions file `results/predictions/<model>__<prompt>.json`:
   ```json
   { "model": "gpt-4o", "prompt": "usemention",
     "predictions": [ { "id": "D0001", "pred": "MGY" }, { "id": "D0002", "pred": "NOT" } ] }
   ```
4. Score it (this updates the same results files Option A writes):
   ```
   node src/score_predictions.js results/predictions/gpt-4o__usemention.json
   ```

Classify all 416 items per (model, prompt). If that's a lot, **start with the single most valuable run**: your best model + the `usemention` prompt. That one result already closes the paper's main gap. Then add `minimal` (the naive contrast) to show the consistency gap. Then expand.

### The 8 prompts (in `src/prompts.js`)
`minimal`, `definition`, `usemention`, `cot` (chain-of-thought), `fewshot`, `role`, `translate` (translate-to-English first), `rubric` (5-category then map). The few-shot examples are hand-authored and not from the eval set (no leakage) — keep them fixed.

---

## 4. What the scores mean (already computed for you)

Per (model, prompt) you get: `accuracy`, `macroF1`, `mcc`, **`consistency`** (fraction of the 163 pairs with BOTH halves correct), and `perCategory` accuracy. The two per-category numbers that matter most are **`COUNTER`** (counter-speech — over-flagging it is the false-positive that silences people condemning abuse) and **`USE_SLUR`** (explicit use — under-flagging it lets abuse through).

**The hypothesis to test:** prompts without the use–mention rule (`minimal`, `role`, `definition`) get similar accuracy but **lower consistency and lower `COUNTER` accuracy** than prompts that state it (`usemention`, `fewshot`, `rubric`, `cot`). Report whether this holds, per model. If it does NOT hold (e.g. a strong model is robust regardless of prompt), that is also a clean, honest finding — report it as such.

---

## 5. Fold results into the paper

1. Add **§6.5 "Which prompt survives the use–mention test?"** to `paper/PAPER.md`, right after §6.4. Include:
   - One sentence on setup (which models, which prompts, `temperature 0`, the date, and whether you used Option A or B — state the method honestly).
   - The results table (copy from `results/results_prompts.md`).
   - 2–4 sentences of analysis: does the accuracy-vs-consistency gap appear? Which prompt wins on consistency? Which category drives the misses?
2. Update the **abstract** and **§7 / §10**: the LLM comparison is now (partly) executed arms-length — adjust the wording that currently says the arms-length harness is "provided but not run."
3. Update **Limitation 4** in §8 (it currently says the independent harness was not run) to reflect what you actually ran.
4. Optional figure: a grouped bar chart of consistency per prompt (mirror `src/figures_diag.js` style) → `figures/fig9_prompts.svg`, then reference it in §6.5.
5. Rebuild: `node src/build_paper.js` → regenerates `paper/paper.html` with everything inlined.
6. Add references if you cite prior prompt-engineering work, e.g. *Leveraging the Potential of Prompt Engineering for Hate Speech Detection in Low-Resource Languages* (arXiv:2506.23930, 2025) and HateCheckHIn (Das et al., 2022) — both relevant; position your novelty narrowly (misogyny + Hinglish + **consistency-based** prompt comparison).

---

## 6. Hard rules (research integrity)

- **No fabricated results.** Every number must come from an actual model run. If you can't run something, say so in the paper.
- **Classify blind.** Never feed the `label`/`category` columns into the model or use them to decide; they are ground truth for scoring only.
- **Keep slurs redacted.** The `[FSLUR_xx]` tokens stay as-is. Never reconstruct a real slur.
- **Disclose.** In the paper, name the exact models and versions, the date, `temperature`, and whether predictions came via API (Option A) or agent classification (Option B). If a human/AI assisted the writing, keep the existing AI-use disclosure honest.
- **Determinism.** Use `temperature: 0`. The dataset and classical pipeline are seeded (12345 for CV/SGD, 20240614 for generation) — don't change seeds.
- Don't try to "beat" AI-text detectors; that's not the goal and it's not honest.

---

## 7. Definition of done

- [ ] `node src/selftest.js` → 12/12 pass.
- [ ] At least **one arms-length model** scored on the `usemention` prompt (gap-closer). Ideally all 8 prompts × 2–3 models.
- [ ] `results/results_prompts.json` + `results_prompts.md` populated with real runs.
- [ ] **§6.5** written into `paper/PAPER.md` with the table + honest analysis; abstract/§7/§8/§10 updated to reflect the executed runs.
- [ ] `node src/build_paper.js` re-run; `paper/paper.html` current.
- [ ] Models, versions, date, and method disclosed in the paper.

That's the whole job. The harness, scorer, metrics, dataset, and paper are all in place — you're supplying the one ingredient the author's environment lacked: real model inference.
