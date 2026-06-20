# Used, Mentioned, or Condemned? — Hinglish Misogyny Detection (diagnostic study + resource)

A self-contained, dependency-free study extending Yadav, Kaushik & McDaid (2024),
*"Leveraging Weakly Annotated Data for Hate Speech Detection in Code-Mixed Hinglish"* (arXiv:2403.02121).

**Authors:** Ashanvi Yadav (Manipal University Jaipur) · Shubham Bhardwaj (BITS Pilani, Hyderabad Campus).
**Full paper:** [`paper/PAPER.md`](paper/PAPER.md) · reference PDF in [`paper/`](paper/).

---

## 1. The one-paragraph version

A misogyny detector keyed to a slur lexicon cannot tell a slur **used** against a woman from one
**mentioned** in counter-speech (*"don't call her that"*) — the use–mention distinction. We (i) **diagnose**
why standard evaluation on found, redacted Hinglish data hides this (label leakage + register
separability make the task look solved), and (ii) **build and release HINGLISH-MGY-DIAG**, a controlled
contrast-set diagnostic where those shortcuts are removed *by construction*, plus a strict
*contrast-set consistency* metric, five from-scratch baselines, and a ready-to-run LLM harness.

## 2. Two parts

**Part A — Diagnosis (found 500-row corpus).** A no-learning rule "MGY iff a redaction token is present"
scores **1.000** (anonymization leaks the label); even masked, misogynistic vs benign registers are
disjoint so bag-of-words hits macro-F1 ≈1.00 under random CV but collapses (0.66–0.93) under
template-disjoint CV. A learning curve shows the subtype signal is **data-starved**.

**Part B — Remedy (HINGLISH-MGY-DIAG, generated).** 416 items, **163 minimal pairs**, 5 linguistically
defined categories, **40 frames**. By design, slur presence and gendered register are **decorrelated from
the label** (both naïve cues score exactly **0.500**). On the cleanest use-vs-mention contrast the best
model reaches 0.93 accuracy but only **0.82 consistency** — it still mislabels ~1 counter-speech pair in 5,
and *under*-flags explicit slur use (the dangerous direction).

We position this against **HateCheckHIn** (Das et al., LREC 2022, the Hindi HateCheck) with a deliberately
*narrow* novelty claim: not "first behavioral test for Indian-language hate speech" (HateCheckHIn precedes
us), but a **misogyny-specific, Romanised-Hinglish, label-flipping minimal-*pair* set with a paired-consistency
metric** and measured decorrelation of slur-presence/register from the label. Organic external validation is
mapped to Negi et al.'s (2024) 17k-comment corpus (see paper §8b Future Work).

## 3. Folder map

```
research paper/
├── README.md                 ← you are here
├── data/
│   ├── hinglish_mgy_binary_redacted_500.csv   found corpus (Part A)
│   └── hinglish_mgy_diag.csv                   HINGLISH-MGY-DIAG, generated (Part B)
├── src/                      all code — pure Node.js, ZERO dependencies
│   ├── ml.js                 NB, LogReg, Linear SVM, TF-IDF, metrics, MCC, k-fold, RNG (from scratch)
│   ├── selftest.js           12 unit tests validating the library
│   ├── csv.js / data.js / harness.js     parsing, loaders, CV harness + consistency
│   ├── generate.js           THE GENERATOR for HINGLISH-MGY-DIAG (documented, seeded)
│   ├── profile.js / explore2.js          found-corpus profiling
│   ├── experiments.js        Part A: leakage / register / protocol / subtype (Exp 1–5)
│   ├── erroranalysis.js      use–mention error dump
│   ├── learningcurve.js      is the found corpus big enough? (Fig 6)
│   ├── experiments_diag.js   Part B: controlled diagnostic + contrast-set consistency (D1–D3)
│   ├── figures.js / figures_diag.js      SVG figure generation
│   ├── llm_selfrun.js        zero-shot LLM run BY THE AUTHOR-MODEL (executed; ceiling + label validation)
│   └── llm_baseline.js       arms-length zero-shot LLM harness (provided; needs API key)
├── results/
│   ├── results.md/.json                 Part A tables + confusion matrices
│   ├── error_analysis.md                misclassified examples
│   ├── learning_curve.md/.json          performance vs #training frames
│   └── results_diag.md/.json            Part B tables + consistency (results_llm.json when LLM run)
├── figures/  fig1..fig8 .svg
└── paper/    PAPER.md  +  original reference paper
```

## 4. Reproduce (Node ≥ 16; ≥ 18 only for the optional LLM harness; no `npm install`)

```bash
cd "research paper"
node src/selftest.js          # 12/12 unit tests
node src/generate.js          # build HINGLISH-MGY-DIAG (deterministic)
node src/experiments.js       # Part A diagnosis
node src/erroranalysis.js     # use–mention errors
node src/learningcurve.js     # learning-curve probe
node src/experiments_diag.js  # Part B controlled diagnostic + consistency
node src/llm_selfrun.js       # zero-shot LLM ceiling (author-model) + label validation
node src/figures.js && node src/figures_diag.js   # all figures
# optional arms-length zero-shot LLM baseline (needs credentials):
#   $env:LLM_API_KEY="..."; node src/llm_baseline.js
```

Deterministic: seed 12345 for CV/SGD; seed 20240614 for generation.

## 5. Models (all from scratch in `src/ml.js`)

Majority · learned lexicon · a-priori cue lexicon · Multinomial Naive Bayes · TF-IDF Logistic
Regression (SGD+L2) · **Linear SVM (Pegasos hinge-SGD)**.
Evaluation: stratified, **template-disjoint**, and **construction-disjoint** k-fold; accuracy,
macro-P/R/F1, multiclass MCC, confusion matrices, and **contrast-set consistency**.

## 6. Ethics

Slurs are redacted throughout and never reconstructed; models never need a slur's identity. Misogyny
detection is dual-use — lexicon-keyed systems misfire on counter-speech (silencing those condemning
abuse) and can under-flag explicit use. For defensive moderation research only.
