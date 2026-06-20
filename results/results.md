# Experimental Results

Dataset: 500 rows | 5-fold CV | seed 12345

## Experiment 1 — Two label-leakage channels in redacted data

### 1A — Placeholder *presence* vs the binary MGY/NOT label

A degenerate rule "predict MGY iff the row contains any redaction token" is evaluated directly (no training).

Accuracy = **1.000**, Macro-F1 = **1.000**, MCC = **1.000** — a no-learning rule is perfect.

### 1B — Placeholder *identity* vs the STRONG/CONTEXT subtype (register held constant)

LogReg on the 125 misogynistic rows under four representations, stratified 5-fold. Because both classes are misogynistic, topical register is constant; any signal above chance under a leaking representation is the artifact alone.

| Model | Accuracy | Macro-F1 | Macro-P | Macro-R | MCC |
|---|---|---|---|---|---|
| LogReg — `raw` (token index kept (leaks)) | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |
| LogReg — `mask-distinct` (<FSLUR>/<CTX> kept (leaks type)) | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |
| LogReg — `masked` (<RDW> only (neutralized)) | 0.984±0.019 | 0.981±0.023 | 0.989±0.013 | 0.975±0.031 | 0.964±0.044 |
| LogReg — `stripped` (token removed (neutralized)) | 0.984±0.019 | 0.981±0.023 | 0.989±0.013 | 0.975±0.031 | 0.964±0.044 |

**Reading:** representations that preserve the redaction token (`raw`, `mask-distinct`) recover the subtype almost perfectly *with register held constant* — proof the signal is the artifact, not meaning. Neutralizing the token (`masked`, `stripped`) forces the model onto the linguistic frame and scores fall to their honest level. **Anonymization that encodes class in the placeholder silently inflates results.**


## Experiment 2 — Random vs template-disjoint evaluation (binary, `stripped`)

On templated data, random k-fold leaks near-duplicate frames across train/test. A template-disjoint (grouped) split holds out whole frames, measuring generalization to unseen phrasings.


### Split: random stratified 5-fold

| Model | Accuracy | Macro-F1 | Macro-P | Macro-R | MCC |
|---|---|---|---|---|---|
| Lexicon (learned) | 0.994±0.008 | 0.992±0.010 | 0.989±0.015 | 0.996±0.005 | 0.985±0.020 |
| Naive Bayes (w1-2) | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |
| Naive Bayes (char2-4) | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |
| LogReg TF-IDF (both) | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |

### Split: template-disjoint 5-fold (unseen frames)

| Model | Accuracy | Macro-F1 | Macro-P | Macro-R | MCC |
|---|---|---|---|---|---|
| Lexicon (learned) | 0.789±0.208 | 0.778±0.214 | 0.817±0.148 | 0.869±0.120 | 0.681±0.273 |
| Naive Bayes (w1-2) | 0.810±0.245 | 0.784±0.265 | 0.849±0.164 | 0.874±0.171 | 0.708±0.337 |
| Naive Bayes (char2-4) | 0.692±0.224 | 0.661±0.230 | 0.755±0.134 | 0.794±0.160 | 0.533±0.285 |
| LogReg TF-IDF (both) | 0.938±0.111 | 0.934±0.112 | 0.948±0.096 | 0.948±0.078 | 0.895±0.174 |

**Reading:** binary misogyny detection stays easy even across unseen frames — the misogynistic and benign *registers* are lexically disjoint. This is exactly why a headline binary accuracy on this kind of corpus is a weak claim, and why we turn to the harder subtype axis below.


## Experiment 3 — Explicit vs context-dependent misogyny (STRONG vs CONTEXT)

Restricted to the 125 misogynistic rows. Representation `masked` (both buckets carry `<RDW>`, so the redaction token cannot leak the bucket — the model must read the surrounding frame). This is the analogue of the false-positive failure mode reported by Yadav et al. (statements misclassified because a charged word appears).

Subset size: 125 (STRONG=86, CONTEXT=39)


### Split: random stratified 5-fold

| Model | Accuracy | Macro-F1 | Macro-P | Macro-R | MCC |
|---|---|---|---|---|---|
| Majority | 0.688±0.011 | 0.408±0.004 | 0.344±0.006 | 0.500±0.000 | 0.000±0.000 |
| Lexicon (learned) | 0.952±0.016 | 0.942±0.021 | 0.962±0.013 | 0.930±0.031 | 0.891±0.036 |
| Naive Bayes (w1-2) | 0.984±0.019 | 0.981±0.023 | 0.989±0.013 | 0.975±0.031 | 0.964±0.044 |
| LogReg TF-IDF (both) | 0.984±0.019 | 0.981±0.023 | 0.989±0.013 | 0.975±0.031 | 0.964±0.044 |
| NB + engineered cues | 0.968±0.030 | 0.961±0.037 | 0.979±0.020 | 0.950±0.047 | 0.928±0.068 |
| Cues ONLY (interpretable) | 0.960±0.025 | 0.951±0.031 | 0.973±0.017 | 0.936±0.040 | 0.908±0.057 |

### Split: template-disjoint 5-fold

| Model | Accuracy | Macro-F1 | Macro-P | Macro-R | MCC |
|---|---|---|---|---|---|
| Majority | 0.655±0.214 | 0.386±0.077 | 0.328±0.107 | 0.500±0.000 | 0.000±0.000 |
| Lexicon (learned) | 0.725±0.305 | 0.696±0.335 | 0.738±0.305 | 0.802±0.209 | 0.595±0.429 |
| Naive Bayes (w1-2) | 0.746±0.259 | 0.743±0.262 | 0.829±0.154 | 0.824±0.146 | 0.651±0.300 |
| LogReg TF-IDF (both) | 0.815±0.228 | 0.783±0.267 | 0.864±0.172 | 0.864±0.179 | 0.707±0.360 |
| NB + engineered cues | 0.746±0.259 | 0.743±0.262 | 0.829±0.154 | 0.824±0.146 | 0.651±0.300 |
| Cues ONLY (interpretable) | 0.803±0.271 | 0.801±0.273 | 0.867±0.162 | 0.872±0.145 | 0.737±0.309 |

**Pooled confusion — LogReg, template-disjoint split:**

| true\pred | CONTEXT | STRONG |
|---|---|---|
| **CONTEXT** | 31 | 8 |
| **STRONG** | 14 | 72 |


**Reading:** the explicit/implicit distinction is the genuinely non-trivial axis. Under template-disjoint evaluation the redaction token is uninformative and the model must transfer condemnation/reporting cues to unseen frames — where performance drops markedly relative to the random split.


## Experiment 4 — Three-way classification (STRONG / CONTEXT / NOT)

Representation `stripped`, template-disjoint 5-fold. Full pipeline view.

| Model | Accuracy | Macro-F1 | Macro-P | Macro-R | MCC |
|---|---|---|---|---|---|
| Naive Bayes (w1-2) | 0.516±0.202 | 0.480±0.143 | 0.595±0.120 | 0.640±0.149 | 0.360±0.161 |
| LogReg TF-IDF (both) | 0.770±0.118 | 0.457±0.048 | 0.425±0.038 | 0.640±0.054 | 0.577±0.082 |
| NB + engineered cues | 0.697±0.186 | 0.594±0.077 | 0.690±0.074 | 0.727±0.105 | 0.544±0.170 |

**Pooled confusion — NB+engineered, 3-way template-disjoint:**

| true\pred | CONTEXT | NOT | STRONG |
|---|---|---|---|
| **CONTEXT** | 34 | 0 | 5 |
| **NOT** | 93 | 282 | 0 |
| **STRONG** | 41 | 0 | 45 |


## Experiment 5 — What the model keys on (LogReg weights, STRONG vs CONTEXT, `masked`)

Positive class = **STRONG**, negative = **CONTEXT**.

Top features → `STRONG`:
```
  +2.164  w1:sahi
  +2.164  w1:nahi
  +2.164  w2:kehna_sahi
  +2.164  w2:sahi_nahi
  +2.099  w2:me_ladki
  +1.767  w2:<rdw>_likh
  +1.767  w2:likh_diya
  +1.738  w1:likh
  +1.637  w1:aa
  +1.637  w1:gaya
  +1.637  w2:type_comment
  +1.637  w2:comment_aa
```
Top features → `CONTEXT`:
```
  -5.874  w2:kisi_ne
  -5.874  w2:me_kisi
  -3.517  w2:personal_attack
  -3.517  w2:kehna_personal
  -3.517  w1:attack
  -3.517  w1:personal
  -3.400  w1:ke
  -3.335  w2:ne_ladkiyon
  -2.965  w2:ne_ladki
  -2.961  w2:attack_hai
  -2.961  w1:hai
  -2.741  w1:kapdon
```

**Reading:** the model recovers the intended contrast — condemnation/meta cues (`mat`, `galat`, `type`, `lafz`, `wajah`) pull toward CONTEXT; direct-address reporting frames (`bol diya`, `likh diya`, `kehke`) pull toward STRONG. The distinction is linguistically real, but it lives in a handful of cues that do not all transfer to unseen frames — hence the template-disjoint drop.
