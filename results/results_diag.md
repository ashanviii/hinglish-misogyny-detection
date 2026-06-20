# Controlled Diagnostic — Results

Corpus: 416 rows | MGY 163 / NOT 253 | 5-fold construction-disjoint CV.

## D1 — Naive surface cues are decorrelated from the label by design

"Slur token present ⇒ MGY", evaluated on the 144 slur-bearing rows: accuracy **0.500** (chance, because the same token appears in USE_SLUR=MGY and COUNTER=NOT).
"Gendered subject present ⇒ MGY", on the 326 gendered rows: accuracy **0.500** (register no longer separates: IMPLICIT=MGY and BENIGN_OVERLAP=NOT share vocabulary).

## D2 — Binary MGY/NOT on the controlled set (construction-disjoint)

Now the model must read the FRAME, not the lexicon. `stripped` representation. Last column = contrast-set consistency.

| Model | Accuracy | Macro-F1 | MCC | Pair-consistency |
|---|---|---|---|---|
| Lexicon (learned) | 0.767±0.114 | 0.760±0.114 | 0.539±0.237 | 0.626 |
| Naive Bayes (w1-2) | 0.767±0.089 | 0.761±0.093 | 0.597±0.162 | 0.528 |
| Linear SVM (both) | 0.806±0.178 | 0.803±0.180 | 0.621±0.358 | 0.613 |
| LogReg TF-IDF (both) | 0.847±0.180 | 0.844±0.182 | 0.689±0.364 | 0.712 |
| Cues-only (interpretable) | 0.825±0.104 | 0.810±0.106 | 0.634±0.210 | 0.589 |

**Reading:** accuracy is no longer near-ceiling, and consistency (both halves of a minimal pair right) is the strictest test. A keyword model cannot be consistent — labeling the slur present pushes both pair members to MGY, so it fails every USE/COUNTER pair on one side.

## D3 — Use–mention contrast subset (USE_SLUR vs COUNTER only)

144 rows, 72 USE / 72 COUNTER; same slur tokens on both sides. Representation `masked` so the token gives nothing — only "bola" vs "mat bolo / galat" can separate.

| Model | Accuracy | Macro-F1 | MCC | Pair-consistency |
|---|---|---|---|---|
| Lexicon (learned) | 0.800±0.292 | 0.787±0.311 | 0.600±0.611 | 0.625 |
| Naive Bayes (w1-2) | 0.922±0.102 | 0.918±0.108 | 0.866±0.173 | 0.806 |
| Linear SVM (both) | 0.928±0.099 | 0.924±0.105 | 0.875±0.168 | 0.819 |
| LogReg TF-IDF (both) | 0.922±0.102 | 0.918±0.108 | 0.866±0.173 | 0.806 |
| Cues-only (interpretable) | 0.750±0.224 | 0.680±0.299 | 0.515±0.448 | 0.625 |

**Pooled confusion — LogReg:**

| true\pred | MGY | NOT |
|---|---|---|
| **MGY** | 58 | 14 |
| **NOT** | 0 | 72 |


**Reading:** this is the cleanest possible test of the use–mention distinction; the slur is held identical across the pair, so any score above chance is genuine frame comprehension and any consistency below 1.0 is a counter-speech failure.

## D4 — Per-category breakdown (LogReg binary, construction-disjoint)

Fraction of items in each category assigned the CORRECT binary label (recall of the gold label), pooled over folds.

| Category | Gold | Correct / Total | Accuracy |
|---|---|---|---|
| USE_SLUR | MGY | 54/72 | 0.750 |
| IMPLICIT | MGY | 78/91 | 0.857 |
| COUNTER | NOT | 49/72 | 0.681 |
| BENIGN_OVERLAP | NOT | 76/91 | 0.835 |
| BENIGN_NEUTRAL | NOT | 90/90 | 1.000 |

**Reading:** neutral chatter is trivial (1.000) and the gendered-but-benign hard negatives are handled reasonably (0.835), confirming the register confound is genuinely removed. The residual difficulty concentrates on exactly the two halves of the use–mention pair — `COUNTER` (0.681, counter-speech still over-flagged as misogynistic) and `USE_SLUR` (0.750, masked explicit use under-flagged) — plus `IMPLICIT` (0.857). The categories that require reading *intent* rather than surface form are where the models bleed.
