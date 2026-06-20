# Learning Curve (template-disjoint, 25 repeats, 30% templates held out)

## Binary MGY/NOT (LogReg, macro-F1)

| train templates | frac | score (mean±SD) |
|---|---|---|
| 6 | 15% | 0.533 ± 0.186 |
| 13 | 30% | 0.642 ± 0.217 |
| 19 | 45% | 0.797 ± 0.182 |
| 26 | 60% | 0.845 ± 0.189 |
| 32 | 75% | 0.877 ± 0.172 |
| 39 | 90% | 0.901 ± 0.112 |
| 43 | 100% | 0.927 ± 0.111 |

**Marginal gain from 90%→100% of templates:** +0.026 (still climbing — more data would help).

## STRONG vs CONTEXT (Cues-only, MCC)

| train templates | frac | score (mean±SD) |
|---|---|---|
| 4 | 15% | 0.336 ± 0.409 |
| 7 | 30% | 0.455 ± 0.398 |
| 11 | 45% | 0.466 ± 0.399 |
| 14 | 60% | 0.464 ± 0.402 |
| 18 | 75% | 0.505 ± 0.449 |
| 22 | 90% | 0.499 ± 0.424 |
| 24 | 100% | 0.489 ± 0.408 |

**Marginal gain from 90%→100% of templates:** -0.009 (roughly saturated).

## STRONG vs CONTEXT (LogReg, MCC)

| train templates | frac | score (mean±SD) |
|---|---|---|
| 4 | 15% | 0.184 ± 0.317 |
| 7 | 30% | 0.283 ± 0.326 |
| 11 | 45% | 0.383 ± 0.309 |
| 14 | 60% | 0.434 ± 0.343 |
| 18 | 75% | 0.481 ± 0.324 |
| 22 | 90% | 0.503 ± 0.312 |
| 24 | 100% | 0.525 ± 0.332 |

**Marginal gain from 90%→100% of templates:** +0.022 (still climbing — more data would help).
