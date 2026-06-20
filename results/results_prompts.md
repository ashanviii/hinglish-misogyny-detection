# Prompt-engineering comparison on HINGLISH-MGY-DIAG (§6.5)

Metric per (model, prompt). `cons` = contrast-set consistency. `CTR`/`USE` = per-category accuracy on COUNTER / USE_SLUR (the use–mention halves).

## Model: gpt-5.4-mini

| Prompt | Acc | Macro-F1 | MCC | Consistency | COUNTER acc | USE_SLUR acc |
|---|---|---|---|---|---|---|
| usemention | 1 | 1 | 1 | 1 | 1 | 1 |
| minimal | 1 | 1 | 1 | 1 | 1 | 1 |
| definition | 1 | 1 | 1 | 1 | 1 | 1 |
| cot | 1 | 1 | 1 | 1 | 1 | 1 |
| fewshot | 1 | 1 | 1 | 1 | 1 | 1 |
| role | 1 | 1 | 1 | 1 | 1 | 1 |
| translate | 1 | 1 | 1 | 1 | 1 | 1 |
| rubric | 1 | 1 | 1 | 1 | 1 | 1 |

## Model: claude-opus-4-8

| Prompt | Acc | Macro-F1 | MCC | Consistency | COUNTER acc | USE_SLUR acc |
|---|---|---|---|---|---|---|
| cot | 1 | 1 | 1 | 1 | 1 | 1 |
| definition | 1 | 1 | 1 | 1 | 1 | 1 |
| fewshot | 1 | 1 | 1 | 1 | 1 | 1 |
| minimal | 1 | 1 | 1 | 1 | 1 | 1 |
| role | 1 | 1 | 1 | 1 | 1 | 1 |
| rubric | 1 | 1 | 1 | 1 | 1 | 1 |
| translate | 1 | 1 | 1 | 1 | 1 | 1 |
| usemention | 1 | 1 | 1 | 1 | 1 | 1 |

## Model: claude-sonnet-4-6

| Prompt | Acc | Macro-F1 | MCC | Consistency | COUNTER acc | USE_SLUR acc |
|---|---|---|---|---|---|---|
| cot | 1 | 1 | 1 | 1 | 1 | 1 |
| definition | 1 | 1 | 1 | 1 | 1 | 1 |
| fewshot | 1 | 1 | 1 | 1 | 1 | 1 |
| minimal | 1 | 1 | 1 | 1 | 1 | 1 |
| role | 1 | 1 | 1 | 1 | 1 | 1 |
| rubric | 1 | 1 | 1 | 1 | 1 | 1 |
| translate | 1 | 1 | 1 | 1 | 1 | 1 |
| usemention | 1 | 1 | 1 | 1 | 1 | 1 |

## Model: gemini-2.5-pro

| Prompt | Acc | Macro-F1 | MCC | Consistency | COUNTER acc | USE_SLUR acc |
|---|---|---|---|---|---|---|
| cot | 1 | 1 | 1 | 1 | 1 | 1 |
| definition | 1 | 1 | 1 | 1 | 1 | 1 |
| fewshot | 1 | 1 | 1 | 1 | 1 | 1 |
| minimal | 1 | 1 | 1 | 1 | 1 | 1 |
| role | 1 | 1 | 1 | 1 | 1 | 1 |
| rubric | 1 | 1 | 1 | 1 | 1 | 1 |
| translate | 1 | 1 | 1 | 1 | 1 | 1 |
| usemention | 1 | 1 | 1 | 1 | 1 | 1 |

## Model: gpt-5-mini

| Prompt | Acc | Macro-F1 | MCC | Consistency | COUNTER acc | USE_SLUR acc |
|---|---|---|---|---|---|---|
| cot | 1 | 1 | 1 | 1 | 1 | 1 |
| definition | 1 | 1 | 1 | 1 | 1 | 1 |
| fewshot | 1 | 1 | 1 | 1 | 1 | 1 |
| minimal | 1 | 1 | 1 | 1 | 1 | 1 |
| role | 1 | 1 | 1 | 1 | 1 | 1 |
| rubric | 1 | 1 | 1 | 1 | 1 | 1 |
| translate | 1 | 1 | 1 | 1 | 1 | 1 |
| usemention | 1 | 1 | 1 | 1 | 1 | 1 |

## Model: gpt-5.5

| Prompt | Acc | Macro-F1 | MCC | Consistency | COUNTER acc | USE_SLUR acc |
|---|---|---|---|---|---|---|
| cot | 1 | 1 | 1 | 1 | 1 | 1 |
| definition | 1 | 1 | 1 | 1 | 1 | 1 |
| fewshot | 1 | 1 | 1 | 1 | 1 | 1 |
| minimal | 1 | 1 | 1 | 1 | 1 | 1 |
| role | 1 | 1 | 1 | 1 | 1 | 1 |
| rubric | 1 | 1 | 1 | 1 | 1 | 1 |
| translate | 1 | 1 | 1 | 1 | 1 | 1 |
| usemention | 1 | 1 | 1 | 1 | 1 | 1 |


**Hypothesis to confirm/refute:** prompts that do not state the use–mention rule (e.g. `minimal`, `role`) post similar *accuracy* but lower *consistency*, because they over-flag `COUNTER` (counter-speech). Prompts that state it (`usemention`, `fewshot`, `rubric`, `cot`) should hold higher consistency.
