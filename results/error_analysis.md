# Error Analysis — STRONG vs CONTEXT (masked, template-disjoint CV)

## LogReg TF-IDF — 22 errors / 125

| gold | predicted | text |
|---|---|---|
| CONTEXT | STRONG | Kapdon ki wajah se ladki ko [CTX_18] kehna galat. |
| CONTEXT | STRONG | Kapdon ki wajah se women ko [CTX_01] kehna galat. |
| CONTEXT | STRONG | Kapdon ki wajah se aurat ko [CTX_02] kehna galat. |
| CONTEXT | STRONG | timeline me kisi ne ladki ko [CTX_06] bol diya. |
| CONTEXT | STRONG | Kapdon ki wajah se mahila ko [CTX_18] kehna galat. |
| CONTEXT | STRONG | Kapdon ki wajah se aurat ko [CTX_10] kehna galat. |
| CONTEXT | STRONG | Kapdon ki wajah se girl ko [CTX_20] kehna galat. |
| CONTEXT | STRONG | Kapdon ki wajah se ladki ko [CTX_11] kehna galat. |
| STRONG | CONTEXT | aurat ko [FSLUR_16] btate rehna बंद kro. |
| STRONG | CONTEXT | Kisi ladki ko [FSLUR_02] kehna sahi nahi. |
| STRONG | CONTEXT | aurat ko [FSLUR_05] btate rehna बंद kro. |
| STRONG | CONTEXT | Kisi ladkiyon ko [FSLUR_11] kehna sahi nahi. |
| STRONG | CONTEXT | Kisi women ko [FSLUR_07] kehna sahi nahi. |
| STRONG | CONTEXT | Kisi girl ko [FSLUR_11] kehna sahi nahi. |
| STRONG | CONTEXT | Kisi women ko [FSLUR_14] kehna sahi nahi. |
| STRONG | CONTEXT | Kisi aurat ko [FSLUR_08] kehna sahi nahi. |
| STRONG | CONTEXT | Kisi girl ko [FSLUR_20] kehna sahi nahi. |
| STRONG | CONTEXT | Kisi girl ko [FSLUR_15] kehna sahi nahi. |
| STRONG | CONTEXT | Kisi girl ko [FSLUR_08] kehna sahi nahi. |
| STRONG | CONTEXT | Kisi aurat ko [FSLUR_07] kehna sahi nahi. |
| STRONG | CONTEXT | Kisi mahila ko [FSLUR_17] kehna sahi nahi. |
| STRONG | CONTEXT | Kisi aurat ko [FSLUR_11] kehna sahi nahi. |

## Cues-only — 29 errors / 125

| gold | predicted | text |
|---|---|---|
| CONTEXT | STRONG | timeline me kisi ne ladki ko [CTX_06] bol diya. |
| CONTEXT | STRONG | group chat me kisi ne aurat ko [CTX_12] bol diya. |
| CONTEXT | STRONG | group chat me kisi ne ladkiyon ko [CTX_11] bol diya. |
| CONTEXT | STRONG | group chat me kisi ne girls ko [CTX_06] bol diya. |
| STRONG | CONTEXT | aurat ko [FSLUR_16] btate rehna बंद kro. |
| STRONG | CONTEXT | women ko [FSLUR_18] btate rehna बंद kro. |
| STRONG | CONTEXT | ladki ko [FSLUR_05] btate rehna बंद kro. |
| STRONG | CONTEXT | Kisi ladki ko [FSLUR_02] kehna sahi nahi. |
| STRONG | CONTEXT | aurat ko [FSLUR_05] btate rehna बंद kro. |
| STRONG | CONTEXT | ladkiyon ko [FSLUR_11] btate rehna बंद kro. |
| STRONG | CONTEXT | mahila ko [FSLUR_08] btate rehna बंद kro. |
| STRONG | CONTEXT | Kisi ladkiyon ko [FSLUR_11] kehna sahi nahi. |
| STRONG | CONTEXT | Kisi women ko [FSLUR_07] kehna sahi nahi. |
| STRONG | CONTEXT | women ko [FSLUR_02] btate rehna बंद kro. |
| STRONG | CONTEXT | Kisi girl ko [FSLUR_11] kehna sahi nahi. |
| STRONG | CONTEXT | ladki ko [FSLUR_19] btate rehna बंद kro. |
| STRONG | CONTEXT | women ko [FSLUR_08] btate rehna बंद kro. |
| STRONG | CONTEXT | Kisi women ko [FSLUR_14] kehna sahi nahi. |
| STRONG | CONTEXT | Kisi aurat ko [FSLUR_08] kehna sahi nahi. |
| STRONG | CONTEXT | Kisi girl ko [FSLUR_20] kehna sahi nahi. |
| STRONG | CONTEXT | Kisi girl ko [FSLUR_15] kehna sahi nahi. |
| STRONG | CONTEXT | girl ko [FSLUR_03] btate rehna बंद kro. |
| STRONG | CONTEXT | Kisi girl ko [FSLUR_08] kehna sahi nahi. |
| STRONG | CONTEXT | Kisi aurat ko [FSLUR_07] kehna sahi nahi. |
| STRONG | CONTEXT | Kisi mahila ko [FSLUR_17] kehna sahi nahi. |
| STRONG | CONTEXT | Kisi aurat ko [FSLUR_11] kehna sahi nahi. |
| STRONG | CONTEXT | women ko [FSLUR_12] btate rehna बंद kro. |
| STRONG | CONTEXT | girl ko [FSLUR_01] btate rehna बंद kro. |
| CONTEXT | STRONG | public post me kisi ne ladkiyon ko [CTX_07] bol diya. |

## Error direction counts
LogReg: {"CONTEXT->STRONG":8,"STRONG->CONTEXT":14}
Cues-only: {"CONTEXT->STRONG":5,"STRONG->CONTEXT":24}