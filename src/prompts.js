// ============================================================================
// Prompt-template library for the prompt-engineering comparison (§6.5).
// Each template returns { system, user(text) }. The classifier must answer with
// a binary label MGY (misogynistic) or NOT. Parsing is handled by the runner.
//
// Few-shot examples are HAND-AUTHORED and are NOT drawn from the evaluation set
// (no leakage). Keep them fixed so runs are comparable across models.
// ============================================================================

const FEWSHOT = [
  { text: 'aurat ko kitchen me hi rehna chahiye, bahar kya kaam.', label: 'MGY' }, // implicit
  { text: 'kisi bhi ladki ko gaali dena bilkul galat hai, mat karo.', label: 'NOT' }, // counter-speech
  { text: 'kal ka demo kitne baje hai?', label: 'NOT' }, // neutral
  { text: 'usko sab online slur bol ke harass kar rahe the.', label: 'MGY' }, // explicit use (reported w/ harassment)
];

const USEMENTION_RULE =
`Decide whether a code-mixed Hinglish (Romanised Hindi + English) message is MISOGYNISTIC (MGY) or NOT.
Rules:
- A slur or sexist claim directed at / used against a woman is MGY.
- Implicit misogyny (stereotyping, moral-policing, objectification) with no slur is still MGY.
- A slur that is only mentioned, quoted, or condemned (counter-speech such as "don't call her that", "calling her X is wrong") is NOT.
Answer with exactly one token: MGY or NOT.`;

const prompts = {
  // 1. Minimal / direct — naive baseline
  minimal: {
    id: 'minimal',
    system: 'You classify messages as misogynistic or not. Answer with exactly one token: MGY or NOT.',
    user: t => `Message: "${t}"\nLabel:`,
  },

  // 2. Definition-injected
  definition: {
    id: 'definition',
    system: 'You are a misogyny classifier for code-mixed Hinglish. Misogyny = hostility, contempt, slurs, stereotyping, moral-policing, or objectification targeting women. Answer with exactly one token: MGY or NOT.',
    user: t => `Message: "${t}"\nLabel (MGY or NOT):`,
  },

  // 3. Use–mention-explicit (the harness default; the hypothesis-confirming prompt)
  usemention: {
    id: 'usemention',
    system: USEMENTION_RULE,
    user: t => `Message: "${t}"\nLabel (MGY or NOT):`,
  },

  // 4. Chain-of-thought
  cot: {
    id: 'cot',
    system: USEMENTION_RULE + '\nFirst reason briefly, then end with a final line exactly "LABEL: MGY" or "LABEL: NOT".',
    user: t => `Message: "${t}"\nThink step by step about who the slur/claim targets and whether it is used or merely condemned, then give the LABEL line.`,
  },

  // 5. Few-shot (includes a counter-speech NOT and an implicit MGY)
  fewshot: {
    id: 'fewshot',
    system: USEMENTION_RULE,
    user: t => {
      const ex = FEWSHOT.map(e => `Message: "${e.text}"\nLabel: ${e.label}`).join('\n\n');
      return `${ex}\n\nMessage: "${t}"\nLabel:`;
    },
  },

  // 6. Role-based
  role: {
    id: 'role',
    system: 'You are an experienced Trust & Safety content moderator fluent in Hinglish. You protect women from abuse but you must not flag people who are condemning or reporting abuse. Answer with exactly one token: MGY or NOT.',
    user: t => `Moderate this message. Message: "${t}"\nLabel (MGY or NOT):`,
  },

  // 7. Translate-first (tests whether code-mixing is the bottleneck)
  translate: {
    id: 'translate',
    system: USEMENTION_RULE + '\nFirst translate the message to English, then classify. End with "LABEL: MGY" or "LABEL: NOT".',
    user: t => `Message (Hinglish): "${t}"\nTranslate to English, then decide, then give the LABEL line.`,
  },

  // 8. Category-rubric (classify into the 5 fine categories, then map)
  rubric: {
    id: 'rubric',
    system:
`Classify a Hinglish message into ONE category, then map to a binary label.
Categories:
- USE_SLUR: an explicit slur used against a woman          -> MGY
- IMPLICIT: stereotyping / moral-policing / objectification -> MGY
- COUNTER: a slur only condemned / reported (counter-speech) -> NOT
- BENIGN_GENDERED: a neutral/positive statement about a woman -> NOT
- NEUTRAL: unrelated content                                 -> NOT
End with a final line exactly "LABEL: MGY" or "LABEL: NOT".`,
    user: t => `Message: "${t}"\nPick the category, then give the LABEL line.`,
  },
};

module.exports = { prompts, FEWSHOT, USEMENTION_RULE };
