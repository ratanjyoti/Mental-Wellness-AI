# Error Analysis From `prediction.csv`

This review is based on:

- `new_code/predictions.csv`
- `new_code/Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv`

I aligned the two files by row order, not by `id`, because the prediction log has 1,200 rows but only 1,080 unique IDs. IDs `721-840` appear twice, so row order is the reliable key.

## Headline Findings

- State prediction is fairly strong: `1114 / 1200` correct (`92.8%`).
- Intensity prediction is much weaker: `348 / 1200` exact (`29.0%`).
- The biggest systematic problem is intensity collapse toward the middle: the model predicts intensity `3` for `756 / 1200` rows.
- Short inputs are especially fragile. For texts with `<= 3` words, state error rises to `21.7%` and intensity error to `80.9%`.

## 10 Failure Cases

### 1. CSV row 100 (ID 100) - Ambiguous text
**Text:** "The forest track helped a little, though something still feels off underneath."  
**Actual:** `mixed`, intensity `2`  
**Predicted:** `neutral`, intensity `3`, confidence `0.2908`

**What went wrong:** The model flattened a mixed emotional state into `neutral` and also pushed severity up by one level.

**Why the model failed:** The sentence contains a classic contrast pattern: one positive clause ("helped a little") and one unresolved negative clause ("still feels off underneath"). A TF-IDF style representation sees those words, but not the discourse relation between them, so it averages the signal.

**How to improve:** Add explicit features for contrast markers like `but`, `though`, and `still`, or switch the text encoder for this stage to a sentence model that can represent mixed sentiment rather than bag-of-words counts.

### 2. CSV row 90 (ID 90) - Conflicting signals
**Text:** "I couldn't really settle into the cafe track; I kept thinking of everything at once. Part of me wants to do everything at once."  
**Actual:** `restless`, intensity `4`  
**Predicted:** `overwhelmed`, intensity `4`, confidence `0.7348`

**What went wrong:** The model over-escalated the state from `restless` to `overwhelmed`.

**Why the model failed:** The text contains strong distress phrases like "couldn't really settle" and "everything at once", and metadata includes `stress_level = 5`. That combination likely pushed the classifier into the nearest more-severe class instead of keeping it in the adjacent `restless` bucket.

**How to improve:** Train specifically on borderline pairs like `restless` vs `overwhelmed`, and consider a hierarchical setup where the model predicts arousal/severity separately before choosing the final state label.

### 3. CSV row 13 (ID 13) - Conflicting signals
**Text:** "The forest session made me calmer, but part of me still feels uneasy. Part of me wants rest, part of me wants action."  
**Actual:** `mixed`, intensity `2`  
**Predicted:** `mixed`, intensity `4`, confidence `0.8769`

**What went wrong:** The state was correct, but intensity was overestimated by two levels.

**Why the model failed:** The regressor seems to treat multiple emotional cues as stronger distress, even when those cues are about ambivalence rather than crisis. The words `uneasy`, `rest`, and `action` raise emotional complexity, but they do not necessarily mean high intensity.

**How to improve:** Replace the free-form regressor with an ordinal intensity classifier, and add a separate "mixedness" or "polarity conflict" feature so ambivalence is not automatically translated into severity.

### 4. CSV row 14 (ID 14) - Conflicting signals
**Text:** "The cafe session helped a little, though I still feel pulled in too many directions. Part of me wants to do everything at once."  
**Actual:** `restless`, intensity `4`  
**Predicted:** `restless`, intensity `3`, confidence `0.7940`

**What went wrong:** The state was correct, but the model softened the severity.

**Why the model failed:** This is the opposite of Case 3. The early recovery phrase "helped a little" seems to dilute the later distress phrase "pulled in too many directions", so the model lands on the safe middle value.

**How to improve:** Add clause-position features or sentence-level weighting so trailing distress phrases can override weak positive openers when they carry the final emotional meaning.

### 5. CSV row 673 (ID 673) - Short input
**Text:** "it was fine"  
**Actual:** `restless`, intensity `1`  
**Predicted:** `focused`, intensity `3`, confidence `0.4270`

**What went wrong:** The model read a vague phrase as positive and missed both the actual state and the low intensity.

**Why the model failed:** On a three-word input, the token `fine` dominates. TF-IDF has no way to know whether `fine` means genuinely okay, emotionally flat, dismissive, or resigned. The model defaults to a more common positive class and a middling intensity.

**How to improve:** Treat phrases like `fine`, `okay`, and `alright` as low-information rather than positive evidence, especially when the input has fewer than four words.

### 6. CSV row 1085 (ID 965) - Short input
**Text:** "okay session"  
**Actual:** `calm`, intensity `2`  
**Predicted:** `calm`, intensity `3`, confidence `0.1950`

**What went wrong:** The classifier held onto the right state, but the intensity drifted upward even though the model was clearly uncertain.

**Why the model failed:** This is almost a textbook low-information input. The confidence score is only `0.1950`, and the uncertainty flags already mark it as both low-confidence and short text. The regressor still emits a concrete intensity and falls back to `3`.

**How to improve:** When confidence is this low on a two-word input, the system should abstain, ask for another sentence, or return an uncertainty-aware range instead of a hard intensity number.

### 7. CSV row 1192 (ID 1072) - Short input
**Text:** "still heavy"  
**Actual:** `overwhelmed`, intensity `4`  
**Predicted:** `overwhelmed`, intensity `3`, confidence `0.2989`

**What went wrong:** The model caught the direction of distress but underestimated the severity.

**Why the model failed:** With only two words, the text carries negative polarity but almost no context about duration, cause, or escalation. The classifier can still infer "not okay", but the intensity regressor again snaps back toward the center.

**How to improve:** Add a severity lexicon for high-risk phrases like `heavy`, `spiraling`, `can't settle`, and `shut down`, and penalize underestimation of true `4-5` cases more heavily during training.

### 8. CSV row 524 (ID 524) - Probable noisy label
**Text:** "felt heavy"  
**Actual:** `calm`, intensity `5`  
**Predicted:** `overwhelmed`, intensity `3`, confidence `0.4830`

**What went wrong:** This looks like a model error on paper, but the ground-truth label itself is suspicious.

**Why the model failed:** The text says "felt heavy", metadata shows `stress_level = 5`, and the face hint is `tense_face`. Those cues do not line up naturally with `calm`. The model may be getting penalized for disagreeing with a mislabeled example.

**How to improve:** Audit rows where text sentiment and metadata strongly disagree with the label. High-disagreement rows like this should go into a manual relabel queue before retraining.

### 9. CSV row 970 (ID 850) - Probable noisy label
**Text:** "more clear today"  
**Actual:** `overwhelmed`, intensity `4`  
**Predicted:** `calm`, intensity `3`, confidence `0.3827`

**What went wrong:** The label says `overwhelmed`, but both the text and metadata lean calmer than that.

**Why the model failed:** The phrase is clearly positive or at least improving, metadata shows `stress_level = 2`, the face hint is `calm_face`, and reflection quality is `clear`. This is a strong sign that the training or evaluation label may be noisy.

**How to improve:** Add consistency checks between label, text polarity, and metadata. High-confidence disagreements with semantically positive text should be reviewed by an annotator.

### 10. CSV row 977 (ID 857) - Noisy label plus metadata leakage
**Text:** "still heavy"  
**Actual:** `focused`, intensity `5`  
**Predicted:** `focused`, intensity `3`, confidence `0.3490`

**What went wrong:** The model reproduced a suspicious state label and still underestimated intensity.

**Why the model failed:** The text is plainly negative, yet the label is `focused`. That suggests label noise. The more worrying part is that the model copied the noisy state label anyway, which means metadata or learned class priors are overriding the text.

**How to improve:** Clean contradictory labels first, then reduce metadata dominance with stronger text weighting or a gating rule that prevents clearly negative text from being classified as `focused` without stronger evidence.

## Main Patterns Behind These Errors

### Ambiguous text
The model struggles when a sentence contains both relief and unease. Words like `helped`, `better`, `off`, and `underneath` often get averaged together instead of being interpreted as emotional conflict.

### Conflicting signals
When text, metadata, and label point in slightly different directions, the model often drifts into a neighboring class or defaults to medium intensity. Adjacent emotional states like `restless`, `mixed`, and `overwhelmed` are the hardest boundaries.

### Short inputs
Very short inputs are the highest-risk failure mode. They do not carry enough context for TF-IDF to know whether `fine`, `okay`, or `heavy` are mild, sarcastic, guarded, or severe. The system then backs into its priors.

### Noisy labels
Some rows appear internally inconsistent: positive text paired with distressed labels, or clearly distressed text paired with calm/focused labels. Those rows make evaluation noisy and also teach the model the wrong pattern.

## Recommended Fixes

1. Replace the intensity regressor with an ordinal classifier so the model stops collapsing most outputs to `3`.
2. Add a short-text fallback: if the input has `<= 3` words, either ask for more text or switch to a metadata-aware uncertainty mode.
3. Add explicit features for contrast markers such as `but`, `though`, `still`, `yet`, and `not fully`.
4. Audit likely noisy labels before retraining, especially rows where text polarity and metadata sharply disagree with the ground truth.
5. Fix duplicated IDs in the dataset so future error analysis can safely use `id` as a join key.
