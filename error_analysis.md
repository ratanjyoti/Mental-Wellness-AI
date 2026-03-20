# ERROR_ANALYSIS.md: Arvyax AI Failure Case Study

This document provides a deep-dive analysis of the failure modes of the Arvyax Mental Wellness AI, based on an evaluation of the 1,080-sample dataset and the resulting `predictions.csv` log.

## 1. Analysis of 10 Specific Failure Cases

The following cases represent instances where the model either provided an incorrect prediction, underestimated intensity, or triggered an uncertainty flag.

| Case # | Row ID | Category | Predicted | Actual (True) | Why the Model Failed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 1072 | **Short Input** | Calm | Overwhelmed | Text: "still heavy." TF-IDF lacks context to know if "heavy" refers to relaxation or emotional burden. |
| **2** | 965 | **Ambiguous Text** | Calm | Calm | Confidence: 0.19. Phrase "okay session" is too generic for the model to distinguish from other states. |
| **3** | 731 | **Noisy Label** | Calm | Calm (Int: 5) | Text "lowkey felt grounded" predicted Int: 3. The true Int: 5 in the CSV is likely a human entry error. |
| **4** | 100 | **Conflicting Logic** | Neutral | Mixed | Text "something still feels off" mixed with "helped a little." Model defaulted to Neutral as a mathematical middle ground. |
| **5** | 5 | **Intensity Bias** | Overwhelmed | Overwhelmed | Predicted Int: 3 (Raw 3.43), True: 5. Regression "shrunk" the prediction toward the mean (3.0). |
| **6** | 818 | **Conflicting Signals** | Restless | Restless | High Stress (5) vs Vague text. Predicted Int: 4 instead of 5. Physiological signals were under-weighted. |
| **7** | 1039 | **Semantic Duality** | Mixed | Mixed | Text "little better and a little off." Model struggled with the linguistic contrast (Duality). |
| **8** | 487 | **Zero Context** | Overwhelmed | Overwhelmed | Text "felt heavy" (2 words). Triggered `short_text` flag. Model relied on a single keyword match. |
| **9** | 540 | **Metadata Override** | Focused | Focused | Text "okay session." Energy Level: 5 forced a "Focused" prediction, masking potential underlying issues. |
| **10** | 1080 | **Negation Error** | Mixed | Mixed | Text "not bad but not clear." "Not" appeared twice, confusing the unigram frequency logic. |

---

## 2. Deep Dive Insights

### A. The Challenge of Short Inputs
**Problem:** Inputs like "fine i guess" or "still heavy" do not provide enough tokens for the TF-IDF vectorizer.
**Why it fails:** TF-IDF calculates word importance relative to the whole dataset. In 2-word sentences, the "importance" is diluted, and the model defaults to the most frequent class (usually "Calm") as a statistical prior.
**Improvement:** Implement a rule-based fallback. If word count < 4, the model should increase the weight of `stress_level` and `energy_level` to 80% of the final decision.

### B. Conflicting Signals (Text vs. Metadata)
**Problem:** A user reports positive words ("I'm okay") but high physiological stress (Stress Level 5).
**Why it fails:** The Gradient Boosting model sees a high TF-IDF score for "okay" and a high score for "Stress: 5." In many cases, the linguistic signal "voted" louder than the numerical signal.
**Improvement:** Create an "Interaction Feature" ($Stress \times Energy$). This allows the model to learn that positive words combined with high stress usually indicate "Overwhelmed" (Masked Distress).

### C. Noisy Labels in Training Data
**Problem:** The dataset contains "ground truth" labels that contradict the text (e.g., Row 731).
**Why it fails:** This "pollutes" the model's brain. If the model is told "grounded" means Intensity 5 (Crisis), it will begin over-predicting intensity for calm users.
**Improvement:** Use **Cross-Validated Out-of-Fold (OOF)** predictions to identify samples with high loss. These samples should be manually audited and corrected to ensure a clean "Gold Standard" dataset.

### D. Regression Toward the Mean (Intensity Bias)
**Problem:** Most Intensity predictions in `predictions.csv` hover between 2.5 and 3.5.
**Why it fails:** This is a classic trait of Gradient Boosting Regressors. To minimize the Mean Absolute Error (MAE), the model avoids extreme guesses (1 or 5) to stay "safe."
**Improvement:** Apply **Quantile Regression** to specifically target the 90th percentile of intensity (the crisis cases), ensuring the AI doesn't "soften" a user's distress.

