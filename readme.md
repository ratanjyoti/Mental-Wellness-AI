# Arvyax Mental Wellness AI — Complete Pipeline

## Overview

A fully local, end-to-end mental wellness AI system that:
1. **Understands** emotional state from journal text + biometric context
2. **Decides** what action to recommend and when
3. **Reasons under uncertainty** — knows when it's unsure
4. **Communicates** with warm, human-like supportive messages

---

## Project Structure

```
arvyax_wellness_ai/
├── mental_wellness_pipeline.py   ← Main pipeline (all 9 Parts)
├── app.py                        ← Flask REST API
├── predictions.csv               ← Output predictions
├── ERROR_ANALYSIS.md             ← 10 failure case analysis
├── EDGE_PLAN.md                  ← Mobile deployment plan
└── README.md                     ← This file
```

---

## Setup

### Requirements
```bash
pip install scikit-learn pandas numpy scipy flask flask-cors
```

> **No external LLMs, no API keys.** Runs 100% locally.

### Run the Pipeline
```bash
python main.ipynb
```
This trains all models, runs all 9 parts, saves `predictions.csv` and `model_artifacts.pkl`.

### Run the API
```bash
python app.py
# API available at http://localhost:5000
```

### Test the API
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "journal_text": "Feeling really anxious today, heart is racing",
    "sleep_hours": 4.5,
    "stress_level": 5,
    "energy_level": 2,
    "time_of_day": "morning"
  }'
```

---

## Approach

### Text Representation
- **TF-IDF** (Term Frequency–Inverse Document Frequency) with bigrams
- Captures key emotional vocabulary without any model downloads
- `ngram_range=(1,2)` catches phrases like "not calm", "very anxious"
- `sublinear_tf=True` dampens overly frequent terms
- Upgrade path: swap TF-IDF for `all-MiniLM-L6-v2` (23 MB, local) when available

### Models
| Task | Model | Why |
|------|-------|-----|
| Emotional state | GradientBoostingClassifier | Sequential boosting handles mixed text+metadata, outputs class probabilities |
| Intensity | GradientBoostingRegressor | Ordinal 1–5 scale → regression preserves distance between values |

### Feature Engineering
| Feature | Processing |
|---------|------------|
| `journal_text` | TF-IDF (1000 dims, bigrams) |
| Numerical metadata | Median imputation → StandardScaler |
| Categorical metadata | Mode imputation → OneHotEncoder |
| Combined | `scipy.sparse.hstack` → dense for GBM |

---

## Part-by-Part Summary

### Part 1 — Emotional State (Classification)
GradientBoostingClassifier, 6 classes, outputs probabilities for uncertainty.

### Part 2 — Intensity (Regression)
GradientBoostingRegressor. Ordinal scale → regression. Raw float output (e.g., 3.7) signals borderline uncertainty. Clamp + round to [1, 5].

### Part 3 — Decision Engine (What + When)
4-layer priority cascade:
1. Urgency (high stress → breathe now)
2. State-specific action mapping
3. Time-of-day adjustments
4. Intensity-based timing urgency

### Part 4 — Uncertainty Modeling
4 signals combined:
- Classification confidence < 0.45
- Borderline intensity (|raw − rounded| > 0.4)
- Short text (≤ 3 words)
- Missing metadata (≥ 2 fields)

### Part 5 — Feature Importance
Text features: ~80% of importance for state detection.
Metadata (especially `energy_level`, `sleep_hours`): critical for intensity + decision layer.

### Part 6 — Ablation Study
Three configurations: Text-Only, Metadata-Only, Hybrid.
Metadata alone is weak for state detection but essential for safe decisions.

### Part 7 — Error Analysis
10 failure case categories: vague text, contradictory signals, label noise, mixed states, intensity under-prediction, time-of-day logic errors, metaphorical language, previous mood ignored, energy-state mismatch, missing data.

### Part 8 — Edge Deployment
~15 MB total, < 30ms inference, full offline capability.
ONNX export → CoreML (iOS) / TFLite (Android) for 3–5× speedup.

### Part 9 — Robustness
Short text → uncertain_flag=1, metadata fallback.
Missing values → imputed, flagged.
Contradictory signals → detected, safety-weighted toward metadata.

---

## API Reference

### POST /predict
**Request:**
```json
{
  "journal_text": "string (required)",
  "ambience_type": "forest|ocean|rain|mountain|café",
  "duration_min": 30,
  "sleep_hours": 7.0,
  "energy_level": 3,
  "stress_level": 2,
  "time_of_day": "morning|afternoon|evening|night",
  "previous_day_mood": "positive|neutral|negative|mixed",
  "face_emotion_hint": "happy|neutral|sad|anxious|surprised|angry",
  "reflection_quality": "high|medium|low"
}
```

**Response:**
```json
{
  "predicted_state": "anxious",
  "predicted_intensity": 4,
  "confidence": 0.87,
  "uncertain_flag": 0,
  "uncertainty_reasons": [],
  "what_to_do": "box_breathing",
  "when_to_do": "now",
  "supportive_message": "You seem a little on edge...",
  "state_probabilities": {"anxious": 0.87, "calm": 0.05, ...}
}
```

### GET /health
Returns API status and available classes.

### GET /states
Returns valid states, actions, and timings.

---

## What We Optimized For
- **Real-world messiness**: handles vague text, missing data, contradictions
- **Safety-first decisions**: high stress always triggers immediate intervention
- **Honest uncertainty**: doesn't over-claim confidence
- **Product thinking**: warm messages, not cold labels
- **Edge-ready**: 15 MB, 30ms, fully offline