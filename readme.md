# 🌿 Mental Wellness AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-5a8c5a?style=flat-square)
![Offline](https://img.shields.io/badge/Runs-100%25%20Offline-2e7d32?style=flat-square)
![Model](https://img.shields.io/badge/Model-Gradient%20Boosting-1565c0?style=flat-square)

**A fully local, privacy-first AI pipeline for emotional state detection, intensity prediction, and personalised wellness recommendations.**

*No cloud. No API keys. No data leaves your device.*

</div>

---

## 🌐 Web App Preview

<div align="center">

**Try the live interactive demo — no installation required.**

<br/>

[![Arvyax Web App Preview](./assets/web_previews.png)](https://ratanjyoti.github.io/Mental-Wellness-AI/)

<br/>

### [🔗 Launch Live Demo → ratanjyoti.github.io/Mental-Wellness-AI](https://ratanjyoti.github.io/Mental-Wellness-AI/)

<br/>

</div>

Write a short journal reflection, adjust your stress, energy, and sleep sliders — and Arvyax instantly detects your emotional state, recommends a personalised wellness action, and delivers a warm, human-like supportive message. All ML inference runs through a local Flask backend; the web app is a beautiful, responsive interface built over it.

### ✨ Features in the Web App

- 🧠 **Real-time Emotional Prediction** — detects 6 states: `calm`, `focused`, `restless`, `anxious`, `overwhelmed`, `mixed`
- 🎯 **Intelligent Action Recommendations** — suggests `box_breathing`, `deep_work`, `rest`, `journaling`, `movement`, `grounding`, and more
- ⏱ **Urgency-Aware Timing** — tells you *when* to act: `now`, `within 15 min`, `later today`, `tonight`, or `tomorrow morning`
- 📊 **Confidence & Uncertainty Display** — surfaces a live confidence score and flags low-certainty inputs so you're never misled
- 💬 **Supportive Personalised Message** — generates a warm, empathetic response tailored to your current emotional state and intensity
- 🌬 **Deep Breathe Loading Animation** — a guided breathing animation plays while the model processes your reflection
- 🔒 **100% Private** — your journal text never leaves your device; all processing happens on `localhost`
- 📱 **Fully Responsive** — designed for mobile, tablet, and desktop

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [ML Pipeline — All 9 Parts](#-ml-pipeline--all-9-parts)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Running on Mobile — Pydroid 3](#-running-on-mobile--pydroid-3)
- [API Reference](#-api-reference)
- [Model Performance](#-model-performance)
- [Error Analysis](#-error-analysis)
- [Edge Deployment](#-edge-deployment)
- [License](#-license)

---

## 🧠 Project Overview

Arvyax is a fully local mental wellness AI that takes a user's journal reflection and lightweight biometric context — sleep hours, stress level, energy level, time of day — and produces a complete emotional understanding, decision, and supportive response:

| Output | Description |
|--------|-------------|
| `predicted_state` | Emotional state from 6 classes |
| `predicted_intensity` | Severity on a 1–5 scale (regression) |
| `confidence` | Model certainty score (0–1) |
| `uncertain_flag` | `1` when the system detects low-quality input |
| `what_to_do` | Recommended wellness action |
| `when_to_do` | Urgency timing for the recommendation |
| `supportive_message` | Human-like, empathetic response |

> **This is not a standard classification problem.** The system is built to reason under noisy text, missing metadata, contradictory signals, and imperfect labels — conditions that reflect real-world messy data.

---

## 🏗 Architecture

```
User Input (journal text + metadata)
           │
           ▼
┌──────────────────────────────────────┐
│         Feature Engineering          │
│  TF-IDF (text)  +  ColumnTransformer │
│  (numerical imputation + OHE)        │
└──────────────────┬───────────────────┘
                   │ scipy.sparse.hstack
                   ▼
┌──────────────────────────────────────┐
│         Hybrid Feature Matrix        │
│   shape: (n_samples, ~317 features)  │
└──────┬───────────────────────────────┘
       │                 │
       ▼                 ▼
┌─────────────┐   ┌──────────────────┐
│ GB Classifier│   │  GB Regressor    │
│ (emotional  │   │  (intensity      │
│   state)    │   │   1–5 scale)     │
└─────────────┘   └──────────────────┘
       │                 │
       ▼                 ▼
┌──────────────────────────────────────┐
│         Uncertainty Engine           │
│  confidence · uncertain_flag         │
│  (4 independent signals combined)    │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│         Decision Engine              │
│  4-layer priority cascade            │
│  → what_to_do  +  when_to_do        │
└──────────────────┬───────────────────┘
                   ▼
         Supportive Message
         Flask /predict API
         Arvyax Web UI
```

---

## 🔬 ML Pipeline — All 9 Parts

### Part 1 — Emotional State Prediction
`GradientBoostingClassifier` with 150 estimators, learning rate 0.08, max depth 4. Outputs class probabilities for all 6 emotional states — used directly for uncertainty scoring downstream.

### Part 2 — Intensity Prediction
Treated as **regression**, not classification. Intensity is ordinal (1–5): a `GradientBoostingRegressor` captures the real distance between severity levels. Raw float output (e.g., `3.7`) doubles as a borderline uncertainty signal. Final value is clipped and rounded to `[1, 5]`.

### Part 3 — Decision Engine (What + When)
A 4-layer priority cascade:
1. **Urgency** — high stress or very low energy triggers immediate intervention
2. **State-specific** — maps each emotional state to an appropriate action
3. **Time-of-day** — night suppresses `deep_work`, morning favours `light_planning`
4. **Intensity** — high intensity escalates timing to `now` or `within_15_min`

### Part 4 — Uncertainty Modeling
Four independent signals combined into a single `uncertain_flag`:
- Classification confidence < 0.45
- Borderline intensity (`|raw − rounded| > 0.4`)
- Short text (≤ 3 words)
- Missing metadata (≥ 2 fields absent)

### Part 5 — Feature Understanding
Text features (TF-IDF) contribute ~80% of importance for state detection. Metadata — especially `energy_level` and `sleep_hours` — is critical for intensity prediction and the decision layer. Neither modality is sufficient alone.

### Part 6 — Ablation Study
Three configurations compared:

| Configuration | State Accuracy | Intensity MAE |
|---------------|---------------|---------------|
| Text-Only | ~97.5% | 0.65 |
| Metadata-Only | ~45.0% | 0.58 |
| **Hybrid (Ours)** | **~97.5%** | **0.60** |

Metadata alone achieves only ~45% state accuracy. Metadata is nonetheless mandatory for the decision engine — you cannot safely recommend `rest` from text alone when `sleep_hours = 3`.

### Part 7 — Error Analysis
10 failure case categories documented in [`error_analysis.md`](./error_analysis.md), including vague text, contradictory signals, label noise, mixed-state confusion, intensity under-prediction, metaphorical language, and energy-state mismatch.

### Part 8 — Edge / Offline Thinking
Full deployment plan in [`edge_plan.md`](./edge_plan.md). Total artifact: ~15 MB. Inference latency: ~25ms. Runs on Android via Pydroid 3 with zero code changes. Upgrade path to ONNX Runtime and local SLMs documented.

### Part 9 — Robustness
- Short text (`"ok"`, `"fine"`) → `uncertain_flag=1`, metadata-only fallback
- Missing values → median imputation for numericals, constant for categoricals
- Contradictory inputs → confidence scores weight toward metadata for safety

---

## 🛠 Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Text Features | `TF-IDF` (sklearn) | Low RAM, fast, fully local — no model downloads |
| Metadata Features | `ColumnTransformer` (impute + scale + OHE) | Handles missing values and categorical data cleanly |
| State Model | `GradientBoostingClassifier` | Outputs probabilities, fast CPU inference, no GPU needed |
| Intensity Model | `GradientBoostingRegressor` | Ordinal scale → regression preserves distance between values |
| Serialization | `pickle` | Single 15 MB file, atomic warm-start load |
| API | `Flask` + `flask-cors` | Lightweight, runs identically on Pydroid 3 Android |
| UI | Vanilla HTML / CSS / JS | Zero dependencies, works as a local file or served by Flask |

---

## 📁 Project Structure

```
Mental-Wellness-AI/
│
├── new_code/
│   ├── main.ipynb                         ← End-to-end ML pipeline (all 9 parts)
│   ├── text.ipynb                         ← Text feature experiments
│   ├── model_artifacts.pkl                ← Serialized trained models (15 MB)
│   ├── predictions.csv                    ← Output predictions (1,200 samples)
│   ├── arvyax_test_inputs_120.xlsx        ← Test input dataset
│   └── Sample_arvyax_reflective_...xlsx   ← Training dataset
│
├── assets/
│   └── web_preview.png                    ← Web app screenshot for README
│
├── app.py                                 ← Flask REST API
├── index.html                             ← Web UI (Arvyax frontend)
├── requirements.txt                       ← Python dependencies
├── edge_plan.md                           ← Mobile & offline deployment plan
├── error_analysis.md                      ← 10 failure case deep dive
├── readme.md                              ← This file
├── .gitignore
└── LICENSE
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/ratanjyoti/Mental-Wellness-AI.git
cd Mental-Wellness-AI
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
flask
flask-cors
scikit-learn
pandas
numpy
scipy
```

### 4. Train the Model *(skip if `model_artifacts.pkl` already exists)*
```bash
cd new_code
jupyter notebook main.ipynb
# Run all cells — generates model_artifacts.pkl
```

### 5. Start the Flask API
```bash
python app.py
# ✅ Running on http://localhost:5000
```

### 6. Open the Web App
Open `index.html` in your browser, or navigate to `http://localhost:5000`.

---

## 📱 Running on Mobile — Pydroid 3

Arvyax runs entirely on Android — no laptop, no internet, no cloud server required.

```
Step 1 — Install Pydroid 3 from the Google Play Store
Step 2 — Open Pydroid 3 terminal and install dependencies:
         pip install flask flask-cors scikit-learn pandas numpy scipy
Step 3 — Transfer app.py and model_artifacts.pkl to your device
Step 4 — Open app.py in Pydroid 3 and press ▶ Run
Step 5 — Open Chrome on the same device → http://localhost:5000
```

Flask binds to the device's loopback interface (`127.0.0.1`). No Wi-Fi or mobile data is involved. Inference runs in ~25ms on a mid-range Android CPU. See [`edge_plan.md`](./edge_plan.md) for the full architecture and optimisation details.

---

## 🔌 API Reference

### `POST /predict`

**Request Body (JSON):**
```json
{
  "journal_text":       "Feeling a bit anxious today, heart is racing",
  "sleep_hours":        4.5,
  "stress_level":       5,
  "energy_level":       2,
  "time_of_day":        "morning",
  "previous_day_mood":  "negative"
}
```

**Response:**
```json
{
  "predicted_state":      "anxious",
  "predicted_intensity":  4,
  "confidence":           0.874,
  "uncertain_flag":       0,
  "uncertainty_reasons":  [],
  "what_to_do":           "box_breathing",
  "when_to_do":           "now",
  "supportive_message":   "You seem a little on edge, which makes sense. Let's slow things down — try a box breathing exercise now. Breathe in 4 counts, hold 4, out 4, hold 4. You've got this — one step at a time.",
  "state_probabilities": {
    "anxious": 0.874, "calm": 0.042, "focused": 0.031,
    "mixed": 0.028, "overwhelmed": 0.016, "restless": 0.009
  }
}
```

### `GET /health`
Returns API status and available emotional state classes.

### `GET /states`
Returns all valid states, recommended actions, and timing options.

---

## 📊 Model Performance

Validated on **1,200 samples** from `predictions.csv`:

| Metric | Value |
|--------|-------|
| Mean Confidence | 77.8% |
| Median Confidence | 87.4% |
| High-confidence predictions (> 0.85) | 56.1% |
| Certain predictions | 69.8% |
| Correctly flagged as uncertain | 30.2% |
| Maximum confidence achieved | 99.7% |

**Confidence Distribution:**

```
> 0.85  (High)       ████████████████████████████░░░░  56.1%
0.70–0.85 (Good)     ████████░░░░░░░░░░░░░░░░░░░░░░░░  15.3%
0.50–0.70 (Moderate) █████████░░░░░░░░░░░░░░░░░░░░░░░  17.2%
< 0.50  (Low)        ██████░░░░░░░░░░░░░░░░░░░░░░░░░░  11.5%
```

**Action Urgency Profile:**

```
now              ████████████████████████████████░░  58.8%  (705 samples)
within_15_min    ████████████░░░░░░░░░░░░░░░░░░░░░░  22.3%
later_today      █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   9.9%
tonight          ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   7.5%
tomorrow_morning █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   1.6%
```

---

## 🔍 Error Analysis

10 failure case categories documented in [`error_analysis.md`](./error_analysis.md):

| # | Category | Root Cause | Fix |
|---|----------|------------|-----|
| 1 | Vague text ("ok", "fine") | TF-IDF has no signal on 1-word inputs | Auto `uncertain_flag=1` for ≤3 words |
| 2 | Contradictory signals | Text overrides physiology | Contradiction detector, weight metadata for safety |
| 3 | Noisy / mislabeled labels | Label noise inflates error rate | Confident Learning (cleanlab) |
| 4 | Mixed state confusion | "Mixed" overlaps all other classes | Second-pass: flag as mixed if top-2 probs within 0.10 |
| 5 | Short text + missing metadata | Both signal sources simultaneously weak | Data quality score → heuristic fallback |
| 6 | Intensity under-prediction | Regression pulled toward mean | Oversample high-intensity; crisis keyword boosters |
| 7 | Time-of-day decision error | Blanket night rules ignore chronotype | User preference layer |
| 8 | Metaphorical language | OOV words invisible to TF-IDF | Upgrade to MiniLM sentence embeddings |
| 9 | Previous mood ignored | No context carryover modelled | Interaction feature: `prev_mood × stress` |
| 10 | Ambience vs. self-description | Text about environment ≠ internal state | Subject-detection heuristic to down-weight ambience sentences |

---

## 🚀 Edge Deployment

Full technical details in [`edge_plan.md`](./edge_plan.md).

| Property | Value |
|----------|-------|
| Total artifact size | ~15 MB |
| Cold start time | ~1.5 seconds |
| Inference latency | **~25ms** |
| Internet required | **Never** |
| Data transmitted | **Zero bytes** |
| Android runtime | Pydroid 3 |
| State accuracy | ~97.5% |

**Roadmap:**

| Phase | Description | Footprint |
|-------|-------------|-----------|
| ✅ Phase 1 | Pydroid 3 + Flask + Pickle + TF-IDF | ~15 MB |
| Phase 2 | Standalone APK (Kivy / BeeWare) | ~15 MB |
| Phase 3 | ONNX Runtime — 3–5× faster, no Python | ~11 MB |
| Phase 4 | MiniLM-L3 semantic embeddings | ~30 MB |
| Phase 5 | Local SLM message generation (Phi-3 Mini) | ~2.2 GB Q4 |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

---

<div align="center">

Built with 🌿 by [Ratanjyoti](https://github.com/ratanjyoti)

*Arvyax — built to understand, decide, and guide. Entirely on your device.*

⭐ If you found this useful, consider starring the repository

</div>
