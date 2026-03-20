import pickle
import numpy as np
import pandas as pd
import random
import sys
import os
from scipy.sparse import hstack

# ── Load Flask ────────────────────────────────────────────────────────────────
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    print("[ERROR] Flask not installed. Run: pip install flask flask-cors")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for UI demo

# ── Load Model Artifacts ──────────────────────────────────────────────────────
ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "model_artifacts.pkl")

try:
    with open(ARTIFACT_PATH, "rb") as f:
        arts = pickle.load(f)
    state_model      = arts["state_model"]
    intensity_model  = arts["intensity_model"]
    meta_preprocessor = arts["metadata_preprocessor"]
    tfidf            = arts["tfidf"]
    state_labels     = arts["state_labels"]
    le               = arts["le"]
    numerical_cols   = arts["numerical_cols"]
    categorical_cols = arts["categorical_cols"]
    print("[API] Models loaded successfully.")
except FileNotFoundError:
    print(f"[ERROR] {ARTIFACT_PATH} not found. Run mental_wellness_pipeline.py first.")
    sys.exit(1)

# ── Helpers (same as pipeline) ────────────────────────────────────────────────

def clean_text(text):
    if not text or str(text).strip() == "" or text != text:
        return "missing reflection"
    return str(text).strip().lower()


def compute_uncertainty(state_probs, intensity_raw, text, metadata_row):
    reasons = []
    max_prob = float(np.max(state_probs))
    if max_prob < 0.45:
        reasons.append(f"low_class_confidence ({max_prob:.2f})")
    intensity_dist = abs(intensity_raw - round(intensity_raw))
    if intensity_dist > 0.4:
        reasons.append(f"borderline_intensity ({intensity_raw:.2f})")
    word_count = len(str(text).split())
    if word_count <= 3:
        reasons.append(f"short_text ({word_count} words)")
    missing_count = sum(1 for v in metadata_row if v is None or (isinstance(v, float) and np.isnan(v)))
    if missing_count >= 2:
        reasons.append(f"missing_metadata ({missing_count} fields)")
    uncertain_flag = 1 if len(reasons) > 0 else 0
    return max_prob, uncertain_flag, reasons


class DecisionEngine:
    def decide(self, state, intensity, stress, energy, time_of_day):
        if stress >= 4 and state == "overwhelmed":
            return "rest", "now"
        if stress >= 4:
            return "box_breathing", "now"
        if energy <= 1:
            return "rest", "now"
        if state == "focused" and energy >= 3:
            action = "deep_work"
        elif state == "focused":
            action = "light_planning"
        elif state == "anxious":
            action = "sound_therapy"
        elif state == "restless":
            action = "movement"
        elif state == "overwhelmed":
            action = "journaling"
        elif state == "calm":
            action = "light_planning" if energy >= 3 else "rest"
        elif state == "mixed":
            action = "grounding"
        else:
            action = "pause"

        if time_of_day == "night":
            if action in ["deep_work", "movement"]:
                action = "rest"
                timing = "tonight"
            elif action == "light_planning":
                timing = "tomorrow_morning"
            else:
                timing = "tonight"
        elif time_of_day == "morning":
            timing = "now" if state in ["focused", "calm"] else "within_15_min"
        elif time_of_day == "afternoon":
            timing = "now" if state in ["anxious", "restless"] else "within_15_min"
        else:
            timing = "later_today" if action not in ["rest"] else "tonight"

        if intensity >= 4 and timing != "now":
            timing = "within_15_min"
        return action, timing

    def generate_message(self, state, action, timing, intensity):
        openers = {
            "calm":        ["You seem to be in a peaceful headspace right now.", "It's great to see you feeling so centered."],
            "focused":     ["You're in a great flow state right now.", "Your concentration seems really sharp today."],
            "restless":    ["You seem a bit restless right now — and that's okay.", "I sense some extra energy looking for an outlet."],
            "anxious":     ["You seem a little on edge, which makes sense.", "I sense some tension in your reflection."],
            "overwhelmed": ["Things seem quite heavy for you right now.", "It sounds like there's a lot weighing on you."],
            "mixed":       ["You're navigating some complex feelings today.", "It sounds like a bit of a mixed bag right now."],
        }
        instructions = {
            "box_breathing":  f"Let's slow things down — try a box breathing exercise {timing}. Breathe in 4 counts, hold 4, out 4, hold 4.",
            "deep_work":      f"This is the perfect moment for deep work. Block out distractions and dive in {timing}.",
            "rest":           f"Be kind to yourself. Your body is asking for rest — please honour that {timing}.",
            "grounding":      f"Let's bring you back to the present. Try a 5-4-3-2-1 grounding exercise {timing} to settle your mind.",
            "movement":       f"Let's channel that energy. A short walk or stretch {timing} would be the perfect outlet.",
            "light_planning": f"Let's clear the mental clutter. Spending 10 minutes on light planning {timing} will help you feel in control.",
            "journaling":     f"Writing it out can help you process. Try journaling {timing} — even a few lines makes a difference.",
            "sound_therapy":  f"Let's shift the atmosphere. Some calming sounds or music {timing} can soothe an anxious mind.",
            "yoga":           f"A gentle yoga session {timing} would help reconnect your mind and body.",
            "pause":          f"Just pause for a moment {timing}. Take 3 deep breaths and give yourself permission to reset.",
        }
        state_key = state.lower() if state.lower() in openers else "mixed"
        opener = random.choice(openers[state_key])
        instruction = instructions.get(action, f"Try some {action} {timing}.")
        suffix = " You've got this — one step at a time." if intensity >= 4 else (" Small steps forward still count." if intensity <= 2 else "")
        return f"{opener} {instruction}{suffix}"


engine = DecisionEngine()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": "GradientBoosting", "classes": list(state_labels)})


@app.route("/states", methods=["GET"])
def get_states():
    """Return valid emotional states and actions."""
    return jsonify({
        "emotional_states": list(state_labels),
        "actions": ["box_breathing", "journaling", "grounding", "deep_work",
                    "yoga", "sound_therapy", "light_planning", "rest", "movement", "pause"],
        "timings": ["now", "within_15_min", "later_today", "tonight", "tomorrow_morning"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Request JSON:
    {
        "journal_text":    "I feel a bit anxious today",    // required
        "ambience_type":   "forest",                        // optional
        "duration_min":    30,                              // optional
        "sleep_hours":     6.5,                             // optional
        "energy_level":    3,                               // optional
        "stress_level":    4,                               // optional
        "time_of_day":     "morning",                       // optional
        "previous_day_mood": "neutral",                     // optional
        "face_emotion_hint": "anxious",                     // optional
        "reflection_quality": "medium"                      // optional
    }

    Response JSON:
    {
        "predicted_state":      "anxious",
        "predicted_intensity":  4,
        "confidence":           0.87,
        "uncertain_flag":       0,
        "uncertainty_reasons":  [],
        "what_to_do":           "box_breathing",
        "when_to_do":           "now",
        "supportive_message":   "You seem a little on edge..."
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        journal_text = data.get("journal_text", "")
        if not journal_text:
            return jsonify({"error": "journal_text is required"}), 400

        # Build a one-row DataFrame matching the training schema
        row = {
            "journal_text":      journal_text,
            "ambience_type":     data.get("ambience_type", np.nan),
            "duration_min":      data.get("duration_min", np.nan),
            "sleep_hours":       data.get("sleep_hours", np.nan),
            "energy_level":      data.get("energy_level", np.nan),
            "stress_level":      data.get("stress_level", np.nan),
            "time_of_day":       data.get("time_of_day", "morning"),
            "previous_day_mood": data.get("previous_day_mood", np.nan),
            "face_emotion_hint": data.get("face_emotion_hint", np.nan),
            "reflection_quality":data.get("reflection_quality", np.nan),
        }
        df = pd.DataFrame([row])

        # Text features
        text_clean = clean_text(journal_text)
        text_feats = tfidf.transform([text_clean])

        # Metadata features
        meta_feats = meta_preprocessor.transform(df[numerical_cols + categorical_cols])

        # Combine and predict
        X = hstack([text_feats, meta_feats]).toarray()
        state_probs = state_model.predict_proba(X)[0]
        state_enc   = state_model.predict(X)[0]
        pred_state  = le.inverse_transform([state_enc])[0]
        int_raw     = float(intensity_model.predict(X)[0])
        pred_intensity = int(np.clip(round(int_raw), 1, 5))

        # Uncertainty
        stress = float(row["stress_level"]) if pd.notna(row["stress_level"]) else 3.0
        energy = float(row["energy_level"]) if pd.notna(row["energy_level"]) else 3.0
        conf, unc_flag, reasons = compute_uncertainty(
            state_probs, int_raw, journal_text,
            [row[c] for c in numerical_cols]
        )

        # Decision
        time_of_day = str(row["time_of_day"]) if pd.notna(row["time_of_day"]) else "morning"
        what_to_do, when_to_do = engine.decide(pred_state, pred_intensity, stress, energy, time_of_day)
        message = engine.generate_message(pred_state, what_to_do, when_to_do, pred_intensity)

        # State probability breakdown
        state_probs_dict = {state_labels[i]: round(float(state_probs[i]), 4)
                            for i in range(len(state_labels))}

        return jsonify({
            "predicted_state":      pred_state,
            "predicted_intensity":  pred_intensity,
            "confidence":           round(conf, 4),
            "uncertain_flag":       unc_flag,
            "uncertainty_reasons":  reasons,
            "what_to_do":           what_to_do,
            "when_to_do":           when_to_do,
            "supportive_message":   message,
            "state_probabilities":  state_probs_dict,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  ARVYAX MENTAL WELLNESS API — RUNNING")
    print("  http://localhost:5000/predict")
    print("  http://localhost:5000/health")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)