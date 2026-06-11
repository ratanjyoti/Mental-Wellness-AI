# %%
import pandas as pd
import numpy as np
import random
import json
import os
import warnings
import pickle
warnings.filterwarnings("ignore")
 
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, accuracy_score,
                              mean_absolute_error)
from imblearn.over_sampling import SMOTE

# %%
TRAIN_PATH = "Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv"
TEST_PATH  = "arvyax_test_inputs_120.xlsx - Sheet1.csv"
 
data    = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
 
print(f"[DATA] Train: {data.shape}  |  Test: {test_df.shape}")

# %%
print("\n── SECTION 2 — FEATURE ENGINEERING & PREPROCESSING ──")
 
# ── Text preprocessing ──────────────────────────────────────
def clean_text(text):
    if pd.isna(text) or str(text).strip() == "":
        return "missing reflection"
    return str(text).strip().lower()
 
data["journal_text_clean"]    = data["journal_text"].apply(clean_text)
test_df["journal_text_clean"] = test_df["journal_text"].apply(clean_text)
 
print("[TEXT] Loading MiniLM embeddings model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

text_features_train = embedding_model.encode(
    data["journal_text_clean"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)

text_features_test = embedding_model.encode(
    test_df["journal_text_clean"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)

print(f"[TEXT] Embedding Dimension: {text_features_train.shape[1]}")
 
# ── Metadata preprocessing ──────────────────────────────────
numerical_cols   = ["duration_min", "sleep_hours", "energy_level", "stress_level"]
categorical_cols = ["ambience_type", "time_of_day", "previous_day_mood",
                    "face_emotion_hint", "reflection_quality"]
 
metadata_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ]), numerical_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical_cols),
])
 
meta_features_train = metadata_preprocessor.fit_transform(
    data[numerical_cols + categorical_cols])
meta_features_test  = metadata_preprocessor.transform(
    test_df[numerical_cols + categorical_cols])
 
# ── Combine features ────────────────────────────────────────
from numpy import hstack

X_train_full = np.hstack(
    [text_features_train, meta_features_train]
)

X_test_final = np.hstack(
    [text_features_test, meta_features_test]
)
 
print(f"[COMBINED] Training Matrix : {X_train_full.shape}")
print(f"[COMBINED] Test Matrix     : {X_test_final.shape}")
 
# ── Target encoding ─────────────────────────────────────────
le = LabelEncoder()
y_state_encoded = le.fit_transform(data["emotional_state"])
state_labels    = le.classes_
y_intensity     = data["intensity"].values.astype(float)
 
print(f"[LABELS] State classes: {list(state_labels)}")
 
# ── Train / Val split ───────────────────────────────────────
X_tr, X_val, y_s_tr, y_s_val, y_i_tr, y_i_val = train_test_split(
    X_train_full, y_state_encoded, y_intensity,
    test_size=0.2, random_state=42, stratify=y_state_encoded
)

X_tr_dense = X_tr
X_val_dense = X_val
 
smote = SMOTE(random_state=42, k_neighbors=min(3, min(np.bincount(y_s_tr)) - 1))
X_tr_dense_bal, y_s_tr_bal = smote.fit_resample(X_tr_dense, y_s_tr)
 
_, counts_before = np.unique(y_s_tr, return_counts=True)
_, counts_after  = np.unique(y_s_tr_bal, return_counts=True)
print(f"\n[SMOTE] Class counts before: {dict(zip(state_labels, counts_before))}")
print(f"[SMOTE] Class counts after : {dict(zip(state_labels, counts_after))}")

y_i_tr_for_reg = y_i_tr  
 
print(f"\n[DONE] Train: {X_tr.shape[0]}→{X_tr_dense_bal.shape[0]} (after SMOTE) "
      f"| Val: {X_val.shape[0]} | Test: {X_test_final.shape[0]}")
 

# %%

print("\n── PART 1 — EMOTIONAL STATE PREDICTION ──")
 
state_model = GradientBoostingClassifier(
    n_estimators=200,    
    learning_rate=0.07,
    max_depth=4,
    subsample=0.8,
    random_state=42,
)
state_model.fit(X_tr_dense_bal, y_s_tr_bal)
 
y_s_pred = state_model.predict(X_val_dense)
acc = accuracy_score(y_s_val, y_s_pred)
print(f"\n[PART 1] Validation Accuracy: {acc:.4f}")
print("\n[PART 1] Classification Report:")
print(classification_report(y_s_val, y_s_pred, target_names=state_labels))
 
predicted_states = set(le.inverse_transform(state_model.predict(X_val_dense)))
print(f"[PART 1] States predicted on val set: {predicted_states}")
 

# %%
print("\n── PART 2 — INTENSITY PREDICTION ──")
 
intensity_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    min_samples_leaf=5,
    loss="huber",
    alpha=0.9,
    random_state=42,
)
intensity_model.fit(X_tr_dense, y_i_tr_for_reg)
 
y_i_pred_raw = intensity_model.predict(X_val_dense)
y_i_pred     = np.clip(np.round(y_i_pred_raw), 1, 5).astype(int)
 
mae    = mean_absolute_error(y_i_val, y_i_pred)
cv_mae = -cross_val_score(
    intensity_model, X_tr_dense, y_i_tr_for_reg,
    scoring="neg_mean_absolute_error", cv=5
).mean()
 
print(f"[PART 2] Val MAE  : {mae:.4f}")
print(f"[PART 2] CV  MAE  : {cv_mae:.4f}")
print(f"[PART 2] Intensity distribution on val:")
unique, counts = np.unique(y_i_pred, return_counts=True)
print(dict(zip(unique, counts)))

# %%

class DecisionEngine:
 
    # FIX 2: All 10 emotional states now have openers (overwhelmed, neutral added)
    OPENERS = {
        "calm":       ["You seem grounded right now.",
                       "There's a quiet steadiness coming through."],
        "anxious":    ["It sounds like your mind is running fast.",
                       "There's some tension in what you've shared."],
        "restless":   ["Something feels unsettled.",
                       "You seem like you're between things right now."],
        "sad":        ["It sounds like today has been heavy.",
                       "There's a weight in what you've written."],
        "energized":  ["You've got some real momentum right now.",
                       "There's good energy in what you've shared."],
        "tired":      ["Your body seems like it's asking for rest.",
                       "It sounds like today has taken a lot out of you."],
        "focused":    ["You seem clear-headed right now.",
                       "There's good focus coming through."],
        "mixed":      ["Your signals are a bit mixed today.",
                       "Today seems to have a few layers to it."],
        "overwhelmed":["It sounds like a lot is pressing on you right now.",
                       "You're carrying quite a bit — that comes through clearly."],
        "neutral":    ["You seem fairly steady today.",
                       "Things feel balanced for you right now."],
    }
 
    ACTION_TEMPLATES = {
        "box_breathing": {
            "now":           "Take 4 minutes right now — breathe in for 4, hold for 4, out for 4. Just one round.",
            "within_15_min": "In the next 15 minutes, find a quiet spot and do 3–4 rounds of box breathing.",
        },
        "journaling": {
            "now":           "Take 5 minutes to write whatever's on your mind — no structure needed.",
            "later_today":   "Tonight, try writing for 10 minutes. Don't edit. Just let it out.",
            "tonight":       "Before bed, jot down 3 things from today — good, bad, or just notable.",
        },
        "grounding": {
            "now":           "Pause for a moment. Name 5 things you can see, 4 you can touch, 3 you can hear.",
            "within_15_min": "Step outside or to a window soon — a few minutes of fresh air and ground beneath you.",
        },
        "deep_work": {
            "now":           "You're in a good window for focused work. Start with your most important task.",
            "within_15_min": "In a bit, clear your notifications and block out 60–90 minutes of deep focus.",
            "later_today":   "This afternoon could be a good window for deep work once your energy settles.",
        },
        "rest": {
            "now":           "Give yourself permission to rest — even 15–20 minutes makes a difference.",
            "tonight":       "Prioritize sleep tonight. Aim to be in bed an hour earlier than usual.",
            "tomorrow_morning": "Tomorrow morning, take it slow. Don't rush into tasks right away.",
        },
        "movement": {
            "now":           "Get up and move — a 10-minute walk, some stretching, anything that gets you out of your head.",
            "within_15_min": "Soon, step away from your screen and move your body for a bit.",
            "later_today":   "This afternoon, try to fit in a walk or some light movement.",
        },
        "sound_therapy": {
            "now":           "Put on something calming — lo-fi, nature sounds, or silence. Let it just be background.",
            "tonight":       "Tonight, wind down with some soft music or ambient sound instead of screens.",
        },
        "light_planning": {
            "now":           "Spend 5 minutes writing out the 2–3 things that actually matter today. Just those.",
            "within_15_min": "Before diving into work, take a few minutes to map out the day simply.",
        },
        "yoga": {
            "now":           "Even 10 minutes of gentle stretching or a short yoga flow can reset your nervous system.",
            "later_today":   "Try to carve out time this afternoon for some yoga or stretching.",
        },
        "pause": {
            "now":           "Before deciding anything, just pause. Sit with where you are for a minute.",
            "within_15_min": "Give yourself a few minutes before jumping into anything. You don't need to rush.",
        },
    }
 
    INTENSITY_CLOSERS = {
        1: "Even small steps count.",
        2: "Nothing drastic — just a small nudge in the right direction.",
        3: "This is a good moment to make one clear choice for yourself.",
        4: "Right now, one intentional action can shift things meaningfully.",
        5: "This feels important. Give yourself what you need — don't push through.",
    }
 
    UNCERTAINTY_HEDGES = [
        "I'm reading between some mixed lines here, so take this as a gentle suggestion —",
        "Your signals are a bit nuanced today, so this is more of a nudge than a prescription —",
        "I'm not entirely certain what's going on, but if I had to offer one thing —",
    ]
 
    INTENSITY_STATE_CONSTRAINTS = {
        "calm":       (1, 3),
        "focused":    (1, 4),
        "neutral":    (1, 3),
        "overwhelmed": (2, 5),
        "anxious":    (2, 5),
        "sad":        (2, 5),
        "energized":  (1, 4),
        "tired":      (1, 4),
        "restless":   (1, 4),
        "mixed":      (1, 4),
    }
 
    def decide(self, state, intensity, stress, energy, time_of_day, uncertain_flag=0):
        state = state.lower()
 
        # ── LAYER 1: Time-of-day baseline ────────────────────────────────
        if time_of_day == "morning":
            action, timing = "light_planning", "now"
        elif time_of_day == "afternoon":
            action, timing = "deep_work", "now"
        elif time_of_day == "evening":
            action, timing = "journaling", "tonight"
        else:
            action, timing = "rest", "tonight"
 
        # ── LAYER 2: Emotional state override ────────────────────────────
        state_overrides = {
            "anxious":    ("box_breathing",  "now"),
            "restless":   ("grounding",      "now"),
            "sad":        ("journaling",     "now"),
            "tired":      ("rest",           "now"),
            "energized":  ("deep_work",      "now"),
            "focused":    ("deep_work",      "now"),
            "overwhelmed":("box_breathing",  "now"),
        }
        if state in state_overrides:
            action, timing = state_overrides[state]
 
        # ── LAYER 3: Stress × Energy matrix ──────────────────────────────
        if stress >= 4 and energy >= 3:
            action, timing = "movement", "within_15_min"
        elif stress >= 4 and energy < 3:
            action, timing = "box_breathing", "now"
 
        if energy <= 2 and time_of_day in ("evening", "night"):
            action, timing = "rest", "tonight"
        elif energy <= 1:
            action, timing = "rest", "now"
 
        # ── LAYER 4: Intensity amplifier ─────────────────────────────────
        negative_states = {"anxious", "sad", "restless", "overwhelmed"}
        if intensity >= 4 and state in negative_states:
            timing = "now"
        if intensity <= 2 and timing == "now" and action not in ("rest",):
            timing = "within_15_min"
 
        if state == "overwhelmed" and intensity >= 4:
            timing = "now" if intensity == 5 else "within_15_min"
 
        if state == "focused" and action == "rest" and timing == "now":
            action, timing = "light_planning", "within_15_min"
 
        if state == "neutral" and action == "deep_work" and timing == "now":
            timing = "within_15_min"
 
        # ── LAYER 5: Uncertainty softening ───────────────────────────────
        if uncertain_flag == 1:
            if stress < 3 and intensity <= 2:
                action, timing = "pause", "now"
            elif timing == "now" and stress < 4:
                timing = "within_15_min"
 
        return action, timing
 
    def generate_message(self, state, action, timing, intensity, uncertain_flag=0):
        state_key = state.lower() if state.lower() in self.OPENERS else "mixed"
        opener = random.choice(self.OPENERS[state_key])
 
        action_map  = self.ACTION_TEMPLATES.get(action, {})
        instruction = (action_map.get(timing)
                       or action_map.get("now")
                       or f"Consider {action.replace('_', ' ')} {timing.replace('_', ' ')}.")
 
        closer = self.INTENSITY_CLOSERS.get(intensity, self.INTENSITY_CLOSERS[3])
 
        if uncertain_flag == 1:
            hedge    = random.choice(self.UNCERTAINTY_HEDGES)
            full_msg = f"{opener} {hedge} {instruction} {closer}"
        else:
            full_msg = f"{opener} {instruction} {closer}"
 
        return full_msg
 
engine = DecisionEngine()
 

# %%
print("\n── PART 4 — UNCERTAINTY MODELING ──")
 
def compute_uncertainty(state_probs, intensity_raw, text, metadata_row,
                        state_pred=None, intensity_pred=None):
    reasons = []
 
    max_prob = float(np.max(state_probs))
 
    intensity_dist = abs(intensity_raw - round(intensity_raw))
    if intensity_dist > 0.4:
        reasons.append(f"borderline_intensity ({intensity_raw:.2f})")
 
    word_count = len(str(text).split())
    if word_count <= 3:
        reasons.append(f"short_text ({word_count} words)")
        max_prob = min(max_prob, 0.45)   # ← hard cap for near-empty input
 
    missing_count = sum(1 for v in metadata_row if pd.isna(v))
    if missing_count >= 2:
        reasons.append(f"missing_metadata ({missing_count} fields)")
 
    borderline_penalty = intensity_dist * 0.3   # max 0.15 penalty at dist=0.5
    unified_conf = max(0.0, max_prob - borderline_penalty)
 
    if unified_conf < 0.45:
        reasons.append(f"low_class_confidence ({unified_conf:.2f})")
 
    if state_pred is not None and intensity_pred is not None:
        constraints = {
            "calm": (1,3), "neutral": (1,3),
            "overwhelmed": (2,5), "anxious": (2,5), "sad": (2,5),
        }
        if state_pred in constraints:
            lo, hi = constraints[state_pred]
            if not (lo <= intensity_pred <= hi):
                reasons.append(f"state_intensity_conflict ({state_pred},{intensity_pred})")
                unified_conf = min(unified_conf, 0.55)   # cap confidence on contradiction
 
    uncertain_flag = 1 if len(reasons) > 0 else 0
    return round(unified_conf, 4), uncertain_flag, reasons
 
sample_probs = state_model.predict_proba(X_val_dense[:3])
for i in range(3):
    row  = data.iloc[i]
    conf, flag, reasons = compute_uncertainty(
        state_probs    = sample_probs[i],
        intensity_raw  = y_i_pred_raw[i] if i < len(y_i_pred_raw) else 3.0,
        text           = row["journal_text"],
        metadata_row   = [row.get(c, np.nan) for c in numerical_cols],
        state_pred     = le.inverse_transform([y_s_tr[i]])[0] if i < len(y_s_tr) else None,
        intensity_pred = int(y_i_pred[i]) if i < len(y_i_pred) else None,
    )
    print(f"[PART 4] Sample {i+1}: confidence={conf}, uncertain={flag}, reasons={reasons}")
 
 

# %%
print("\n── INFERENCE PIPELINE — run_inference() ──")
 
def run_inference(df, state_model, intensity_model, meta_preprocessor,
                  tfidf, state_labels, le, engine, split_label=""):
    results = []
    df = df.copy().reset_index(drop=True)
 
    texts_clean = df["journal_text"].apply(clean_text).tolist()
    text_feats = embedding_model.encode(
        texts_clean,
        convert_to_numpy=True
    ) 
    meta_cols = numerical_cols + categorical_cols
    for col in meta_cols:
        if col not in df.columns:
            df[col] = np.nan
    meta_feats = meta_preprocessor.transform(df[meta_cols])
 
    X_combined = np.hstack([text_feats, meta_feats])    
    state_probs_all   = state_model.predict_proba(X_combined)
    state_preds_enc   = state_model.predict(X_combined)
    state_preds       = le.inverse_transform(state_preds_enc)
    intensity_raw_all = intensity_model.predict(X_combined)
    intensity_preds   = np.clip(np.round(intensity_raw_all), 1, 5).astype(int)
 
    for i, row in df.iterrows():
        pred_state     = state_preds[i]
        pred_intensity = int(intensity_preds[i])
        int_raw        = float(intensity_raw_all[i])
        probs          = state_probs_all[i]
 
        stress = float(row.get("stress_level", np.nan) or 3.0)
        energy = float(row.get("energy_level", np.nan) or 3.0)
        if np.isnan(stress): stress = 3.0
        if np.isnan(energy): energy = 3.0
 
        conf, unc_flag, reasons = compute_uncertainty(
            state_probs    = probs,
            intensity_raw  = int_raw,
            text           = row.get("journal_text", ""),
            metadata_row   = [row.get(c, np.nan) for c in numerical_cols],
            state_pred     = pred_state,
            intensity_pred = pred_intensity,
        )
 
        time_of_day = str(row.get("time_of_day", "morning") or "morning")
        what_to_do, when_to_do = engine.decide(
            state        = pred_state,
            intensity    = pred_intensity,
            stress       = stress,
            energy       = energy,
            time_of_day  = time_of_day,
            uncertain_flag = unc_flag,
        )
 
        message = engine.generate_message(
            state        = pred_state,
            action       = what_to_do,
            timing       = when_to_do,
            intensity    = pred_intensity,
            uncertain_flag = unc_flag,
        )
 
        results.append({
            "id":                  row.get("id", f"row_{i}"),
            "predicted_state":     pred_state,
            "predicted_intensity": pred_intensity,
            "confidence":          conf,
            "uncertain_flag":      unc_flag,
            "what_to_do":          what_to_do,
            "when_to_do":          when_to_do,
            "supportive_message":  message,
            "uncertainty_reasons": "; ".join(reasons) if reasons else "none",
        })
 
    out = pd.DataFrame(results)
    if split_label:
        print(f"[{split_label}] State distribution:\n{out['predicted_state'].value_counts().to_string()}")
        print(f"[{split_label}] Intensity distribution:\n{out['predicted_intensity'].value_counts().sort_index().to_string()}")
    return out
 
train_results = run_inference(
    data,
    state_model,
    intensity_model,
    metadata_preprocessor,
    embedding_model,
    state_labels,
    le,
    engine,
    split_label="TRAIN"
)

test_results = run_inference(
    test_df,
    state_model,
    intensity_model,
    metadata_preprocessor,
    embedding_model,
    state_labels,
    le,
    engine,
    split_label="TEST"
)
 
print(f"\n[INFERENCE] Train rows: {len(train_results)} | Test rows: {len(test_results)}")

# %%
print("\n── PART 5 — FEATURE IMPORTANCE ──")
 
cat_names          = (metadata_preprocessor
                      .named_transformers_["cat"]["ohe"]
                      .get_feature_names_out(categorical_cols).tolist())
meta_feature_names = numerical_cols + cat_names
text_feat_names = [
    f"embedding_{i}"
    for i in range(text_features_train.shape[1])
]
all_feature_names  = text_feat_names + meta_feature_names
 
importances = state_model.feature_importances_
n_text_feats = len(text_feat_names)
min_len      = min(len(importances), len(all_feature_names))
 
imp_df = pd.DataFrame({
    "feature":    all_feature_names[:min_len],
    "importance": importances[:min_len],
    "type":       ["text" if i < n_text_feats else "metadata" for i in range(min_len)],
}).sort_values("importance", ascending=False)
 
print("\n[PART 5] Top 15 Features (State Model):")
print(imp_df.head(15).to_string(index=False))
 
text_imp  = imp_df[imp_df["type"] == "text"]["importance"].sum()
meta_imp  = imp_df[imp_df["type"] == "metadata"]["importance"].sum()
total_imp = text_imp + meta_imp
print(f"\n[PART 5] Text contribution    : {text_imp/total_imp*100:.1f}%")
print(f"[PART 5] Metadata contribution: {meta_imp/total_imp*100:.1f}%")

# %%

print("\n── PART 6 — ABLATION STUDY ──")
 
def quick_model(X_tr, y_tr, X_va, y_va, task="clf"):
    if task == "clf":
        m = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        m.fit(X_tr, y_tr)
        return accuracy_score(y_va, m.predict(X_va))
    else:
        m = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        m.fit(X_tr, y_tr)
        return mean_absolute_error(y_va, np.clip(np.round(m.predict(X_va)), 1, 5))
 
n_text_feats = text_features_train.shape[1]
X_tr_full    = X_tr_dense_bal
X_val_full   = X_val_dense
 
X_tr_txt     = X_tr_full[:, :n_text]
X_val_txt    = X_val_full[:, :n_text]
X_tr_meta    = X_tr_full[:, n_text:]
X_val_meta   = X_val_full[:, n_text:]
 
acc_txt_only  = quick_model(X_tr_txt,  y_s_tr_bal, X_val_txt,  y_s_val, "clf")
acc_meta_only = quick_model(X_tr_meta, y_s_tr_bal, X_val_meta, y_s_val, "clf")
acc_hybrid    = accuracy_score(y_s_val, state_model.predict(X_val_full))
 
mae_txt_only  = quick_model(X_tr_dense[:, :n_text], y_i_tr_for_reg,
                             X_val_dense[:, :n_text], y_i_val, "reg")
mae_meta_only = quick_model(X_tr_dense[:, n_text:], y_i_tr_for_reg,
                             X_val_dense[:, n_text:], y_i_val, "reg")
mae_hybrid    = mean_absolute_error(
    y_i_val, np.clip(np.round(intensity_model.predict(X_val_full)), 1, 5))
 
print("\n╔══════════════════════════════════════════════════════╗")
print("║         ABLATION STUDY RESULTS                       ║")
print("╠════════════════════╦══════════════╦══════════════════╣")
print("║ Configuration      ║ State Acc ↑  ║ Intensity MAE ↓  ║")
print("╠════════════════════╬══════════════╬══════════════════╣")
print(f"║ Text-Only          ║  {acc_txt_only:.4f}       ║  {mae_txt_only:.4f}          ║")
print(f"║ Metadata-Only      ║  {acc_meta_only:.4f}       ║  {mae_meta_only:.4f}          ║")
print(f"║ Hybrid (Ours) ★    ║  {acc_hybrid:.4f}       ║  {mae_hybrid:.4f}          ║")
print("╚════════════════════╩══════════════╩══════════════════╝")
 

# %%
print("\n── SAVING OUTPUTS ──")
 
output_cols = [
    "id", "predicted_state", "predicted_intensity",
    "confidence", "uncertain_flag",
    "what_to_do", "when_to_do",
    "supportive_message", "uncertainty_reasons",
]
 
test_results[output_cols].to_csv("predictions.csv", index=False)
print(f"[OUTPUT] predictions.csv saved → {len(test_results)} rows (test set)")
print(test_results[output_cols].head(5).to_string(index=False))
 
train_results[output_cols].to_csv("predictions_train.csv", index=False)
print(f"[OUTPUT] predictions_train.csv saved → {len(train_results)} rows")
 

# %%


artifacts = {
    "state_model": state_model,
    "intensity_model": intensity_model,
    "metadata_preprocessor": metadata_preprocessor,
    "state_labels": state_labels,
    "le": le,
    "numerical_cols": numerical_cols,
    "categorical_cols": categorical_cols,
}

with open("model_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

embedding_model.save("./minilm_model")

print("[OUTPUT] model_artifacts.pkl saved")
print("[OUTPUT] MiniLM model saved")
print("\n── PIPELINE COMPLETE ──")

