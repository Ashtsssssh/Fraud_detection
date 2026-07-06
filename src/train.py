import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, f1_score

# ── choose classifier ──────────────────────────────────────────────────────────
# set USE_LGBM = True to use LightGBM, False to use XGBoost
USE_LGBM = False

if USE_LGBM:
    from lightgbm import LGBMClassifier
else:
    from xgboost import XGBClassifier

# ── load the processed data ────────────────────────────────────────────────────
df = pd.read_csv("data/processed/features.csv")

cols_to_drop = ['is_suspicious', 'timestamp', 'Date', 'Time',
                'sender_account', 'receiver_account',
                'sender_currency', 'receiver_currency',
                'sender_country', 'receiver_country',
                'payment_type', 'laundering_type']

cols_to_drop = [col for col in cols_to_drop if col in df.columns]

X = df.drop(columns=cols_to_drop)
y = df['is_suspicious']

# ── train/test split with stratify ────────────────────────────────────────────
# stratify=y ensures fraud cases are equally split between train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
print(f"Features: {list(X.columns)}")

# ── class imbalance ratio ─────────────────────────────────────────────────────
scale = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class imbalance: {scale:.0f}x")

# ── define model ───────────────────────────────────────────────────────────────
if USE_LGBM:
    print("\nUsing LightGBM classifier...")
    model = LGBMClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        scale_pos_weight=31,     # geometric mean: sqrt(962) ≈ 31 (balanced middle ground)
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
else:
    print("\nUsing XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        scale_pos_weight=scale,
        random_state=42
    )

# ── train ──────────────────────────────────────────────────────────────────────
print("Training started...")
if USE_LGBM:
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
else:
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

# ── get predicted probabilities (not just 0/1) ────────────────────────────────
# predict_proba gives the confidence score for each class
# [:, 1] = probability of being fraud (class 1)
y_proba = model.predict_proba(X_test)[:, 1]

# ── find the OPTIMAL threshold using precision-recall curve ───────────────────
# Default threshold = 0.5 → "if model is >50% sure it's fraud, flag it"
# But 0.5 is often wrong for imbalanced data!
# We find the threshold that gives the best F1 score for fraud detection
print("\nFinding optimal decision threshold...")
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Default threshold (0.5): F1 for fraud = {f1_score(y_test, (y_proba >= 0.5).astype(int), pos_label=1):.4f}")
print(f"Optimal threshold ({best_threshold:.4f}): F1 for fraud = {best_f1:.4f}")

# ── evaluate with BOTH thresholds ────────────────────────────────────────────
print("\n--- Results with DEFAULT threshold (0.5) ---")
y_pred_default = (y_proba >= 0.5).astype(int)
print(f"Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
print(classification_report(y_test, y_pred_default))

print(f"\n--- Results with OPTIMAL threshold ({best_threshold:.4f}) ---")
y_pred_optimal = (y_proba >= best_threshold).astype(int)
print(f"Accuracy: {accuracy_score(y_test, y_pred_optimal):.4f}")
print(classification_report(y_test, y_pred_optimal))

# ── save model and threshold ──────────────────────────────────────────────────
joblib.dump({"model": model, "threshold": best_threshold}, "models/aml_model.pkl")
print(f"\nModel + optimal threshold saved to models/aml_model.pkl")
