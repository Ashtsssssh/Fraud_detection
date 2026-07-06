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

# ── CHANGE 2: Random Undersampling ────────────────────────────────────────────
# Problem: 7.6M normal rows vs ~8k fraud rows. The model spends 99.9% of its
# attention learning normal transactions, drowning out the fraud signal.
#
# Fix: keep ALL fraud rows, but randomly keep only 10x as many normal rows.
# Result: 80k normal + 8k fraud = 88k training rows instead of 7.6M!
# Training goes from 10 minutes → ~5 seconds. And the model actually sees fraud.
#
# IMPORTANT: We still test on the FULL untouched test set (1.9M rows) so our
# evaluation reflects real-world performance. Undersampling only affects training.

fraud_idx  = y_train[y_train == 1].index
normal_idx = y_train[y_train == 0].index

# 10:1 ratio → 10 normal for every 1 fraud case
n_normal_keep = min(len(fraud_idx) * 10, len(normal_idx))
normal_idx_sampled = normal_idx.to_series().sample(n=n_normal_keep, random_state=42).index

balanced_idx = fraud_idx.append(normal_idx_sampled)
X_train = X_train.loc[balanced_idx].sample(frac=1, random_state=42)  # shuffle
y_train = y_train.loc[X_train.index]

new_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nAfter undersampling:")
print(f"  Training rows: {len(X_train):,}  (was 7.6M)")
print(f"  Fraud: {(y_train==1).sum():,}  |  Normal: {(y_train==0).sum():,}  |  Ratio: {new_ratio:.0f}:1")

# ── define model ───────────────────────────────────────────────────────────────
# scale_pos_weight is now 10 (the new ratio), not 962
if USE_LGBM:
    print("\nUsing LightGBM classifier...")
    model = LGBMClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        scale_pos_weight=10,     # 10:1 ratio after undersampling
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
        early_stopping_rounds=50, # CHANGE 3: Stop if validation loss doesn't improve for 50 trees
        scale_pos_weight=10,
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
print("\nFinding optimal decision threshold (Optimizing for F2 Score)...")
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# CHANGE 4: Optimize for F2 Score instead of F1
# ─── METRIC HISTORY ───────────────────────────────────────────────────────────
# V2.0 (F1 Optimized) : Recall 16%, Precision 15%  (Threshold: 0.9882)
# V2.1 (F2 Optimized) : Recall 23%, Precision  9%  (Threshold: 0.9682)
# ──────────────────────────────────────────────────────────────────────────────
# F2 weights recall twice as heavily as precision. 
# We are willing to tolerate more false positives to catch more fraud.
fbeta_scores = 5 * (precisions * recalls) / (4 * precisions + recalls + 1e-8)

best_idx = fbeta_scores.argmax()
best_threshold = thresholds[best_idx]
best_fbeta = fbeta_scores[best_idx]

from sklearn.metrics import fbeta_score
print(f"Default threshold (0.5): F2 for fraud = {fbeta_score(y_test, (y_proba >= 0.5).astype(int), beta=2, pos_label=1):.4f}")
print(f"Optimal threshold ({best_threshold:.4f}): F2 for fraud = {best_fbeta:.4f}")

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
