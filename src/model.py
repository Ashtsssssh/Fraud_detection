from xgboost import XGBClassifier

def get_model():
    """
    XGBoost Classifier — Variation log:

    V1 (ACTIVE) : n_estimators=1000, lr=0.05, scale_pos_weight handled in train.py
                  WHY: logloss was still falling at tree 499, so we give it 1000 trees
                  RESULT: TBD

    V2 (next)   : n_estimators=1000, learning_rate=0.01
                  WHY: slower learning rate = more careful correction per tree
                  EXPECTED: better precision, less overfitting

    V3 (next)   : n_estimators=500, scale_pos_weight=50 (instead of ~962)
                  WHY: 962x was too aggressive causing tons of false positives
                  EXPECTED: better precision, lower recall
    """

    # ── V1: More trees (ACTIVE) ──────────────────────────────────────────────
    return XGBClassifier(
        n_estimators=1000,      # was 500 — logloss still falling at tree 499
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        random_state=42
    )

    # ── V2: Lower learning rate ──────────────────────────────────────────────
    # return XGBClassifier(
    #     n_estimators=1000,
    #     max_depth=8,
    #     learning_rate=0.01,    # slower, more precise corrections
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     tree_method="hist",
    #     device="cuda",
    #     eval_metric="logloss",
    #     random_state=42
    # )

    # ── V3: Reduce scale_pos_weight to fix false positives ───────────────────
    # return XGBClassifier(
    #     n_estimators=500,
    #     max_depth=8,
    #     learning_rate=0.05,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     tree_method="hist",
    #     device="cuda",
    #     eval_metric="logloss",
    #     scale_pos_weight=50,   # instead of ~962, less aggressive balancing
    #     random_state=42
    # )
