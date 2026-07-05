from xgboost import XGBClassifier

def get_model():
    """
    Define and return the XGBoost classifier.

    XGBoost (eXtreme Gradient Boosting) is an ensemble method — it builds
    hundreds of decision trees one after another, where each new tree tries
    to correct the mistakes of the previous ones. The final prediction is
    a vote across all trees.

    Hyperparameter explanations:
    - n_estimators=500   : Build 500 trees. More trees = better accuracy but slower.
    - max_depth=8        : Each tree can have at most 8 levels of splits.
                           Deeper = more complex patterns, but risks overfitting.
    - learning_rate=0.05 : How much each new tree corrects the previous errors.
                           Small = slower but more precise learning.
    - subsample=0.8      : Each tree is trained on a random 80% of the data.
                           This prevents overfitting (memorizing training data).
    - colsample_bytree=0.8: Each tree uses a random 80% of features.
                           Forces the model to learn diverse patterns.
    - tree_method="hist" : Uses histogram-based algorithm — much faster for large datasets.
    - device="cuda"      : Run on GPU (NVIDIA CUDA). ~10-20x faster than CPU.
                           Falls back to CPU if no GPU is available.
    - eval_metric="logloss": Measures error using log loss (good for binary classification).
    - random_state=42    : Seed for reproducibility — same results every run.
    """
    return XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        random_state=42
    )
