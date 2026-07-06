import shap
import joblib
import matplotlib.pyplot as plt

def explain_model(X_sample):
    """
    SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain the output of any ML model.
    It tells us exactly how much each feature contributed to the final prediction.
    """
    print("Loading model for SHAP explanation...")
    # Our model file now contains both the model and the optimal threshold
    data = joblib.load("models/aml_model.pkl")
    model = data["model"]
    
    print("Calculating SHAP values... (this might take a moment)")
    explainer = shap.TreeExplainer(model)
    
    # LightGBM/XGBoost output SHAP values slightly differently, we handle the format here
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # Take the fraud class if it returns a list
        
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png")
    print("Saved SHAP plot to models/shap_summary.png")
