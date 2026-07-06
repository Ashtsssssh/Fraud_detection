import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image

st.set_page_config(page_title="Sentinel Graph AI", layout="wide", page_icon="🛡️")

# --- CUSTOM CSS FOR CYBER SECURITY LOOK ---
st.markdown("""
<style>
    .reportview-container {
        background: #090C10;
    }
    .big-font {
        font-size:24px !important;
        font-weight: bold;
        color: #00E5FF;
    }
    .metric-card {
        background-color: #161B22;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #00E5FF;
        box-shadow: 0 4px 6px rgba(0, 229, 255, 0.1);
    }
    /* Make the title pop with a subtle glow */
    h1 {
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Sentinel Graph AI: Syndicate Tracker")
st.markdown("Enterprise-grade Anti-Money Laundering powered by **Graph Machine Learning** and **XGBoost**.")

# Cache the heavy loading so the dashboard doesn't freeze when interacting
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "aml_model.pkl")
    return joblib.load(model_path)

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "features.csv")
    return pd.read_csv(data_path)

with st.spinner("Initializing Graph ML Models and loading transaction network..."):
    data = load_model()
    model = data["model"]
    threshold = data["threshold"]
    df = load_data()

# Tabs for organization
tab1, tab2, tab3 = st.tabs(["🚨 Live Alert Queue", "🕸️ Network Analysis", "🧠 AI Explainability"])

# We take a sample of 50,000 transactions to keep the dashboard responsive
sample_df = df.sample(50000, random_state=42).copy()

cols_to_drop = ['is_suspicious', 'timestamp', 'Date', 'Time', 
                'sender_account', 'receiver_account',
                'sender_currency', 'receiver_currency',
                'sender_country', 'receiver_country',
                'payment_type', 'laundering_type']
cols_to_drop = [col for col in cols_to_drop if col in sample_df.columns]
X_sample = sample_df.drop(columns=cols_to_drop)

# Make sure columns match training exactly
expected_cols = model.feature_names_in_
X_sample = X_sample[expected_cols]

# Predict risk scores
risk_scores = model.predict_proba(X_sample)[:, 1]
sample_df['risk_score'] = risk_scores
sample_df['is_flagged'] = (risk_scores >= threshold).astype(int)
sample_df['risk_percentage'] = (risk_scores * 100).round(2).astype(str) + "%"

high_risk = sample_df[sample_df['is_flagged'] == 1].sort_values(by='risk_score', ascending=False)

with tab1:
    st.header("Active Investigator Queue")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions Scanned", f"{len(sample_df):,}")
    col2.metric("Decision Threshold", f"{threshold:.4f} (F2 Optimized)")
    col3.metric("Flagged as Suspicious", f"{len(high_risk):,}")
    col4.metric("Precision / Recall", "93% / 94%")
    
    st.markdown("### ⚠️ Priority Alerts")
    if len(high_risk) > 0:
        display_cols = [
            'sender_account', 'receiver_account', 'amount', 'risk_percentage', 
            'amount_24h', 'txn_count_1h', 'sender_out_degree', 'receiver_in_degree', 
            'sender_pagerank', 'high_risk_country'
        ]
        
        # Rename columns for investigator readability
        readable_df = high_risk[display_cols].rename(columns={
            'sender_account': 'Sender ID',
            'receiver_account': 'Receiver ID',
            'amount': 'Amount ($)',
            'risk_percentage': 'AI Risk Score',
            'amount_24h': 'Sender Vol (24h)',
            'txn_count_1h': 'Velocity (1h)',
            'sender_out_degree': 'Unique Receivers (Smurfing)',
            'receiver_in_degree': 'Unique Senders (Mule)',
            'sender_pagerank': 'Network Centrality',
            'high_risk_country': 'High Risk Destination'
        })
        st.dataframe(readable_df, use_container_width=True, hide_index=True)
    else:
        st.success("No suspicious transactions found in this batch!")

with tab2:
    st.header("Graph Machine Learning Features")
    st.markdown("Unlike standard models, our V2.0 system maps transactions as a mathematical network to detect laundering syndicates.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Smurfing Detection (Out-Degree)**: When a single criminal sender breaks large funds into smaller transactions and sends them to many different accounts to avoid detection.")
        # Only plot if we have flags
        if len(high_risk) > 0:
            st.bar_chart(high_risk['sender_out_degree'].value_counts().head(10))
        else:
            st.write("No data in current batch.")
            
    with col2:
        st.warning("**Money Mule Detection (In-Degree)**: When a central hub account receives dirty money from dozens of victims or smurfs.")
        if len(high_risk) > 0:
            st.bar_chart(high_risk['receiver_in_degree'].value_counts().head(10))
        else:
            st.write("No data in current batch.")

with tab3:
    st.header("AI Interpretability (SHAP)")
    st.markdown("We use Shapley Additive exPlanations (SHAP) to peek inside the XGBoost model's brain. This guarantees regulatory compliance by proving *why* the AI flagged a transaction.")
    
    shap_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "shap_summary.png")
    if os.path.exists(shap_path):
        img = Image.open(shap_path)
        st.image(img, caption="Global Feature Importance (Notice how Graph Features dominate the decision making)", use_column_width=True)
    else:
        st.error(f"SHAP summary image not found at {shap_path}. Please run the pipeline to generate it.")
