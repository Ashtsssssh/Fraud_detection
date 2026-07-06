import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AML Dashboard", layout="wide", page_icon="🚨")
st.title("🚨 AML Transaction Monitoring")
st.markdown("Real-time transaction risk analysis using LightGBM.")

# Cache the heavy loading so the dashboard doesn't freeze when interacting
@st.cache_resource
def load_model():
    return joblib.load("models/aml_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/features.csv")

st.info("Loading model and parsing 7.6M transactions... this will take a few seconds on first load.")
data = load_model()
model = data["model"]
threshold = data["threshold"]

df = load_data()

# We take a sample of 50,000 transactions to keep the dashboard responsive
# In a real app, this would be streaming live data
sample_df = df.sample(50000, random_state=42).copy()

cols_to_drop = ['is_suspicious', 'timestamp', 'Date', 'Time', 
                'sender_account', 'receiver_account',
                'sender_currency', 'receiver_currency',
                'sender_country', 'receiver_country',
                'payment_type', 'laundering_type']
cols_to_drop = [col for col in cols_to_drop if col in sample_df.columns]
X_sample = sample_df.drop(columns=cols_to_drop)

# Predict risk scores
risk_scores = model.predict_proba(X_sample)[:, 1]
sample_df['risk_score'] = risk_scores
sample_df['is_flagged'] = (risk_scores >= threshold).astype(int)

# Dashboard UI
col1, col2, col3 = st.columns(3)
col1.metric("Transactions Scanned", len(sample_df))
col2.metric("Decision Threshold", f"{threshold:.4f}")
col3.metric("Flagged as Suspicious", sample_df['is_flagged'].sum())

st.subheader("⚠️ High Risk Transactions Queue")
high_risk = sample_df[sample_df['is_flagged'] == 1].sort_values(by='risk_score', ascending=False)

if len(high_risk) > 0:
    # Show the most important columns to investigators
    display_cols = ['sender_account', 'receiver_account', 'amount', 'risk_score', 'cross_border', 'high_risk_country']
    st.dataframe(high_risk[display_cols], use_container_width=True)
else:
    st.success("No suspicious transactions found in this batch!")
