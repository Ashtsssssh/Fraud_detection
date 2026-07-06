import pandas as pd
import numpy as np

# Step 1: load raw data
print("Loading data...")
df = pd.read_csv("data/raw/SAML-D.csv")

# Step 2: create timestamp
df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# Step 3: rename columns
df = df.rename(columns={
    'Sender_account': 'sender_account',
    'Receiver_account': 'receiver_account',
    'Amount': 'amount',
    'Payment_currency': 'sender_currency',
    'Received_currency': 'receiver_currency',
    'Sender_bank_location': 'sender_country',
    'Receiver_bank_location': 'receiver_country',
    'Payment_type': 'payment_type',
    'Is_laundering': 'is_suspicious',
    'Laundering_type': 'laundering_type'
})

print(f"Loaded {len(df)} transactions")

# Step 4: feature engineering
print("Engineering features...")
df = df.sort_values('timestamp').copy()
df['txn_count'] = df.groupby('sender_account').cumcount() + 1
df['amount_log'] = np.log1p(df['amount'])
df['currency_mismatch'] = (df['sender_currency'] != df['receiver_currency']).astype(int)
df['cross_border'] = (df['sender_country'] != df['receiver_country']).astype(int)
high_risk = ['Mexico', 'Turkey', 'Morocco', 'UAE']
df['high_risk_country'] = df['receiver_country'].isin(high_risk).astype(int)

# Step 5: save processed data
df.to_csv("data/processed/features.csv", index=False)
print("Saved processed data to data/processed/features.csv")

# Step 6: run training
print("\nStarting training...")
exec(open("src/train.py").read())

# Step 7: explain model
print("\nGenerating model explanations (SHAP)...")
exec(open("src/explain.py").read())
# Since explain.py expects X_sample, we'll pass a small sample
X_sample = df.drop(columns=['is_suspicious', 'timestamp', 'Date', 'Time', 
                'sender_account', 'receiver_account',
                'sender_currency', 'receiver_currency',
                'sender_country', 'receiver_country',
                'payment_type', 'laundering_type'], errors='ignore').sample(500, random_state=42)
from src.explain import explain_model
explain_model(X_sample)
