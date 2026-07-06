import pandas as pd
import numpy as np
import sys, os

# make sure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from feature_engineering import engineer_features

# ── Step 1: load raw data ──────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/raw/SAML-D.csv")

# ── Step 2: create timestamp ───────────────────────────────────────────────────
df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# ── Step 3: rename columns ─────────────────────────────────────────────────────
df = df.rename(columns={
    'Sender_account':       'sender_account',
    'Receiver_account':     'receiver_account',
    'Amount':               'amount',
    'Payment_currency':     'sender_currency',
    'Received_currency':    'receiver_currency',
    'Sender_bank_location': 'sender_country',
    'Receiver_bank_location':'receiver_country',
    'Payment_type':         'payment_type',
    'Is_laundering':        'is_suspicious',
    'Laundering_type':      'laundering_type'
})

print(f"Loaded {len(df)} transactions")

# ── Step 4: feature engineering (now uses src/feature_engineering.py) ─────────
print("Engineering features...")
df = engineer_features(df)  # includes velocity features!

# ── Step 5: save processed data ───────────────────────────────────────────────
df.to_csv("data/processed/features.csv", index=False)
print("Saved processed data to data/processed/features.csv")

# ── Step 6: training ──────────────────────────────────────────────────────────
print("\nStarting training...")
exec(open("src/train.py").read())

# ── Step 7: SHAP model explanations ───────────────────────────────────────────
print("\nGenerating model explanations (SHAP)...")
from explain import explain_model
X_sample = df.drop(columns=['is_suspicious', 'timestamp', 'Date', 'Time',
                'sender_account', 'receiver_account',
                'sender_currency', 'receiver_currency',
                'sender_country', 'receiver_country',
                'payment_type', 'laundering_type'], errors='ignore').sample(500, random_state=42)
explain_model(X_sample)
