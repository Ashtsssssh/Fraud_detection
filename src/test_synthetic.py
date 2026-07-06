import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import joblib
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix

# Make sure we can import feature_engineering
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import engineer_features

print("1. Generating BELIEVABLE Synthetic Data...")
np.random.seed(42)
random.seed(42)

transactions = []
start_date = datetime(2023, 1, 1)

currencies = ['USD', 'EUR', 'GBP']
countries = ['USA', 'UK', 'Germany']
high_risk = ['Panama', 'Cayman Islands', 'Russia', 'Uganda', 'Syria']

# --- GENERATE NORMAL TRANSACTIONS ---
print("   -> Generating Normal Users (Regular habits, low network degree, consistent amounts)...")
normal_senders = [f"NORM_SND_{i}" for i in range(2000)]
normal_receivers = [f"NORM_RCV_{i}" for i in range(4000)]

for sender in normal_senders:
    # Normal users have 1 to 3 favorite receivers (e.g. landlord, grocery, friend)
    fav_receivers = random.sample(normal_receivers, random.randint(1, 3))
    # Typical transaction size (e.g. $50 to $1500)
    base_amount = random.uniform(50, 1500)
    
    # 5 to 15 transactions over 30 days
    num_txns = random.randint(5, 15)
    for _ in range(num_txns):
        t = start_date + timedelta(days=random.randint(0, 29), hours=random.randint(0, 23))
        # Normal amount fluctuates slightly around their base amount (low variance)
        amt = max(1, np.random.normal(base_amount, base_amount * 0.1))
        
        transactions.append({
            'timestamp': t,
            'sender_account': sender,
            'receiver_account': random.choice(fav_receivers),
            'amount': amt,
            'sender_currency': 'USD',
            'receiver_currency': 'USD',
            'sender_country': 'USA',
            'receiver_country': 'USA',
            'payment_type': 'Transfer',
            'true_label': 0  # Normal
        })

# --- GENERATE FRAUD TRANSACTIONS ---
print("   -> Generating Fraud Users (Velocity spikes, High network degree / Smurfing)...")

# Fraud Type 1: High Velocity Structuring (Sending many large sums in a short time)
for i in range(50):
    sender = f"FRAUD_VEL_{i}"
    receiver = f"MULE_RCV_{i}"
    # Pick a random 2-hour window
    base_t = start_date + timedelta(days=random.randint(0, 29), hours=random.randint(0, 20))
    # Send 10 transactions in 2 hours of $9,900 (avoiding $10k reporting limit)
    for _ in range(10):
        t = base_t + timedelta(minutes=random.randint(0, 120))
        transactions.append({
            'timestamp': t,
            'sender_account': sender,
            'receiver_account': receiver,
            'amount': random.uniform(9500, 9999),
            'sender_currency': 'USD',
            'receiver_currency': 'USD',
            'sender_country': 'USA',
            'receiver_country': random.choice(high_risk), # Send to high risk country
            'payment_type': 'Transfer',
            'true_label': 1  # Fraud
        })

# Fraud Type 2: Smurfing / High Out-Degree (1 sender, 30 different receivers)
for i in range(50):
    sender = f"FRAUD_SMURF_{i}"
    base_t = start_date + timedelta(days=random.randint(0, 29))
    for j in range(30):
        t = base_t + timedelta(minutes=random.randint(0, 300))
        transactions.append({
            'timestamp': t,
            'sender_account': sender,
            'receiver_account': f"RANDOM_RCV_{i}_{j}",
            'amount': random.uniform(1000, 5000),
            'sender_currency': 'USD',
            'receiver_currency': 'EUR', # Currency mismatch
            'sender_country': 'USA',
            'receiver_country': 'USA',
            'payment_type': 'Transfer',
            'true_label': 1  # Fraud
        })

df = pd.DataFrame(transactions)
df['Date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
df['Time'] = df['timestamp'].dt.strftime('%H:%M:%S')

# Sort chronologically (critical for velocity features)
df = df.sort_values('timestamp').reset_index(drop=True)

# Save true labels and remove them so the feature pipeline doesn't see them
true_labels = df[['sender_account', 'receiver_account', 'timestamp', 'true_label']].copy()
# Deduplicate just in case same timestamp
true_labels = true_labels.drop_duplicates(subset=['sender_account', 'receiver_account', 'timestamp'])
df = df.drop(columns=['true_label'])

print(f"\nCreated {len(df)} realistic transactions.")
print(f"  - Normal Transactions: {len(df) - 2000}")
print(f"  - Fraud Transactions:  2000")

print("\n2. Pushing through Feature Engineering (Graph & Velocity)...")
df = engineer_features(df)

# Re-attach true labels to check accuracy later
df = df.merge(true_labels, on=['sender_account', 'receiver_account', 'timestamp'], how='left')

cols_to_drop = ['is_suspicious', 'timestamp', 'Date', 'Time',
                'sender_account', 'receiver_account',
                'sender_currency', 'receiver_currency',
                'sender_country', 'receiver_country',
                'payment_type', 'laundering_type', 'true_label']
cols_to_drop = [c for c in cols_to_drop if c in df.columns]

X = df.drop(columns=cols_to_drop)
y_true = df['true_label']

print("\n3. Loading the V2.0 AML Model...")
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "aml_model.pkl")
data = joblib.load(model_path)
model = data["model"]
threshold = data["threshold"]

print(f"\n4. Predicting blindly (Model does not know the true labels)...")
# Make sure columns match training exactly
expected_cols = model.feature_names_in_
X = X[expected_cols]

y_proba = model.predict_proba(X)[:, 1]
y_pred = (y_proba >= threshold).astype(int)

print("\n========================================")
print("       REALITY CHECK RESULTS              ")
print("========================================")
print("Confusion Matrix:")
print("                 Predicted Normal | Predicted Fraud")
cm = confusion_matrix(y_true, y_pred)
print(f"Actual Normal  |       {cm[0][0]:<10} |      {cm[0][1]:<10}")
print(f"Actual Fraud   |       {cm[1][0]:<10} |      {cm[1][1]:<10}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Normal", "Fraud"]))
print("========================================")
