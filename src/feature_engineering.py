import pandas as pd
import numpy as np

def engineer_features(df):
    # Step 1: Sort all transactions by time (chronological order)
    # This is critical for computing "how many transactions has this sender made SO FAR"
    df = df.sort_values('timestamp').copy()

    # Step 2: Transaction frequency per sender
    # cumcount() counts 0,1,2,3... for each sender. We add 1 to start from 1.
    # Example: Alice's 1st txn -> txn_count=1, 2nd -> 2, 3rd -> 3
    # Suspicious accounts tend to have MANY transactions in a short time
    df['txn_count'] = df.groupby('sender_account').cumcount() + 1

    # Step 3: Log-transform the amount
    # Transaction amounts range from $1 to $1,000,000+
    # Log smooths this out so the model doesn't get confused by extreme values
    # log1p means log(1 + amount), which handles amount=0 safely
    df['amount_log'] = np.log1p(df['amount'])

    # Step 4: Currency mismatch flag
    # If you send USD but the receiver gets EUR, that's suspicious (value can be hidden)
    # 1 = mismatch (suspicious), 0 = same currency (normal)
    df['currency_mismatch'] = (df['sender_currency'] != df['receiver_currency']).astype(int)

    # Step 5: Cross-border transaction flag
    # Money leaving the country is harder to trace — a common laundering tactic
    # 1 = cross-border (suspicious), 0 = domestic (normal)
    df['cross_border'] = (df['sender_country'] != df['receiver_country']).astype(int)

    # Step 6: High-risk country flag
    # Some countries have weak AML regulations — known laundering hotspots
    high_risk = ['Mexico', 'Turkey', 'Morocco', 'UAE']
    df['high_risk_country'] = df['receiver_country'].isin(high_risk).astype(int)

    return df
