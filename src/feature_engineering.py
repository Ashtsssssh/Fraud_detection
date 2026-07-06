import pandas as pd
import numpy as np

def engineer_features(df):
    # Step 1: Sort chronologically — CRITICAL for velocity features
    # Everything needs to be in time order before we compute rolling windows
    df = df.sort_values('timestamp').copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # ── ORIGINAL FEATURES ────────────────────────────────────────────────────

    # Lifetime transaction count per sender
    df['txn_count'] = df.groupby('sender_account').cumcount() + 1

    # Log-transform the amount (smooths extreme values)
    df['amount_log'] = np.log1p(df['amount'])

    # Currency mismatch: sender sends USD but receiver gets EUR = suspicious
    df['currency_mismatch'] = (df['sender_currency'] != df['receiver_currency']).astype(int)

    # Cross-border: money leaving the country is harder to trace
    df['cross_border'] = (df['sender_country'] != df['receiver_country']).astype(int)

    # High-risk destination country
    high_risk = ['Mexico', 'Turkey', 'Morocco', 'UAE']
    df['high_risk_country'] = df['receiver_country'].isin(high_risk).astype(int)

    # ── NEW VELOCITY FEATURES ────────────────────────────────────────────────
    # Money laundering = move money FAST. These features capture that urgency.
    # We set timestamp as the index so pandas can roll over real time windows.

    print("  Computing velocity features (this takes a few minutes on 9.5M rows)...")

    df_indexed = df.set_index('timestamp')

    # NEW FEATURE 1: amount_24h
    # How much has this sender moved in the last 24 hours (BEFORE and including this txn)?
    # Launderers "burst" — they move large amounts rapidly before being detected.
    # A normal person might send $200 in a day; a launderer might send $500,000.
    df['amount_24h'] = (
        df_indexed.groupby('sender_account')['amount']
        .rolling('24h')
        .sum()
        .reset_index(level=0, drop=True)
        .values
    )

    # NEW FEATURE 2: txn_count_1h
    # How many transactions has this sender made in the last 1 hour?
    # "Structuring" / "smurfing" = many small transactions to avoid detection thresholds.
    # Normal: 1-2 transactions per hour. Launderer: 20+ transactions per hour.
    df['txn_count_1h'] = (
        df_indexed.groupby('sender_account')['amount']
        .rolling('1h')
        .count()
        .reset_index(level=0, drop=True)
        .values
    )

    # NEW FEATURE 3: amount_vs_avg
    # Is THIS transaction much larger than what this sender normally does?
    # e.g. if Alice usually sends $100, but today sent $50,000 → ratio = 500x → very suspicious
    # We compute the sender's historical average amount first
    sender_avg = df.groupby('sender_account')['amount'].transform('mean')
    df['amount_vs_avg'] = df['amount'] / (sender_avg + 1e-9)  # +1e-9 avoids divide by zero

    print("  Velocity features done!")
    return df
