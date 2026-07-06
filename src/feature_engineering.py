import pandas as pd
import numpy as np
import networkx as nx

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

    # ── NEW GRAPH ML FEATURES (NETWORKX) ─────────────────────────────────────
    print("  Building transaction graph for Network features...")
    
    # 1. Fast Degree Features via Pandas (Unique counterparties)
    # How many UNIQUE accounts does this sender send to? (Launderers disburse money widely)
    out_degree = df.groupby('sender_account')['receiver_account'].nunique()
    df['sender_out_degree'] = df['sender_account'].map(out_degree).fillna(0)
    
    # How many UNIQUE accounts does this receiver get money from? (Mules receive from many victims)
    in_degree = df.groupby('receiver_account')['sender_account'].nunique()
    df['receiver_in_degree'] = df['receiver_account'].map(in_degree).fillna(0)
    
    # 2. PageRank via NetworkX
    print("  Computing PageRank (this may take a few minutes)...")
    # Build a directed graph from the transactions
    G = nx.from_pandas_edgelist(df, source='sender_account', target='receiver_account', create_using=nx.DiGraph())
    
    # Calculate PageRank (max_iter=30 to keep it fast on 9.5M rows)
    try:
        pageranks = nx.pagerank(G, alpha=0.85, max_iter=30)
    except nx.PowerIterationFailedConvergence:
        print("  Warning: PageRank failed to converge fully, using fallback...")
        pageranks = nx.pagerank(G, alpha=0.85, max_iter=10) # Fallback if it fails

    # Map PageRank scores back to the dataframe
    df['sender_pagerank'] = df['sender_account'].map(pageranks).fillna(0)
    df['receiver_pagerank'] = df['receiver_account'].map(pageranks).fillna(0)

    print("  Graph features done!")
    return df
