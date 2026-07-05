import pandas as pd
from src.config import DATA_PATH

def load_data():
    # Load the raw dataset
    df = pd.read_csv(DATA_PATH)
    
    # Create timestamp from Date and Time columns
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Rename columns to lowercase with underscores for clean Python code
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
    
    return df
