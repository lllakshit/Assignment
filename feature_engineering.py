import pandas as pd
import numpy as np

def build_features(df):
    """
    Build wallet-level features from raw transaction dataframe.
    """
    # Normalize wallet column as index
    df['wallet'] = df['wallet'].str.lower()

    # Aggregation features per wallet
    aggregations = {
        'action': ['count'],
        'amount': ['sum', 'mean', 'std'],
    }

    # Count transactions per action type per wallet
    action_counts = df.pivot_table(index='wallet', columns='action', values='amount', aggfunc='count', fill_value=0)

    # Sum volumes per action
    action_sums = df.pivot_table(index='wallet', columns='action', values='amount', aggfunc='sum', fill_value=0)

    # Total transactions count
    total_tx = df.groupby('wallet')['action'].count()

    features = pd.DataFrame(index=total_tx.index)
    features['total_tx'] = total_tx

    # Add action counts
    for col in action_counts.columns:
        features[f'{col}_count'] = action_counts[col]

    # Add action sums
    for col in action_sums.columns:
        features[f'{col}_sum'] = action_sums[col]

    # Repay to borrow ratio
    features['repay_to_borrow_ratio'] = features.apply(
        lambda x: x['repay_count']/x['borrow_count'] if x['borrow_count'] > 0 else 0, axis=1)

    # Borrow to deposit ratio
    features['borrow_to_deposit_ratio'] = features.apply(
        lambda x: x['borrow_sum']/x['deposit_sum'] if x['deposit_sum'] > 0 else 0, axis=1)

    # Liquidation indicator
    features['has_liquidation'] = features['liquidationcall_count'].apply(lambda x: 1 if x > 0 else 0)

    # Average transaction amount std (volatility)
    amt_std = df.groupby('wallet')['amount'].std().fillna(0)
    features['amount_std'] = amt_std

    # Activity duration (days)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    first_time = df.groupby('wallet')['time'].min()
    last_time = df.groupby('wallet')['time'].max()
    features['activity_duration_days'] = (last_time - first_time).dt.days

    # Repay frequency proxy
    first_repay = df[df['action']=='repay'].groupby('wallet')['time'].min()
    first_borrow = df[df['action']=='borrow'].groupby('wallet')['time'].min()
    repay_delay = (first_repay - first_borrow).dt.days.fillna(9999)
    features['repay_delay_days'] = repay_delay

    # Fill NaNs
    features = features.fillna(0)

    return features
