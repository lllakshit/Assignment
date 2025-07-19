import argparse
import pandas as pd
import json
from feature_engineering import build_features
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import matplotlib.pyplot as plt

def train_model(X, y):
    from sklearn.model_selection import train_test_split, cross_val_score
    import numpy as np

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    scores = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc')
    print(f'ROC AUC scores (5-fold CV): {scores}')
    print(f'Mean ROC AUC: {np.mean(scores)}')
    return model

def run(args):
    print(f'Loading data from {args.input}')
    with open(args.input, 'r') as f:
        data_json = json.load(f)
    df = pd.DataFrame(data_json)

    print('Building features...')
    features_df = build_features(df)

    # Simple heuristic labels: risky if any liquidationcall
    if 'liquidationcall' in df['action'].values:
        label_data = df.groupby('wallet').apply(lambda x: 1 if 'liquidationcall' in x['action'].values else 0)
        labels = label_data.reindex(features_df.index).fillna(0).astype(int)
    else:
        labels = pd.Series(0, index=features_df.index)

    print('Training model...')
    model = train_model(features_df, labels)

    print('Predicting risk probabilities...')
    probs = model.predict_proba(features_df)[:, 1]
    scores = (1 - probs) * 1000  # Higher score = safer
    features_df['credit_score'] = scores.astype(int)
    features_df['risk_probability'] = probs

    output_df = features_df[['credit_score', 'risk_probability']]
    output_df.index.name = 'wallet'

    print(f'Saving scores to {args.output}')
    output_df.to_csv(args.output)

    print('Plotting score distribution...')
    plt.hist(features_df['credit_score'], bins=50, color='c')
    plt.title('Distribution of Wallet Credit Scores')
    plt.xlabel('Credit Score (0-1000)')
    plt.ylabel('Number of Wallets')
    plt.savefig('score_distribution.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score Aave V2 wallets based on transaction data')
    parser.add_argument('--input', required=True, help='Input JSON file with transaction data')
    parser.add_argument('--output', required=True, help='Output CSV file with wallet scores')
    args = parser.parse_args()
    run(args)
