#!/usr/bin/env python3
"""
train.py
Trains a baseline bot detection classifier using graph-based features.
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import numpy as np


def create_labels(df, ratio=0.05):
    """Simulate 5% bots randomly"""
    nodes = df["node"].tolist()
    np.random.seed(42)
    num_bots = int(len(nodes) * ratio)
    bots = set(np.random.choice(nodes, num_bots, replace=False))

    df["label"] = df["node"].apply(lambda x: 1 if x in bots else 0)
    return df, bots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="results/features_baseline.csv")
    parser.add_argument("--model_out", default="results/model_baseline.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    df, bots = create_labels(df)

    feature_cols = [c for c in df.columns if c not in ["node", "label"]]

    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))

    with open(args.model_out, "wb") as f:
        pickle.dump(clf, f)

    print(f"[+] Model saved to {args.model_out}")
