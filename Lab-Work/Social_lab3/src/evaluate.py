#!/usr/bin/env python3
"""
evaluate.py
Evaluates a trained model on new features (after attack).
"""

import argparse
import pickle
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="results/model_baseline.pkl")
    parser.add_argument("--features", required=True)
    args = parser.parse_args()

    # Load model & data
    clf = pickle.load(open(args.model, "rb"))
    df = pd.read_csv(args.features)

    feature_cols = [c for c in df.columns if c not in ["node", "label"]]
    X = df[feature_cols].values
    y = df["label"].values

    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]

    print(classification_report(y, y_pred))
    print("AUC:", roc_auc_score(y, y_prob))
