import argparse
import pickle
import networkx as nx
import pandas as pd
import numpy as np

def compute_anomaly_scores(G):
    print("[+] Computing node anomaly scores...")

    degree = dict(G.degree())
    clustering = nx.clustering(G)
    betweenness = nx.betweenness_centrality(G, k=500, seed=42)  # approximate for speed

    df = pd.DataFrame({
        "node": list(G.nodes()),
        "degree": [degree[n] for n in G.nodes()],
        "clustering": [clustering[n] for n in G.nodes()],
        "betweenness": [betweenness[n] for n in G.nodes()]
    })

    # Normalize
    for col in ["degree", "clustering", "betweenness"]:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)

    # High degree or high betweenness or low clustering => anomaly
    df["anomaly_score"] = (
        np.abs(df["degree"]) * 0.4 +
        np.abs(df["betweenness"]) * 0.4 +
        (1 - df["clustering"]) * 0.2
    )

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bot_rate", type=float, default=0.03)  # 3% bots
    args = parser.parse_args()

    with open(args.graph, "rb") as f:
        G = pickle.load(f)

    df = compute_anomaly_scores(G)

    # Pick top X% as bots
    cutoff = df["anomaly_score"].quantile(1 - args.bot_rate)
    df["label"] = (df["anomaly_score"] >= cutoff).astype(int)

    print(f"[+] Generated labels: {df['label'].sum()} bots out of {len(df)} nodes")

    df[["node", "label"]].to_csv(args.out, index=False)
    print(f"[+] Saved labels to {args.out}")

if __name__ == "__main__":
    main()
