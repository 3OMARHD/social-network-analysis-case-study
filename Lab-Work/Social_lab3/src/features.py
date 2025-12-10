#!/usr/bin/env python3
"""
features.py
Extracts graph-based features for each node and saves to CSV.
"""

import argparse
import pickle
import networkx as nx
import pandas as pd
import numpy as np


def neighbor_stats(G, node):
    neighbors = list(G.neighbors(node))
    degs = [G.degree(n) for n in neighbors] or [0]
    return np.mean(degs), np.std(degs) if len(degs) > 1 else 0.0


def extract_features(G: nx.Graph) -> pd.DataFrame:
    print("[+] Extracting features...")

    deg = dict(G.degree())
    clust = nx.clustering(G)
    pagerank = nx.pagerank(G)
    triangles = nx.triangles(G)

    df = pd.DataFrame({
        "node": list(G.nodes()),
        "degree": [deg[n] for n in G.nodes()],
        "clustering": [clust[n] for n in G.nodes()],
        "pagerank": [pagerank[n] for n in G.nodes()],
        "triangles": [triangles[n] for n in G.nodes()],
    })

    # Neighbor stats
    mean_list, std_list = [], []
    for n in G.nodes():
        m, s = neighbor_stats(G, n)
        mean_list.append(m)
        std_list.append(s)

    df["avg_neighbor_degree"] = mean_list
    df["std_neighbor_degree"] = std_list

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="results/graph_baseline.pkl")
    parser.add_argument("--out", default="results/features_baseline.csv")
    args = parser.parse_args()

    with open(args.graph, "rb") as f:
        G = pickle.load(f)

    df = extract_features(G)
    df.to_csv(args.out, index=False)

    print(f"[+] Features saved to {args.out}")
