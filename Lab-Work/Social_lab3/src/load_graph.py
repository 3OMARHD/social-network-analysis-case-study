#!/usr/bin/env python3
"""
load_graph.py
Loads the SNAP Facebook combined graph and saves it as a NetworkX graph object.
"""

import argparse
import networkx as nx
import gzip
import pickle


def load_graph(path: str) -> nx.Graph:
    print(f"[+] Loading graph from {path}")
    G = nx.read_edgelist(path, nodetype=int)
    G = G.to_undirected()
    print(f"[+] Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def save_graph(G: nx.Graph, out_path: str):
    print(f"[+] Saving graph to {out_path}")
    with open(out_path, "wb") as f:
        pickle.dump(G, f)
    print("[+] Graph saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/facebook_combined.txt.gz")
    parser.add_argument("--output", default="results/graph_baseline.pkl")
    args = parser.parse_args()

    G = load_graph(args.input)
    save_graph(G, args.output)
