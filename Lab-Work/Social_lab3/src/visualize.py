#!/usr/bin/env python3
"""
visualize.py
Simple visualization utilities for drawing small subgraphs.
"""

import argparse
import pickle
import networkx as nx
import matplotlib.pyplot as plt


def draw_ego(G, node, bots, out):
    ego = nx.ego_graph(G, node, radius=2)
    pos = nx.spring_layout(ego)

    colors = ["red" if n in bots else "skyblue" for n in ego.nodes()]

    plt.figure(figsize=(10, 8))
    nx.draw(ego, pos, node_color=colors, node_size=80, edge_color="gray")
    plt.title(f"Ego Network of Node {node}")
    plt.savefig(out)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="results/graph_baseline.pkl")
    parser.add_argument("--bots", default="results/bots.txt")
    parser.add_argument("--node", type=int, required=True)
    parser.add_argument("--out", default="results/ego.png")
    args = parser.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    bots = set(int(x.strip()) for x in open(args.bots))

    draw_ego(G, args.node, bots, args.out)
    print(f"[+] Saved ego network to {args.out}")
