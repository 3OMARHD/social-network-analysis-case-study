#!/usr/bin/env python3
"""
attacks.py
Implements Structural Evasion and Graph Poisoning attacks.
"""

import argparse
import pickle
import networkx as nx
import numpy as np


def structural_evasion(G, bot_nodes, budget=3):
    """Add edges from bots to high-degree nodes."""
    print("[+] Running Structural Evasion Attack...")

    deg_sorted = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)

    for bot in bot_nodes:
        added = 0
        for target in deg_sorted:
            if added >= budget:
                break
            if bot != target and not G.has_edge(bot, target):
                G.add_edge(bot, target)
                added += 1
    return G


def poisoning_attack(G, num_inject=100, deg=8, start_id=1_000_000):
    """Inject attacker-controlled nodes into the graph."""
    print("[+] Running Graph Poisoning Attack...")

    for i in range(num_inject):
        v = start_id + i
        G.add_node(v)

        targets = np.random.choice(list(G.nodes()), deg, replace=False)
        for t in targets:
            if v != t:
                G.add_edge(v, t)
    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="results/graph_baseline.pkl")
    parser.add_argument("--out", default="results/graph_attacked.pkl")
    parser.add_argument("--bots_file", default="results/bots.txt")
    parser.add_argument("--attack", choices=["evasion", "poison"], required=True)
    parser.add_argument("--budget", type=int, default=3)
    args = parser.parse_args()

    # Load graph
    with open(args.graph, "rb") as f:
        G = pickle.load(f)

    # Load bots
    with open(args.bots_file) as f:
        bot_nodes = [int(x.strip()) for x in f.readlines()]

    if args.attack == "evasion":
        G = structural_evasion(G, bot_nodes, args.budget)

    elif args.attack == "poison":
        G = poisoning_attack(G)

    with open(args.out, "wb") as f:
        pickle.dump(G, f)

    print(f"[+] Attacked graph saved to {args.out}")
