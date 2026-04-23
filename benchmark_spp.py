"""
SPP Benchmark: Spectral Path-Product vs. baseline methods.

Generates multiple synthetic directed graphs (Barabasi-Albert, Scale-Free,
Erdos-Renyi) at different sizes, runs all methods, and prints a comparison
table ranked by Δρ (spectral radius decrease).

Run:
    python benchmark_spp.py
"""

import sys
import time
import random
import numpy as np
import networkx as nx

sys.path.insert(0, ".")

from algorithms.spp import spectral_radius, spp_selection, find_nontrivial_sccs
from algorithms.sonic import sonic
from algorithms.measures import compute_katz_out, spp_score
from algorithms.source_inference import rumor_centrality
from algorithms.eppr import source_risk


# ─────────────────────────────────────────────────────────────────────────────
# Graph generators
# ─────────────────────────────────────────────────────────────────────────────

def make_ba_directed(n, m=3, seed=42):
    """Barabasi-Albert directed graph (preferential attachment)."""
    G_undirected = nx.barabasi_albert_graph(n, m, seed=seed)
    G = nx.DiGraph()
    G.add_nodes_from(G_undirected.nodes())
    for u, v in G_undirected.edges():
        if random.random() < 0.5:
            G.add_edge(u, v)
        else:
            G.add_edge(v, u)
    return G


def make_scale_free(n, seed=42):
    """NetworkX scale_free_graph (directed)."""
    G = nx.scale_free_graph(n, seed=seed)
    return nx.DiGraph(G)


def make_erdos_renyi_directed(n, p=0.04, seed=42):
    """Directed Erdos-Renyi graph."""
    return nx.erdos_renyi_graph(n, p, seed=seed, directed=True)


def simulate_epidemic(G, seed=0):
    """SI spread from the highest in-degree node for ~10 steps."""
    random.seed(seed)
    np.random.seed(seed)
    source = max(G.nodes(), key=lambda v: G.in_degree(v))
    infected = {source}
    order = [source]
    frontier = [source]
    for _ in range(10):
        next_frontier = []
        for u in frontier:
            for v in G.successors(u):
                if v not in infected and random.random() < 0.3:
                    infected.add(v)
                    order.append(v)
                    next_frontier.append(v)
        frontier = next_frontier
        if not frontier:
            break
    Gn = G.subgraph(order).copy()
    if Gn.number_of_nodes() < 2:
        Gn = G
    return Gn, order


# ─────────────────────────────────────────────────────────────────────────────
# Method runners
# ─────────────────────────────────────────────────────────────────────────────

def run_method(name, G, Gn, tau, k):
    """Run one immunisation method and return (nodes_removed, delta_rho, runtime_s)."""
    t0 = time.time()

    if name == "SPP (ours)":
        L, delta = spp_selection(G, k, tau, return_delta_rho=True, verbose=False)

    elif name == "DINO (structural)":
        # Uniform tau → SPP degenerates to pure Katz-structural selection
        uniform_tau = {v: 1.0 for v in G.nodes()}
        L, delta = spp_selection(G, k, uniform_tau, return_delta_rho=True, verbose=False)

    elif name == "Degree":
        nodes = sorted(G.nodes(), key=lambda v: (G.out_degree(v), G.in_degree(v)), reverse=True)
        L = nodes[:k]
        rho_b = spectral_radius(G)
        G2 = G.copy()
        G2.remove_nodes_from(L)
        delta = rho_b - spectral_radius(G2)

    elif name == "Katz":
        rho = spectral_radius(G)
        alpha = min(0.9 / max(rho, 1e-9), 0.01)
        try:
            katz = nx.katz_centrality(G, alpha=alpha, normalized=True,
                                      max_iter=1000, tol=1e-6)
        except nx.PowerIterationFailedConvergence:
            katz = {v: float(G.out_degree(v)) for v in G.nodes()}
        L = sorted(katz, key=katz.get, reverse=True)[:k]
        rho_b = spectral_radius(G)
        G2 = G.copy()
        G2.remove_nodes_from(L)
        delta = rho_b - spectral_radius(G2)

    elif name == "SourceOnly":
        L = sorted(tau, key=tau.get, reverse=True)[:k]
        rho_b = spectral_radius(G)
        G2 = G.copy()
        G2.remove_nodes_from(L)
        delta = rho_b - spectral_radius(G2)

    elif name == "Random":
        rng = np.random.default_rng(99)
        nodes = list(G.nodes())
        L = list(rng.choice(nodes, size=min(k, len(nodes)), replace=False))
        rho_b = spectral_radius(G)
        G2 = G.copy()
        G2.remove_nodes_from(L)
        delta = rho_b - spectral_radius(G2)

    else:
        raise ValueError(name)

    elapsed = time.time() - t0
    return L, delta, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

METHODS = ["SPP (ours)", "DINO (structural)", "Degree", "Katz", "SourceOnly", "Random"]

def benchmark_graph(graph_name, G, k, seed=42):
    """Run all methods on one graph and return results dict."""
    print(f"\n{'─'*60}")
    print(f"  Graph : {graph_name}")
    print(f"  |V|={G.number_of_nodes()}  |E|={G.number_of_edges()}  k={k}")
    rho0 = spectral_radius(G)
    print(f"  ρ₀   = {rho0:.4f}")

    # Build epidemic subgraph and tau
    Gn, _ = simulate_epidemic(G, seed=seed)
    pi = rumor_centrality(Gn)
    tau = source_risk(G, pi, K=min(10, len(pi)), alpha=0.15)

    results = {}
    for method in METHODS:
        try:
            L, delta, t = run_method(method, G, Gn, tau, k)
            results[method] = {"delta": delta, "t": t, "L": L}
            tag = "★" if method == "SPP (ours)" else " "
            print(f"  {tag} {method:<22} Δρ={delta:>7.4f}  ({t:.2f}s)")
        except Exception as e:
            results[method] = {"delta": float("nan"), "t": 0.0, "L": []}
            print(f"    {method:<22} ERROR: {e}")

    return results, rho0


def print_comparison_table(all_results):
    """Print a final ranked summary across all graphs."""
    print(f"\n{'='*72}")
    print("  FINAL COMPARISON TABLE  —  Δρ (higher = better containment)")
    print(f"{'='*72}")

    graph_names = list(all_results.keys())
    col_w = 22

    header = f"  {'Method':<{col_w}}"
    for gn in graph_names:
        header += f"  {gn[:14]:>14}"
    print(header)
    print("  " + "─" * (col_w + 16 * len(graph_names)))

    for method in METHODS:
        tag = "★" if method == "SPP (ours)" else " "
        row = f"  {tag}{method:<{col_w - 1}}"
        for gn in graph_names:
            delta = all_results[gn]["results"].get(method, {}).get("delta", float("nan"))
            if np.isnan(delta):
                row += f"  {'N/A':>14}"
            else:
                row += f"  {delta:>14.4f}"
        print(row)

    # SPP improvement over each baseline
    print(f"\n  {'SPP improvement over baselines (avg Δρ)':}")
    spp_deltas = [all_results[gn]["results"].get("SPP (ours)", {}).get("delta", 0.0)
                  for gn in graph_names]

    for method in METHODS[1:]:
        other_deltas = [all_results[gn]["results"].get(method, {}).get("delta", 0.0)
                        for gn in graph_names]
        improvements = [s - o for s, o in zip(spp_deltas, other_deltas)
                        if not (np.isnan(s) or np.isnan(o))]
        if improvements:
            avg_imp = np.mean(improvements)
            pct = np.mean([(s - o) / max(abs(o), 1e-9) * 100
                           for s, o in zip(spp_deltas, other_deltas)
                           if not (np.isnan(s) or np.isnan(o))])
            sign = "+" if avg_imp >= 0 else ""
            print(f"    vs {method:<22}  {sign}{avg_imp:.4f} Δρ  ({sign}{pct:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(0)

    GRAPHS = [
        ("BA-200",  make_ba_directed(200,  m=3, seed=1),  20),
        ("BA-500",  make_ba_directed(500,  m=3, seed=2),  50),
        ("SF-300",  make_scale_free(300,         seed=3),  30),
        ("SF-500",  make_scale_free(500,         seed=4),  50),
        ("ER-400",  make_erdos_renyi_directed(400, p=0.015, seed=5), 40),
    ]

    print("\n" + "="*72)
    print("  Spectral Path-Product (SPP) vs. Baselines — Benchmark")
    print("="*72)

    all_results = {}
    for graph_name, G, k in GRAPHS:
        res, rho0 = benchmark_graph(graph_name, G, k)
        all_results[graph_name] = {"results": res, "rho0": rho0, "k": k}

    print_comparison_table(all_results)
    print(f"\n{'='*72}")
    print("  ★ = SPP (ours)")
    print("  Metric: Δρ = ρ(G) - ρ(G after immunisation)  [higher is better]")
    print("="*72)
