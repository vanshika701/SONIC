"""
SPP Benchmark: Spectral Path-Product vs. baseline methods on real-world datasets.

Runs SPP, Degree, HITS-Authority, HITS-Hub, Acquaintance, SourceOnly, and
Random on HIV, Gnutella, and Reddit datasets.

No training required — all baselines are pure graph algorithms.

Run:
    python benchmark_datasets.py
"""

import sys
import time
import random
import numpy as np
import networkx as nx

sys.path.insert(0, ".")

from algorithms.spp import spectral_radius, spp_selection
from algorithms.measures import compute_katz_out, spp_score
from algorithms.source_inference import rumor_centrality
from algorithms.eppr import source_risk
from data.loaders import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

def simulate_epidemic(G, seed=42):
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


from experiments.baselines import (
    random_immunization, degree_immunization,
    hits_authority_immunization, hits_hub_immunization,
    acquaintance_immunization, source_only,
)


def _eval_delta(G, L):
    rho_b = spectral_radius(G)
    G2 = G.copy()
    G2.remove_nodes_from(L)
    return rho_b - spectral_radius(G2)


def run_method(name, G, Gn, tau, k):
    """Run one immunisation method and return (nodes_removed, delta_rho, runtime_s)."""
    t0 = time.time()

    if name == "SPP (ours)":
        L, delta = spp_selection(G, k, tau, return_delta_rho=True, verbose=False)

    elif name == "Degree":
        L = degree_immunization(G, k)
        delta = _eval_delta(G, L)

    elif name == "HITS-Authority":
        L = hits_authority_immunization(G, k)
        delta = _eval_delta(G, L)

    elif name == "HITS-Hub":
        L = hits_hub_immunization(G, k)
        delta = _eval_delta(G, L)

    elif name == "Acquaintance":
        L = acquaintance_immunization(G, k)
        delta = _eval_delta(G, L)

    elif name == "SourceOnly":
        L = sorted(tau, key=tau.get, reverse=True)[:k]
        delta = _eval_delta(G, L)

    elif name == "Random":
        L = random_immunization(G, k)
        delta = _eval_delta(G, L)

    else:
        raise ValueError(name)

    elapsed = time.time() - t0
    return L, delta, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

METHODS = ["SPP (ours)", "Degree", "HITS-Authority", "HITS-Hub", "Acquaintance", "SourceOnly", "Random"]

# Define graphs and their target k-budget (roughly scalable with |V|)
DATASETS = {
    "hiv": {"name": "hiv", "k": 100},
    "gnutella": {"name": "gnutella", "k": 200},
    "reddit": {"name": "reddit", "k": 300},
}


def benchmark_dataset(dataset_key):
    info = DATASETS[dataset_key]
    dataset_name = info["name"]
    k = info["k"]
    
    print(f"\n{'─'*60}")
    print(f"  Dataset : {dataset_name.upper()}")
    
    G = load_dataset(dataset_name)
    if G is None:
        print("  [Failed to load dataset]")
        return {}, 0.0

    print(f"  |V|={G.number_of_nodes()}  |E|={G.number_of_edges()}  k={k}")
    rho0 = spectral_radius(G)
    print(f"  ρ₀   = {rho0:.4f}")

    # Build epidemic subgraph and tau
    Gn, _ = simulate_epidemic(G, seed=42)
    pi = rumor_centrality(Gn)
    tau = source_risk(G, pi, K=10, alpha=0.15)

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
    print(f"\n{'='*72}")
    print("  FINAL COMPARISON TABLE  —  Real-world Datasets (Δρ)")
    print(f"{'='*72}")

    dataset_keys = list(all_results.keys())
    col_w = 22

    header = f"  {'Method':<{col_w}}"
    for ds in dataset_keys:
        name_k = f"{ds.upper()} (k={DATASETS[ds]['k']})"
        header += f"  {name_k:>16}"
    print(header)
    print("  " + "─" * (col_w + 18 * len(dataset_keys)))

    for method in METHODS:
        tag = "★" if method == "SPP (ours)" else " "
        row = f"  {tag}{method:<{col_w - 1}}"
        for ds in dataset_keys:
            delta = all_results[ds]["results"].get(method, {}).get("delta", float("nan"))
            if np.isnan(delta):
                row += f"  {'N/A':>16}"
            else:
                row += f"  {delta:>16.4f}"
        print(row)

    print(f"\n  {'SPP improvement over baselines (avg Δρ)':}")
    spp_deltas = [all_results[ds]["results"].get("SPP (ours)", {}).get("delta", 0.0)
                  for ds in dataset_keys]

    for method in METHODS[1:]:
        other_deltas = [all_results[ds]["results"].get(method, {}).get("delta", 0.0)
                        for ds in dataset_keys]
        improvements = [s - o for s, o in zip(spp_deltas, other_deltas)
                        if not (np.isnan(s) or np.isnan(o))]
        if improvements:
            avg_imp = np.mean(improvements)
            pct = np.mean([(s - o) / max(abs(o), 1e-9) * 100
                           for s, o in zip(spp_deltas, other_deltas)
                           if not (np.isnan(s) or np.isnan(o))])
            sign = "+" if avg_imp >= 0 else ""
            print(f"    vs {method:<22}  {sign}{avg_imp:.4f} Δρ  ({sign}{pct:.1f}%)")


if __name__ == "__main__":
    print("\n" + "="*72)
    print("  Spectral Path-Product (SPP) vs. Baselines — Real-world Datasets")
    print("="*72)

    all_results = {}
    for ds in ["hiv", "gnutella", "reddit"]:
        res, rho0 = benchmark_dataset(ds)
        if res:
            all_results[ds] = {"results": res, "rho0": rho0}

    print_comparison_table(all_results)
