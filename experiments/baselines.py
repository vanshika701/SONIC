"""
Baseline immunization strategies for comparison with SONIC / SPP.

Baselines:
1. Random       — random node removal (lower bound)
2. Degree       — remove highest out-degree nodes
3. Katz         — remove highest Katz centrality nodes
4. DINO         — structural-only greedy KSCC (uniform tau, SPP reduces to degree proxy)
5. SourceOnly   — pure source-risk selection (tau-ranked, skip KSCC)
6. Betweenness  — remove highest betweenness centrality nodes
"""

import numpy as np
import networkx as nx
from algorithms.spp import spectral_radius, spp_selection
from algorithms.sonic import sonic


# ---------------------------------------------------------------------------
# Simple centrality baselines
# ---------------------------------------------------------------------------

def random_immunization(G, k, seed=42):
    """Select k nodes uniformly at random."""
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    chosen = rng.choice(nodes, size=min(k, len(nodes)), replace=False)
    return list(chosen)


def degree_immunization(G, k):
    """
    Select top-k nodes by out-degree (directed).
    Ties broken by in-degree descending.
    """
    nodes = sorted(
        G.nodes(),
        key=lambda v: (G.out_degree(v), G.in_degree(v)),
        reverse=True
    )
    return nodes[:k]


def katz_immunization(G, k, beta=0.01):
    """
    Select top-k nodes by Katz centrality.
    beta should be < 1/spectral_radius(G) for convergence.
    Falls back to smaller beta if needed.
    """
    rho = spectral_radius(G)
    if rho > 0 and beta >= 1.0 / rho:
        beta = 0.9 / rho

    try:
        katz = nx.katz_centrality_numpy(G, alpha=beta, normalized=True)
    except Exception:
        # Fall back to degree if Katz fails on disconnected/trivial graphs
        return degree_immunization(G, k)

    nodes = sorted(katz, key=katz.get, reverse=True)
    return nodes[:k]


def betweenness_immunization(G, k, k_approx=None):
    """
    Select top-k nodes by betweenness centrality.
    Uses exact computation; for large graphs pass k_approx to use sampling.
    """
    if k_approx is not None:
        bc = nx.betweenness_centrality(G, k=k_approx, normalized=True)
    else:
        bc = nx.betweenness_centrality(G, normalized=True)
    nodes = sorted(bc, key=bc.get, reverse=True)
    return nodes[:k]


# ---------------------------------------------------------------------------
# SONIC ablation variants
# ---------------------------------------------------------------------------

def dino_only(G, Gn, k, **kwargs):
    """
    DINO structural baseline: greedy KSCC immunization with uniform tau.
    Uniform source_risk removes the source-risk signal, leaving a
    Katz-weighted structural selector — closest equivalent to the
    original DINO heuristic within the SPP framework.
    """
    # Uniform tau → SPP reduces to pure Katz / structural order
    uniform_tau = {v: 1.0 for v in G.nodes()}
    result = spp_selection(G, k, source_risk=uniform_tau,
                           return_delta_rho=False, **kwargs)
    return result


def source_only(G, Gn, k, **kwargs):
    """SourceOnly: select top-k nodes purely by E-PPR SourceRisk (tau)."""
    from algorithms.source_inference import infer_source_posterior
    from algorithms.eppr import source_risk as compute_sr
    pi = infer_source_posterior(Gn=Gn, method="rumor", G=G)
    tau = compute_sr(G, pi, K=10, alpha=0.15)
    sorted_nodes = sorted(tau, key=tau.get, reverse=True)
    return sorted_nodes[:k]


# ---------------------------------------------------------------------------
# Run all baselines for a given graph + budget
# ---------------------------------------------------------------------------

def run_all_baselines(G, Gn, k, seed=42, verbose=True, run_betweenness=False):
    """
    Run all baseline immunization strategies.

    Parameters
    ----------
    G    : nx.DiGraph, full network
    Gn   : nx.DiGraph, observed epidemic subgraph
    k    : int, immunization budget
    seed : int
    verbose : bool

    Returns
    -------
    results : dict {method_name: list_of_immunized_nodes}
    """
    results = {}

    methods = {
        "Random":       lambda: random_immunization(G, k, seed=seed),
        "Degree":       lambda: degree_immunization(G, k),
        "Katz":         lambda: katz_immunization(G, k),
        "DINO":         lambda: dino_only(G, Gn, k),
        "SourceOnly":   lambda: source_only(G, Gn, k),
        "SPP":          lambda: sonic(G, Gn, k, source_method="rumor",
                                      return_delta_rho=False),
    }

    if run_betweenness:
        # Expensive for large graphs
        n = G.number_of_nodes()
        k_approx = min(200, n) if n > 1000 else None
        methods["Betweenness"] = lambda: betweenness_immunization(G, k, k_approx=k_approx)

    for name, fn in methods.items():
        if verbose:
            print(f"  Running baseline: {name} (k={k})...", end=" ", flush=True)
        immunized = fn()
        results[name] = immunized
        if verbose:
            print(f"done ({len(immunized)} nodes)")

    return results


# ---------------------------------------------------------------------------
# Convenience: evaluate baselines and return metric dicts
# ---------------------------------------------------------------------------

def evaluate_baselines(G, Gn, k, seed=42, run_sis=True, verbose=True,
                       run_betweenness=False):
    """
    Run baselines and compute delta_rho (+ SIS if run_sis=True).

    Returns
    -------
    metrics_list : list of metric dicts (one per method)
    """
    from evaluation.metrics import evaluate_method

    baseline_nodes = run_all_baselines(
        G, Gn, k, seed=seed, verbose=verbose,
        run_betweenness=run_betweenness
    )

    metrics_list = []
    for name, immunized in baseline_nodes.items():
        m = evaluate_method(
            G, immunized,
            method_name=name,
            run_sis=run_sis,
            verbose=verbose,
        )
        metrics_list.append(m)

    return metrics_list
