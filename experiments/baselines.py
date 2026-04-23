"""
Baseline immunization strategies for comparison with SONIC / SPP.

Baselines:
1. Random               — random node removal (lower bound)
2. Degree               — remove highest out-degree nodes
3. HITS (Authority)     — remove highest HITS authority-score nodes
                          (Kleinberg 1999; no training required)
4. HITS (Hub)           — remove highest HITS hub-score nodes
5. Acquaintance         — immunize neighbors of random nodes
                          (friendship-paradox exploit; no training required)
6. SourceOnly           — pure source-risk selection (τ-ranked, skip KSCC)
7. Betweenness          — remove highest betweenness centrality nodes
                          (expensive; disabled by default on large graphs)
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


# ---------------------------------------------------------------------------
# HITS baselines  (Kleinberg 1999 — no training required)
# ---------------------------------------------------------------------------

def hits_authority_immunization(G, k, max_iter=1000, tol=1e-6):
    """
    Select top-k nodes by HITS *authority* score.

    In epidemic containment, authority nodes receive many links from
    high-hub nodes — they are the "targets" that spreading hubs point
    to, making them critical bridge points for propagation.

    Uses NetworkX power-iteration HITS.  O(|E| · iterations).
    Falls back to in-degree if HITS fails to converge.

    Parameters
    ----------
    G        : nx.DiGraph
    k        : int
    max_iter : int
    tol      : float

    Returns
    -------
    list of k node ids
    """
    try:
        _, authorities = nx.hits(G, max_iter=max_iter, tol=tol, normalized=True)
        nodes = sorted(authorities, key=authorities.get, reverse=True)
        return nodes[:k]
    except nx.PowerIterationFailedConvergence:
        # Fallback: in-degree as authority proxy
        nodes = sorted(G.nodes(), key=lambda v: G.in_degree(v), reverse=True)
        return nodes[:k]


def hits_hub_immunization(G, k, max_iter=1000, tol=1e-6):
    """
    Select top-k nodes by HITS *hub* score.

    Hub nodes point to many high-authority nodes — they are the primary
    spreaders in a directed epidemic network.  Removing them directly
    disrupts multi-hop transmission chains.

    Falls back to out-degree if HITS fails to converge.

    Parameters
    ----------
    G        : nx.DiGraph
    k        : int
    max_iter : int
    tol      : float

    Returns
    -------
    list of k node ids
    """
    try:
        hubs, _ = nx.hits(G, max_iter=max_iter, tol=tol, normalized=True)
        nodes = sorted(hubs, key=hubs.get, reverse=True)
        return nodes[:k]
    except nx.PowerIterationFailedConvergence:
        # Fallback: out-degree as hub proxy
        nodes = sorted(G.nodes(), key=lambda v: G.out_degree(v), reverse=True)
        return nodes[:k]


# ---------------------------------------------------------------------------
# Acquaintance Immunization  (Cohen et al. 2003 — no training required)
# ---------------------------------------------------------------------------

def acquaintance_immunization(G, k, seed=42):
    """
    Acquaintance Immunization (Cohen, Havlin & ben-Avraham, PRL 2003).

    Exploits the *friendship paradox*: a random neighbor of a random node
    is statistically more connected than the random node itself.

    Algorithm
    ---------
    1. Sample k random "probe" nodes (with replacement allowed).
    2. For each probe, follow one random *outgoing* edge (or incoming if
       no out-edges) to land on a neighbor.
    3. Collect the k unique neighbor-targets as the immunization set.
       If fewer than k unique targets are found, fill with degree fallback.

    Complexity: O(k + |degree lookups|) — effectively O(k).
    No model training required.

    Parameters
    ----------
    G    : nx.DiGraph
    k    : int
    seed : int

    Returns
    -------
    list of k node ids
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n = len(nodes)

    immunized = set()
    max_attempts = k * 10  # prevent infinite loop on sparse graphs
    attempts = 0

    while len(immunized) < k and attempts < max_attempts:
        attempts += 1
        # Step 1: pick a random probe node
        probe = nodes[rng.integers(0, n)]

        # Step 2: follow a random outgoing edge (friendship-paradox exploit)
        out_nbrs = list(G.successors(probe))
        if out_nbrs:
            target = out_nbrs[rng.integers(0, len(out_nbrs))]
        else:
            # No out-edges: try incoming neighbors
            in_nbrs = list(G.predecessors(probe))
            if in_nbrs:
                target = in_nbrs[rng.integers(0, len(in_nbrs))]
            else:
                continue  # isolated node — skip

        if target not in immunized:
            immunized.add(target)

    # Fill up to k if not enough unique targets found (sparse graph edge case)
    if len(immunized) < k:
        fallback = degree_immunization(G, k)
        for v in fallback:
            if len(immunized) >= k:
                break
            immunized.add(v)

    return list(immunized)[:k]


# ---------------------------------------------------------------------------
# SONIC ablation variants
# ---------------------------------------------------------------------------

def source_only(G, Gn, k, **kwargs):
    """SourceOnly: select top-k nodes purely by E-PPR SourceRisk (tau)."""
    from algorithms.source_inference import infer_source_posterior
    from algorithms.eppr import source_risk as compute_sr
    pi = infer_source_posterior(Gn=Gn, method="rumor", G=G)
    tau = compute_sr(G, pi, K=10, alpha=0.15)
    sorted_nodes = sorted(tau, key=tau.get, reverse=True)
    return sorted_nodes[:k]


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
        "Random":           lambda: random_immunization(G, k, seed=seed),
        "Degree":           lambda: degree_immunization(G, k),
        "HITS-Authority":   lambda: hits_authority_immunization(G, k),
        "HITS-Hub":         lambda: hits_hub_immunization(G, k),
        "Acquaintance":     lambda: acquaintance_immunization(G, k, seed=seed),
        "SourceOnly":       lambda: source_only(G, Gn, k),
        "SPP":              lambda: sonic(G, Gn, k, source_method="rumor",
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
