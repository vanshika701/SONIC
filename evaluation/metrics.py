"""
Evaluation metrics for SONIC experiments.

Metrics from SONIC paper Section VII:
1. Δρ  — Spectral Radius Decrease (primary metric, from DINO paper)
2. I_T — Total Infected at Time T (SIS simulation)
3. T_contain — Time to Containment (first t where infected < 1%)
4. Top-k Source Detection Accuracy
5. SRA — SourceRisk Alignment Score (cosine similarity, novel metric)
"""

import numpy as np
import networkx as nx
from algorithms.dino import spectral_radius


# ---------------------------------------------------------------------------
# 1. Spectral Radius Decrease
# ---------------------------------------------------------------------------

def delta_rho(G, immunized_nodes):
    """
    delta_rho = rho(G) - rho(G minus L)

    Parameters
    ----------
    G               : nx.DiGraph (original)
    immunized_nodes : list of nodes

    Returns
    -------
    delta : float
    rho_before : float
    rho_after  : float
    """
    rho_before = spectral_radius(G)
    G_after = G.copy()
    G_after.remove_nodes_from([v for v in immunized_nodes if v in G_after])
    rho_after = spectral_radius(G_after)
    return rho_before - rho_after, rho_before, rho_after


# ---------------------------------------------------------------------------
# 2 & 3. I_T and T_contain (from SIS simulation)
# ---------------------------------------------------------------------------

def sis_metrics(G, immunized_nodes, beta=0.03, delta=0.1, I0=0.95, T=200, n_trials=20):
    """
    Compute I_T and T_contain via SIS simulation.
    Wrapper around simulation.sis.simulate_sis.
    """
    from simulation.sis import simulate_sis
    curve, I_T, T_contain = simulate_sis(
        G, immunized_nodes=immunized_nodes,
        beta=beta, delta=delta, I0=I0, T=T, n_trials=n_trials,
    )
    return {
        "curve": curve,
        "I_T": I_T,
        "T_contain": T_contain if T_contain is not None else T,
    }


# ---------------------------------------------------------------------------
# 4. Top-k Source Detection Accuracy
# ---------------------------------------------------------------------------

def topk_source_accuracy(pi, true_source, k=1):
    """
    Returns 1.0 if true_source is among top-k nodes by π(v), else 0.0.

    Parameters
    ----------
    pi          : dict {node: float}
    true_source : node id
    k           : int
    """
    if true_source not in pi:
        return 0.0
    sorted_nodes = sorted(pi, key=pi.get, reverse=True)
    return 1.0 if true_source in sorted_nodes[:k] else 0.0


def batch_topk_accuracy(pi_list, true_sources, k=1):
    """Average top-k accuracy over multiple trials."""
    accs = [topk_source_accuracy(pi, src, k) for pi, src in zip(pi_list, true_sources)]
    return float(np.mean(accs))


# ---------------------------------------------------------------------------
# 5. SourceRisk Alignment Score (SRA) — SONIC paper Eq. 12
# ---------------------------------------------------------------------------

def source_risk_alignment(source_risk_vector, infection_order):
    """
    SRA = (r · g) / (||r|| ||g||)

    Cosine similarity between:
      r : SourceRisk vector (dict {node: float})
      g : ground-truth infection-order vector

    g[v] is set to 1/(rank of v in infection order) — earlier = higher value.
    Nodes not in infection_order get g[v] = 0.

    Parameters
    ----------
    source_risk_vector : dict {node: float}
    infection_order    : list of nodes in order they were infected

    Returns
    -------
    sra : float in [-1, 1], higher = better alignment
    """
    nodes = list(source_risk_vector.keys())
    if not nodes or not infection_order:
        return 0.0

    rank_map = {v: 1.0 / (i + 1) for i, v in enumerate(infection_order)}

    r = np.array([source_risk_vector.get(v, 0.0) for v in nodes])
    g = np.array([rank_map.get(v, 0.0) for v in nodes])

    norm_r = np.linalg.norm(r)
    norm_g = np.linalg.norm(g)

    if norm_r < 1e-12 or norm_g < 1e-12:
        return 0.0

    return float(np.dot(r, g) / (norm_r * norm_g))


# ---------------------------------------------------------------------------
# Full evaluation suite
# ---------------------------------------------------------------------------

def evaluate_method(
    G,
    immunized_nodes,
    method_name="method",
    run_sis=True,
    source_risk_vec=None,
    infection_order=None,
    pi=None,
    true_source=None,
    budgets=None,
    verbose=True,
):
    """
    Compute all evaluation metrics for a single immunization strategy.

    Returns
    -------
    metrics : dict with keys: delta_rho, rho_before, rho_after,
              I_T (if run_sis), T_contain (if run_sis),
              SRA (if source_risk_vec and infection_order provided),
              top1_acc, top5_acc, top10_acc (if pi and true_source provided)
    """
    metrics = {"method": method_name, "k": len(immunized_nodes)}

    # Δρ
    dr, rho_b, rho_a = delta_rho(G, immunized_nodes)
    metrics["delta_rho"] = dr
    metrics["rho_before"] = rho_b
    metrics["rho_after"] = rho_a

    # SIS metrics
    if run_sis:
        sim = sis_metrics(G, immunized_nodes)
        metrics["I_T"] = sim["I_T"]
        metrics["T_contain"] = sim["T_contain"]

    # SRA
    if source_risk_vec is not None and infection_order is not None:
        metrics["SRA"] = source_risk_alignment(source_risk_vec, infection_order)

    # Source detection accuracy
    if pi is not None and true_source is not None:
        for k in [1, 5, 10]:
            metrics[f"top{k}_acc"] = topk_source_accuracy(pi, true_source, k)

    if verbose:
        print(f"\n[{method_name}] k={len(immunized_nodes)}")
        print(f"  Δρ={metrics['delta_rho']:.4f}  "
              f"(ρ: {metrics['rho_before']:.4f} → {metrics['rho_after']:.4f})")
        if "I_T" in metrics:
            print(f"  I_T={metrics['I_T']:.1f}  T_contain={metrics['T_contain']:.1f}")
        if "SRA" in metrics:
            print(f"  SRA={metrics['SRA']:.4f}")

    return metrics


def print_results_table(all_metrics, budgets=None):
    """Print a formatted comparison table like DINO paper Table 2."""
    if not all_metrics:
        return

    methods = list({m["method"] for m in all_metrics})
    ks = sorted(list({m["k"] for m in all_metrics}))

    print(f"\n{'Method':<20}", end="")
    for k in ks:
        print(f"  Δρ(k={k})", end="")
    print()
    print("-" * (20 + 12 * len(ks)))

    for method in methods:
        print(f"{method:<20}", end="")
        for k in ks:
            matches = [m for m in all_metrics if m["method"] == method and m["k"] == k]
            if matches:
                print(f"  {matches[0]['delta_rho']:>8.2f}", end="")
            else:
                print(f"  {'N/A':>8}", end="")
        print()
