"""
Phase 2 of SONIC: Expected Personalized PageRank (E-PPR).

SONIC Equations:
  Eq. 4: π_s = α·e_s + (1-α)·D^{-1}·A·π_s   (PPR seeded at source s)
  Eq. 5: π_s = α·(I - (1-α)·D^{-1}·A)^{-1}·e_s  (closed form)
  Eq. 6: SourceRisk(v) = Σ_{s in Gn} π(s) · π_s(v)

Computation: power iteration over top-K sources by π weight.
Complexity: O(K · |E| / α)
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Personalized PageRank via power iteration
# ---------------------------------------------------------------------------

def personalized_pagerank(G, source, alpha=0.15, tol=1e-6, max_iter=200):
    """
    Compute Personalized PageRank seeded at source node using power iteration.

    π_s = α·e_s + (1-α)·D^{-1}·A·π_s

    For directed graphs, D^{-1}A is the row-normalized adjacency (transition matrix):
    information flows ALONG directed edges (from u to v if edge u→v exists).
    A node with no out-edges uses a uniform teleport.

    Parameters
    ----------
    G      : nx.DiGraph
    source : node id
    alpha  : float, teleport probability (0.15 as in PageRank literature)
    tol    : float, convergence tolerance
    max_iter : int

    Returns
    -------
    pi_s : dict {node: float}  (sums to 1)
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}

    node_idx = {v: i for i, v in enumerate(nodes)}
    src_idx = node_idx.get(source)
    if src_idx is None:
        return {v: 0.0 for v in nodes}

    # Build sparse row-normalized adjacency D^{-1}·A
    # Edge u→v: u can spread to v (column v in row u)
    out_deg = dict(G.out_degree())

    rows, cols, data = [], [], []
    for u, v in G.edges():
        u_idx = node_idx[u]
        v_idx = node_idx[v]
        d_out_u = out_deg.get(u, 0)
        if d_out_u > 0:
            rows.append(u_idx)
            cols.append(v_idx)
            data.append(1.0 / d_out_u)

    if rows:
        D_inv_A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    else:
        D_inv_A = sp.csr_matrix((n, n))

    # Teleport vector e_s
    e_s = np.zeros(n)
    e_s[src_idx] = 1.0

    # Power iteration
    pi = e_s.copy()
    for _ in range(max_iter):
        pi_new = alpha * e_s + (1 - alpha) * D_inv_A.T.dot(pi)
        # Note: PPR propagates ALONG directed edges.
        # If edge u→v, then v receives mass from u.
        # Matrix form: π_new[v] += (1-α) * Σ_{u: u→v} π[u] / out_deg[u]
        # This is D_inv_A.T · π (column: v, row: u means u→v)

        if np.linalg.norm(pi_new - pi, 1) < tol:
            pi = pi_new
            break
        pi = pi_new

    pi = np.maximum(pi, 0)
    total = pi.sum()
    if total > 0:
        pi /= total

    return {v: float(pi[node_idx[v]]) for v in nodes}


# ---------------------------------------------------------------------------
# SourceRisk via Expected PPR
# ---------------------------------------------------------------------------

def source_risk(G, pi_posterior, K=10, alpha=0.15, tol=1e-6):
    """
    Compute SourceRisk(v) for all nodes in G.

    SourceRisk(v) = Σ_{s in top-K} π(s) · π_s(v)

    Answers: "Given where the epidemic likely started, what fraction of
    spreading probability flows through node v?"

    Parameters
    ----------
    G            : nx.DiGraph
    pi_posterior : dict {node: float}  (source posterior from Phase 1)
    K            : int, number of top sources to use
    alpha        : float, PPR teleport probability

    Returns
    -------
    sr : dict {node: float}  (SourceRisk score, unnormalized)
    """
    nodes = list(G.nodes())
    sr = {v: 0.0 for v in nodes}

    if not pi_posterior:
        return sr

    # Select top-K sources by posterior weight
    sorted_sources = sorted(pi_posterior.items(), key=lambda x: x[1], reverse=True)
    top_K = sorted_sources[:K]

    for s, pi_s_weight in top_K:
        if s not in G or pi_s_weight <= 0:
            continue

        # Compute PPR seeded at s
        pi_s = personalized_pagerank(G, s, alpha=alpha, tol=tol)

        # Accumulate weighted contribution
        for v in nodes:
            sr[v] += pi_s_weight * pi_s.get(v, 0.0)

    return sr


# ---------------------------------------------------------------------------
# Novelty: Entropy-gated weight selection
# (Suggest auto-tuning αw/βw based on source uncertainty)
# ---------------------------------------------------------------------------

def source_entropy(pi):
    """
    Compute entropy H(π) of source posterior.
    High entropy = high uncertainty about source = should weight structural term more.
    Low entropy = confident source detection = should weight SourceRisk more.

    Returns float in [0, log(n)].
    """
    vals = np.array(list(pi.values()))
    vals = vals[vals > 0]
    return float(-np.sum(vals * np.log(vals)))


def entropy_gated_weights(pi, alpha_w_base=0.5, beta_w_base=0.5):
    """
    Novelty: Auto-tune αw/βw based on source posterior entropy.

    When entropy is high (uncertain source): increase αw (trust structure more).
    When entropy is low (confident source): increase βw (trust SourceRisk more).

    Returns (alpha_w, beta_w) that sum to 1.
    """
    n = len(pi)
    if n <= 1:
        return alpha_w_base, beta_w_base

    H = source_entropy(pi)
    H_max = np.log(n)  # max entropy = uniform distribution

    if H_max == 0:
        return alpha_w_base, beta_w_base

    # Confidence = 1 - normalized entropy
    confidence = 1.0 - (H / H_max)  # 0=uncertain, 1=certain

    # Smoothly interpolate: high confidence → more SourceRisk weight
    beta_w = beta_w_base * (0.5 + 0.5 * confidence)
    alpha_w = 1.0 - beta_w

    return float(alpha_w), float(beta_w)
