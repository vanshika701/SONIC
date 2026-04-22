"""
Phase 1 of SONIC: Source Posterior Inference.

Two implementations:
1. DeepTrace GNN (Tan et al., 2025) — primary method
2. Rumor Centrality (Shah & Zaman, 2012) — fast fallback when GNN unavailable

Both return π(v): a probability distribution over nodes in Gn,
concentrating mass on the most likely epidemic source.

SONIC Eq. 3:  π(v) = exp(h^(L)_v) / Σ_u exp(h^(L)_u)
"""

import numpy as np
import networkx as nx
from collections import deque


# ---------------------------------------------------------------------------
# Rumor Centrality (Shah & Zaman, SIGMETRICS 2012)
# Fast O(N) heuristic; DeepTrace paper uses this as a baseline.
# ---------------------------------------------------------------------------

def rumor_centrality(Gn):
    """
    Compute rumor centrality score for each node in Gn.
    Rumor centrality R(v) is proportional to |Ω(Gn|v)| — the number of
    permitted permutations rooted at v.

    For trees: R(v) = n! / Π_u T_v(u), where T_v(u) is the subtree size
    of u in the tree rooted at v.

    For general graphs: use the undirected spanning tree approximation.

    Returns
    -------
    pi : dict {node: float}  (normalized to sum=1)
    """
    nodes = list(Gn.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: 1.0}

    Gu = Gn.to_undirected()

    log_scores = {}
    for v in nodes:
        # BFS tree rooted at v
        subtree_sizes = _compute_subtree_sizes(Gu, v)
        log_rc = _log_rumor_centrality(subtree_sizes, n)
        log_scores[v] = log_rc

    # Softmax normalization for numerical stability
    scores = np.array([log_scores[v] for v in nodes])
    scores -= scores.max()  # shift for stability
    exp_scores = np.exp(scores)
    exp_scores /= exp_scores.sum()

    return {v: float(exp_scores[i]) for i, v in enumerate(nodes)}


def _compute_subtree_sizes(G, root):
    """
    Compute subtree sizes for BFS tree of G rooted at root.
    Returns dict {node: subtree_size}.
    """
    parent = {root: None}
    order = []  # BFS order
    queue = deque([root])
    visited = {root}

    while queue:
        v = queue.popleft()
        order.append(v)
        for u in G.neighbors(v):
            if u not in visited:
                visited.add(u)
                parent[u] = v
                queue.append(u)

    # Compute subtree sizes bottom-up
    subtree_size = {v: 1 for v in G.nodes()}
    for v in reversed(order):
        p = parent.get(v)
        if p is not None:
            subtree_size[p] += subtree_size[v]

    return subtree_size


def _log_rumor_centrality(subtree_sizes, n):
    """
    log R(v) = log(n!) - Σ_u log(T_v(u))
    """
    log_n_fact = sum(np.log(i) for i in range(1, n + 1))
    log_denom = sum(np.log(max(s, 1)) for s in subtree_sizes.values())
    return log_n_fact - log_denom


# ---------------------------------------------------------------------------
# DeepTrace-based source posterior
# ---------------------------------------------------------------------------

def deeptrace_posterior(Gn, model, G=None, node_features=None):
    """
    Compute source posterior π(v) using trained DeepTrace GNN.

    Parameters
    ----------
    Gn            : nx.DiGraph (observed epidemic subgraph)
    model         : DeepTraceGNN instance
    G             : nx.DiGraph (full network — used for correct r_hat computation)
    node_features : dict {node: np.array} or None (auto-computed)

    Returns
    -------
    pi : dict {node: float}
    """
    if node_features is None:
        from data.synthetic import compute_deeptrace_features
        node_features = compute_deeptrace_features(Gn, G=G)

    return model.predict_source_posterior(Gn, node_features)


# ---------------------------------------------------------------------------
# Unified source inference interface
# ---------------------------------------------------------------------------

def infer_source_posterior(Gn, model=None, method="auto", G=None):
    """
    Infer source posterior π(v) over nodes in Gn.

    Parameters
    ----------
    Gn     : nx.DiGraph (epidemic subgraph)
    model  : DeepTraceGNN or None
    method : str
        'deeptrace' — use GNN (requires trained model)
        'rumor'     — use Rumor Centrality (always available)
        'auto'      — use DeepTrace if model available, else Rumor Centrality
    G      : nx.DiGraph (full network, used for correct r_hat in DeepTrace features)

    Returns
    -------
    pi : dict {node: float}  (sums to ~1)
    """
    if method == "auto":
        method = "deeptrace" if model is not None else "rumor"

    if method == "deeptrace":
        if model is None:
            raise ValueError("DeepTrace model required for method='deeptrace'")
        return deeptrace_posterior(Gn, model, G=G)

    if method == "rumor":
        return rumor_centrality(Gn)

    raise ValueError(f"Unknown method '{method}'. Choose 'deeptrace', 'rumor', or 'auto'.")


# ---------------------------------------------------------------------------
# Top-K source selection (used in E-PPR phase)
# ---------------------------------------------------------------------------

def top_k_sources(pi, K=10):
    """
    Return top-K nodes by posterior weight π(v).

    Parameters
    ----------
    pi : dict {node: float}
    K  : int

    Returns
    -------
    List of (node, weight) sorted descending by weight, length min(K, |pi|)
    """
    sorted_nodes = sorted(pi.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:K]


# ---------------------------------------------------------------------------
# Evaluation: Top-k source detection accuracy
# ---------------------------------------------------------------------------

def topk_accuracy(pi, true_source, k=1):
    """
    Returns 1 if true_source is among top-k nodes by π, else 0.
    """
    top = [node for node, _ in top_k_sources(pi, K=k)]
    return int(true_source in top)
