"""
Spectral measures for the Spectral Path-Product (SPP) algorithm.

Mathematical grounding
----------------------
The reduction in spectral radius from removing node v is proportional to

    Δρ ∝ u_i · v_i

where:
  u_i  = i-th entry of the principal LEFT  eigenvector of A  (receives spread)
  v_i  = i-th entry of the principal RIGHT eigenvector of A  (drives spread)

Key identity: left eigenvector of A  =  right eigenvector of Aᵀ.
So we compute it via a single sparse eigs() call on the TRANSPOSED adjacency.

Proxy choices
-------------
  v_i  →  Downstream Katz Centrality C_out on G_rev  (infinite-path spreading power)
  u_i  →  Principal left eigenvector of the subgraph adjacency  (TRUE spectral proxy)
           Supersedes the old E-PPR / τ approach, which failed on large graphs
           because PPR seeded on a tiny epidemic subgraph assigns τ≈0 to most
           KSCC nodes, collapsing SPP to a tiny PPR-reachable pocket.

Source-awareness (optional)
----------------------------
τ(v) from E-PPR is retained only as a soft *tiebreaker* bonus, not as a
hard gate.  SPP_v2(v) = u_i · v_i · (1 + γ·τ̃(v)) where γ controls how
much source-awareness biases the spectral criterion.

References
----------
[4] Katz, L. (1953). A new status index derived from sociometric analysis.
[5] Benzi & Klymko (2015). Total communicability as a centrality measure.
[6] Van Mieghem et al. (2011). Decreasing the spectral radius of a graph
    by link removals. IEEE Trans. Netw.
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# LEFT eigenvector of the adjacency matrix  (true u_i proxy)
# ---------------------------------------------------------------------------

def compute_left_eigenvec(G, normalized=True):
    """
    Compute the principal LEFT eigenvector of G's adjacency matrix.

    Left eigenvector of A  =  right eigenvector of Aᵀ.
    Computed via one sparse eigs() call on the transposed matrix — O(|E|).

    Nodes whose entry is negative (sign ambiguity) are flipped so that all
    entries are non-negative (standard convention for the Perron vector).

    Falls back to in-degree normalised as a proxy when eigs() fails.

    Parameters
    ----------
    G          : nx.DiGraph  (subgraph of the KSCC step)
    normalized : bool, if True scale entries to [0, 1]

    Returns
    -------
    left_vec : dict {node: float}   all values ≥ 0
    """
    n = G.number_of_nodes()
    if n == 0:
        return {}
    if n == 1:
        return {list(G.nodes())[0]: 1.0}

    nodes = list(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}

    rows = [node_idx[u] for u, v in G.edges()]
    cols = [node_idx[v] for u, v in G.edges()]

    if not rows:
        # No edges — all nodes equal
        return {v: 1.0 / n for v in nodes}

    A = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n, n), dtype=float
    )

    try:
        # Left eigenvector of A  =  right eigenvector of Aᵀ
        vals, vecs = spla.eigs(A.T, k=1, which="LM",
                               return_eigenvectors=True, maxiter=2000)
        u = np.real(vecs[:, 0])
        # Ensure non-negative (Perron–Frobenius: flip if majority negative)
        if u.sum() < 0:
            u = -u
        u = np.maximum(u, 0.0)
    except Exception:
        # Fallback: in-degree as left-eigenvector proxy
        u = np.array([float(G.in_degree(v)) for v in nodes])

    if normalized:
        max_u = u.max()
        if max_u > 1e-12:
            u = u / max_u

    return {v: float(u[node_idx[v]]) for v in nodes}


# ---------------------------------------------------------------------------
# Downstream Katz Centrality  (right-eigenvector proxy)
# ---------------------------------------------------------------------------

def compute_katz_out(G, alpha=None, tol=1.0e-6, max_iter=1000, normalized=True):
    """
    Compute Downstream Katz Centrality for all nodes in G.

    Katz centrality on G.reverse() captures global downstream spreading power
    over infinite directed paths — the right-eigenvector proxy v_i.

    α is set to 0.95 / ρ(G) when not provided.  Explicit alpha skips the
    eigensolver (use precomputed global_alpha for efficiency in the greedy loop).

    Complexity: O(|E| · iterations) via power iteration.

    Parameters
    ----------
    G          : nx.DiGraph
    alpha      : float or None
    tol        : float
    max_iter   : int
    normalized : bool

    Returns
    -------
    katz_out : dict {node: float}
    """
    n = G.number_of_nodes()
    if n == 0:
        return {}

    if alpha is None:
        nodes_l = list(G.nodes())
        node_idx = {v: i for i, v in enumerate(nodes_l)}
        rows = [node_idx[u] for u, v in G.edges()]
        cols = [node_idx[v] for u, v in G.edges()]
        rho = 1.0
        if rows:
            try:
                A = sp.csr_matrix(
                    (np.ones(len(rows)), (rows, cols)),
                    shape=(n, n), dtype=float
                )
                rho = float(np.max(np.abs(
                    spla.eigs(A, k=1, which="LM",
                              return_eigenvectors=False, maxiter=1000)
                )))
            except Exception:
                pass
        alpha = 0.95 / max(rho, 1e-9)

    G_rev = G.reverse(copy=False)

    try:
        katz_out = nx.katz_centrality(
            G_rev,
            alpha=alpha,
            beta=1.0,
            normalized=normalized,
            tol=tol,
            max_iter=max_iter,
        )
    except nx.PowerIterationFailedConvergence:
        katz_out = {v: float(G.out_degree(v)) for v in G.nodes()}
        if normalized:
            max_val = max(katz_out.values(), default=1.0)
            if max_val > 0:
                katz_out = {v: s / max_val for v, s in katz_out.items()}

    return katz_out


# ---------------------------------------------------------------------------
# SPP Score  (left × right eigenvector product, with optional τ bonus)
# ---------------------------------------------------------------------------

def spp_score(left_vec, katz_out, source_risk=None, gamma=0.5):
    """
    Compute the Spectral Path-Product score for every node.

    SPP_v2(v) = u_i(v) · C_out(v) · (1 + γ · τ̃(v))

    where:
        u_i(v)    = left_vec[v]      — TRUE left-eigenvector entry (spectral proxy)
        C_out(v)  = katz_out[v]      — right-eigenvector proxy (spreading power)
        τ̃(v)      = source_risk[v]   — E-PPR source-risk as *soft* tiebreaker bonus
                                        (not a hard gate; zero τ no longer zeroes score)
        γ         = source-awareness weight (default 0.5)

    The τ bonus steers SPP toward source-reachable nodes when there is a tie
    in the spectral product, without blocking structurally critical nodes that
    happen to lie outside the observed epidemic subgraph.

    Parameters
    ----------
    left_vec    : dict {node: float}  — principal left eigenvector entries (u_i)
    katz_out    : dict {node: float}  — Downstream Katz Centrality (v_i)
    source_risk : dict {node: float} or None  — E-PPR τ(v); if None, γ=0
    gamma       : float, source-awareness weight ∈ [0, ∞)

    Returns
    -------
    scores : dict {node: float}
    """
    all_nodes = set(left_vec) | set(katz_out)
    scores = {}

    # Normalise τ to [0,1] for the bonus term
    if source_risk is not None and gamma > 0:
        tau_max = max(source_risk.values(), default=1.0)
        if tau_max < 1e-12:
            tau_max = 1.0
        tau_norm = {v: source_risk.get(v, 0.0) / tau_max for v in all_nodes}
    else:
        tau_norm = {}
        gamma = 0.0

    for v in all_nodes:
        u_v = left_vec.get(v, 0.0)
        c_v = katz_out.get(v, 0.0)
        bonus = 1.0 + gamma * tau_norm.get(v, 0.0)
        scores[v] = u_v * c_v * bonus

    return scores
