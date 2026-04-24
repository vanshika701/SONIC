"""
Spectral measures for the Spectral Path-Product (SPP) algorithm.

Mathematical grounding
----------------------
The reduction in spectral radius from removing node v is proportional to

    Δρ ∝ u_i · v_i

where u_i is the left eigenvector entry (proxy: Personalized PageRank / tau,
representing infection-arrival probability) and v_i is the right eigenvector
entry (proxy: Downstream Katz Centrality, representing global spreading power).

References
----------
[3] DeepTrace (source-risk arrival proxy via PPR)
[4] Katz, L. (1953). A new status index derived from sociometric analysis.
[5] Benzi & Klymko (2015). Total communicability as a centrality measure.
"""

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Downstream Katz Centrality  (right-eigenvector proxy)
# ---------------------------------------------------------------------------

def compute_katz_out(G, alpha=None, tol=1.0e-6, max_iter=1000, normalized=True):
    """
    Compute Downstream Katz Centrality for all nodes in G.

    This is Katz centrality computed on the *reversed* graph, so it captures
    global downstream spreading power over infinite directed paths.

    Formally:
        C_out(v) = Σ_{l=1}^{∞} α^l (A^T)^l · 1
    which, on the reversed graph G_rev, reduces to standard Katz centrality.

    Attenuation factor α is automatically set to

        α = 0.95 / ρ(G)

    so that the power series converges and captures multi-hop structure without
    being dominated by the leading eigenvalue.  If α is supplied explicitly it
    overrides the auto-selection.

    Complexity: O(|E| · iterations) — power iteration, never dense.

    Parameters
    ----------
    G          : nx.DiGraph
    alpha      : float or None
        Katz attenuation factor. If None, set to 0.95 / ρ(G).
    tol        : float, convergence tolerance for power iteration
    max_iter   : int, maximum iterations
    normalized : bool, if True normalise scores to [0, 1]

    Returns
    -------
    katz_out : dict {node: float}
    """
    n = G.number_of_nodes()
    if n == 0:
        return {}

    # Auto-select alpha from spectral radius
    if alpha is None:
        try:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla

            nodes = list(G.nodes())
            node_idx = {v: i for i, v in enumerate(nodes)}
            rows = [node_idx[u] for u, v in G.edges()]
            cols = [node_idx[v] for u, v in G.edges()]

            if rows:
                A = sp.csr_matrix(
                    (np.ones(len(rows)), (rows, cols)),
                    shape=(n, n), dtype=float
                )
                rho = float(np.max(np.abs(
                    spla.eigs(A, k=1, which="LM",
                              return_eigenvectors=False, maxiter=1000)
                )))
            else:
                rho = 1.0
        except Exception:
            rho = 1.0

        alpha = 0.95 / max(rho, 1e-9)

    # Katz on reversed graph = downstream Katz on original
    G_rev = G.reverse(copy=False)

    try:
        katz_out = nx.katz_centrality(
            G_rev,
            alpha=alpha,
            beta=1.0,
            normalized=normalized,
            tol=tol,
            max_iter=max_iter,
            nstart=None,
        )
    except nx.PowerIterationFailedConvergence:
        # Graceful fallback: use out-degree as surrogate
        katz_out = {v: float(G.out_degree(v)) for v in G.nodes()}
        if normalized:
            max_val = max(katz_out.values(), default=1.0)
            if max_val > 0:
                katz_out = {v: s / max_val for v, s in katz_out.items()}

    return katz_out


# ---------------------------------------------------------------------------
# SPP Score  (left × right eigenvector product)
# ---------------------------------------------------------------------------

def spp_score(source_risk, katz_out):
    """
    Compute the Spectral Path-Product score for every node.

    SPP(v) = τ(v) × C_out(v)

    where
        τ(v)      = source_risk[v]  — left-eigenvector proxy (infection arrival)
        C_out(v)  = katz_out[v]     — right-eigenvector proxy (spreading power)

    A node with τ(v) = 0 scores exactly 0 regardless of C_out, enforcing the
    hard constraint that nodes unreachable from the source cannot contribute to
    the epidemic spectral radius.

    Parameters
    ----------
    source_risk : dict {node: float}
        E-PPR output τ(v) — probability of infection arriving at v from the
        inferred source(s).
    katz_out    : dict {node: float}
        Downstream Katz Centrality C_out(v) for each node.

    Returns
    -------
    scores : dict {node: float}
        SPP(v) for every node present in *both* dicts.
        Nodes missing from either dict receive score 0.
    """
    all_nodes = set(source_risk) | set(katz_out)
    scores = {}
    for v in all_nodes:
        tau_v = source_risk.get(v, 0.0)
        if tau_v == 0.0:
            scores[v] = 0.0
        else:
            scores[v] = tau_v * katz_out.get(v, 0.0)
    return scores
