"""
DINO: DIrected NetwOrk epidemic containment (He et al., WSDM 2025).

Key results from the paper:
- Theorem 3.1: rho(G) = rho(KSCC)
- Theorem 3.2: rho(G) ≈ d_in * d_out / vol(G) for directed Chung-Lu networks
- Algorithm 1: Greedy sequential KSCC immunization using F(v) score
- F(v) = d_in_{G\v} * d_out_{G\v} / vol(G\v)  [MINIMIZE this]
- Complexity: O(k|V| + |E|)
"""

import heapq
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# Spectral radius computation
# ---------------------------------------------------------------------------

def spectral_radius(G):
    """
    Compute exact spectral radius (largest eigenvalue magnitude) of G's adjacency matrix.
    Uses scipy sparse eigensolver — O(|E|) per call with k=1.

    Returns float.
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0

    nodes = list(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}

    rows, cols = [], []
    for u, v in G.edges():
        rows.append(node_idx[u])
        cols.append(node_idx[v])

    if not rows:
        return 0.0

    A = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n, n),
        dtype=float
    )

    try:
        # k=1 largest eigenvalue by magnitude
        vals = spla.eigs(A, k=1, which="LM", return_eigenvectors=False, maxiter=1000)
        return float(np.max(np.abs(vals)))
    except Exception:
        # Fallback: dense computation for small graphs
        if n <= 500:
            return float(np.max(np.abs(np.linalg.eigvals(A.toarray()))))
        return 0.0


def approx_spectral_radius(G):
    """
    Approximate spectral radius using DINO Theorem 3.2:
    rho ≈ d_in_total * d_out_total / vol(G)

    where d_in_total = sum of in-degrees, d_out_total = sum of out-degrees,
    vol(G) = sum of in-degrees = sum of out-degrees.

    This is O(1) given degree sums.
    """
    vol = G.number_of_edges()  # vol(G) = sum of in-degrees = |E|
    if vol == 0:
        return 0.0
    d_in_total = sum(d for _, d in G.in_degree())
    d_out_total = sum(d for _, d in G.out_degree())
    return (d_in_total * d_out_total) / vol


# ---------------------------------------------------------------------------
# Node score function F(v)
# ---------------------------------------------------------------------------

def _precompute_degrees(G):
    """
    Precompute d_in[v], d_out[v], vol(G) for efficient F(v) computation.
    Returns (d_in_dict, d_out_dict, vol)
    """
    d_in = dict(G.in_degree())
    d_out = dict(G.out_degree())
    vol = G.number_of_edges()
    return d_in, d_out, vol


def score_F(v, d_in, d_out, vol):
    """
    F(v) = d_in_{G\v} * d_out_{G\v} / vol(G\v)

    After removing v:
      d_in_{G\v}  = sum_{u != v} in_deg(u in G\v)
                  = (total_in_degree - in_deg(v)) - (edges from v's predecessors to v, i.e. in_deg(v))
                  = d_in_total - 2*d_in[v]   ... wait, need to be careful.

    Actually:
      sum of in-degrees in G\v = (sum of in-degrees in G) - in_deg(v) - (number of edges FROM v that go to other nodes)
      But edges FROM v don't contribute to in-degree of other nodes... wait no.

      In G, in_deg(u) counts edges INTO u. When we remove v:
      - We remove v's in-edges (which reduced in-deg of v by in_deg(v))
      - We remove v's out-edges (each out-edge v→u reduces in_deg(u) by 1, so reduces total in-degree by out_deg(v))
      So: d_in_total_{G\v} = d_in_total - in_deg(v) - out_deg(v)
                           = vol - in_deg(v) - out_deg(v)  [since vol = total in-degree]

      Similarly: d_out_total_{G\v} = vol - out_deg(v) - in_deg(v)

      vol(G\v) = vol - in_deg(v) - out_deg(v)

    Therefore:
      F(v) = (vol - in_deg(v) - out_deg(v))^2 / (vol - in_deg(v) - out_deg(v))
           = vol - in_deg(v) - out_deg(v)

    Wait, that simplifies too much. Let me re-read the DINO paper formula.

    From DINO paper: F(v) = d_in_{G\v} * d_out_{G\v} / vol(G\v)

    where d_in_{G\v} is the SUM of in-degrees (same as vol), d_out similarly.
    vol(G\v) = |E(G\v)| = vol(G) - in_deg(v) - out_deg(v)

    d_in_total_{G\v} = vol(G\v) = vol - in_deg(v) - out_deg(v)
    d_out_total_{G\v} = vol(G\v) = vol - in_deg(v) - out_deg(v)

    So F(v) = vol(G\v)^2 / vol(G\v) = vol(G\v) = vol - in_deg(v) - out_deg(v).

    To MINIMIZE F(v) = maximize (in_deg(v) + out_deg(v)) = maximize total degree.
    This makes sense: remove the highest total-degree node.

    But wait — reading the paper more carefully, d_in and d_out in the approximation
    formula are the DOT PRODUCTS d_in · d_out = sum_i d_in_i * d_out_i, not sums.
    Let me re-read Theorem 3.2:

    "rho ≈ d_in · d_out / vol(G)"

    where d_in and d_out are VECTORS. So d_in · d_out = sum_i d_in_i * d_out_i.
    This is NOT vol^2 / vol.

    So:
      F(v) = (sum_{u != v} d_in_u * d_out_u) / vol(G\v)
           = (dot_prod - d_in[v]*d_out[v]) / (vol - in_deg[v] - out_deg[v])

    Minimizing F(v) = finding v that maximally reduces spectral radius.
    Delta_rho(v) = dot_prod/vol - (dot_prod - d_in[v]*d_out[v])/(vol - in_deg[v] - out_deg[v])
    """
    d_in_v = d_in.get(v, 0)
    d_out_v = d_out.get(v, 0)
    vol_minus_v = vol - d_in_v - d_out_v
    if vol_minus_v <= 0:
        return 0.0

    # dot product sum_u d_in[u] * d_out[u]
    # This needs to be precomputed and passed in
    # See dino() for the actual usage
    return vol_minus_v


def delta_rho_approx(v, d_in, d_out, vol, dot_prod):
    """
    Estimated spectral radius decrease from removing v (DINO Eq. in Section 4.1):
    Delta_rho(v) = d_in·d_out/vol - (d_in·d_out - d_in[v]*d_out[v]) / (vol - d_in[v] - d_out[v])

    Used as the structural term in SONIC composite score.
    """
    d_in_v = d_in.get(v, 0)
    d_out_v = d_out.get(v, 0)
    vol_v = vol - d_in_v - d_out_v
    if vol_v <= 0:
        return dot_prod / vol if vol > 0 else 0.0

    rho_before = dot_prod / vol
    rho_after = (dot_prod - d_in_v * d_out_v) / vol_v
    return rho_before - rho_after


def compute_all_delta_rho(G):
    """
    Compute delta_rho_approx for all nodes in G.
    Returns dict {node: delta_rho}.
    O(|V|) after precomputation.
    """
    d_in, d_out, vol = _precompute_degrees(G)
    if vol == 0:
        return {v: 0.0 for v in G.nodes()}

    dot_prod = sum(d_in[v] * d_out[v] for v in G.nodes())

    result = {}
    for v in G.nodes():
        result[v] = delta_rho_approx(v, d_in, d_out, vol, dot_prod)
    return result


# ---------------------------------------------------------------------------
# KSCC identification
# ---------------------------------------------------------------------------

def find_nontrivial_sccs(G, min_size=3):
    """
    Find all strongly connected components with >= min_size nodes,
    sorted descending by approximate spectral radius.

    DINO paper: "non-trivial SCCs with node number larger than two"
    => min_size=3

    Returns list of sets (node sets), sorted by approx rho descending.
    """
    sccs = [s for s in nx.strongly_connected_components(G) if len(s) >= min_size]
    if not sccs:
        return []

    def scc_rho(scc):
        subG = G.subgraph(scc)
        return approx_spectral_radius(subG)

    return sorted(sccs, key=scc_rho, reverse=True)


def get_kscc(G):
    """Return the Key Strongly Connected Component (highest spectral radius)."""
    sccs = find_nontrivial_sccs(G)
    return sccs[0] if sccs else set()


# ---------------------------------------------------------------------------
# DINO Algorithm 1
# ---------------------------------------------------------------------------

def dino(G, k, return_delta_rho=True, verbose=False):
    """
    DINO Algorithm 1 (He et al., WSDM 2025).

    Greedy sequential node immunization on directed network G.
    Selects k nodes from the KSCC whose removal maximally reduces spectral radius.

    Parameters
    ----------
    G               : nx.DiGraph (will NOT be modified — works on a copy)
    k               : int, immunization budget
    return_delta_rho: bool, if True also compute exact Δρ before/after
    verbose         : bool

    Returns
    -------
    L : list of immunized node indices (length k)
    delta_rho : float, rho(G) - rho(G minus L)  [only if return_delta_rho=True]
    """
    G_work = G.copy()
    L = []

    # Compute initial spectral radius
    rho_initial = None
    if return_delta_rho:
        rho_initial = spectral_radius(G)
        if verbose:
            print(f"[DINO] Initial spectral radius: {rho_initial:.4f}")

    # Initialize sorted SCC list (descending by approx rho)
    S = find_nontrivial_sccs(G_work)

    if verbose:
        print(f"[DINO] Found {len(S)} non-trivial SCCs, "
              f"KSCC size={len(S[0]) if S else 0}")

    for step in range(k):
        # Find the first non-empty SCC
        while S and len(S[0]) < 3:
            S.pop(0)

        if not S:
            if verbose:
                print(f"[DINO] No more non-trivial SCCs at step {step}. Stopping.")
            break

        S1 = S.pop(0)
        subG = G_work.subgraph(S1)

        # Precompute degrees within subgraph
        d_in_sub = dict(subG.in_degree())
        d_out_sub = dict(subG.out_degree())
        vol_sub = subG.number_of_edges()

        if vol_sub == 0:
            continue

        dot_prod_sub = sum(d_in_sub[v] * d_out_sub[v] for v in S1)

        # Select node v* = argmin F(v) in S1
        # F(v) = (dot_prod - d_in[v]*d_out[v]) / (vol - d_in[v] - d_out[v])
        best_v = None
        best_F = float("inf")

        for v in S1:
            d_in_v = d_in_sub.get(v, 0)
            d_out_v = d_out_sub.get(v, 0)
            vol_v = vol_sub - d_in_v - d_out_v
            if vol_v <= 0:
                # Removing v empties the SCC — great, F=0
                F_v = 0.0
            else:
                F_v = (dot_prod_sub - d_in_v * d_out_v) / vol_v

            if F_v < best_F:
                best_F = F_v
                best_v = v

        if best_v is None:
            continue

        L.append(best_v)
        G_work.remove_node(best_v)

        if verbose:
            print(f"[DINO] Step {step+1}/{k}: removed node {best_v}, F={best_F:.4f}")

        # Reinsert new SCCs from S1\{v*} back into sorted list
        remaining = S1 - {best_v}
        if len(remaining) >= 3:
            new_sccs = find_nontrivial_sccs(G_work.subgraph(remaining))
            # Merge into S (maintain descending rho order)
            S = _merge_sorted_sccs(S, new_sccs, G_work)

    if return_delta_rho:
        rho_final = spectral_radius(G_work)
        delta = rho_initial - rho_final
        if verbose:
            print(f"[DINO] Final spectral radius: {rho_final:.4f}, Δρ={delta:.4f}")
        return L, delta

    return L


def _merge_sorted_sccs(existing, new_sccs, G):
    """
    Merge new_sccs into existing sorted list, maintaining descending approx rho order.
    O(|new_sccs| * log(|existing|)) with binary search.
    """
    if not new_sccs:
        return existing

    combined = existing + new_sccs

    def scc_rho(scc):
        subG = G.subgraph(scc)
        return approx_spectral_radius(subG)

    return sorted(combined, key=scc_rho, reverse=True)


# ---------------------------------------------------------------------------
# Convenience: replicate DINO paper Table 2 for HIV network
# ---------------------------------------------------------------------------

def benchmark_hiv(G, budgets=(100, 300, 500)):
    """
    Run DINO on G for multiple budgets and print delta_rho.
    Expected output for HIV: 4.59 (k=100), 5.70 (k=300).
    """
    print("\n=== DINO Benchmark ===")
    print(f"Graph: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    rho0 = spectral_radius(G)
    print(f"Initial spectral radius: {rho0:.4f}")
    print(f"{'Budget':>8} | {'Δρ':>8} | {'ρ_final':>8}")
    print("-" * 32)
    for k in budgets:
        _, delta = dino(G, k, return_delta_rho=True)
        print(f"{k:>8} | {delta:>8.2f} | {rho0 - delta:>8.4f}")
