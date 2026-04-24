"""
Spectral Path-Product (SPP) node selection for directed network containment.

Mathematical background
-----------------------
For a directed network G, the spectral radius ρ(G) = ρ(KSCC(G)) (Theorem 3.1,
He et al. WSDM 2025).  The eigen-drop from immunising node v is proportional to

    Δρ ∝ u_i · v_i

where u_i is the left-eigenvector entry (proxied by τ(v) — the E-PPR source
risk, capturing infection-*arrival* probability) and v_i is the right-eigenvector
entry (proxied by C_out(v) — Downstream Katz Centrality on G_rev, capturing
global downstream spreading power).

SPP(v) = τ(v) × C_out(v)

replaces the additive DINO heuristic  F(v) = d_in · d_out / vol  with a
theoretically grounded multiplicative product.

Complexity: O(k|V| + |E|) — Katz centrality uses power iteration.
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from algorithms.measures import compute_katz_out, spp_score


# ---------------------------------------------------------------------------
# Spectral radius helpers  (kept for evaluation / Δρ reporting)
# ---------------------------------------------------------------------------

def spectral_radius(G):
    """
    Exact spectral radius ρ(G) via sparse eigensolver.  O(|E|) with k=1.
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0

    nodes = list(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    rows = [node_idx[u] for u, v in G.edges()]
    cols = [node_idx[v] for u, v in G.edges()]

    if not rows:
        return 0.0

    A = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n, n), dtype=float
    )
    try:
        vals = spla.eigs(A, k=1, which="LM",
                         return_eigenvectors=False, maxiter=1000)
        return float(np.max(np.abs(vals)))
    except Exception:
        if n <= 500:
            return float(np.max(np.abs(np.linalg.eigvals(A.toarray()))))
        return 0.0


def approx_spectral_radius(G):
    """
    Approximate ρ via DINO Theorem 3.2: d_in·d_out / vol.
    Used only for SCC ranking (O(|V|+|E|) after degree precomputation).
    """
    vol = G.number_of_edges()
    if vol == 0:
        return 0.0
    d_in_total = sum(d for _, d in G.in_degree())
    d_out_total = sum(d for _, d in G.out_degree())
    return (d_in_total * d_out_total) / vol


# ---------------------------------------------------------------------------
# KSCC identification
# ---------------------------------------------------------------------------

def find_nontrivial_sccs(G, min_size=3):
    """
    Return all SCCs with >= min_size nodes, sorted descending by approx ρ.
    """
    sccs = [s for s in nx.strongly_connected_components(G) if len(s) >= min_size]
    if not sccs:
        return []

    def scc_rho(scc):
        return approx_spectral_radius(G.subgraph(scc))

    return sorted(sccs, key=scc_rho, reverse=True)


def get_kscc(G):
    """Return the Key Strongly Connected Component (highest approx ρ)."""
    sccs = find_nontrivial_sccs(G)
    return sccs[0] if sccs else set()


def _merge_sorted_sccs(existing, new_sccs, G):
    """Merge new_sccs into existing sorted list, maintaining descending approx ρ."""
    if not new_sccs:
        return existing
    combined = existing + new_sccs

    def scc_rho(scc):
        return approx_spectral_radius(G.subgraph(scc))

    return sorted(combined, key=scc_rho, reverse=True)


# ---------------------------------------------------------------------------
# SPP Selection  (replaces DINO greedy loop)
# ---------------------------------------------------------------------------

class SPPSelection:
    """
    Spectral Path-Product (SPP) greedy node selection.

    Selects k nodes from the KSCC whose removal maximally reduces the
    spectral radius, using SPP(v) = τ(v) × C_out(v) as the per-step score.

    Parameters
    ----------
    G            : nx.DiGraph  (not mutated — a copy is used internally)
    k            : int, immunisation budget
    source_risk  : dict {node: float}
        τ(v) from E-PPR Phase 2 — infection-arrival probability seeded at the
        inferred source.  This is the left-eigenvector proxy.
    return_delta_rho : bool  (default True)
        If True, compute exact Δρ before / after selection.
    verbose      : bool
    """

    def __init__(self, G, k, source_risk, return_delta_rho=True, verbose=False):
        self.G_orig = G
        self.k = k
        self.source_risk = source_risk
        self.return_delta_rho = return_delta_rho
        self.verbose = verbose

    # ------------------------------------------------------------------
    def select(self):
        """
        Run the greedy SPP immunisation loop.

        Returns
        -------
        L          : list of selected node ids
        delta_rho  : float  (only when return_delta_rho=True)
        """
        G_work = self.G_orig.copy()
        L = []

        rho_initial = None
        if self.return_delta_rho:
            rho_initial = spectral_radius(self.G_orig)
            if self.verbose:
                print(f"[SPP] Initial ρ = {rho_initial:.4f}")

        # Initialise KSCC priority queue (sorted by approx ρ, descending)
        S = find_nontrivial_sccs(G_work)
        if self.verbose:
            print(f"[SPP] Non-trivial SCCs: {len(S)}, "
                  f"KSCC size = {len(S[0]) if S else 0}")
                  
        # Precompute global alpha to avoid O(|E|) eigensolver every step
        global_alpha = 0.01
        if S:
            kscc_rho = spectral_radius(G_work.subgraph(S[0]))
            global_alpha = min(0.95 / max(kscc_rho, 1e-9), 0.05)

        for step in range(self.k):
            # Skip SCCs that have shrunk below threshold
            while S and len(S[0]) < 3:
                S.pop(0)

            if not S:
                if self.verbose:
                    print(f"[SPP] No non-trivial SCCs remaining at step {step}. "
                          "Stopping early.")
                break

            S1 = S.pop(0)          # current KSCC (set of node ids)
            subG = G_work.subgraph(S1)

            # --- Compute Downstream Katz Centrality on this subgraph ---
            katz_out = compute_katz_out(subG, alpha=global_alpha)

            # --- Restrict τ to nodes in S1 ---
            tau_s1 = {v: self.source_risk.get(v, 0.0) for v in S1}

            # --- SPP score for every node in S1 ---
            scores = spp_score(tau_s1, katz_out)

            # --- Select node v* = argmax SPP(v) ---
            best_v = max(scores, key=lambda v: scores[v], default=None)
            if best_v is None:
                continue

            if self.verbose:
                print(f"[SPP] Step {step + 1}/{self.k}: "
                      f"selected node {best_v}  "
                      f"SPP={scores[best_v]:.6f}  "
                      f"τ={tau_s1[best_v]:.4f}  "
                      f"C_out={katz_out.get(best_v, 0.0):.4f}")

            L.append(best_v)
            G_work.remove_node(best_v)

            # Re-insert child SCCs of S1 \ {best_v} back into sorted list
            remaining = S1 - {best_v}
            if len(remaining) >= 3:
                new_sccs = find_nontrivial_sccs(G_work.subgraph(remaining))
                S = _merge_sorted_sccs(S, new_sccs, G_work)

        if self.return_delta_rho:
            rho_final = spectral_radius(G_work)
            delta = rho_initial - rho_final
            if self.verbose:
                print(f"[SPP] Final ρ = {rho_final:.4f}  Δρ = {delta:.4f}")
            return L, delta

        return L


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def spp_selection(G, k, source_risk, return_delta_rho=True, verbose=False):
    """
    Functional wrapper around SPPSelection.

    Parameters
    ----------
    G            : nx.DiGraph
    k            : int, immunisation budget
    source_risk  : dict {node: float}  — τ(v) from E-PPR
    return_delta_rho : bool
    verbose      : bool

    Returns
    -------
    L          : list of immunised node ids
    delta_rho  : float (only when return_delta_rho=True)
    """
    selector = SPPSelection(
        G=G, k=k, source_risk=source_risk,
        return_delta_rho=return_delta_rho, verbose=verbose
    )
    return selector.select()
