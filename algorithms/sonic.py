"""
SONIC: Source-Oriented Network Immunization and Containment.

Refactored to use the Spectral Path-Product (SPP) algorithm instead of
the additive heuristic (αw · Δρ + βw · SourceRisk).

Pipeline
--------
  Phase 1  —  Source Posterior π via DeepTrace GNN or Rumor Centrality
  Phase 2  —  Source Risk τ(v) via Expected Personalized PageRank (E-PPR)
  Phase 3  —  Spectral Path-Product (SPP) Optimization
               SPP(v) = τ(v) × C_out(v)
               maximised greedily within the KSCC

Mathematical grounding
----------------------
Eigen-drop  Δρ ∝ u_i · v_i  (left × right eigenvector).

  τ(v)     = E-PPR SourceRisk  →  left-eigenvector proxy (arrival probability)
  C_out(v) = Downstream Katz  →  right-eigenvector proxy (spreading power)
"""

import numpy as np
import networkx as nx

from algorithms.spp import (
    find_nontrivial_sccs,
    approx_spectral_radius,
    spectral_radius,
    _merge_sorted_sccs,
    spp_selection,
)
from algorithms.source_inference import infer_source_posterior
from algorithms.eppr import source_risk as compute_source_risk_eppr


# ---------------------------------------------------------------------------
# SONIC Algorithm (SPP Edition)
# ---------------------------------------------------------------------------

def sonic(
    G,
    Gn,
    k,
    source_method="auto",
    deeptrace_model=None,
    K_sources=10,
    ppr_alpha=0.15,
    adaptive=False,
    return_delta_rho=True,
    verbose=False,
):
    """
    SONIC: Source-Oriented Network Immunization and Containment.

    Phase 1  —  infer source posterior π over nodes in Gn
    Phase 2  —  compute SourceRisk τ via E-PPR (seeded at top-K sources)
    Phase 3  —  Spectral Path-Product (SPP) Optimization over the KSCC

    Parameters
    ----------
    G                : nx.DiGraph, full network
    Gn               : nx.DiGraph, observed infection subgraph (subset of G)
    k                : int, immunisation budget
    source_method    : str, 'deeptrace', 'rumor', or 'auto'
    deeptrace_model  : DeepTraceGNN or None
    K_sources        : int, top-K sources used for E-PPR
    ppr_alpha        : float, teleport probability for PPR  (default 0.15)
    adaptive         : bool, if True recompute π and τ after each removal
    return_delta_rho : bool, if True return Δρ = ρ_before − ρ_after
    verbose          : bool

    Returns
    -------
    L         : list of immunised node ids (length <= k)
    delta_rho : float  (only when return_delta_rho=True)
    """
    G_work = G.copy()
    L = []

    rho_initial = spectral_radius(G) if return_delta_rho else None
    if verbose and rho_initial is not None:
        print(f"[SONIC] Initial ρ = {rho_initial:.4f}")
        print(f"[SONIC] Pipeline: DeepTrace/RumorCentrality → E-PPR → "
              "Spectral Path-Product (SPP) Optimization")

    # -------------------------------------------------------------------
    # Phase 1: Source Posterior Inference (π)
    # -------------------------------------------------------------------
    def _infer_pi(obs_graph):
        return infer_source_posterior(
            Gn=obs_graph,
            model=deeptrace_model,
            method=source_method,
            G=G,
        )

    pi = _infer_pi(Gn)
    if verbose:
        top_src = max(pi, key=pi.get, default=None)
        print(f"[SONIC] Phase 1 complete — top source: "
              f"{top_src} (π={pi.get(top_src, 0):.4f})")

    # -------------------------------------------------------------------
    # Phase 2: SourceRisk τ via E-PPR
    # -------------------------------------------------------------------
    def _compute_tau(g_work, pi_current):
        return compute_source_risk_eppr(
            g_work, pi_current, K=K_sources, alpha=ppr_alpha
        )

    tau = _compute_tau(G_work, pi)
    if verbose:
        print(f"[SONIC] Phase 2 complete — E-PPR SourceRisk computed "
              f"over {K_sources} top sources.")

    # -------------------------------------------------------------------
    # Phase 3: Spectral Path-Product (SPP) Optimization
    # -------------------------------------------------------------------
    if verbose:
        print("[SONIC] Phase 3: Spectral Path-Product (SPP) Optimization")

    S = find_nontrivial_sccs(G_work)
    if verbose:
        print(f"[SONIC] Non-trivial SCCs: {len(S)}, "
              f"KSCC size = {len(S[0]) if S else 0}")
              
    global_alpha = 0.01
    if S:
        kscc_rho = spectral_radius(G_work.subgraph(S[0]))
        global_alpha = min(0.95 / max(kscc_rho, 1e-9), 0.05)

    for step in range(k):
        # Skip SCCs that have shrunk below threshold
        while S and len(S[0]) < 3:
            S.pop(0)

        if not S:
            if verbose:
                print(f"[SONIC] No non-trivial SCCs remaining at step {step}. "
                      "Stopping early.")
            break

        S1 = S.pop(0)

        # Adaptive mode: refresh π and τ on reduced graph
        if adaptive and step > 0:
            obs_nodes = [v for v in Gn.nodes() if v in G_work]
            pi = _infer_pi(G_work.subgraph(obs_nodes))
            tau = _compute_tau(G_work, pi)

        # --- Run one SPP step on the current KSCC ---
        # We delegate to spp_selection for a single-step subgraph selection
        from algorithms.measures import compute_left_eigenvec, compute_katz_out, spp_score

        subG = G_work.subgraph(S1)
        # True left eigenvector of the current KSCC subgraph
        left_vec = compute_left_eigenvec(subG)
        katz_out = compute_katz_out(subG, alpha=global_alpha)
        tau_s1 = {v: tau.get(v, 0.0) for v in S1}
        # SPP_v2 = u_i · v_i · (1 + γ·τ̃)
        scores = spp_score(left_vec, katz_out, source_risk=tau_s1)

        best_v = max(scores, key=lambda v: scores[v], default=None)
        if best_v is None:
            continue

        if verbose:
            print(f"[SONIC][SPP] Step {step + 1}/{k}: "
                  f"selected node {best_v}  "
                  f"SPP={scores[best_v]:.6f}  "
                  f"u_i={left_vec.get(best_v, 0.0):.4f}  "
                  f"C_out={katz_out.get(best_v, 0.0):.4f}  "
                  f"τ={tau_s1.get(best_v, 0.0):.4f}")

        L.append(best_v)
        G_work.remove_node(best_v)

        # Re-insert child SCCs
        remaining = S1 - {best_v}
        if len(remaining) >= 3:
            new_sccs = find_nontrivial_sccs(G_work.subgraph(remaining))
            S = _merge_sorted_sccs(S, new_sccs, G_work)

    if return_delta_rho:
        rho_final = spectral_radius(G_work)
        delta = rho_initial - rho_final
        if verbose:
            print(f"[SONIC] Final ρ = {rho_final:.4f}  Δρ = {delta:.4f}")
        return L, delta

    return L


# ---------------------------------------------------------------------------
# Convenience: ablation sweep over K_sources values
# ---------------------------------------------------------------------------

def sonic_sweep(G, Gn, k, K_sources_values=None, **kwargs):
    """
    Run SONIC for multiple K_sources values (ablation study).

    Parameters
    ----------
    G, Gn, k      : as in sonic()
    K_sources_values : list of ints, default [5, 10, 20, 50]
    **kwargs       : forwarded to sonic()

    Returns
    -------
    results : list of (K_sources, L, delta_rho)
    """
    if K_sources_values is None:
        K_sources_values = [5, 10, 20, 50]

    results = []
    for ks in K_sources_values:
        L, delta = sonic(G, Gn, k, K_sources=ks,
                         return_delta_rho=True, **kwargs)
        results.append((ks, L, delta))
        print(f"  K_sources={ks:3d} | Δρ={delta:.4f}")

    return results
