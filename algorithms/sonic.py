"""
SONIC: Source-Oriented Network Immunization and Containment.
Full Algorithm 1 from the SONIC paper.

Combines:
  Phase 1: Source posterior π via DeepTrace / Rumor Centrality
  Phase 2: SourceRisk via Expected Personalized PageRank (E-PPR)
  Phase 3: Composite scoring + KSCC immunization

SONIC Eq. 7:
  Score_SONIC(v) = αw · Δρ̃_spec(v) + βw · SourceRisk̃(v)
  (tildes = min-max normalized to [0,1])

Setting βw=0 recovers pure DINO.
Setting αw=0 yields pure source-driven selection.
"""

import numpy as np
import networkx as nx

from algorithms.dino import (
    find_nontrivial_sccs,
    approx_spectral_radius,
    compute_all_delta_rho,
    spectral_radius,
    _merge_sorted_sccs,
)
from algorithms.source_inference import infer_source_posterior
from algorithms.eppr import source_risk, entropy_gated_weights


# ---------------------------------------------------------------------------
# Min-max normalization
# ---------------------------------------------------------------------------

def _minmax_normalize(scores_dict):
    """
    Min-max normalize a dict of scores to [0, 1].
    Returns new dict. Handles constant arrays (returns 0.5 for all).
    """
    if not scores_dict:
        return {}
    vals = np.array(list(scores_dict.values()), dtype=float)
    v_min, v_max = vals.min(), vals.max()
    if v_max - v_min < 1e-12:
        return {k: 0.5 for k in scores_dict}
    return {k: float((v - v_min) / (v_max - v_min)) for k, v in scores_dict.items()}


# ---------------------------------------------------------------------------
# SONIC Algorithm 1
# ---------------------------------------------------------------------------

def sonic(
    G,
    Gn,
    k,
    alpha_w=0.5,
    beta_w=0.5,
    source_method="auto",
    deeptrace_model=None,
    K_sources=10,
    ppr_alpha=0.15,
    adaptive=False,
    auto_weights=False,
    return_delta_rho=True,
    verbose=False,
):
    """
    SONIC Algorithm 1: Source-Oriented Network Immunization and Containment.

    Parameters
    ----------
    G               : nx.DiGraph, full network
    Gn              : nx.DiGraph, observed infection subgraph (subset of G)
    k               : int, immunization budget
    alpha_w         : float, weight for structural term (Δρ_spec)
    beta_w          : float, weight for SourceRisk term
                      Must satisfy alpha_w + beta_w = 1.
    source_method   : str, 'deeptrace', 'rumor', or 'auto'
    deeptrace_model : DeepTraceGNN or None
    K_sources       : int, top-K sources for E-PPR computation
    ppr_alpha       : float, teleport probability for PPR
    adaptive        : bool, if True recompute π after each removal (Eq. 10)
    auto_weights    : bool, if True use entropy-gated weight auto-tuning (novelty)
    return_delta_rho: bool
    verbose         : bool

    Returns
    -------
    L          : list of immunized nodes
    delta_rho  : float (only if return_delta_rho=True)
    """
    assert abs(alpha_w + beta_w - 1.0) < 1e-6, "alpha_w + beta_w must equal 1"

    G_work = G.copy()
    L = []

    rho_initial = spectral_radius(G) if return_delta_rho else None
    if verbose and rho_initial:
        print(f"[SONIC] Initial ρ={rho_initial:.4f} | αw={alpha_w:.2f} βw={beta_w:.2f}")

    # -----------------------------------------------------------------------
    # Phase 1: Source Posterior Inference (computed once, or adaptively)
    # -----------------------------------------------------------------------
    def _compute_pi(graph_for_source):
        return infer_source_posterior(
            Gn=graph_for_source,
            model=deeptrace_model,
            method=source_method,
            G=G,
        )

    pi = _compute_pi(Gn)

    # Auto-tune weights from entropy if requested (novelty addition)
    if auto_weights:
        alpha_w, beta_w = entropy_gated_weights(pi, alpha_w, beta_w)
        if verbose:
            print(f"[SONIC] Auto-tuned weights: αw={alpha_w:.3f} βw={beta_w:.3f}")

    # -----------------------------------------------------------------------
    # Phase 2: SourceRisk via E-PPR (static, or re-computed adaptively)
    # -----------------------------------------------------------------------
    def _compute_source_risk(g_work, pi_current):
        if beta_w < 1e-6:
            # Pure DINO — skip SourceRisk computation entirely
            return {v: 0.0 for v in g_work.nodes()}
        return source_risk(g_work, pi_current, K=K_sources, alpha=ppr_alpha)

    sr = _compute_source_risk(G_work, pi)

    # -----------------------------------------------------------------------
    # Phase 3: SONIC composite scoring + KSCC immunization loop
    # -----------------------------------------------------------------------
    S = find_nontrivial_sccs(G_work)

    for step in range(k):
        # Skip trivial SCCs
        while S and len(S[0]) < 3:
            S.pop(0)

        if not S:
            if verbose:
                print(f"[SONIC] No non-trivial SCCs remaining at step {step}. Stopping.")
            break

        S1 = S.pop(0)

        # Adaptive variant: recompute π and SourceRisk on updated graph
        if adaptive and step > 0:
            pi = _compute_pi(G_work.subgraph(
                [v for v in Gn.nodes() if v in G_work]
            ))
            sr = _compute_source_risk(G_work, pi)

        # Compute Δρ_spec for all nodes in S1 (using G_work)
        delta_rho_scores = compute_all_delta_rho(G_work.subgraph(S1))

        # Normalize both terms to [0,1]
        dr_norm = _minmax_normalize(delta_rho_scores)
        sr_s1 = {v: sr.get(v, 0.0) for v in S1}
        sr_norm = _minmax_normalize(sr_s1)

        # Composite score (higher = higher priority for removal)
        best_v = None
        best_score = -1.0

        for v in S1:
            score = alpha_w * dr_norm.get(v, 0.0) + beta_w * sr_norm.get(v, 0.0)
            if score > best_score:
                best_score = score
                best_v = v

        if best_v is None:
            continue

        L.append(best_v)
        G_work.remove_node(best_v)

        if verbose:
            print(f"[SONIC] Step {step+1}/{k}: removed node {best_v}, "
                  f"score={best_score:.4f}")

        # Reinsert new SCCs from S1\{v*}
        remaining = S1 - {best_v}
        if len(remaining) >= 3:
            new_sccs = find_nontrivial_sccs(G_work.subgraph(remaining))
            S = _merge_sorted_sccs(S, new_sccs, G_work)

    if return_delta_rho:
        rho_final = spectral_radius(G_work)
        delta = rho_initial - rho_final
        if verbose:
            print(f"[SONIC] Final ρ={rho_final:.4f}, Δρ={delta:.4f}")
        return L, delta

    return L


# ---------------------------------------------------------------------------
# Convenience: run SONIC with a specific βw value
# ---------------------------------------------------------------------------

def sonic_sweep(G, Gn, k, beta_w_values=None, **kwargs):
    """
    Run SONIC for multiple βw values (ablation study).

    Parameters
    ----------
    G, Gn, k : as in sonic()
    beta_w_values : list of floats, default [0, 0.1, ..., 1.0]
    **kwargs : passed to sonic()

    Returns
    -------
    results : list of (beta_w, L, delta_rho)
    """
    if beta_w_values is None:
        beta_w_values = [round(x * 0.1, 1) for x in range(11)]

    results = []
    for bw in beta_w_values:
        aw = round(1.0 - bw, 1)
        L, delta = sonic(G, Gn, k, alpha_w=aw, beta_w=bw,
                         return_delta_rho=True, **kwargs)
        results.append((bw, L, delta))
        print(f"  βw={bw:.1f} | Δρ={delta:.4f}")

    return results
