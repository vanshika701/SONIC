"""
Ablation study: sweep βw ∈ {0, 0.1, ..., 1.0} for SONIC.

Reproduces Figure 6 / Table 3 pattern from the paper:
  - αw=1, βw=0  →  DINO (structural only)
  - αw=0, βw=1  →  SourceOnly
  - Optimal βw* typically in [0.3, 0.7]

Also tests the entropy-gated auto_weights novelty.
"""

import numpy as np
import pandas as pd
import networkx as nx

from algorithms.sonic import sonic, sonic_sweep
from evaluation.metrics import delta_rho as compute_delta_rho, sis_metrics


# ---------------------------------------------------------------------------
# βw sweep (ablation)
# ---------------------------------------------------------------------------

def ablation_beta_sweep(
    G,
    Gn,
    k,
    beta_w_values=None,
    run_sis=False,
    source_method="rumor",
    verbose=True,
):
    """
    Sweep βw values and record Δρ (and optionally I_T, T_contain).

    Parameters
    ----------
    G, Gn         : nx.DiGraph
    k             : int, budget
    beta_w_values : list of floats, default [0, 0.1, ..., 1.0]
    run_sis       : bool, also compute SIS metrics (slower)
    source_method : str, 'rumor' or 'deeptrace'
    verbose       : bool

    Returns
    -------
    df : pd.DataFrame with columns [beta_w, alpha_w, delta_rho, I_T, T_contain]
    """
    if beta_w_values is None:
        beta_w_values = [round(x * 0.1, 1) for x in range(11)]

    rows = []

    for bw in beta_w_values:
        aw = round(1.0 - bw, 10)
        if verbose:
            print(f"  βw={bw:.1f} αw={aw:.1f} ...", end=" ", flush=True)

        L, dr = sonic(
            G, Gn, k,
            alpha_w=aw,
            beta_w=bw,
            source_method=source_method,
            return_delta_rho=True,
            verbose=False,
        )

        row = {"beta_w": bw, "alpha_w": aw, "delta_rho": dr, "k": k}

        if run_sis:
            sim = sis_metrics(G, L)
            row["I_T"] = sim["I_T"]
            row["T_contain"] = sim["T_contain"]

        rows.append(row)

        if verbose:
            msg = f"Δρ={dr:.4f}"
            if run_sis:
                msg += f"  I_T={row['I_T']:.1f}  T_contain={row['T_contain']:.1f}"
            print(msg)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entropy-gated auto_weights test
# ---------------------------------------------------------------------------

def ablation_auto_weights(G, Gn, k, source_method="rumor", verbose=True):
    """
    Compare fixed βw=0.5 vs. entropy-gated auto_weights.

    Returns
    -------
    results : dict
    """
    results = {}

    # Fixed equal weights
    L_fixed, dr_fixed = sonic(
        G, Gn, k, alpha_w=0.5, beta_w=0.5,
        source_method=source_method,
        return_delta_rho=True, verbose=False,
    )
    results["fixed_0.5"] = {"L": L_fixed, "delta_rho": dr_fixed}

    # Entropy-gated
    L_auto, dr_auto = sonic(
        G, Gn, k, alpha_w=0.5, beta_w=0.5,
        source_method=source_method,
        auto_weights=True,
        return_delta_rho=True, verbose=False,
    )
    results["auto_weights"] = {"L": L_auto, "delta_rho": dr_auto}

    if verbose:
        print(f"  Fixed (βw=0.5):     Δρ={dr_fixed:.4f}")
        print(f"  Auto-weights:       Δρ={dr_auto:.4f}")

    return results


# ---------------------------------------------------------------------------
# Budget sweep (k = 10, 20, ..., 100)
# ---------------------------------------------------------------------------

def ablation_budget_sweep(
    G,
    Gn,
    budgets=None,
    alpha_w=0.5,
    beta_w=0.5,
    source_method="rumor",
    verbose=True,
):
    """
    Evaluate SONIC (fixed αw, βw) over varying budgets k.

    Returns
    -------
    df : pd.DataFrame with columns [k, delta_rho, rho_before, rho_after]
    """
    if budgets is None:
        n = G.number_of_nodes()
        step = max(1, n // 20)
        budgets = list(range(step, min(n // 2, 10 * step) + 1, step))

    rows = []
    for k in budgets:
        if verbose:
            print(f"  k={k} ...", end=" ", flush=True)

        L, dr = sonic(
            G, Gn, k,
            alpha_w=alpha_w, beta_w=beta_w,
            source_method=source_method,
            return_delta_rho=True, verbose=False,
        )
        _, rho_b, rho_a = compute_delta_rho(G, L)
        rows.append({"k": k, "delta_rho": dr, "rho_before": rho_b, "rho_after": rho_a})

        if verbose:
            print(f"Δρ={dr:.4f}  ρ: {rho_b:.4f}→{rho_a:.4f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_ablation_table(df, title="βw Ablation"):
    """Pretty-print ablation DataFrame."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    cols = [c for c in ["beta_w", "alpha_w", "k", "delta_rho", "I_T", "T_contain"] if c in df.columns]
    print(df[cols].to_string(index=False, float_format="{:.4f}".format))
    print(f"{'='*60}\n")
