"""
Master experiment runner for SONIC paper results.

Reproduces:
  - Table 2 (DINO paper): Δρ for DINO / baselines on HIV, Reddit, Gnutella
  - Table 3 (SONIC paper): Δρ for SONIC vs. baselines
  - Figure 6: βw ablation curve
  - SIS epidemic curves (I_T and T_contain)

Usage:
    python -m experiments.run_all --dataset hiv --budget 100
    python -m experiments.run_all --all
"""

import argparse
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

from data.loaders import load_dataset
from data.synthetic import (
    make_random_graph, simulate_si, generate_training_batch
)
from algorithms.dino import benchmark_hiv
from algorithms.sonic import sonic
from evaluation.metrics import evaluate_method, print_results_table
from experiments.baselines import run_all_baselines, evaluate_baselines
from experiments.ablation import (
    ablation_beta_sweep, ablation_auto_weights,
    ablation_budget_sweep, print_ablation_table
)
from gnn.train import get_or_train_deeptrace


# ---------------------------------------------------------------------------
# Single-dataset experiment
# ---------------------------------------------------------------------------

def run_experiment(
    dataset_name,
    budgets,
    alpha_w=0.5,
    beta_w=0.5,
    source_method="rumor",
    run_sis=True,
    run_baselines=True,
    run_ablation=False,
    use_deeptrace=False,
    verbose=True,
    results_dir="results",
):
    """
    Full experiment for one dataset.

    Returns
    -------
    all_metrics : list of metric dicts
    """
    Path(results_dir).mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset_name.upper()}  |  budgets: {budgets}")
    print(f"  αw={alpha_w}  βw={beta_w}  source_method={source_method}")
    print(f"{'='*70}")

    # Load graph
    G = load_dataset(dataset_name)
    n, m = G.number_of_nodes(), G.number_of_edges()
    print(f"  Graph: |V|={n}  |E|={m}")

    # Build epidemic subgraph Gn via SI simulation
    # Use a random source (or top-degree node as proxy)
    source = max(G.nodes(), key=lambda v: G.in_degree(v))
    Gn, infection_order, _ = simulate_si(
        G, source=source, beta=0.3, max_steps=10
    )
    if Gn.number_of_nodes() < 2:
        Gn = G  # fallback: use full graph as Gn
    print(f"  Epidemic subgraph Gn: |V|={Gn.number_of_nodes()}  |E|={Gn.number_of_edges()}")

    # Optionally load/train DeepTrace
    model = None
    if use_deeptrace:
        print("  Loading/training DeepTrace GNN...")
        model = get_or_train_deeptrace()

    all_metrics = []

    for k in budgets:
        print(f"\n--- Budget k={k} ---")

        # SONIC
        t0 = time.time()
        L_sonic, dr_sonic = sonic(
            G, Gn, k,
            alpha_w=alpha_w, beta_w=beta_w,
            source_method=source_method,
            deeptrace_model=model,
            return_delta_rho=True,
            verbose=verbose,
        )
        t_sonic = time.time() - t0
        m_sonic = evaluate_method(
            G, L_sonic,
            method_name="SONIC",
            run_sis=run_sis,
            verbose=verbose,
        )
        m_sonic["runtime_s"] = t_sonic
        all_metrics.append(m_sonic)

        # Baselines
        if run_baselines:
            bm = evaluate_baselines(
                G, Gn, k,
                run_sis=run_sis,
                verbose=verbose,
            )
            all_metrics.extend(bm)

    # Print table
    print_results_table(all_metrics, budgets=budgets)

    # Save
    out_path = Path(results_dir) / f"{dataset_name}_results.json"
    # Convert numpy types for JSON serialization
    def _to_py(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        return x

    saveable = [{k: _to_py(v) for k, v in m.items() if k != "curve"}
                for m in all_metrics]
    with open(out_path, "w") as f:
        json.dump(saveable, f, indent=2)
    print(f"\n  Results saved → {out_path}")

    # Ablation
    if run_ablation:
        print("\n--- βw Ablation ---")
        abl_df = ablation_beta_sweep(G, Gn, k=budgets[-1],
                                      run_sis=False,
                                      source_method=source_method,
                                      verbose=True)
        print_ablation_table(abl_df, title=f"{dataset_name} βw Ablation")
        abl_path = Path(results_dir) / f"{dataset_name}_ablation.csv"
        abl_df.to_csv(abl_path, index=False)
        print(f"  Ablation saved → {abl_path}")

    return all_metrics


# ---------------------------------------------------------------------------
# HIV benchmark (validate against DINO paper Table 2)
# ---------------------------------------------------------------------------

def run_hiv_benchmark(budgets=None, verbose=True):
    """
    Validate DINO numbers on HIV network.
    Expected: Δρ ≈ 4.59 at k=100, 5.70 at k=300.
    """
    if budgets is None:
        budgets = [100, 200, 300]

    print("\n=== HIV Benchmark (DINO paper Table 2) ===")
    G = load_dataset("hiv")
    results = benchmark_hiv(G, budgets=budgets)
    print("\nk     Δρ      ρ_before  ρ_after")
    print("-" * 35)
    for k, (dr, rb, ra) in results.items():
        print(f"{k:<6}{dr:>8.4f}  {rb:>8.4f}  {ra:>8.4f}")
    return results


# ---------------------------------------------------------------------------
# Synthetic experiment (quick smoke-test)
# ---------------------------------------------------------------------------

def run_synthetic_experiment(n=500, k=20, seed=42, verbose=True):
    """Quick test on a random BA graph."""
    from data.synthetic import make_ba

    print(f"\n=== Synthetic Experiment (BA, n={n}, k={k}) ===")
    G = make_ba(n=n, m=3, seed=seed)
    source = list(G.nodes())[0]
    Gn, order, _ = simulate_si(G, source=source, beta=0.3, max_steps=8, seed=seed)
    if Gn.number_of_nodes() < 2:
        Gn = G

    L, dr = sonic(G, Gn, k, alpha_w=0.5, beta_w=0.5,
                  source_method="rumor", return_delta_rho=True, verbose=verbose)

    print(f"  SONIC Δρ={dr:.4f}  |L|={len(L)}")
    return L, dr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SONIC Experiment Runner")
    p.add_argument("--dataset", default="hiv",
                   choices=["hiv", "reddit", "gnutella", "synthetic"],
                   help="Dataset to run experiments on")
    p.add_argument("--budgets", nargs="+", type=int, default=[50, 100, 200],
                   help="Immunization budgets k")
    p.add_argument("--alpha_w", type=float, default=0.5)
    p.add_argument("--beta_w",  type=float, default=0.5)
    p.add_argument("--source_method", default="rumor",
                   choices=["rumor", "deeptrace", "auto"])
    p.add_argument("--no_sis", action="store_true",
                   help="Skip SIS simulation (faster)")
    p.add_argument("--no_baselines", action="store_true")
    p.add_argument("--ablation", action="store_true",
                   help="Run βw ablation sweep")
    p.add_argument("--hiv_benchmark", action="store_true",
                   help="Run HIV benchmark to validate DINO numbers")
    p.add_argument("--synthetic", action="store_true",
                   help="Run quick synthetic smoke-test")
    p.add_argument("--all", action="store_true",
                   help="Run all datasets")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    if args.hiv_benchmark:
        run_hiv_benchmark()
        return

    if args.synthetic:
        run_synthetic_experiment()
        return

    datasets = ["hiv", "reddit", "gnutella"] if args.all else [args.dataset]

    for ds in datasets:
        run_experiment(
            dataset_name=ds,
            budgets=args.budgets,
            alpha_w=args.alpha_w,
            beta_w=args.beta_w,
            source_method=args.source_method,
            run_sis=not args.no_sis,
            run_baselines=not args.no_baselines,
            run_ablation=args.ablation,
            verbose=args.verbose,
            results_dir=args.results_dir,
        )


if __name__ == "__main__":
    main()
