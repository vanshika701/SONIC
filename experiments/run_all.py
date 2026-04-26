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
from algorithms.spp import spectral_radius
from algorithms.sonic import sonic, sonic_sweep
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
    print(f"  source_method={source_method}  pipeline=SPP")
    print(f"{'='*70}")

    # Load graph
    G = load_dataset(dataset_name)
    if G is None:
        print(f"  [skip] no graph for {dataset_name} (missing data file)")
        return []
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

    # Create a unique timestamped run directory
    import datetime
    import os
    import io
    from contextlib import redirect_stdout
    import subprocess
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(results_dir) / f"{dataset_name}_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = run_dir / "results.json"
    table_path = run_dir / "summary_table.txt"

    # Print table to both console and text file
    print_results_table(all_metrics, budgets=budgets)
    with open(table_path, "w") as f:
        with redirect_stdout(f):
            print_results_table(all_metrics, budgets=budgets)

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
    print(f"\n  Results & Table saved → {run_dir}/")
    
    # Auto-generate IEEE plots for this run
    print("  Auto-generating IEEE plots...")
    try:
        subprocess.run(["python", "generate_ieee_plots.py", str(out_path), str(run_dir)], check=True)
    except Exception as e:
        print(f"  Failed to generate plots: {e}")

    # Ablation
    if run_ablation:
        print("\n--- K_sources Ablation ---")
        abl_results = sonic_sweep(G, Gn, k=budgets[-1],
                                  source_method=source_method,
                                  verbose=True)
        abl_path = Path(results_dir) / f"{dataset_name}_spp_ablation.csv"
        import csv
        with open(abl_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["K_sources", "delta_rho"])
            for ks, _L, dr in abl_results:
                w.writerow([ks, dr])
        print(f"  Ablation saved → {abl_path}")

    return all_metrics


# ---------------------------------------------------------------------------
# HIV benchmark (validate against DINO paper Table 2)
# ---------------------------------------------------------------------------

def run_hiv_benchmark(budgets=None, verbose=True):
    """
    Run SPP on HIV network for multiple budgets.
    """
    if budgets is None:
        budgets = [100, 200, 300]

    print("\n=== HIV Benchmark (SPP) ===")
    from algorithms.source_inference import infer_source_posterior
    from algorithms.eppr import source_risk
    from algorithms.spp import spp_selection, spectral_radius
    from data.loaders import load_dataset
    from data.synthetic import simulate_si

    G = load_dataset("hiv")
    if G is None:
        print("[skip] HIV dataset not found")
        return {}

    source = max(G.nodes(), key=lambda v: G.in_degree(v))
    Gn, _, _ = simulate_si(G, source=source, beta=0.3, max_steps=10)
    if Gn.number_of_nodes() < 2:
        Gn = G

    pi = infer_source_posterior(Gn=Gn, method="rumor", G=G)
    tau = source_risk(G, pi, K=10, alpha=0.15)

    rho0 = spectral_radius(G)
    print(f"Initial ρ = {rho0:.4f}")
    print(f"{'Budget':>8} | {'Δρ':>8} | {'ρ_final':>8}")
    print("-" * 32)
    out = {}
    for k in budgets:
        _, delta = spp_selection(G, k, tau, return_delta_rho=True)
        print(f"{k:>8} | {delta:>8.2f} | {rho0 - delta:>8.4f}")
        out[k] = (delta, rho0, rho0 - delta)
    return out


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

    L, dr = sonic(G, Gn, k, source_method="rumor",
                  return_delta_rho=True, verbose=verbose)

    print(f"  SPP Δρ={dr:.4f}  |L|={len(L)}")
    return L, dr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SONIC/SPP Experiment Runner")
    p.add_argument("--dataset", default="hiv",
                   choices=["hiv", "reddit", "gnutella", "synthetic", "enron"],
                   help="Dataset to run experiments on")
    p.add_argument("--budgets", nargs="+", type=int, default=[50, 100, 200],
                   help="Immunization budgets k")
    p.add_argument("--source_method", default="rumor",
                   choices=["rumor", "deeptrace", "auto"])
    p.add_argument("--no_sis", action="store_true",
                   help="Skip SIS simulation (faster)")
    p.add_argument("--no_baselines", action="store_true")
    p.add_argument("--ablation", action="store_true",
                   help="Run K_sources ablation sweep")
    p.add_argument("--hiv_benchmark", action="store_true",
                   help="Run HIV benchmark")
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

    datasets = ["hiv", "reddit", "gnutella", "enron"] if args.all else [args.dataset]

    for ds in datasets:
        run_experiment(
            dataset_name=ds,
            budgets=args.budgets,
            source_method=args.source_method,
            run_sis=not args.no_sis,
            run_baselines=not args.no_baselines,
            run_ablation=args.ablation,
            verbose=args.verbose,
            results_dir=args.results_dir,
        )


if __name__ == "__main__":
    main()
