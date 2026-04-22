"""
SONIC: Source-Oriented Network Immunization and Containment
CLI entry point.

Usage examples:
    python main.py --dataset hiv --budget 100
    python main.py --dataset hiv --budget 100 --alpha_w 0.5 --beta_w 0.5
    python main.py --dataset hiv --budget 100 --method sonic --source_method rumor
    python main.py --dataset gnutella --budget 50 --method dino
    python main.py --dataset hiv --budget 100 --ablation
    python main.py --hiv_benchmark
    python main.py --synthetic
"""

import argparse
import sys
import time
import json
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="SONIC: Source-Oriented Network Immunization and Containment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset
    p.add_argument("--dataset", default="hiv",
                   choices=["hiv", "reddit", "gnutella", "synthetic"],
                   help="Network dataset to use (default: hiv)")

    # Budget
    p.add_argument("--budget", type=int, default=100,
                   help="Immunization budget k (default: 100)")
    p.add_argument("--budgets", nargs="+", type=int,
                   help="Multiple budgets for sweep")

    # Method
    p.add_argument("--method", default="sonic",
                   choices=["sonic", "dino", "source_only", "degree",
                            "katz", "random", "betweenness"],
                   help="Immunization method (default: sonic)")

    # SONIC weights
    p.add_argument("--alpha_w", type=float, default=0.5,
                   help="Weight for structural term αw (default: 0.5)")
    p.add_argument("--beta_w", type=float, default=0.5,
                   help="Weight for SourceRisk term βw (default: 0.5)")
    p.add_argument("--auto_weights", action="store_true",
                   help="Use entropy-gated weight auto-tuning (novelty)")
    p.add_argument("--adaptive", action="store_true",
                   help="Recompute source posterior adaptively (Eq. 10)")

    # Source inference
    p.add_argument("--source_method", default="rumor",
                   choices=["rumor", "deeptrace", "auto"],
                   help="Source inference method (default: rumor)")
    p.add_argument("--K_sources", type=int, default=10,
                   help="Top-K sources for E-PPR (default: 10)")
    p.add_argument("--ppr_alpha", type=float, default=0.15,
                   help="PPR teleport probability (default: 0.15)")

    # Evaluation
    p.add_argument("--no_sis", action="store_true",
                   help="Skip SIS simulation (faster)")
    p.add_argument("--sis_trials", type=int, default=20,
                   help="SIS Monte Carlo trials (default: 20)")

    # Experiments
    p.add_argument("--ablation", action="store_true",
                   help="Run βw ablation sweep")
    p.add_argument("--hiv_benchmark", action="store_true",
                   help="Reproduce DINO paper HIV benchmark (Table 2)")
    p.add_argument("--synthetic", action="store_true",
                   help="Quick smoke-test on synthetic BA graph")

    # Output
    p.add_argument("--results_dir", default="results",
                   help="Directory to save results (default: results)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress verbose output")

    return p.parse_args()


def run_sonic_method(G, Gn, k, args):
    """Dispatch to the requested immunization method."""
    from algorithms.sonic import sonic
    from algorithms.dino import dino
    from experiments.baselines import (
        degree_immunization, katz_immunization,
        random_immunization, betweenness_immunization
    )

    verbose = not args.quiet

    # Load DeepTrace model if requested
    deeptrace_model = None
    if args.source_method in ("deeptrace", "auto"):
        from gnn.train import get_or_train_deeptrace
        if verbose:
            print("  Loading DeepTrace model...")
        deeptrace_model = get_or_train_deeptrace()

    if args.method == "sonic":
        alpha_w = args.alpha_w
        beta_w = args.beta_w
        assert abs(alpha_w + beta_w - 1.0) < 1e-6, \
            f"alpha_w + beta_w must equal 1, got {alpha_w + beta_w}"

        L, dr = sonic(
            G, Gn, k,
            alpha_w=alpha_w, beta_w=beta_w,
            source_method=args.source_method,
            deeptrace_model=deeptrace_model,
            K_sources=args.K_sources,
            ppr_alpha=args.ppr_alpha,
            adaptive=args.adaptive,
            auto_weights=args.auto_weights,
            return_delta_rho=True,
            verbose=verbose,
        )
        return L

    elif args.method == "dino":
        return dino(G, k, verbose=verbose)

    elif args.method == "source_only":
        L, _ = sonic(G, Gn, k, alpha_w=0.0, beta_w=1.0,
                     source_method=args.source_method,
                     deeptrace_model=deeptrace_model,
                     return_delta_rho=False, verbose=verbose)
        return L

    elif args.method == "degree":
        return degree_immunization(G, k)

    elif args.method == "katz":
        return katz_immunization(G, k)

    elif args.method == "random":
        return random_immunization(G, k)

    elif args.method == "betweenness":
        return betweenness_immunization(G, k)

    else:
        raise ValueError(f"Unknown method: {args.method}")


def main():
    args = parse_args()
    verbose = not args.quiet

    # --- Special modes ---
    if args.hiv_benchmark:
        from experiments.run_all import run_hiv_benchmark
        run_hiv_benchmark(budgets=[100, 200, 300], verbose=verbose)
        return 0

    if args.synthetic:
        from experiments.run_all import run_synthetic_experiment
        run_synthetic_experiment(verbose=verbose)
        return 0

    if args.ablation:
        from data.loaders import load_dataset
        from data.synthetic import simulate_si
        from experiments.ablation import ablation_beta_sweep, print_ablation_table

        G = load_dataset(args.dataset)
        source = max(G.nodes(), key=lambda v: G.in_degree(v))
        Gn, _, _ = simulate_si(G, source=source, beta=0.3, max_steps=10)
        if Gn.number_of_nodes() < 2:
            Gn = G

        k = args.budget
        print(f"\nRunning βw ablation on {args.dataset.upper()} (k={k})...")
        df = ablation_beta_sweep(G, Gn, k=k,
                                  source_method=args.source_method,
                                  verbose=verbose)
        print_ablation_table(df, title=f"{args.dataset} βw Ablation")

        Path(args.results_dir).mkdir(exist_ok=True)
        out = Path(args.results_dir) / f"{args.dataset}_ablation.csv"
        df.to_csv(out, index=False)
        print(f"Saved → {out}")
        return 0

    # --- Main experiment ---
    from data.loaders import load_dataset
    from data.synthetic import simulate_si
    from evaluation.metrics import evaluate_method

    G = load_dataset(args.dataset)
    n, m = G.number_of_nodes(), G.number_of_edges()

    if verbose:
        print(f"\nDataset : {args.dataset.upper()}")
        print(f"Graph   : |V|={n}  |E|={m}")
        print(f"Method  : {args.method}  (αw={args.alpha_w}  βw={args.beta_w})")

    # Build epidemic subgraph
    source = max(G.nodes(), key=lambda v: G.in_degree(v))
    Gn, order, _ = simulate_si(G, source=source, beta=0.3, max_steps=10)
    if Gn.number_of_nodes() < 2:
        Gn = G
    if verbose:
        print(f"Gn      : |V|={Gn.number_of_nodes()}  |E|={Gn.number_of_edges()}")

    budgets = args.budgets if args.budgets else [args.budget]
    all_metrics = []

    for k in budgets:
        if verbose:
            print(f"\n--- k={k} ---")

        t0 = time.time()
        L = run_sonic_method(G, Gn, k, args)
        elapsed = time.time() - t0

        metrics = evaluate_method(
            G, L,
            method_name=args.method,
            run_sis=not args.no_sis,
            verbose=verbose,
        )
        metrics["runtime_s"] = elapsed
        all_metrics.append(metrics)

        if verbose:
            print(f"  Runtime: {elapsed:.2f}s")

    # Save results
    Path(args.results_dir).mkdir(exist_ok=True)
    out_path = Path(args.results_dir) / f"{args.dataset}_{args.method}_k{budgets[-1]}.json"

    def _to_py(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        return x

    saveable = [{k2: _to_py(v) for k2, v in m.items() if k2 != "curve"}
                for m in all_metrics]
    with open(out_path, "w") as f:
        json.dump(saveable, f, indent=2)

    print(f"\nResults saved → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
