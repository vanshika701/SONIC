"""
SONIC: Source-Oriented Network Immunization and Containment
CLI entry point — Spectral Path-Product (SPP) Edition.

Usage examples:
    python main.py --dataset hiv --budget 100
    python main.py --dataset hiv --budget 100 --method sonic --source_method rumor
    python main.py --dataset gnutella --budget 50 --method spp
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
        description="SONIC: Source-Oriented Network Immunization and Containment "
                    "(Spectral Path-Product Edition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset
    p.add_argument("--dataset", default="hiv",
                   choices=["hiv", "reddit", "gnutella", "synthetic", "enron"],
                   help="Network dataset to use (default: hiv)")

    # Budget
    p.add_argument("--budget", type=int, default=100,
                   help="Immunisation budget k (default: 100)")
    p.add_argument("--budgets", nargs="+", type=int,
                   help="Multiple budgets for sweep")

    # Method
    p.add_argument("--method", default="sonic",
                   choices=["sonic", "spp", "source_only", "degree",
                            "katz", "random", "betweenness"],
                   help="Immunisation method (default: sonic)")

    # Source inference
    p.add_argument("--source_method", default="rumor",
                   choices=["rumor", "deeptrace", "auto"],
                   help="Source inference method (default: rumor)")
    p.add_argument("--K_sources", type=int, default=10,
                   help="Top-K sources for E-PPR (default: 10)")
    p.add_argument("--ppr_alpha", type=float, default=0.15,
                   help="PPR teleport probability (default: 0.15)")
    p.add_argument("--adaptive", action="store_true",
                   help="Recompute source posterior adaptively after each removal")

    # Evaluation
    p.add_argument("--no_sis", action="store_true",
                   help="Skip SIS simulation (faster)")
    p.add_argument("--sis_trials", type=int, default=20,
                   help="SIS Monte Carlo trials (default: 20)")

    # Experiments
    p.add_argument("--ablation", action="store_true",
                   help="Run K_sources ablation sweep")
    p.add_argument("--hiv_benchmark", action="store_true",
                   help="Run SPP benchmark on HIV network")
    p.add_argument("--synthetic", action="store_true",
                   help="Quick smoke-test on synthetic BA graph")

    # Output
    p.add_argument("--results_dir", default="results",
                   help="Directory to save results (default: results)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress verbose output")

    return p.parse_args()


def run_method(G, Gn, k, args):
    """Dispatch to the requested immunisation method."""
    from algorithms.sonic import sonic
    from algorithms.spp import spp_selection
    from experiments.baselines import (
        degree_immunization, hits_authority_immunization,
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

    # ----------------------------------------------------------------
    # SONIC / SPP  — full three-phase pipeline
    # ----------------------------------------------------------------
    if args.method in ("sonic", "spp"):
        L, dr = sonic(
            G, Gn, k,
            source_method=args.source_method,
            deeptrace_model=deeptrace_model,
            K_sources=args.K_sources,
            ppr_alpha=args.ppr_alpha,
            adaptive=args.adaptive,
            return_delta_rho=True,
            verbose=verbose,
        )
        return L

    # ----------------------------------------------------------------
    # Source-only (Phase 1+2 only, skip Phase 3 KSCC optimisation)
    # ----------------------------------------------------------------
    elif args.method == "source_only":
        from algorithms.source_inference import infer_source_posterior
        from algorithms.eppr import source_risk

        pi = infer_source_posterior(
            Gn=Gn, model=deeptrace_model,
            method=args.source_method, G=G
        )
        tau = source_risk(G, pi, K=args.K_sources, alpha=args.ppr_alpha)
        # Select top-k by τ directly
        sorted_nodes = sorted(tau, key=tau.get, reverse=True)
        return sorted_nodes[:k]

    # ----------------------------------------------------------------
    # Structural baselines
    # ----------------------------------------------------------------
    elif args.method == "degree":
        return degree_immunization(G, k)

    elif args.method == "katz":
        return hits_authority_immunization(G, k)

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
        from algorithms.spp import spp_selection, spectral_radius
        from algorithms.source_inference import infer_source_posterior
        from algorithms.eppr import source_risk
        from data.loaders import load_dataset

        print("\n=== Spectral Path-Product (SPP) HIV Benchmark ===")
        G = load_dataset("hiv")
        if G is None:
            print("[error] HIV dataset not found — see README for data paths.")
            return 1

        print(f"Graph: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
        rho0 = spectral_radius(G)
        print(f"Initial ρ = {rho0:.4f}")

        # Build a minimal Gn for source inference
        from data.synthetic import simulate_si
        source = max(G.nodes(), key=lambda v: G.in_degree(v))
        Gn, _, _ = simulate_si(G, source=source, beta=0.3, max_steps=10)
        if Gn.number_of_nodes() < 2:
            Gn = G

        pi = infer_source_posterior(Gn=Gn, method="rumor", G=G)
        tau = source_risk(G, pi, K=10, alpha=0.15)

        print(f"{'Budget':>8} | {'Δρ':>8} | {'ρ_final':>8}")
        print("-" * 32)
        for k in (100, 300, 500):
            L, delta = spp_selection(G, k, tau, return_delta_rho=True)
            print(f"{k:>8} | {delta:>8.2f} | {rho0 - delta:>8.4f}")
        return 0

    if args.synthetic:
        from experiments.run_all import run_synthetic_experiment
        run_synthetic_experiment(verbose=verbose)
        return 0

    if args.ablation:
        from data.loaders import load_dataset
        from data.synthetic import simulate_si
        from algorithms.sonic import sonic_sweep

        G = load_dataset(args.dataset)
        source = max(G.nodes(), key=lambda v: G.in_degree(v))
        Gn, _, _ = simulate_si(G, source=source, beta=0.3, max_steps=10)
        if Gn.number_of_nodes() < 2:
            Gn = G

        k = args.budget
        print(f"\nRunning K_sources ablation on "
              f"{args.dataset.upper()} (k={k}, SPP)...")
        results = sonic_sweep(G, Gn, k,
                              source_method=args.source_method,
                              verbose=verbose)

        Path(args.results_dir).mkdir(exist_ok=True)
        out = Path(args.results_dir) / f"{args.dataset}_spp_ablation.csv"
        import csv
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["K_sources", "delta_rho"])
            for ks, _L, dr in results:
                w.writerow([ks, dr])
        print(f"Saved → {out}")
        return 0

    # --- Main experiment ---
    from data.loaders import load_dataset
    from data.synthetic import simulate_si
    from evaluation.metrics import evaluate_method

    G = load_dataset(args.dataset)
    if G is None:
        print(f"\n[error] dataset '{args.dataset}' not loaded "
              "— see README for data paths.")
        return 1

    n, m = G.number_of_nodes(), G.number_of_edges()

    if verbose:
        print(f"\nDataset  : {args.dataset.upper()}")
        print(f"Graph    : |V|={n}  |E|={m}")
        print(f"Method   : {args.method}")
        print(f"Pipeline : DeepTrace/RumorCentrality "
              "→ E-PPR → Spectral Path-Product (SPP) Optimization")

    # Build epidemic subgraph
    source = max(G.nodes(), key=lambda v: G.in_degree(v))
    Gn, order, _ = simulate_si(G, source=source, beta=0.3, max_steps=10)
    if Gn.number_of_nodes() < 2:
        Gn = G
    if verbose:
        print(f"Gn       : |V|={Gn.number_of_nodes()}  "
              f"|E|={Gn.number_of_edges()}")

    budgets = args.budgets if args.budgets else [args.budget]
    all_metrics = []

    for k in budgets:
        if verbose:
            print(f"\n--- k={k} ---")

        t0 = time.time()
        L = run_method(G, Gn, k, args)
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
            print(f"  Runtime : {elapsed:.2f}s")

    # Save results
    Path(args.results_dir).mkdir(exist_ok=True)
    out_path = (
        Path(args.results_dir)
        / f"{args.dataset}_{args.method}_k{budgets[-1]}.json"
    )

    def _to_py(x):
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    saveable = [
        {k2: _to_py(v) for k2, v in m.items() if k2 != "curve"}
        for m in all_metrics
    ]
    with open(out_path, "w") as f:
        json.dump(saveable, f, indent=2)

    print(f"\nResults saved → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
