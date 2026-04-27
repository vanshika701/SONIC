import sys
import os
import time
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import Namespace
import multiprocessing as mp

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from data.loaders import load_dataset
from data.synthetic import simulate_si
from evaluation.metrics import evaluate_method, delta_rho
from main import run_method
from algorithms.spp import spectral_radius
from simulation.sis import simulate_sis

def run_single_trial(G, immunized_nodes, beta, delta, I0, T, seed):
    """Worker function for parallel SIS trials."""
    # We only need the curve for one trial here, then we average later.
    # But simulate_sis already does n_trials. Let's make a version that does 1 trial.
    rng = np.random.default_rng(seed)
    
    # Remove immunized nodes
    G_sim = G.copy()
    if immunized_nodes:
        G_sim.remove_nodes_from([v for v in immunized_nodes if v in G_sim])
    
    nodes = list(G_sim.nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros(T)
    
    node_idx = {v: i for i, v in enumerate(nodes)}
    in_neighbors = {v: list(G_sim.predecessors(v)) for v in nodes}
    
    state = rng.random(n) < I0
    curve = np.zeros(T)
    
    for t in range(T):
        curve[t] = state.sum()
        if curve[t] == 0:
            break
            
        new_state = state.copy()
        for i, v in enumerate(nodes):
            if state[i]:
                if rng.random() < delta:
                    new_state[i] = False
            else:
                inf_nbrs = [node_idx[u] for u in in_neighbors[v] if state[node_idx[u]]]
                if inf_nbrs:
                    p_infect = 1.0 - (1.0 - beta) ** len(inf_nbrs)
                    if rng.random() < p_infect:
                        new_state[i] = True
        state = new_state
    return curve

def parallel_simulate_sis(G, immunized_nodes, n_trials=20, T=200, beta=0.03, delta=0.1, I0=0.95):
    """Run SIS trials in parallel."""
    print(f"  Running {n_trials} SIS trials in parallel...")
    seeds = np.random.randint(0, 1000000, size=n_trials)
    
    pool = mp.Pool(processes=min(mp.cpu_count(), n_trials))
    tasks = [(G, immunized_nodes, beta, delta, I0, T, seed) for seed in seeds]
    curves = pool.starmap(run_single_trial, tasks)
    pool.close()
    pool.join()
    
    all_curves = np.array(curves)
    mean_curve = all_curves.mean(axis=0)
    I_T = mean_curve[-1]
    
    # Containment time
    threshold = 0.01 * G.number_of_nodes()
    contain_times = []
    for trial_curve in all_curves:
        t_c = np.where(trial_curve <= threshold)[0]
        if len(t_c) > 0:
            contain_times.append(t_c[0])
        else:
            contain_times.append(T)
    T_contain = np.mean(contain_times)
    
    return mean_curve, I_T, T_contain

def main():
    print("=== Fast Reddit Run: DeepTrace + SPP (SONIC) ===")
    
    # 1. Load Dataset
    print("1. Loading Reddit dataset...")
    G = load_dataset("reddit")
    if G is None:
        print("Error: Reddit dataset not found.")
        return
    
    n, m = G.number_of_nodes(), G.number_of_edges()
    rho0 = spectral_radius(G)
    print(f"   Nodes: {n}, Edges: {m}, Initial ρ: {rho0:.4f}")
    
    # 2. Initial Epidemic Spread (SI)
    print("2. Simulating initial epidemic (SI)...")
    source = max(G.nodes(), key=lambda v: G.in_degree(v))
    Gn, _, _ = simulate_si(G, source=source, beta=0.3, max_steps=10)
    print(f"   Infected nodes in Gn: {Gn.number_of_nodes()}")
    
    # 3. Immunization (DeepTrace + SPP)
    budget = 100
    print(f"3. Running SONIC (DeepTrace + SPP) with budget k={budget}...")
    args = Namespace(
        method="sonic",
        source_method="deeptrace",
        K_sources=10,
        ppr_alpha=0.15,
        adaptive=False,
        quiet=False
    )
    t0 = time.time()
    L = run_method(G, Gn, budget, args)
    runtime = time.time() - t0
    print(f"   Immunization completed in {runtime:.2f}s")
    
    # 4. Evaluation
    print("4. Evaluating results...")
    dr, rho_b, rho_a = delta_rho(G, L)
    print(f"   Δρ: {dr:.4f}  ({rho_b:.4f} -> {rho_a:.4f})")
    
    # Parallel SIS
    curve, I_T, T_contain = parallel_simulate_sis(G, L, n_trials=mp.cpu_count())
    print(f"   Final Infected (I_T): {I_T:.2f}")
    print(f"   Containment Time (T_contain): {T_contain:.2f}")
    
    # 5. Save Results
    res_dir = "results_fast"
    os.makedirs(res_dir, exist_ok=True)
    results = {
        "dataset": "reddit",
        "method": "sonic_deeptrace",
        "budget": budget,
        "delta_rho": float(dr),
        "rho_before": float(rho_b),
        "rho_after": float(rho_a),
        "runtime": runtime,
        "I_T": float(I_T),
        "T_contain": float(T_contain),
        "curve": curve.tolist()
    }
    with open(f"{res_dir}/reddit_sonic_deeptrace.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 6. Generate Graphs
    print("6. Generating graphs...")
    sns.set_theme(style="whitegrid")
    
    # Infection Curve
    plt.figure(figsize=(10, 6))
    plt.plot(curve, linewidth=2, color='tab:blue', label='SONIC (DeepTrace)')
    plt.axhline(y=0.01*n, color='r', linestyle='--', label='1% Threshold')
    plt.xlabel('Time Steps')
    plt.ylabel('Infected Nodes')
    plt.title(f'Reddit Epidemic Containment (k={budget}, DeepTrace+SPP)')
    plt.legend()
    plt.savefig(f"{res_dir}/reddit_infection_curve.png", dpi=300)
    plt.close()
    
    # Degree Distribution (Impact)
    plt.figure(figsize=(10, 6))
    in_degrees = [G.in_degree(v) for v in G.nodes()]
    removed_in_degrees = [G.in_degree(v) for v in L if G.has_node(v)]
    sns.histplot(in_degrees, bins=50, color='gray', alpha=0.3, label='Full Network', log_scale=(True, True))
    sns.histplot(removed_in_degrees, bins=20, color='red', alpha=0.8, label='Removed Nodes', log_scale=(True, True))
    plt.xlabel('In-Degree')
    plt.ylabel('Frequency (log)')
    plt.title('Degree Distribution of Removed Nodes vs. Full Network')
    plt.legend()
    plt.savefig(f"{res_dir}/reddit_degree_impact.png", dpi=300)
    plt.close()
    
    print(f"Done! Results and graphs saved in '{res_dir}' directory.")

if __name__ == "__main__":
    main()
