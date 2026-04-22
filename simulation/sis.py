"""
Discrete-time SIS epidemic simulator.

Parameters from DINO paper (Section 5.2):
  I0   = 0.95  (initial infection rate)
  beta = 0.03  (transmission probability per edge per step)
  Recovery rate delta = 0.1 (chosen to allow epidemic to propagate for meaningful steps)
"""

import numpy as np
import networkx as nx
from collections import defaultdict


def simulate_sis(
    G,
    immunized_nodes=None,
    beta=0.03,
    delta=0.1,
    I0=0.95,
    T=200,
    n_trials=20,
    seed=None,
):
    """
    Run SIS epidemic simulation on G with given immunized nodes removed.

    At each timestep:
      S → I with probability 1 - (1-beta)^(infected_neighbors)
      I → S with probability delta

    Parameters
    ----------
    G               : nx.DiGraph
    immunized_nodes : list of nodes to remove before simulation
    beta            : float, per-edge transmission probability
    delta           : float, recovery probability per step
    I0              : float, initial infection probability per node
    T               : int, simulation timesteps
    n_trials        : int, Monte Carlo trials
    seed            : int

    Returns
    -------
    infected_curve  : np.array (T,), mean infected count at each timestep
    I_T             : float, mean infected at final timestep T
    T_contain       : float or None, mean first timestep when infected < 1% of |V|
                      (None if epidemic never contained within T steps)
    """
    rng = np.random.default_rng(seed)

    # Remove immunized nodes
    if immunized_nodes:
        G_sim = G.copy()
        G_sim.remove_nodes_from([v for v in immunized_nodes if v in G_sim])
    else:
        G_sim = G

    nodes = list(G_sim.nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros(T), 0.0, 0

    node_idx = {v: i for i, v in enumerate(nodes)}
    threshold = max(1, int(0.01 * n))  # 1% containment threshold

    # Precompute adjacency for fast lookup
    # For directed: node v can be infected by its in-neighbors
    in_neighbors = {v: list(G_sim.predecessors(v)) for v in nodes}

    all_curves = np.zeros((n_trials, T))
    contain_times = []

    for trial in range(n_trials):
        state = rng.random(n) < I0  # True = Infected
        contained_at = None

        for t in range(T):
            infected_count = state.sum()
            all_curves[trial, t] = infected_count

            if contained_at is None and infected_count <= threshold:
                contained_at = t

            if infected_count == 0:
                # Epidemic died out — fill remaining steps with 0
                all_curves[trial, t:] = 0
                if contained_at is None:
                    contained_at = t
                break

            new_state = state.copy()

            for i, v in enumerate(nodes):
                if state[i]:
                    # Infected → Susceptible with prob delta
                    if rng.random() < delta:
                        new_state[i] = False
                else:
                    # Susceptible → Infected
                    inf_nbrs = [node_idx[u] for u in in_neighbors[v]
                                if node_idx[u] < n and state[node_idx[u]]]
                    if inf_nbrs:
                        # Probability of infection = 1 - (1-beta)^k
                        k = len(inf_nbrs)
                        p_infect = 1.0 - (1.0 - beta) ** k
                        if rng.random() < p_infect:
                            new_state[i] = True

            state = new_state

        contain_times.append(contained_at)

    infected_curve = all_curves.mean(axis=0)
    I_T = float(infected_curve[-1])
    T_contain = (
        float(np.mean([t for t in contain_times if t is not None]))
        if any(t is not None for t in contain_times)
        else None
    )

    return infected_curve, I_T, T_contain


def compare_methods_sis(G, methods_immunized, beta=0.03, delta=0.1, I0=0.95,
                         T=200, n_trials=10, seed=42):
    """
    Compare multiple immunization strategies via SIS simulation.

    Parameters
    ----------
    G                 : nx.DiGraph
    methods_immunized : dict {method_name: list_of_immunized_nodes}

    Returns
    -------
    results : dict {method_name: {'curve': np.array, 'I_T': float, 'T_contain': float}}
    """
    results = {}
    for name, immunized in methods_immunized.items():
        curve, I_T, T_contain = simulate_sis(
            G, immunized_nodes=immunized,
            beta=beta, delta=delta, I0=I0, T=T, n_trials=n_trials, seed=seed
        )
        results[name] = {
            "curve": curve,
            "I_T": I_T,
            "T_contain": T_contain if T_contain is not None else T,
        }
    return results
