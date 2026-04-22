"""
Synthetic directed graph generation and SI epidemic simulation.

Used for:
1. DeepTrace GNN pre-training / fine-tuning data
2. Unit-testing DINO and SONIC with known ground-truth sources
"""

import random
import networkx as nx
import numpy as np
from collections import deque


# ---------------------------------------------------------------------------
# Graph generators (6 types from DeepTrace paper Section IV-C)
# ---------------------------------------------------------------------------

def make_er(n=500, p=1e-3, seed=None):
    """Erdos-Renyi directed random graph."""
    G = nx.erdos_renyi_graph(n, p, directed=True, seed=seed)
    return nx.DiGraph(G)


def make_ba(n=500, m=1, seed=None):
    """Barabasi-Albert directed graph (scale-free)."""
    G = nx.scale_free_graph(n, seed=seed)
    G = nx.DiGraph(G)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return G


def make_ws(n=500, k=2, p=0.1, seed=None):
    """Watts-Strogatz small-world (converted to directed)."""
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return G.to_directed()


def make_regular(n=500, d=3, seed=None):
    """Random d-regular directed graph."""
    if n * d % 2 != 0:
        n += 1
    G = nx.random_regular_graph(d, n, seed=seed)
    return G.to_directed()


def make_sbm(n=500, n_communities=3, p_in=2e-4, p_out=5e-4, seed=None):
    """Stochastic block model directed graph."""
    sizes = [n // n_communities] * n_communities
    sizes[-1] += n - sum(sizes)
    p_matrix = [[p_in if i == j else p_out for j in range(n_communities)]
                for i in range(n_communities)]
    G = nx.stochastic_block_model(sizes, p_matrix, seed=seed, directed=True)
    return nx.DiGraph(G)


def make_random_graph(n=500, graph_type=None, seed=None):
    """Pick a random graph type if none specified."""
    types = ["er", "ba", "ws", "regular", "sbm"]
    if graph_type is None:
        graph_type = random.choice(types)
    makers = {
        "er": lambda: make_er(n, seed=seed),
        "ba": lambda: make_ba(n, seed=seed),
        "ws": lambda: make_ws(n, seed=seed),
        "regular": lambda: make_regular(n, seed=seed),
        "sbm": lambda: make_sbm(n, seed=seed),
    }
    return makers[graph_type](), graph_type


# ---------------------------------------------------------------------------
# SI Epidemic Simulation
# ---------------------------------------------------------------------------

def simulate_si(G, source=None, n_infected=None, beta=1.0, max_steps=10000, seed=None):
    """
    Simulate SI epidemic on directed graph G starting from source.

    At each step, each susceptible node with at least one infected neighbor
    is infected with probability proportional to infected in-neighbors / in-degree.
    (Equivalent to: each infected→susceptible edge fires independently with prob beta.)

    Parameters
    ----------
    G        : nx.DiGraph
    source   : int, node to start from (random if None)
    n_infected : int, stop when this many nodes are infected (None = full spread)
    beta     : float, per-edge transmission probability per step
    seed     : int

    Returns
    -------
    Gn            : nx.DiGraph, induced subgraph of infected nodes
    infection_order : list of nodes in order they were infected
    true_source   : int
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    nodes = list(G.nodes())
    if source is None:
        source = random.choice(nodes)

    infected = {source}
    infection_order = [source]
    susceptible = set(nodes) - infected

    if n_infected is None:
        n_infected = len(nodes)

    for _ in range(max_steps):
        if len(infected) >= n_infected or not susceptible:
            break

        newly_infected = []
        for node in list(susceptible):
            # Count infected in-neighbors (directed: edge u→node means u can infect node)
            infected_in_nbrs = [u for u in G.predecessors(node) if u in infected]
            if not infected_in_nbrs:
                continue
            # Each infected in-neighbor independently tries to infect
            for _ in infected_in_nbrs:
                if random.random() < beta:
                    newly_infected.append(node)
                    break

        if not newly_infected:
            # Epidemic stalled — force one random spread from infected boundary
            boundary = [n for n in susceptible
                        if any(u in infected for u in G.predecessors(n))]
            if boundary:
                newly_infected = [random.choice(boundary)]
            else:
                break

        for node in newly_infected:
            if node in susceptible:
                infected.add(node)
                infection_order.append(node)
                susceptible.discard(node)

    Gn = G.subgraph(list(infected)).copy()
    return Gn, infection_order, source


# ---------------------------------------------------------------------------
# Permitted permutation probability (DeepTrace Eq. 4 — for trees)
# ---------------------------------------------------------------------------

def permutation_prob_tree(G, sigma):
    """
    Compute P(sigma | v=sigma[0]) for a tree network using DeepTrace Eq. 4.
    sigma : list of nodes in infection order
    Returns log probability.
    """
    log_prob = 0.0
    for i in range(1, len(sigma)):
        denom = sum(G.degree(sigma[j]) for j in range(i + 1)) - 2 * (i - 1)
        if denom <= 0:
            return -np.inf
        log_prob -= np.log(denom)
    return log_prob


def compute_log_likelihood_tree(Gn, v):
    """
    Compute log P(Gn | v) for source v on a tree Gn using Eq. 4.
    Sums over all permitted permutations — tractable for small N.
    For large N, uses rumor centrality approximation.
    """
    n = Gn.number_of_nodes()
    if n > 15:
        # Approximation: use rumor centrality (proportional to |Omega(Gn|v)|)
        return _rumor_centrality_score(Gn, v)

    nodes = list(Gn.nodes())
    if v not in nodes:
        return -np.inf

    from itertools import permutations
    log_probs = []
    for perm in permutations(nodes):
        if perm[0] != v:
            continue
        # Check if perm is a valid permitted permutation
        if _is_valid_permutation(Gn, list(perm)):
            log_probs.append(permutation_prob_tree(Gn, list(perm)))

    if not log_probs:
        return -np.inf
    # log-sum-exp
    max_lp = max(log_probs)
    return max_lp + np.log(sum(np.exp(lp - max_lp) for lp in log_probs))


def _is_valid_permutation(G, sigma):
    """Check if sigma is a valid permitted permutation of G rooted at sigma[0]."""
    infected = {sigma[0]}
    for i in range(1, len(sigma)):
        node = sigma[i]
        # Node must be adjacent to at least one already-infected node
        nbrs = set(G.predecessors(node)) | set(G.successors(node))
        if not nbrs.intersection(infected):
            return False
        infected.add(node)
    return True


def _rumor_centrality_score(G, v):
    """
    Rumor centrality approximation: log(|Omega(Gn|v)|).
    Uses the message-passing algorithm from Shah & Zaman (2012).
    Proportional to number of permitted permutations rooted at v.
    """
    if v not in G:
        return -np.inf
    # BFS tree rooted at v on undirected version
    Gu = G.to_undirected()
    try:
        # Subtree sizes via DFS
        subtree_size = {}
        visited = set()
        stack = [(v, None)]
        order = []
        while stack:
            node, parent = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append((node, parent))
            for nbr in Gu.neighbors(node):
                if nbr not in visited:
                    stack.append((nbr, node))

        # Compute subtree sizes bottom-up
        for node, _ in reversed(order):
            subtree_size[node] = 1 + sum(
                subtree_size.get(c, 0)
                for c in Gu.neighbors(node)
                if subtree_size.get(c, 0) > 0 and
                   order.index((c, node)) > order.index((node, _))
            )

        n = G.number_of_nodes()
        # Rumor centrality: n! / prod(subtree_size[u]) for all u
        log_rc = sum(np.log(range(1, n + 1)))  # log(n!)
        for node in G.nodes():
            s = subtree_size.get(node, 1)
            if s > 0:
                log_rc -= np.log(s)
        return log_rc
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Batch data generation for GNN training
# ---------------------------------------------------------------------------

def generate_training_batch(
    n_graphs=100,
    n_nodes=500,
    n_infected_range=(50, 200),
    exact_labels=False,
    seed=None,
):
    """
    Generate a batch of (Gn, node_features, log_likelihood_labels) for DeepTrace training.

    Parameters
    ----------
    n_graphs       : int, number of graphs to generate
    n_nodes        : int, nodes in underlying graph G
    n_infected_range : (min, max) infected nodes in epidemic subgraph
    exact_labels   : bool, use exact P(Gn|v) (slow, for fine-tuning only)
    seed           : int

    Yields
    ------
    (Gn, features_dict, labels_dict) per graph
    """
    rng = np.random.default_rng(seed)

    for i in range(n_graphs):
        s = int(rng.integers(0, 10000))
        G, gtype = make_random_graph(n=n_nodes, seed=s)

        n_inf = int(rng.integers(*n_infected_range))
        n_inf = min(n_inf, G.number_of_nodes())

        Gn, infection_order, true_source = simulate_si(
            G, n_infected=n_inf, seed=s
        )

        if Gn.number_of_nodes() < 5:
            continue

        # Pass full graph G so r_hat uses correct total neighbor count
        features = compute_deeptrace_features(Gn, G=G)

        # Compute labels
        labels = {}
        for v in Gn.nodes():
            if exact_labels and Gn.number_of_nodes() <= 50:
                labels[v] = compute_log_likelihood_tree(Gn, v)
            else:
                labels[v] = _rumor_centrality_score(Gn, v)

        yield Gn, features, labels, true_source


def compute_deeptrace_features(Gn, G=None):
    """
    Compute DeepTrace node features h0_v = [1, r_hat(v), r_check(v)].

    r_hat(v)   = infected_neighbors(v) / total_neighbors(v) in full graph G
                 (DeepTrace paper definition)
                 If G not provided, approximated as degree_Gn / degree_Gn
                 (all neighbors visible are infected)
    r_check(v) = boundary_distance(v) / max_boundary_distance
                 boundary = nodes in Gn that have at least one neighbor NOT in Gn

    Returns
    -------
    features : dict {node: np.array([1, r_hat, r_check])}
    """
    nodes = list(Gn.nodes())
    gn_node_set = set(nodes)

    # -------------------------------------------------------------------
    # r_hat(v): fraction of v's neighbors in full G that are infected (in Gn)
    # If full G is provided, use it exactly (paper definition)
    # If not, use undirected degree in Gn as numerator and denominator
    # (all visible neighbors are infected since we only see Gn)
    # -------------------------------------------------------------------
    Gu = Gn.to_undirected()

    r_hat = {}
    if G is not None:
        Gu_full = G.to_undirected()
        for v in nodes:
            total_nbrs = set(Gu_full.neighbors(v))
            infected_nbrs = total_nbrs & gn_node_set
            denom = len(total_nbrs)
            r_hat[v] = len(infected_nbrs) / denom if denom > 0 else 1.0
    else:
        # Fallback: all visible neighbors are infected → r_hat = 1
        # unless node is isolated in Gn
        for v in nodes:
            deg = Gu.degree(v)
            r_hat[v] = 1.0 if deg > 0 else 0.0

    # -------------------------------------------------------------------
    # Boundary: nodes in Gn with at least one neighbor NOT in Gn
    # If G provided, use it to find real boundary
    # If not, leaf nodes of Gn (degree=1 in undirected Gn) are boundary
    # -------------------------------------------------------------------
    boundary = set()
    if G is not None:
        Gu_full = G.to_undirected()
        for v in nodes:
            nbrs = set(Gu_full.neighbors(v))
            if nbrs - gn_node_set:  # has at least one non-infected neighbor
                boundary.add(v)
    else:
        for v in nodes:
            if Gu.degree(v) <= 1:
                boundary.add(v)

    if not boundary:
        min_deg = min(Gu.degree(v) for v in nodes)
        boundary = {v for v in nodes if Gu.degree(v) == min_deg}

    # BFS inward from boundary to compute boundary distance
    b_dist = {v: float("inf") for v in nodes}
    queue = deque()
    for v in boundary:
        b_dist[v] = 0
        queue.append(v)

    while queue:
        node = queue.popleft()
        for nbr in Gu.neighbors(node):
            if b_dist[nbr] == float("inf"):
                b_dist[nbr] = b_dist[node] + 1
                queue.append(nbr)

    max_b = max((d for d in b_dist.values() if d != float("inf")), default=1)
    if max_b == 0:
        max_b = 1

    features = {}
    for v in nodes:
        r_check = b_dist[v] / max_b if b_dist[v] != float("inf") else 1.0
        features[v] = np.array([1.0, r_hat[v], r_check], dtype=np.float32)

    return features
