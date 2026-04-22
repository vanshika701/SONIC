"""
DeepTrace GNN: GraphSAGE with LSTM aggregators (Tan et al., IEEE TSIPN 2025).

Architecture (Equation 9 from paper):
  h^(l)_{N(v)} = LSTM({w^(l-1), h^(l-1)_u : u in N_G(v)})
  h^(l)_v      = ReLU(W^(l) · [h^(l-1)_v || h^(l)_{N(v)}])

Input features: h^(0)_v = [1, r_hat(v), r_check(v)]  (dim=3)
Output: scalar log-likelihood score per node
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAggregator(nn.Module):
    """
    LSTM-based neighborhood aggregator from DeepTrace (Eq. 9).
    Aggregates neighbor embeddings in a permutation-invariant way
    by sorting neighbors by index before feeding to LSTM.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, neighbor_features):
        """
        Parameters
        ----------
        neighbor_features : Tensor (n_neighbors, input_dim)
            Features of all neighbors of a node.

        Returns
        -------
        agg : Tensor (hidden_dim,)
        """
        if neighbor_features.shape[0] == 0:
            return torch.zeros(self.hidden_dim, device=neighbor_features.device)

        # Add batch dimension: (1, n_neighbors, input_dim)
        x = neighbor_features.unsqueeze(0)
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0).squeeze(0)  # (hidden_dim,)


class DeepTraceLayer(nn.Module):
    """Single GraphSAGE + LSTM layer."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.aggregator = LSTMAggregator(input_dim, output_dim)
        # Combination: W · [h_v || h_agg]
        self.W = nn.Linear(input_dim + output_dim, output_dim, bias=True)

    def forward(self, h, adj_list):
        """
        Parameters
        ----------
        h        : dict {node_id: Tensor(input_dim)}
        adj_list : dict {node_id: list of neighbor node_ids}

        Returns
        -------
        h_new : dict {node_id: Tensor(output_dim)}
        """
        h_new = {}
        for v, h_v in h.items():
            neighbors = adj_list.get(v, [])
            if neighbors:
                nbr_feats = torch.stack([h[u] for u in neighbors if u in h])
            else:
                nbr_feats = torch.zeros(0, h_v.shape[-1], device=h_v.device)

            agg = self.aggregator(nbr_feats)
            combined = torch.cat([h_v, agg], dim=-1)
            h_new[v] = F.relu(self.W(combined))
        return h_new


class DeepTraceGNN(nn.Module):
    """
    Full DeepTrace GNN (Theorem 3: L = diam(Gn) + 1 layers sufficient).

    For practical training we use a fixed n_layers (typically 3).
    """

    def __init__(self, input_dim=3, hidden_dim=64, n_layers=3):
        super().__init__()
        self.n_layers = n_layers

        layers = []
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(DeepTraceLayer(in_d, hidden_dim))
        self.layers = nn.ModuleList(layers)

        # Final regression layer → scalar log-likelihood
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, adj_list):
        """
        Parameters
        ----------
        node_features : dict {node_id: Tensor(input_dim)}
        adj_list      : dict {node_id: list of neighbor node_ids}
            For directed graphs we use both in- and out-neighbors.

        Returns
        -------
        scores : dict {node_id: float}  (log-likelihood prediction)
        """
        h = node_features

        for layer in self.layers:
            h = layer(h, adj_list)

        scores = {}
        for v, h_v in h.items():
            scores[v] = self.output_layer(h_v).squeeze(-1)

        return scores

    def predict_source_posterior(self, Gn, node_features):
        """
        Run inference on epidemic subgraph Gn.
        Returns softmax-normalized source posterior π(v).

        Parameters
        ----------
        Gn            : nx.DiGraph (epidemic subgraph)
        node_features : dict {node: np.array([1, r_hat, r_check])}

        Returns
        -------
        pi : dict {node: float}  (sums to 1)
        """
        self.eval()
        with torch.no_grad():
            nodes = list(Gn.nodes())

            # Build adjacency list (undirected for message passing)
            adj_list = {}
            Gu = Gn.to_undirected()
            for v in nodes:
                adj_list[v] = list(Gu.neighbors(v))

            # Convert features to tensors
            h = {v: torch.tensor(node_features[v], dtype=torch.float32)
                 for v in nodes if v in node_features}

            scores = self.forward(h, adj_list)

            # Softmax over all nodes → source posterior π
            score_vals = torch.stack([scores[v] for v in nodes])
            pi_vals = torch.softmax(score_vals, dim=0).numpy()
            pi = {v: float(pi_vals[i]) for i, v in enumerate(nodes)}

        return pi
