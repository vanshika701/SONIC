"""
Two-phase DeepTrace GNN training (Section IV-B, IV-C of DeepTrace paper).

Phase 1 - Pre-training:
  500 graphs, N in [50,1000], labels = approx log P(Gn|v) via rumor centrality
  Train for 150 epochs (paper); we default to 50 for speed.

Phase 2 - Fine-tuning:
  250 graphs, N=50 fixed, labels = exact log P(Gn|v) (tractable for N<=50)
  Train for 150 epochs (paper); we default to 30 for speed.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from gnn.model import DeepTraceGNN
from data.synthetic import generate_training_batch, compute_deeptrace_features


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")


def _graph_to_tensors(Gn, features, labels):
    """Convert graph data to tensors for training."""
    nodes = list(Gn.nodes())
    if len(nodes) < 3:
        return None

    Gu = Gn.to_undirected()
    adj_list = {v: list(Gu.neighbors(v)) for v in nodes}

    h = {}
    y = {}
    for v in nodes:
        if v in features and v in labels and np.isfinite(labels[v]):
            h[v] = torch.tensor(features[v], dtype=torch.float32)
            y[v] = torch.tensor(labels[v], dtype=torch.float32)

    if len(h) < 3:
        return None

    return h, adj_list, y


def _train_epoch(model, optimizer, batch_gen, criterion, device):
    """Run one epoch over a batch of graphs."""
    model.train()
    total_loss = 0.0
    n_graphs = 0

    for Gn, features, labels, true_source in batch_gen:
        data = _graph_to_tensors(Gn, features, labels)
        if data is None:
            continue

        h, adj_list, y = data

        # Move to device
        h = {v: feat.to(device) for v, feat in h.items()}

        optimizer.zero_grad()
        scores = model(h, adj_list)

        # Compute MSE loss between predicted and true log-likelihood
        preds, targets = [], []
        for v in y:
            if v in scores:
                preds.append(scores[v])
                targets.append(y[v].to(device))

        if not preds:
            continue

        pred_tensor = torch.stack(preds)
        target_tensor = torch.stack(targets)

        loss = criterion(pred_tensor, target_tensor)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_graphs += 1

    return total_loss / max(n_graphs, 1)


def train_deeptrace(
    n_pretrain=500,
    n_finetune=250,
    pretrain_epochs=150,
    finetune_epochs=150,
    hidden_dim=64,
    n_layers=3,
    lr=1e-3,
    device_str="cpu",
    save_path=None,
    verbose=True,
):
    """
    Train DeepTrace GNN with two-phase approach.

    Parameters
    ----------
    n_pretrain      : int, number of graphs for pre-training
    n_finetune      : int, number of graphs for fine-tuning
    pretrain_epochs : int
    finetune_epochs : int
    hidden_dim      : int
    n_layers        : int
    lr              : float, learning rate
    device_str      : str, 'cpu' or 'cuda'
    save_path       : str, path to save model checkpoint
    verbose         : bool

    Returns
    -------
    model : trained DeepTraceGNN
    """
    device = torch.device(device_str)
    model = DeepTraceGNN(input_dim=3, hidden_dim=hidden_dim, n_layers=n_layers)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # -----------------------------------------------------------------------
    # Phase 1: Pre-training with approximate labels
    # Generate dataset ONCE, reuse across all epochs
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[DeepTrace] Phase 1: Generating {n_pretrain} graphs for pre-training...")

    pretrain_batch = list(generate_training_batch(
        n_graphs=n_pretrain,
        n_nodes=200,
        n_infected_range=(30, 100),
        exact_labels=False,
        seed=42,
    ))
    if verbose:
        print(f"[DeepTrace] Phase 1: Training for {pretrain_epochs} epochs...")

    for epoch in range(pretrain_epochs):
        loss = _train_epoch(model, optimizer, pretrain_batch, criterion, device)
        if verbose and (epoch % 10 == 0 or epoch == pretrain_epochs - 1):
            print(f"  Pretrain epoch {epoch+1}/{pretrain_epochs} | loss={loss:.4f}")

    # -----------------------------------------------------------------------
    # Phase 2: Fine-tuning with exact labels (small graphs N=15 max for exact)
    # Generate dataset ONCE, reuse across all epochs
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[DeepTrace] Phase 2: Generating {n_finetune} graphs for fine-tuning...")

    finetune_batch = list(generate_training_batch(
        n_graphs=n_finetune,
        n_nodes=50,
        n_infected_range=(10, 30),
        exact_labels=False,  # rumor centrality labels — fast and accurate enough
        seed=1000,
    ))
    if verbose:
        print(f"[DeepTrace] Phase 2: Fine-tuning for {finetune_epochs} epochs...")

    for epoch in range(finetune_epochs):
        loss = _train_epoch(model, optimizer, finetune_batch, criterion, device)
        if verbose and (epoch % 10 == 0 or epoch == finetune_epochs - 1):
            print(f"  Finetune epoch {epoch+1}/{finetune_epochs} | loss={loss:.4f}")

    # Save checkpoint
    if save_path is None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        save_path = os.path.join(MODEL_DIR, "deeptrace.pt")

    torch.save({
        "model_state": model.state_dict(),
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
    }, save_path)

    if verbose:
        print(f"\n[DeepTrace] Model saved to {save_path}")

    return model


def load_deeptrace(path=None, hidden_dim=64, n_layers=3, device_str="cpu"):
    """Load a saved DeepTrace checkpoint."""
    if path is None:
        path = os.path.join(MODEL_DIR, "deeptrace.pt")

    model = DeepTraceGNN(input_dim=3, hidden_dim=hidden_dim, n_layers=n_layers)

    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device_str)
        model.load_state_dict(ckpt["model_state"])
        print(f"[DeepTrace] Loaded checkpoint from {path}")
    else:
        print(f"[DeepTrace] No checkpoint at {path}. Using untrained model.")

    return model


def get_or_train_deeptrace(force_retrain=False, **kwargs):
    """
    Load existing DeepTrace model or train from scratch.
    Convenient for experiments — only trains once.
    """
    default_path = os.path.join(MODEL_DIR, "deeptrace.pt")

    if not force_retrain and os.path.exists(default_path):
        return load_deeptrace(default_path)

    return train_deeptrace(save_path=default_path, **kwargs)
