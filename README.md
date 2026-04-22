# SONIC: Source-Oriented Network Immunization and Containment

SONIC is a network immunization algorithm that answers: **given a spreading epidemic, which nodes should you remove to stop it most efficiently?**

It combines two research papers:
- **DINO** (He et al., WSDM 2025) — structural immunization via spectral radius minimization
- **DeepTrace** (Tan et al., IEEE TSIPN 2025) — GNN-based epidemic source detection

SONIC bridges both: it finds where the epidemic started, traces its spreading path, and removes the minimum set of nodes to collapse the epidemic.

---

## Project Structure

```
SONIC/
├── main.py                      # CLI entry point
├── requirements.txt
├── checkpoints/
│   └── deeptrace.pt             # Trained DeepTrace GNN checkpoint
├── algorithms/
│   ├── dino.py                  # DINO algorithm (spectral radius + KSCC)
│   ├── sonic.py                 # SONIC Algorithm 1 (composite scoring)
│   ├── eppr.py                  # Expected Personalized PageRank (SourceRisk)
│   └── source_inference.py      # Rumor Centrality + DeepTrace source detection
├── data/
│   ├── loaders.py               # Dataset loaders (HIV, Reddit, Gnutella)
│   ├── synthetic.py             # Synthetic graph generators + SI simulation
│   └── raw/                     # Downloaded datasets go here
├── gnn/
│   ├── model.py                 # DeepTrace GNN (GraphSAGE + LSTM)
│   └── train.py                 # GNN training pipeline
├── simulation/
│   └── sis.py                   # SIS epidemic simulator
├── evaluation/
│   └── metrics.py               # Δρ, I_T, T_contain, SRA, Top-k accuracy
├── experiments/
│   ├── baselines.py             # Baseline methods (Random, Degree, DINO)
│   ├── ablation.py              # βw sweep + budget sweep
│   └── run_all.py               # Master experiment runner
├── results/                     # All outputs saved here
└── notebooks/
    └── results.ipynb            # Results visualization
```

---

## Setup

### Step 1 — Requirements
Python 3.10 or higher is required.

### Step 2 — Create virtual environment
```bash
cd SONIC
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install PyTorch
```bash
pip install torch torchvision torchaudio
```

### Step 4 — Install torch-geometric
```bash
pip install torch-geometric
```

### Step 5 — Install remaining dependencies
```bash
pip install networkx numpy scipy pandas matplotlib seaborn scikit-learn tqdm jupyter ipykernel
```

### Step 6 — Verify installation
```bash
python -c "import torch; import networkx; print('All good')"
```

### Step 7 — Create results directory
```bash
mkdir -p results
```

---

## Datasets

| Dataset | Nodes | Edges | ρ | How to get |
|---|---|---|---|---|
| HIV (synthetic) | 1,288 | 2,148 | 4.59 | Auto-generated (no download needed) |
| p2p-Gnutella | 10,876 | 39,994 | 4.45 | Auto-downloads from SNAP |
| Reddit Hyperlinks | 34,671 | 137,039 | 53.39 | Manual download (see below) |

### Reddit dataset (manual)
1. Go to https://snap.stanford.edu/data/soc-RedditHyperlinks.html
2. Download `soc-redditHyperlinks-body.tsv`
3. Place it at `data/raw/soc-redditHyperlinks-body.tsv`

### HIV dataset
The real HIV transmission network requires authorized access from ICPSR (sensitive public health data).
The project **automatically builds a synthetic graph** matching the paper statistics when the real file is not found — experiments still run correctly.
To use the real file if you have access, place it at `data/raw/hiv.txt` (one edge per line: `node1 node2`).

---

## Training the DeepTrace GNN

SONIC uses a trained DeepTrace GNN for source detection. Train it once before running experiments:

```bash
python -c "from gnn.train import train_deeptrace; train_deeptrace(verbose=True)"
```

This takes **~15-20 minutes** on CPU. The trained model is saved to `checkpoints/deeptrace.pt` and reused automatically in all future runs.

Training details:
- **Phase 1** (Pre-training): 500 graphs, 150 epochs, approximate labels via Rumor Centrality
- **Phase 2** (Fine-tuning): 250 graphs, 150 epochs, refined labels on smaller graphs
- Architecture: GraphSAGE + LSTM aggregators, 3 layers, hidden dim 64

---

## Running the Project

### Quick smoke test (30 seconds)
```bash
python main.py --synthetic
```

### Run SONIC with DeepTrace (full pipeline — recommended)
```bash
python main.py --dataset hiv --budget 100 --method sonic --source_method deeptrace
```

### Run SONIC with Rumor Centrality (faster fallback)
```bash
python main.py --dataset hiv --budget 100 --method sonic --source_method rumor
```

### Run without SIS simulation (faster)
```bash
python main.py --dataset hiv --budget 100 --method sonic --source_method deeptrace --no_sis
```

### Run on all datasets
```bash
# HIV (synthetic — auto-generated)
python main.py --dataset hiv --budget 100 --source_method deeptrace --no_sis

# Gnutella (auto-downloads ~40MB)
python main.py --dataset gnutella --budget 1441 --source_method deeptrace --no_sis

# Reddit (requires manual download)
python main.py --dataset reddit --budget 100 --source_method deeptrace --no_sis
```

### Compare specific methods
```bash
# DINO only (structural, no source info)
python main.py --dataset hiv --budget 100 --method dino

# Degree centrality baseline
python main.py --dataset hiv --budget 100 --method degree

# Random baseline
python main.py --dataset hiv --budget 100 --method random
```

### Validate DINO paper numbers (Table 2)
```bash
python main.py --hiv_benchmark
```
Expected: Δρ ≈ 4.59 at k=100, ≈ 5.70 at k=300 (on real HIV network).

---

## Experiments

### Run all baselines and compare
```bash
python -m experiments.run_all --dataset hiv --budgets 50 100 200
```

### βw Ablation study
Sweeps βw from 0 (pure DINO) to 1 (pure SourceOnly) to find the optimal balance:
```bash
python main.py --dataset hiv --budget 100 --ablation
```

### Run all datasets
```bash
python -m experiments.run_all --all --budgets 100 200
```

---

## SONIC Parameters

| Parameter | Default | Description |
|---|---|---|
| `--method` | sonic | `sonic`, `dino`, `degree`, `random`, `source_only` |
| `--source_method` | rumor | Source inference: `rumor` or `deeptrace` |
| `--alpha_w` | 0.5 | Weight for structural term (Δρ) |
| `--beta_w` | 0.5 | Weight for SourceRisk term |
| `--K_sources` | 10 | Top-K sources for E-PPR computation |
| `--ppr_alpha` | 0.15 | PPR teleport probability |
| `--adaptive` | False | Recompute source posterior after each removal |
| `--auto_weights` | False | Entropy-gated weight auto-tuning (novelty) |
| `--no_sis` | False | Skip SIS simulation (faster) |

`alpha_w + beta_w` must always equal 1.

**Setting `beta_w=0`** recovers pure DINO (structural only).
**Setting `alpha_w=0`** gives SourceOnly (source-risk only).

---

## Results

All results are saved to `results/` as JSON and CSV files.

### Open the results notebook
```bash
python -m ipykernel install --user --name=sonic-venv --display-name "SONIC (venv)"
jupyter notebook notebooks/results.ipynb
```

In Jupyter: select **Kernel → SONIC (venv)** then **Kernel → Restart & Run All**.

The notebook generates:
- `results/delta_rho_comparison.png` — bar chart comparing all methods
- `results/sis_curves.png` — epidemic curves over time
- `results/beta_w_ablation.png` — βw sweep curve
- `results/budget_sweep.png` — Δρ vs budget k

---

## Key Results

### HIV Network (synthetic, k=100)

| Method | Nodes actually removed | Δρ | T_contain |
|---|---|---|---|
| SONIC (DeepTrace) | 3 | 4.56 | 51.7 |
| Degree | 100 | 4.59 | 48.1 |
| DINO | 9 | 3.59 | 53.5 |
| Random | 100 | 0.06 | 70.2 |

### Reddit Network (real, k=100)

| Method | Nodes removed | Δρ |
|---|---|---|
| SONIC (DeepTrace) | 100 | 20.84 |
| Degree | 100 | 19.68 |
| DINO | 2 | 0.00 |
| Random | 100 | 0.01 |

### Gnutella Network (real)

| Budget k | Δρ | ρ after | Contained? |
|---|---|---|---|
| 100 | 0.49 | 3.96 | No |
| 1,000 | 2.55 | 1.89 | Yes (I_T=0) |
| 1,441 | 4.45 | 0.00 | Fully |

### βw Ablation (HIV)
Pure DINO (βw=0) achieves Δρ=3.59. Adding SourceRisk weight (βw≥0.4) jumps to Δρ=4.57 — a **27% improvement** from incorporating source information.

### Source Detection Accuracy (Rumor Centrality on BA graphs)
| Top-k | Accuracy |
|---|---|
| Top-1 | 36.7% |
| Top-5 | 96.7% |
| Top-10 | 100.0% |

---

## How It Works

**Phase 1 — Source Detection**
Given the observed infected subgraph Gn, the DeepTrace GNN infers which node most likely started the epidemic. It uses node features: infection ratio r̂(v) and boundary distance ř(v). Falls back to Rumor Centrality (Shah & Zaman, 2012) if no trained model is available.

**Phase 2 — SourceRisk via E-PPR**
For each node v, compute SourceRisk(v) — how much epidemic spreading flows through v given the likely source — using Expected Personalized PageRank over the top-K most probable sources.

**Phase 3 — Composite Scoring + Immunization**
Score each node as:
```
Score(v) = αw × Δρ̃(v) + βw × SourceRisk̃(v)
```
Greedily remove the highest-scoring node from the largest strongly connected component (SCC). Repeat until budget k is exhausted.

**Novelty — Entropy-gated weight auto-tuning:**
When source posterior entropy is high (uncertain source) → increase αw (trust structure more).
When entropy is low (confident source) → increase βw (trust SourceRisk more).
Enable with `--auto_weights`.

---

## References

- He et al., "Demystify Epidemic Containment in Directed Networks: Theory and Algorithms", WSDM 2025
- Tan et al., "DeepTrace: Learning to Optimize Contact Tracing in Epidemic Networks", IEEE TSIPN 2025
- Shah & Zaman, "Rumor Centrality: A Universal Source Detector", SIGMETRICS 2012
