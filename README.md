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
├── algorithms/
│   ├── dino.py                  # DINO algorithm (spectral radius + KSCC)
│   ├── sonic.py                 # SONIC Algorithm 1 (composite scoring)
│   ├── eppr.py                  # Expected Personalized PageRank (SourceRisk)
│   └── source_inference.py      # Rumor Centrality + DeepTrace source detection
├── data/
│   ├── loaders.py               # Dataset loaders (HIV, Reddit, Gnutella)
│   └── synthetic.py             # Synthetic graph generators + SI simulation
├── gnn/
│   ├── model.py                 # DeepTrace GNN (GraphSAGE + LSTM)
│   └── train.py                 # GNN training pipeline
├── simulation/
│   └── sis.py                   # SIS epidemic simulator
├── evaluation/
│   └── metrics.py               # Δρ, I_T, T_contain, SRA, Top-k accuracy
├── experiments/
│   ├── baselines.py             # Baseline methods (Random, Degree, Katz, DINO)
│   ├── ablation.py              # βw sweep + budget sweep
│   └── run_all.py               # Master experiment runner
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
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install PyTorch
```bash
pip install torch torchvision torchaudio
```

### Step 4 — Install torch-geometric
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
```

### Step 5 — Install remaining dependencies
```bash
pip install networkx numpy scipy pandas matplotlib seaborn scikit-learn tqdm
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

| Dataset | Nodes | Edges | How to get |
|---|---|---|---|
| HIV (synthetic) | 1,288 | 2,148 | Auto-generated (no download needed) |
| p2p-Gnutella | 10,876 | 39,994 | Auto-downloads from SNAP |
| Reddit Hyperlinks | 34,671 | 137,039 | Manual download (see below) |

### Reddit dataset (manual)
1. Go to https://snap.stanford.edu/data/soc-RedditHyperlinks.html
2. Download `soc-redditHyperlinks-body.tsv`
3. Place it at `data/raw/soc-redditHyperlinks-body.tsv`

### HIV dataset (real)
The real HIV transmission network requires authorized access from ICPSR.
The project automatically builds a synthetic graph matching the paper statistics when the real file is not found.
To use the real file, place it at `data/raw/hiv.txt` (one edge per line: `node1 node2`).

---

## Running the Project

### Quick smoke test (30 seconds)
```bash
python main.py --synthetic
```

### Run SONIC on HIV network
```bash
python main.py --dataset hiv --budget 100 --method sonic
```

### Run without SIS simulation (faster)
```bash
python main.py --dataset hiv --budget 100 --method sonic --no_sis
```

### Run on Gnutella (auto-downloads)
```bash
python main.py --dataset gnutella --budget 1200 --method sonic
```

### Run on Reddit
```bash
python main.py --dataset reddit --budget 100 --method sonic
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
Expected output: Δρ ≈ 4.59 at k=100, ≈ 5.70 at k=300 (on real HIV network).

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
| `--alpha_w` | 0.5 | Weight for structural term (Δρ) |
| `--beta_w` | 0.5 | Weight for SourceRisk term |
| `--source_method` | rumor | Source inference: `rumor` or `deeptrace` |
| `--K_sources` | 10 | Top-K sources for E-PPR computation |
| `--ppr_alpha` | 0.15 | PPR teleport probability |
| `--adaptive` | False | Recompute source posterior after each removal |
| `--auto_weights` | False | Entropy-gated weight auto-tuning (novelty) |

`alpha_w + beta_w` must always equal 1.

**Setting `beta_w=0`** recovers pure DINO (structural only).
**Setting `alpha_w=0`** gives SourceOnly (source-risk only).

---

## Results

All results are saved to `results/` as JSON and CSV files.

### Open the results notebook
```bash
pip install jupyter ipykernel
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

### HIV Network (synthetic)

| Method | Nodes removed | Δρ | T_contain |
|---|---|---|---|
| SONIC | 3 | 4.56 | 51.7 |
| Degree | 100 | 4.59 | 48.1 |
| DINO | 9 | 3.59 | 53.5 |
| Random | 100 | 0.06 | 70.2 |

### Reddit Network (real)

| Method | Nodes removed | Δρ |
|---|---|---|
| SONIC | 100 | 20.84 |
| Degree | 100 | 19.68 |
| DINO | 2 | 0.00 |
| Random | 100 | 0.01 |

### βw Ablation (HIV)
Pure DINO (βw=0) achieves Δρ=3.59. Adding SourceRisk weight (βw≥0.4) jumps to Δρ=4.57 — a **27% improvement** from source information alone.

---

## How It Works

**Phase 1 — Source Detection**
Given the observed infected subgraph Gn, infer which node most likely started the epidemic using Rumor Centrality (Shah & Zaman, 2012) or DeepTrace GNN (Tan et al., 2025).

**Phase 2 — SourceRisk via E-PPR**
For each node v, compute how much epidemic spreading flows through it given the likely source locations, using Expected Personalized PageRank.

**Phase 3 — Composite Scoring + Immunization**
Score each node as:
```
Score(v) = αw × Δρ̃(v) + βw × SourceRisk̃(v)
```
Greedily remove the highest-scoring node from the largest SCC. Repeat until budget k is exhausted.

**Novelty addition:** Entropy-gated weight auto-tuning — when the source is uncertain (high entropy), increase αw to trust structure more. When source is confidently identified, increase βw to trust SourceRisk more. Enable with `--auto_weights`.

---

## References

- He et al., "Demystify Epidemic Containment in Directed Networks: Theory and Algorithms", WSDM 2025
- Tan et al., "DeepTrace: Learning to Optimize Contact Tracing in Epidemic Networks", IEEE TSIPN 2025
- Shah & Zaman, "Rumor Centrality: A Universal Source Detector", SIGMETRICS 2012
