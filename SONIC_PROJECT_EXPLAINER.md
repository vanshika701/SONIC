# SONIC: Source-Oriented Network Immunization and Containment
### Complete Project Explainer — Presentation Ready

---

## TABLE OF CONTENTS
1. [Introduction](#1-introduction)
2. [Background — What Existed Before](#2-background)
3. [Methodology and Approach](#3-methodology--approach)
4. [Implementation / Work Done](#4-implementation--work-done)
5. [Results and Analysis](#5-results-and-analysis)
6. [Conclusion and Future Work](#6-conclusion--future-work)
7. [References](#7-references)

---

---

# 1. INTRODUCTION

## Project Title
**SONIC — Source-Oriented Network Immunization and Containment**

## Problem Statement
Imagine a virus spreading through a social network — like COVID spreading through a city's contact graph, or a computer virus spreading across the internet. The big question is:

> **"Given that an epidemic is already spreading, which people (or computers) should you vaccinate/remove to stop it as fast as possible, using the fewest resources?"**

This is called the **Network Immunization Problem**. It is extremely hard because:
- Networks can have millions of nodes (people, computers, servers).
- You can't remove everyone — you have a limited **budget** (called `k`) of how many nodes you can immunize.
- Even knowing *which* nodes matter most is computationally hard.
- Existing methods are either too slow, or ignore WHERE the epidemic started — which turns out to be crucial information.

## Objective
SONIC's goal is to answer: **given a spreading epidemic in a network, find the minimum set of nodes to remove so that the epidemic collapses — using the SOURCE of the epidemic as a guide to be smarter about which nodes to remove.**

SONIC combines two key ideas:
1. **Find where the epidemic likely started** (source detection).
2. **Use that source info to identify the most critical spreading paths** and cut them.

## Team Members
*(Fill in your team names here)*

---

---

# 2. BACKGROUND

## What is a Network / Graph?
Before diving in, let's build up the vocabulary from scratch:

- A **graph** (also called a network) is a set of **nodes** (dots) connected by **edges** (lines).
- Example: Think of Facebook — each person is a node, and a friendship is an edge.
- A **directed graph** means edges have direction (like Twitter follows — you can follow someone who doesn't follow back). Epidemics spread along directed edges.

## What is an Epidemic on a Network?
In epidemic modeling, a disease spreads from infected nodes to their neighbors. In our project, we use the **SIS model**:
- **S** = Susceptible (healthy but can get infected)
- **I** = Infected (sick and can infect others)

At each time step:
- A healthy neighbor of an infected node gets infected with probability **β (beta)** = 0.03 (3% chance per connection per day).
- An infected person recovers with probability **δ (delta)** = 0.1 (10% chance per day).

The epidemic keeps bouncing around (you can get sick again after recovering — hence S→I→S→I...).

## What is Spectral Radius (ρ)?
This is the MOST important concept in this project — and it looks scary but here's the intuition:

> **The spectral radius ρ (rho) of a network is a single number that tells you how fast a disease will spread through that network.**

- **ρ > 1** → The epidemic will keep growing and never die out (endemic state). 
- **ρ ≤ 1** → The epidemic will naturally fade and die (contained!).

Mathematically, ρ is the **largest eigenvalue** of the network's adjacency matrix. The adjacency matrix is just a table of zeros and ones — row i, column j is 1 if node i connects to node j.

**Think of it this way:** If ρ = 4.59 (our HIV network), the epidemic can spread to 4.59 neighbors per step on average in the worst case. If we can drive ρ below 1, the epidemic dies on its own.

**Δρ (Delta rho)** = how much we REDUCED the spectral radius by removing our chosen nodes. Bigger Δρ = better job.

## What is the KSCC?
The **Key Strongly Connected Component (KSCC)** is the most important subgroup in a directed network. 

In a directed graph, a **Strongly Connected Component (SCC)** is a group of nodes where you can get from any node to any other node by following directed edges. The KSCC is the largest/most densely connected such group.

The critical theorem that SONIC uses: **ρ(entire network) = ρ(KSCC)**. This means you only need to target the KSCC, not the whole network. This saves enormous amounts of computation.

## Existing Work SONIC Builds On

### 1. DINO (He et al., WSDM 2025)
DINO stands for **DIrected NetwOrk** epidemic containment. It was one of the first algorithms to formally prove that the KSCC is the only part of the network you need to target. 

DINO's scoring function for each node `v`:
```
F(v) = d_in(v) × d_out(v) / vol(KSCC)
```
Where:
- `d_in(v)` = number of edges coming INTO node v (how many people can infect v)
- `d_out(v)` = number of edges going OUT of node v (how many people v can infect)
- `vol(KSCC)` = total number of edges in the KSCC

DINO says: pick the node with the biggest `d_in × d_out` ratio. This approximates which node contributes most to the spectral radius.

**DINO's limitation:** It's purely structural — it doesn't know WHERE the epidemic started. It treats all nodes with high degree equally, even if some are nowhere near the actual spreading path of the current epidemic.

### 2. DeepTrace (Tan et al., IEEE TSIPN 2025)
DeepTrace uses a **Graph Neural Network (GNN)** to detect where the epidemic started. Given the observed infected subgraph, it outputs a probability distribution over nodes: π(v) = "probability that v is the source."

### 3. Rumor Centrality (Shah & Zaman, SIGMETRICS 2012)
A classical mathematical method for source detection. For trees (simple networks with no cycles), it has a beautiful formula:
```
R(v) = n! / Π_u T_v(u)
```
Where:
- `n!` = n factorial (n × (n-1) × ... × 1)
- `T_v(u)` = size of the subtree below node u, when we root the tree at v

The idea: the true source should be the "most central" node in the spreading tree. Rumor Centrality quantifies this geometrically.

**SONIC's contribution:** It is the FIRST method to combine epidemic source detection with structural immunization. Neither DINO nor DeepTrace does immunization with source-awareness the way SONIC does.

---

---

# 3. METHODOLOGY / APPROACH

## High-Level Idea
SONIC reasons like a detective:
1. **"Where did this epidemic start?"** (Source Detection)
2. **"Given where it started, which nodes are most important for spreading?"** (SourceRisk via E-PPR)
3. **"Remove those nodes first, in the smartest order."** (SPP Optimization)

## Tools and Techniques

| Tool/Tech | What it does in SONIC |
|---|---|
| Python 3.10 | Main programming language |
| NetworkX | Library for creating and analyzing graphs |
| PyTorch | Deep learning framework (for training DeepTrace GNN) |
| PyTorch Geometric | Graph Neural Network operations |
| NumPy / SciPy | Fast math — matrix operations, eigenvalue computation |
| Sparse matrices | Efficiently store large networks (most entries are 0) |

## System Architecture / Pipeline Flowchart

```
INPUT: Network G (full)  +  Infected subgraph Gn  +  Budget k
          |
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: SOURCE DETECTION                                       │
│                                                                 │
│  Method A: DeepTrace GNN (preferred)                           │
│    → Graph Neural Network reads infected subgraph Gn           │
│    → Outputs π(v) = probability each node v is the source      │
│                                                                 │
│  Method B: Rumor Centrality (fast fallback)                    │
│    → Mathematical formula on BFS spanning tree                 │
│    → Also outputs π(v)                                         │
│                                                                 │
│  OUTPUT: Source posterior π = {node: probability}              │
└─────────────────────────────────────────────────────────────────┘
          |
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: SOURCE RISK (E-PPR)                                    │
│                                                                 │
│  For each of the top-K most probable sources:                  │
│    → Run Personalized PageRank seeded at that source           │
│    → Combine weighted by their source probability              │
│                                                                 │
│  SourceRisk(v) = τ(v) = how much epidemic flow passes          │
│                         through node v                         │
│                                                                 │
│  OUTPUT: τ = {node: source_risk_score}                         │
└─────────────────────────────────────────────────────────────────┘
          |
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: SPECTRAL PATH-PRODUCT (SPP) OPTIMIZATION              │
│                                                                 │
│  Repeat until budget k exhausted:                              │
│    1. Find KSCC (Key Strongly Connected Component)             │
│    2. For each node v in KSCC:                                 │
│         score(v) = Katz_in(v) × Katz_out(v) × (1 + 0.3×τ(v)) │
│    3. Remove node with highest score                           │
│    4. Re-check if KSCC broke into smaller pieces               │
│                                                                 │
│  OUTPUT: List L of k immunized nodes                           │
└─────────────────────────────────────────────────────────────────┘
          |
          ▼
OUTPUT: L (which nodes to remove), Δρ (how much ρ dropped)
        + SIS simulation curves (if enabled)
```

## Detailed Explanation of Each Phase

---

### PHASE 1: Source Detection

**Goal:** Given that we can see which nodes are infected (the "infected subgraph" Gn), figure out which node MOST LIKELY started the epidemic.

#### Method A: DeepTrace GNN

A **Graph Neural Network (GNN)** is a type of deep learning model that operates on graphs instead of images or text.

**Architecture: GraphSAGE + LSTM**

Each node v gets an initial feature vector:
```
h⁰(v) = [1, r̂(v), ř(v)]
```
Where:
- `1` = constant (bias term, always 1)
- `r̂(v)` = infection ratio = (number of infected neighbors of v) / (total neighbors of v). High means v is surrounded by infected people → likely not the source (it got infected later).
- `ř(v)` = boundary distance = distance from v to the nearest boundary of the infected region. Small means v is at the "center" of the epidemic → more likely to be the source.

The GNN has 3 layers. At each layer, each node collects information from its neighbors and updates its own representation:

**Equation 9 from DeepTrace paper:**
```
h^(l)_{N(v)} = LSTM({w^(l-1), h^(l-1)_u : u ∈ N(v)})   ← aggregate neighbors
h^(l)_v      = ReLU(W^(l) · [h^(l-1)_v || h^(l)_{N(v)}]) ← update node
```
- The `||` symbol means concatenation (just putting vectors side by side)
- `ReLU` is an activation function: ReLU(x) = max(0, x) — keeps positives, zeros out negatives
- `LSTM` is a special type of neural cell that handles sequences

After 3 layers, each node has a learned score. These are passed through **softmax** to get probabilities:
```
π(v) = exp(h^(L)_v) / Σ_u exp(h^(L)_u)   [Equation 3 in paper]
```
Softmax ensures all probabilities sum to 1. The node with the highest π is the predicted source.

**Training:** The model is trained in two phases:
- Phase 1: 500 synthetic graphs, 150 epochs, with Rumor Centrality as approximate labels
- Phase 2: 250 graphs, 150 epochs, refined on smaller networks

#### Method B: Rumor Centrality

For each candidate source v, compute:
```
R(v) = n! / Π_u T_v(u)
```
Where T_v(u) = size of the subtree rooted at u when tree is rooted at v.

**Intuition:** Imagine the infected graph as a tree. The true source should be "geometrically balanced" — with roughly equal-sized groups of infected people in each direction. Rumor Centrality measures this balance. The most balanced node (smallest denominators) gets the highest score.

In log space (for numerical stability):
```
log R(v) = log(n!) - Σ_u log(T_v(u))
```

---

### PHASE 2: SourceRisk via Expected Personalized PageRank (E-PPR)

**Goal:** Given the source probability distribution π, compute for EVERY node v in the network: "How much of the epidemic's spreading power flows through v?"

#### What is PageRank?
PageRank (invented by Google) measures the "importance" of a node in a network by simulating a random walk:
- At each step, with probability α (the **teleport probability**), jump to a specific starting node.
- With probability (1-α), follow a random outgoing edge.

**Personalized PageRank (PPR)** seeded at source node s:
```
π_s = α·e_s + (1-α)·D⁻¹·A·π_s   [Equation 4]
```
Where:
- `π_s` = the probability vector (π_s[v] = probability of being at node v)
- `α` = 0.15 (teleport probability — same as original Google PageRank)
- `e_s` = teleport vector with 1 at the source node, 0 everywhere else
- `D⁻¹·A` = the row-normalized adjacency matrix (each row divided by out-degree)
  - This represents: if you're at node u with 3 outgoing edges, you go to each neighbor with probability 1/3

Closed-form solution:
```
π_s = α·(I - (1-α)·D⁻¹·A)⁻¹·e_s   [Equation 5]
```

In practice, this is solved via **power iteration**: start with a guess, keep multiplying by the matrix until convergence.

#### Expected PPR Over Multiple Sources

Since we're not 100% sure who started the epidemic (we have a probability distribution π over sources), we compute SourceRisk as a weighted sum:

```
SourceRisk(v) = τ(v) = Σ_{s in top-K} π(s) · π_s(v)   [Equation 6]
```

- We only sum over the **top-K most probable sources** (default K=10) for efficiency
- π(s) is the probability that s is the source (from Phase 1)
- π_s(v) is the probability that a random walk from source s visits node v

**Intuition:** τ(v) = "Given our uncertainty about where the epidemic started, how often does epidemic flow actually pass through node v?"

**Computational complexity:** O(K × |E| / α) — manageable even for large networks.

---

### PHASE 3: Spectral Path-Product (SPP) Optimization

**Goal:** Use τ(v) from Phase 2 and structural information to greedily select which nodes to remove.

#### The Core Mathematical Insight

From spectral graph theory, when you remove node v from a network, the drop in spectral radius is approximately:
```
Δρ ∝ u_i · v_i
```
Where:
- `u_i` = the **left eigenvector** entry for node v ← this is related to "how much epidemic arrives at v" → **approximated by τ(v)**
- `v_i` = the **right eigenvector** entry for node v ← this is related to "how much can v spread downstream" → **approximated by Downstream Katz Centrality C_out(v)**

#### Katz Centrality

Katz Centrality measures how important a node is by counting ALL paths to it (not just direct edges), but penalizing longer paths with a decay factor α:
```
C_katz(v) = Σ_{l=1}^{∞} α^l × (number of paths of length l to v)
```

**Downstream Katz C_out(v):** Computed on the REVERSED graph (flip all edge directions). This captures how many nodes v can REACH downstream — its spreading power.

**Upstream Katz C_in(v):** Standard Katz on the original graph — captures how accessible v is to infection from upstream.

**Attenuation factor α** is set to `0.95 / ρ(G)` to ensure mathematical convergence (the infinite sum must terminate).

#### The SPP Score Formula

```
SPP(v) = Katz_in(v)  ×  Katz_out(v)  ×  (1 + 0.3 × τ_normalized(v))
         ─────────────────────────────   ────────────────────────────
         Left × Right eigenvector proxy   Source-aware boost factor
```

Breaking down each term:
- **`Katz_in(v)`** — how strongly v is reachable from the rest of the network (upstream connectivity)
- **`Katz_out(v)`** — how far v can spread infection downstream  
- **`τ_normalized(v)`** — source risk, normalized to [0.1, 1.0] range. The 0.3 coefficient means source risk gives up to a 30% bonus score boost

**Why multiply instead of add (unlike DINO)?** Multiplication (product) means BOTH conditions must be true simultaneously: high spreading power AND on the epidemic path. Addition (like DINO's F(v)) can compensate — a node with very high structure but no epidemic connection could still score high. Multiplication enforces that both factors matter jointly.

#### The Greedy Loop

```
Repeat k times:
1. Find all non-trivial SCCs (size ≥ 3) in remaining network
2. Pick the KSCC (highest approximate ρ)  
3. Compute SPP score for every node in KSCC
4. Remove node v* = argmax SPP(v)
5. Check if removing v* split the KSCC into smaller SCCs
6. Re-sort all SCCs by approximate ρ, repeat
```

After step 6, the algorithm always targets the current "most dangerous" SCC. As the epidemic is dismantled, the KSCC breaks into smaller pieces.

**Early stopping:** Algorithm stops early when no non-trivial SCCs remain (all groups have < 3 nodes). The epidemic has been fully contained. You can see this in the run output where it says:
```
[SONIC] No non-trivial SCCs remaining at step 1537. Stopping early.
```

#### Novelty: Entropy-Gated Auto-Weights

A novel contribution of SONIC that automatically adjusts trust in source detection:

**Entropy of source posterior:**
```
H(π) = -Σ_v π(v) · log(π(v))
```
- High H means we're UNCERTAIN about the source (distribution is spread out)
- Low H means we're CERTAIN about the source (distribution is peaked at one node)

**Confidence score:**
```
confidence = 1 - H(π) / H_max
```
Where H_max = log(n) = maximum possible entropy (completely uniform distribution).

**Auto-tuned weights:**
```
β_w = β_base × (0.5 + 0.5 × confidence)
α_w = 1 - β_w
```
- When confident about source → higher β_w → trust SourceRisk more
- When uncertain about source → higher α_w → trust structural information more
- Activated with `--auto_weights` flag

---

### SIS Epidemic Simulation

After immunization, the project validates results using a discrete-time SIS simulation (Monte Carlo).

**Infection probability for node v at step t:**
```
P(v gets infected) = 1 - (1 - β)^(number of infected neighbors of v)
```
This formula comes from probability theory: if each neighbor independently infects v with probability β, the probability that NONE of them infect v is (1-β)^k, so at least one infects v with probability 1-(1-β)^k.

**Parameters:**
- β = 0.03 (3% transmission rate per edge)
- δ = 0.1 (10% recovery rate per step)
- I₀ = 0.95 (initially 95% of nodes are infected)
- T = 200 time steps
- 20 Monte Carlo trials (run 20 independent simulations and average)

**Evaluation metrics from simulation:**
- **I_T** = average number of infected nodes at final time step T. Lower = better immunization.
- **T_contain** = first time step where infected < 1% of network. Lower = faster containment.

---

### Datasets

| Dataset | Nodes | Edges | Domain | ρ (spectral radius) | How loaded |
|---|---|---|---|---|---|
| HIV (synthetic) | 1,288 | 2,148 | HIV transmission network | 4.59 | Auto-generated to match real stats |
| p2p-Gnutella | 10,876 | 39,994 | Peer-to-peer file sharing network | 4.45 | Auto-downloads from Stanford SNAP |
| Reddit Hyperlinks | 34,671 | 137,039 | Reddit community links | 53.39 | Manual download |
| Enron Email | ~36,000 | ~200,000 | Corporate email network | varies | Stanford SNAP |

---

---

# 4. IMPLEMENTATION / WORK DONE

## File Structure and What Each File Does

```
SONIC/
├── main.py                    ← Command-line interface (the "front door")
├── requirements.txt           ← Python packages needed
│
├── algorithms/
│   ├── sonic.py               ← Main SONIC 3-phase algorithm
│   ├── dino.py                ← DINO baseline (structural only)
│   ├── eppr.py                ← Phase 2: Expected Personalized PageRank
│   ├── source_inference.py    ← Phase 1: Rumor Centrality + DeepTrace
│   ├── spp.py                 ← Phase 3: SPP scoring + KSCC management
│   └── measures.py            ← Katz centrality computations
│
├── gnn/
│   ├── model.py               ← DeepTrace GNN architecture (PyTorch)
│   └── train.py               ← Training pipeline (2-phase)
│
├── simulation/
│   └── sis.py                 ← SIS epidemic simulator
│
├── data/
│   ├── loaders.py             ← Load HIV/Gnutella/Reddit/Enron datasets
│   └── synthetic.py           ← Generate synthetic graphs + SI simulation
│
├── evaluation/
│   └── metrics.py             ← Δρ, I_T, T_contain, SRA, Top-k accuracy
│
├── experiments/
│   ├── baselines.py           ← Degree, Katz, Random, Betweenness baselines
│   ├── ablation.py            ← Parameter sweep experiments
│   └── run_all.py             ← Master script to run all experiments
│
├── results/                   ← All JSON/PNG output files saved here
├── checkpoints/
│   └── deeptrace.pt           ← Saved trained GNN model
└── notebooks/
    └── results.ipynb          ← Jupyter notebook for visualizations
```

## Key Features

### 1. Three-Phase Pipeline
- Phase 1 (Source Detection) → Phase 2 (SourceRisk Computation) → Phase 3 (Greedy SPP Immunization)
- Modular design — each phase can be swapped independently.

### 2. Dual Source Detection Methods
- **DeepTrace GNN** — Deep learning based, more accurate, takes 15-20 min to train once
- **Rumor Centrality** — Fast mathematical method, no training needed, good fallback

### 3. Multiple Baselines for Comparison
- **Random** — Remove k random nodes. Worst expected performance.
- **Degree** — Remove nodes with highest total degree (in+out). Strong but dumb.
- **Katz** — Remove nodes with highest Katz centrality. Smarter than degree.
- **Betweenness** — Remove nodes that appear on most shortest paths. Expensive to compute.
- **DINO** — State-of-art structural method. SONIC beats it by adding source info.
- **SourceOnly** — Only use SourceRisk (τ), ignore structural component.

### 4. Multi-Dataset Support
HIV, Gnutella, Reddit, Enron — each with its own loader.

### 5. Evaluation Suite
Five metrics computed automatically after immunization:
- **Δρ** (delta rho) — Primary metric: how much spectral radius dropped
- **I_T** — Infected count at end of SIS simulation
- **T_contain** — How fast epidemic was contained
- **SRA** — SourceRisk Alignment Score (novel metric, see below)
- **Top-k Source Accuracy** — How often the true source was in top-k predicted nodes

### 6. SourceRisk Alignment Score (SRA) — Novel Metric

```
SRA = (r · g) / (||r|| · ||g||)
```
This is cosine similarity between:
- **r** = SourceRisk vector (τ values for all nodes)
- **g** = Ground-truth infection order vector (g[v] = 1/rank where rank is order of infection)

SRA measures whether SourceRisk correctly identifies early-infected nodes as high-risk. Higher SRA = better alignment between predicted risk and actual epidemic path.

### 7. Adaptive Mode
With `--adaptive` flag: after each node removal, recompute the source posterior π and SourceRisk τ on the updated network. More accurate but slower.

## How to Run (Key Commands)

```bash
# Quick test
python main.py --synthetic

# Full SONIC on HIV network
python main.py --dataset hiv --budget 100

# Full SONIC on Gnutella (bigger network)
python main.py --dataset gnutella --budget 1441 --no_sis

# Compare methods
python main.py --dataset hiv --budget 100 --method degree
python main.py --dataset hiv --budget 100 --method dino

# Run ablation study (sweep K_sources values)
python main.py --dataset hiv --budget 100 --ablation
```

---

---

# 5. RESULTS AND ANALYSIS

## Metric Definitions (Recap)
- **Δρ** (goal: maximize) — How much the spectral radius dropped. Higher = epidemic more contained.
- **k** = how many nodes were actually removed
- **T_contain** (goal: minimize) — How many time steps before epidemic drops below 1% of network
- **I_T** (goal: minimize) — How many nodes are infected at the final time step

---

## Result 1: HIV Network (Synthetic, 1,288 nodes)

| Method | Nodes Removed (k) | Δρ | ρ_before | ρ_after |
|---|---|---|---|---|
| **SONIC** | **3** | **4.567** | 4.591 | **0.023** |
| SourceOnly | 3 | 4.565 | 4.591 | 0.026 |
| Degree | 100 | 4.591 | 4.591 | ~0.0 |
| Katz | 100 | 4.591 | 4.591 | ~0.0 |
| DINO | 9 | 3.591 | 4.591 | **1.000** |
| Random | 100 | 0.057 | 4.591 | 4.534 |

**Key insight:** SONIC needed only **3 nodes** to achieve Δρ = 4.567 (near-complete containment). DINO needed **9 nodes** for only Δρ = 3.591. Random needed **100 nodes** and barely moved the needle (Δρ = 0.057).

This demonstrates the core value proposition: **knowing where the epidemic started tells you exactly which 3 nodes to attack, instead of blindly trying 9+ or 100.**

---

## Result 2: Gnutella Network (Real P2P network, 10,876 nodes)

| Budget k | Nodes Used | Δρ | ρ_after | Contained? |
|---|---|---|---|---|
| 100 | 100 | 0.49 | 3.96 | No |
| 1,000 | 1,000 | 2.55 | 1.89 | Partially |
| 1,441 | 1,190 | 4.45 | 0.00 | **Fully!** |
| 1,800 | 1,537 | 4.45 | 0.00 | **Fully!** |

At budget k=1,441, only **1,190 nodes** were actually needed (algorithm stopped early!) to fully contain it. Δρ = 4.45 means the spectral radius went from 4.45 all the way to 0.00 — **total eradication of the epidemic**.

---

## Result 3: Source Detection Accuracy (on Barabási-Albert synthetic graphs)

| Metric | Value |
|---|---|
| Top-1 accuracy | 36.7% |
| Top-5 accuracy | 96.7% |
| Top-10 accuracy | 100.0% |

**Interpretation:** The true source is almost never perfectly identified (36.7% top-1), but it is almost ALWAYS in the top-10 candidates (100%). This is why SONIC uses top-K=10 sources — it hedges against uncertainty.

---

## βw Ablation Study (HIV network, k=100)

This experiment sweeps the weight β_w (how much to weight SourceRisk vs pure structure):

| β_w (SourceRisk weight) | Δρ |
|---|---|
| 0.0 (pure DINO — structure only) | 3.59 |
| 0.2 | 3.90 |
| 0.4 | 4.57 |
| 0.5 (SONIC default) | **4.57** |
| 0.8 | 4.50 |
| 1.0 (pure SourceOnly) | 4.56 |

**Key finding:** Adding ANY source-risk information (β_w > 0) significantly improves performance. The sweet spot is around β_w = 0.4–0.5, giving a **27% improvement over pure DINO** (4.57 vs 3.59).

---

## Epidemic Curves (SIS Simulation)

The SIS simulation shows how the infection count evolves over 200 time steps:
- **Without immunization:** Infection stays high and oscillates (never dies)
- **After SONIC immunization:** Infection drops rapidly to near-zero
- **After Random immunization:** Almost no change

See: `results/sis_curves.png` and `results/sis_curves(ENRON).png`

---

## Runtime Performance

| Dataset | Budget | Runtime |
|---|---|---|
| HIV | 100 | ~0.05 seconds |
| Gnutella | 1,441 | ~104 seconds |
| Gnutella | 1,800 | ~742 seconds |

Runtime scales roughly O(k × |V| + |E|) — manageable for networks up to ~tens of thousands of nodes.

---

---

# 6. CONCLUSION & FUTURE WORK

## Summary

SONIC successfully demonstrates that **combining epidemic source detection with structural network immunization is significantly more effective than either approach alone.**

Key contributions:
1. **First algorithm to integrate source detection (DeepTrace/Rumor Centrality) with network immunization (DINO-style KSCC targeting)**
2. **Spectral Path-Product (SPP) scoring** — a theoretically grounded node scoring that multiplies left and right eigenvector proxies (instead of DINO's additive heuristic)
3. **Novel entropy-gated auto-weight system** that automatically adjusts reliance on source information based on detection confidence
4. **Novel SRA metric** (SourceRisk Alignment Score) for evaluating how well SourceRisk aligns with actual infection order
5. **Demonstrated real-world applicability** on three different network types: HIV transmission, P2P file sharing, social media

## Possible Improvements and Future Work

### 1. Real-time / Online Operation
Currently, SONIC assumes you observe the infected subgraph all at once. In reality, epidemic data trickles in over time. Future work: online SONIC that updates its source estimate as new infected nodes are reported.

### 2. Partial Immunization
Currently, immunization is binary (full removal). In reality, vaccination is probabilistic and partial. Future: fractional immunization where each node is vaccinated with some probability.

### 3. Better GNN for Source Detection
The current DeepTrace uses only 2 node features (r̂, ř). Including temporal features (when did each node get infected?), spatial features, or node attributes could improve source detection accuracy significantly.

### 4. Scalability to Billion-Scale Networks
Current algorithm handles ~35K nodes well. For Facebook-scale networks (billions of nodes), we'd need:
- Approximate spectral radius computation
- Streaming or distributed graph processing
- Sketch-based SCC detection

### 5. Dynamic Networks
Real networks change over time (people join/leave). SONIC currently assumes a static graph. Handling temporal/dynamic networks is an open problem.

### 6. Multiple Sources
SONIC assumes one epidemic source. In reality (e.g., COVID imported by multiple travelers), there may be multiple parallel sources. Multi-source E-PPR is a natural extension.

### 7. Heterogeneous Networks
Different types of nodes and edges (e.g., a hospital network where doctors, nurses, and patients have different infection rates). Incorporating node/edge attributes into the scoring function is a natural next step.

---

---

# 7. REFERENCES

1. **He et al. (2025)** — "Demystify Epidemic Containment in Directed Networks: Theory and Algorithms"  
   *WSDM 2025*  
   → Foundation paper for DINO algorithm; proved ρ(G) = ρ(KSCC) and the greedy F(v) scoring heuristic.

2. **Tan et al. (2025)** — "DeepTrace: Learning to Optimize Contact Tracing in Epidemic Networks"  
   *IEEE Transactions on Signal and Information Processing over Networks (TSIPN) 2025*  
   → Source of the DeepTrace GNN architecture (GraphSAGE + LSTM), infection ratio features r̂(v) and ř(v), and the two-phase training procedure.

3. **Shah & Zaman (2012)** — "Rumor Centrality: A Universal Source Detector"  
   *ACM SIGMETRICS 2012*  
   → Original paper deriving the Rumor Centrality formula for tree networks; proved it is the ML estimator of the true source under the SI model on regular trees.

4. **Katz (1953)** — "A New Status Index Derived from Sociometric Analysis"  
   *Psychometrika, 1953*  
   → Original definition of Katz centrality; basis for C_out(v) computation in SONIC.

5. **Benzi & Klymko (2015)** — "Total Communicability as a Centrality Measure"  
   *Journal of Complex Networks*  
   → Mathematical justification for Katz as a proxy for eigenvector entries.

6. **SNAP Datasets** — Stanford Network Analysis Project  
   → Source of the p2p-Gnutella08 and Reddit Hyperlinks datasets used in experiments.  
   URL: https://snap.stanford.edu/

7. **Page et al. (1999)** — "The PageRank Citation Ranking: Bringing Order to the Web"  
   *Stanford Technical Report*  
   → Original PageRank paper; Personalized PageRank (Phase 2 of SONIC) is a direct extension.

---

## Quick Glossary (All Key Terms)

| Term | Plain-English Meaning |
|---|---|
| **Graph / Network** | Set of nodes (people/computers) connected by edges (relationships/connections) |
| **Directed graph** | Edges have direction (A→B ≠ B→A), like Twitter follows or email sends |
| **Spectral radius ρ** | A single number measuring how fast epidemic spreads; must drop below 1 to contain |
| **Eigenvalue** | A special number associated with a matrix; spectral radius = largest eigenvalue |
| **Adjacency matrix** | Grid of 0s and 1s showing which nodes connect to which |
| **SCC** | Strongly Connected Component — group where you can reach everyone from everyone |
| **KSCC** | The most "dangerous" SCC — where the epidemic lives; ρ(G) = ρ(KSCC) |
| **Immunization** | Removing nodes from graph (vaccinating people; blocking servers) |
| **Budget k** | Maximum number of nodes we're allowed to remove |
| **Δρ** | Drop in spectral radius = how much better we made the network |
| **SIS model** | Epidemic model where recovered people can get sick again |
| **β (beta)** | Infection probability per edge per time step (0.03 = 3%) |
| **δ (delta)** | Recovery probability per time step (0.1 = 10%) |
| **Source posterior π** | Probability distribution over which node started the epidemic |
| **PPR** | Personalized PageRank — random walk seeded at one specific node |
| **SourceRisk τ** | How likely epidemic flow passes through a given node, given likely source |
| **E-PPR** | Expected PPR — weighted average PPR over top-K possible sources |
| **Katz centrality** | Importance score counting all paths (with exponential decay for long paths) |
| **GNN** | Graph Neural Network — deep learning model that works on graph-structured data |
| **LSTM** | Long Short-Term Memory — type of neural network for processing sequences |
| **Softmax** | Function converting raw scores into probabilities that sum to 1 |
| **Entropy H** | Measure of uncertainty; high entropy = spread-out distribution = uncertain |
| **SRA** | SourceRisk Alignment Score — how well predicted risk aligns with actual infection order |
| **Greedy algorithm** | Step-by-step strategy that always picks the locally best option at each step |
| **Convergence** | When an iterative process stops changing significantly (found the answer) |
| **Power iteration** | Algorithm for finding eigenvalues by repeatedly multiplying a matrix |

---

*Document prepared for SONIC project presentation. Last updated: April 2026.*
