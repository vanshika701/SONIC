# SONIC: Source-Oriented Network Immunization and Containment in Directed Graphs

## Abstract
Recent advances in network epidemiology address containment strategies by identifying and immunizing critical nodes to reduce the spectral radius of the underlying contact network. However, most existing spectral methods, including state-of-the-art algorithms designed for directed networks, ignore a crucial piece of information dynamically available during an ongoing outbreak: the location of the epidemic source. In this project, we introduce **SONIC (Source-Oriented Network Immunization and Containment)**, an algorithm that directly integrates source inference into spectral network immunization. Our solution operates in three synergistic phases: inferring the source posterior using Rumor Centrality or DeepTrace GNN algorithms, mapping the spread vulnerability via Expected Personalized PageRank (E-PPR) Source Risk, and applying a localized Spectral Path-Product (SPP) Optimization routine to minimize the spectral radius of the dominant network strongly connected components. Empirical results on complex real-world datasets show that SONIC drastically outperforms prior methods like DINO in spectral radius reduction and total epidemic curtailment.

---

## 1 Introduction
Epidemic containment within complex networks is an active area of research motivated by diverse domains such as disease mapping, curbing misinformation spread, and securing critical digital infrastructure. Many recent studies connect the mitigation of epidemics to minimizing the spectral radius (the largest eigenvalue magnitude) of network adjacency matrices. If the spectral radius can be driven below a specific epidemic threshold, widespread transmission is inherently stopped.

While significant work has been aimed at analyzing symmetric (undirected) networks, real-world epidemic transmissions are primarily directional. The application of spectral approximation logic to asymmetric networks introduces large mathematical hurdles—such as non-orthogonality in the eigenspace and the loss of Lipschitz continuity. Methods like DINO exist to bridge this gap via greedily pruning the network to drop the spectral radius. 

However, previous spectral optimizations are completely blind to the actual dynamics of the ongoing cascade. They treat the graph statically, failing to weight the importance of nodes nearer to the *actual* source(s) of the epidemic. **SONIC** is introduced to solve this. SONIC recognizes that deleting nodes with high general spectral impact does not help containment if those nodes are nowhere near the active disease front. By grounding immunization probabilistically toward the true source, SONIC ensures optimal spectral decay directly uncoupling the active transmission paths.

## 2 Preliminaries

### 2.1 Epidemic Models & Spectral Correlation
Epidemic spread is broadly modeled with non-linear dynamical systems including the Susceptible-Infected-Susceptible (SIS) framework. Under realistic conditions, the spread behavior maps closely to network properties. The epidemic threshold guarantees that a disease will organically collapse if the largest eigenvalue of the network architecture transitions below $\tau_c$. Thus, deleting nodes (e.g., through isolation, patching, or vaccination) is modeled as a matrix perturbation problem constrained by a designated budget $k$.

### 2.2 Directed Graph Constraints
In continuous space undirected graphs, classic tools like the Davis-Kahan theorem provide solid bounds for node immunization. In directed networks, asymmetric adjacency matrices possess complex eigenvalues and asymmetric eigenvectors, limiting simple rank-1 derivations.

### 2.3 Epidemic Source Inference
Source detection strategies model a sub-graph $G_n$ representing the detected infected state at a given time $t$. Using structural algorithms (such as Rumor Centrality) or generative Deep Learning algorithms (DeepTrace GNNs), we deduce the probability distribution $\pi(v)$ that a specified node $v$ was the true originator.

## 3 Theory

### 3.1 Spectral Formulations on Subgraphs
To limit spectral radius approximation complexities, we target the main transmission reservoirs inside the graph. By recursively diminishing the spectral radius of the network's main Strongly Connected Components (KSCC), we globally suppress the epidemic capacity for the overall network topology.

### 3.2 Coupling Spectral Drop with Source Probability
Instead of independently maximizing simple spectral drop $\Delta \rho$, we introduce the mechanism of **Epidemic Source Risk $\tau(v)$**. Source risk represents the mathematical likelihood that node $v$ lies directly on the active downstream infection vector originating from the probability cloud of inferred sources. We utilize Expected Personalized PageRank (E-PPR), configured with the source posterior $\pi(v)$ as the restart vector, to evaluate this local threat topology. This guarantees spatial grounding in our matrix pruning operations.

## 4 Algorithm and Analysis

### 4.1 SONIC: The Proposed Algorithm
SONIC executes network defense via a unified 3-phase pipeline targeting the highest-threat nodes within the available cost budget $k$.

**Phase 1: Source Posterior $\pi$ Inference** 
Using the observed infection snapshot $G_n$, SONIC extracts the structural origin of the disease, mapping a probability distribution $\pi(v)$ onto the network layout.

**Phase 2: Expected Personalized PageRank (E-PPR) Source Risk $\tau$** 
We generate the Source Risk score for every node using E-PPR, distributing weight recursively downstream from the top $K$ predicted sources utilizing a defined teleport probability scalar.

**Phase 3: Spectral Path-Product (SPP) Optimization** 
Within the largest cyclic clusters (KSCC) of the network, SONIC assesses each node $v$ using an optimized heuristic equation that binds the local Source Risk with left and right eigenvector constraints representing arrival probability and spreading influence:
$SPP(v) = \tau(v) \times C_{out}(v)$ 

SONIC iterates this in an adaptive, greedy framework $k$ times.

### 4.2 Effectiveness and Efficiency
SONIC reduces runtime dependencies compared to heavily uncoupled eigenvalue recalculations. The use of Katz centralities as eigenvector proxies and SCC localized isolation limits the computational demand dramatically.

## 5 Experiments

### 5.1 Experimental Settings
We contrast SONIC against numerous state-of-the-art benchmark approaches—including DINO, HITS-Authority, HITS-Hub, SourceOnly, Degree, and Random immunization protocols. Experiments were conducted tracking exact SIS spreading metrics across diverse graph datasets targeting biological networks (HIV) and internet topology networks (Gnutella), parameterized roughly at $I_0 = 0.95, \beta = 0.03$.

### 5.2 Effectiveness of SONIC
The inclusion of source topology generates massive performance jumps. At subset extraction ($k = 100$) inside the Gnutella directed network simulation, **SONIC achieves a maximum $\Delta \rho$ spectral decomposition of 0.83**. In contrast, DINO yields $\Delta \rho = 0.69$, and simple structural baselines like Degree Centrality fall to $\Delta \rho = 0.62$. 

Due to early localized cutoff actions near the epidemic source, SONIC also forces a massive dampening of total infections overtime ($I_T$), dropping aggregate casualty metrics from 181.5 (highest competing curve) to 144.25 units.

### 5.3 Efficiency and Scalability
By compartmentalizing calculations within KSCC blocks and utilizing proxy spectral algorithms instead of recalculating full graph decompositions, the SPP pipeline scales rapidly. On dense clusters, our testing confirms SONIC operations finish sequentially in roughly 36 seconds natively while offering radically upgraded $\Delta \rho$ yield arrays relative to structural baseline times.

### 5.4 Detailed Study
We monitored variation over expanding budget capabilities. Incrementing $k$ consistently scaled spectral containment linearly for SONIC without suffering the asymptote drop-offs seen in non-localized degree algorithms. Even on smaller budgets $k \le 20$, SONIC restricts local diffusion heavily due to targeting initial transmission branches near the disease origin. 

## 6. Related Work

### 6.1 Epidemiology on Directed Networks
Early efforts modeled disease strictly through degree-based pruning. Advanced algorithms utilized matrix perturbations (like Netshield). However, most classical equations map strictly to symmetrical arrays and collapse under directional constraints. 

### 6.2 Matrix Spectral Baselines (DINO)
Recent efforts like DINO solved directional graph epidemic boundaries manually minimizing independent strongly connected components. Yet DINO acts without source awareness, spending budget capacity removing core elements functionally separated from the realistic disease route. 

### 6.3 Source Inference Formulations
Recent deep-learning tracking measures like DeepTrace and probabilistic matrices like Rumor Centrality allow excellent single-node inference. SONIC forms the first bridge seamlessly merging this predictive analytics layer with functional graph manipulation via E-PPR matrices. 

## 7. Conclusion
In this report we present **SONIC** (Source-Oriented Network Immunization and Containment), an algorithmic breakthrough targeting disease suppression over directed networks. Rather than abstracting disease flow natively as a graph problem, SONIC operates predictively, merging inferred disease sources and network spectral radius optimization. Our experiments validate comprehensive dominance in early-outbreak mitigation, fundamentally suppressing spectral thresholds superiorly to cutting-edge models like DINO.
