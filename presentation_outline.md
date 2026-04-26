# Presentation Outline: SONIC & Spectral Path-Product (SPP)

## 1. Introduction (2 minutes)
**Project Title:** SONIC: Spectral Path-Product Algorithm for Network Epidemic Containment
**Team Members:** [Insert Names]

**Problem Statement and Objective:**
When an outbreak occurs in a complex network (like a virus spreading through a contact network, or malware through a computer network), we are forced to contain it using a very limited budget of vaccines, quarantines, or patches. 
*The Objective:* We must identify and immunize a small set of "super-spreader" bottleneck nodes to mathematically guarantee the outbreak dies out as quickly as possible. Doing this randomly or purely by out-degree is highly inefficient.

---

## 2. Background (2 minutes)
**Brief overview of existing technology/work:**
Existing containment strategies largely fall into two categories:
1. **Pure Structural Methods:** Things like *Degree Centrality*, *Katz Centrality*, or the recent *DINO* algorithm. These look at the graph structure to find bottlenecks, but they are "blind" to where the infection actually started, leading to wasted immunizations far from the outbreak.
   - *DINO Objective Function:* $$\arg\min F(v) = \frac{\sum (d_{in} \cdot d_{out}) - d_{in}(v) \cdot d_{out}(v)}{vol - d_{in}(v) - d_{out}(v)}$$
2. **Pure Source-Aware Methods:** These predict where the virus is heading but fail to account for the infinite possible downstream branching paths a virus could take over time.

---

## 3. Methodology / Approach (3 minutes)
**Tools & Techniques:**
Our project builds the **SONIC framework**, which mathematically unites source-awareness with structural graph theory. We rely heavily on algebraic graph theory, specifically exploiting the *spectral radius* ($\rho$), which acts as the tipping point for epidemic survival.

**System Architecture / Flowchart:**
SONIC operates in three distinct phases:
1. **Source Inference:** We observe a snapshot of the infection ($G_n$) and use *Rumor Centrality* or Graph Neural Networks to estimate the probability distribution ($\pi$) of the outbreak's origin (Patient Zero).
2. **E-PPR (Epidemic Personalized PageRank):** We run random walks forward from the estimated source to calculate $\tau(v)$, the probability the virus arrives at any node $v$.
3. **SPP (Spectral Path-Product) Engine:** To force the epidemic to die out, we must minimize the network's spectral radius ($\rho$). Mathematical theory guarantees that the drop in spectral radius upon removing a node is proportional to the product of its left and right eigenvectors:
   $$\Delta\rho \propto u_i \cdot v_i$$

---

## 4. Implementation / Work Done (4 minutes)
**Key Features & The Algorithm:**
We completely rewrote the previous state-of-the-art heuristic using the theoretically correct SPP formula. Inside the Katz Strongly Connected Component (KSCC) of the network, we calculate our custom criterion for every node:

$$\text{SPP}(v) = u_i(v) \times C_{out}(v) \times (1 + \gamma \cdot \tau(v))$$

*Where:*
- **$u_i(v)$ (Left Eigenvector):** How much infection the node *receives* from the topological structure. Computed via Sparse SciPy Eigensolvers on $A^T$.
- **$C_{out}(v)$ (Right Eigenvector Proxy):** Downstream Katz Centrality computed on the reversed graph. How efficiently the node *spreads* the infection infinitely downstream.
- **$\tau(v)$ (Source-Risk):** Our E-PPR arrival probability acting as a synergistic bonus to prioritize structural bottlenecks that are *in the active path* of the virus.

**Demonstration:**
*(Speaker Note: Here you can mention the NetLogo visualizer built to demonstrate how SPP surrounds the infection compared to the Random or Degree methods which scatter defenses randomly, or show the Python CLI terminal running live.)*

---

## 5. Results and Analysis (2 minutes)
**Outcomes, Graphs, and Comparisons:**
We benchmarked SPP against robust, zero-training graph algorithms including *Degree*, *Katz*, *DINO*, *HITS-Authority*, and *Acquaintance* immunization across diverse datasets.

*Key Takeaways:*
- On synthetic graphs (Scale-Free, Barabasi-Albert) and real-world graphs (HIV, Gnutella, Reddit), **SPP consistently outperforms or ties true DINO**. 
- Because SPP utilizes the true left eigenvector rather than a hard-gated probability metric, it scales beautifully to massive graphs like Reddit (35,000+ nodes) where other methods plateau.
- *Metric Showcase:* Mention the $\Delta\rho$ improvement metrics. For example, on certain budgets, SPP achieves strict +1.0 $\Delta\rho$ advantages over DINO.

---

## 6. Conclusion & Future Work (1 minute)
**Summary:**
The Spectral Path-Product algorithm successfully bridges the gap between proactive structural topology analysis and reactive, real-time source inference. By isolating the exact mathematical drivers of the spectral radius ($u_i \times v_i$), we achieve optimal network immunization.

**Possible Improvements:**
- Replacing the baseline Rumor Centrality with deep-learning Graph Masking (DeepTrace module scaling).
- Adapting the SPP formula for dynamically changing temporal graphs where edges appear and disappear over time.

---

## 7. References
1. Van Mieghem, P., et al. (2011). Decreasing the spectral radius of a graph by link removals. *IEEE/ACM Transactions on Networking.*
2. Katz, L. (1953). A new status index derived from sociometric analysis. *Psychometrika.*
3. He, et al. (2025). DINO: Structural Bottleneck Identification. *WSDM.*
4. Kleinberg, J. M. (1999). Authoritative sources in a hyperlinked environment. *Journal of the ACM.*
