Here is a clear summary of how **SONIC** and **SPP** work together. 

Think of **SONIC** as the overarching *pipeline framework* (the car), while **SPP** is the mathematical *engine* that powers its decision-making. 

The framework operates in three sequential phases:

### Phase 1: Where did the outbreak start? (Source Inference)
The algorithm looks at the *partially observed* infected subgraph ($G_n$) and looks backwards in time. Using either Rumor Centrality or a Graph Neural Network (DeepTrace), it calculates a probability distribution ($\pi$) to identify the most likely "Patient Zero" nodes where the infection originated.

### Phase 2: Where is the outbreak going? (E-PPR)
Once it has the likely sources, SONIC looks forwards. It runs **Epidemic Personalized PageRank (E-PPR)** seeded at those origins. This calculates a value $\tau(v)$ for every susceptible node in the entire network, representing the geometric probability that the infection will arrive at node $v$. 

### Phase 3: How do we stop it? (SPP Engine)
This is where **SPP (Spectral Path-Product)** comes in. Epidemic theory states that an infection will only die out if the network's maximum eigenvalue (the "spectral radius", $\rho$) is pushed below a critical threshold. 

To drop the spectral radius as fast as possible, linear algebra dictates that we must remove nodes that maximize the product of their left and right eigenvectors ($\Delta\rho \propto u_i \cdot v_i$).

SPP isolates the most dangerous structural core of the network (the Katz Strongly Connected Component, or KSCC) and greedily removes nodes one-by-one by calculating:

1. **Left Eigenvector ($u_i$)**: How much infection the node *receives* from the structure around it.
2. **Right Eigenvector Proxy ($C_{out}$)**: Downstream Katz Centrality. This measures how rapidly the node *spreads* something along infinite downstream paths.
3. **The Bonus ($\tau$)**: SPP uses the arrival probability from Phase 2 as a multiplier tie-breaker.

**The Final SPP Formula:**
$$\text{SPP}(v) = u_i(v) \times C_{out}(v) \times (1 + \gamma \cdot \tau(v))$$

By multiplying these together, SPP perfectly combines structural containment (removing global network super-spreaders) with source-awareness (prioritizing the paths the virus is actively travelling down right now).