import streamlit as st
import networkx as nx
import numpy as np
import time
from argparse import Namespace
import matplotlib.pyplot as plt

# Import SONIC modules
from data.loaders import load_dataset
from data.synthetic import simulate_si
from evaluation.metrics import evaluate_method
from main import run_sonic_method

st.set_page_config(page_title="SONIC: Epidemic Containment", layout="wide")

st.title("SONIC: Source-Oriented Network Immunization and Containment")
st.markdown("""
Interactive visualization of the SONIC algorithm. 
**SONIC** combines structural graph theory (spectral radius) and epidemiological source tracing to find the optimal nodes to remove to stop an epidemic.
""")

# --- Sidebar Configuration ---
st.sidebar.header("1. Dataset Configuration")
dataset_choice = st.sidebar.selectbox("Dataset", ["hiv", "gnutella", "reddit"])

st.sidebar.header("2. Epidemic Simulation")
beta = st.sidebar.slider("Infection Rate (β)", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
max_steps = st.sidebar.slider("SI Simulation Steps", min_value=5, max_value=20, value=10, step=1)

st.sidebar.header("3. Immunization Method")
method = st.sidebar.selectbox("Method", ["sonic", "spp", "source_only", "degree", "random", "betweenness"])
budget = st.sidebar.slider("Budget (k nodes)", min_value=1, max_value=300, value=20, step=1)

# Method specific parameters
alpha_w = 0.5
beta_w = 0.5
source_method = "rumor"

if method == "sonic":
    st.sidebar.subheader("SONIC Parameters")
    alpha_w = st.sidebar.slider("Alpha (Structure Weight)", 0.0, 1.0, 0.5, 0.1)
    beta_w = 1.0 - alpha_w
    st.sidebar.text(f"Beta (Source Weight) = {beta_w:.1f}")
    source_method = st.sidebar.selectbox("Source Inference", ["rumor", "deeptrace"])

run_sis = st.sidebar.checkbox("Run SIS Evaluation (Slow)", value=False)

# --- Main Logic ---
if st.button("Run Simulation & Immunization"):
    with st.spinner(f"Loading {dataset_choice.upper()} dataset..."):
        try:
            G = load_dataset(dataset_choice)
            if G is None:
                st.error(f"Could not load dataset {dataset_choice}. For Reddit, ensure soc-redditHyperlinks-body.tsv is in data/raw.")
                st.stop()
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.stop()
        
    n, m = G.number_of_nodes(), G.number_of_edges()
    st.write(f"**Network Loaded:** {n} nodes, {m} edges")

    with st.spinner("Simulating initial epidemic spread (SI)..."):
        # Source is node with max in-degree
        source = max(G.nodes(), key=lambda v: G.in_degree(v))
        Gn, order, _ = simulate_si(G, source=source, beta=beta, max_steps=max_steps)
        if Gn.number_of_nodes() < 2:
            Gn = G
    st.write(f"**Epidemic Subgraph (Gn):** {Gn.number_of_nodes()} nodes infected initially.")

    with st.spinner(f"Running {method.upper()} immunization..."):
        args = Namespace(
            method=method,
            alpha_w=alpha_w,
            beta_w=beta_w,
            source_method=source_method,
            K_sources=10,
            ppr_alpha=0.15,
            adaptive=False,
            auto_weights=False,
            quiet=True
        )
        t0 = time.time()
        L = run_sonic_method(G, Gn, budget, args)
        elapsed = time.time() - t0
    
    st.success(f"Immunization completed in {elapsed:.2f} seconds.")

    with st.spinner("Evaluating metrics..."):
        metrics = evaluate_method(G, L, method_name=method, run_sis=run_sis, verbose=False)
    
    # --- Display Results ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes Removed (k)", len(L))
    col2.metric("Spectral Radius Decrease (Δρ)", f"{metrics.get('delta_rho', 0):.4f}")
    
    if run_sis:
        col3.metric("Total Infected (I_T)", f"{metrics.get('I_T', 0):.1f}")
        st.write(f"**Time to Containment:** {metrics.get('T_contain', 'N/A')}")
        
    st.subheader("Network Visualization (Removed Nodes & Neighbors)")
    
    # Visualization: Render a small subgraph around removed nodes to avoid hanging
    if len(L) > 0:
        with st.spinner("Generating network plot..."):
            # Get 1-hop neighborhood of removed nodes
            subgraph_nodes = set(L)
            for node in L:
                if G.has_node(node):
                    subgraph_nodes.update(G.successors(node))
                    subgraph_nodes.update(G.predecessors(node))
            
            # Limit size for visualization
            subgraph_nodes = list(subgraph_nodes)[:300]
            if len(subgraph_nodes) == 0:
                st.write("No nodes to visualize.")
            else:
                H = G.subgraph(subgraph_nodes)
                
                pos = nx.spring_layout(H, seed=42)
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Colors: Red for removed, Blue for others
                color_map = ["red" if node in L else "lightblue" for node in H.nodes()]
                sizes = [300 if node in L else 50 for node in H.nodes()]
                
                nx.draw(H, pos, node_color=color_map, node_size=sizes, 
                        edge_color="gray", alpha=0.7, ax=ax, with_labels=False)
                
                # Add legend
                from matplotlib.lines import Line2D
                legend_elements = [Line2D([0], [0], marker='o', color='w', label='Removed (Immunized)', markerfacecolor='red', markersize=10),
                                   Line2D([0], [0], marker='o', color='w', label='Other Nodes', markerfacecolor='lightblue', markersize=10)]
                ax.legend(handles=legend_elements, loc='upper right')
                
                st.pyplot(fig)
    else:
        st.write("No nodes removed, nothing to visualize.")
