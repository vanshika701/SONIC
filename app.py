import streamlit as st
import networkx as nx
import numpy as np
import time
import pandas as pd
from argparse import Namespace
import matplotlib.pyplot as plt
import seaborn as sns

# Import SONIC modules
from data.loaders import load_dataset
from data.synthetic import simulate_si
from evaluation.metrics import evaluate_method, delta_rho
from main import run_method
from algorithms.spp import spectral_radius

st.set_page_config(page_title="SONIC: Epidemic Containment", layout="wide")

st.title("SONIC: Source-Oriented Network Immunization and Containment")
st.markdown("""
Interactive visualization of the SONIC algorithm and baselines. 
**SONIC** combines structural graph theory (spectral radius) and epidemiological source tracing via **Spectral Path-Product (SPP)** optimization to find the optimal nodes to remove.
""")

# --- Sidebar Configuration ---
st.sidebar.header("1. Dataset Configuration")
dataset_choice = st.sidebar.selectbox("Dataset", ["hiv", "gnutella", "reddit"])

st.sidebar.header("2. Epidemic Simulation")
beta = st.sidebar.slider("Infection Rate (β)", min_value=0.01, max_value=0.5, value=0.03, step=0.01)
max_steps = st.sidebar.slider("Initial Spread Steps (SI)", min_value=1, max_value=50, value=10, step=1)

st.sidebar.header("3. Immunization Configuration")
compare_mode = st.sidebar.checkbox("Compare Two Methods", value=False)

methods_list = ["sonic", "spp", "dino", "source_only", "degree", "katz", "random", "betweenness"]

if compare_mode:
    method1 = st.sidebar.selectbox("Method 1", methods_list, index=0)
    method2 = st.sidebar.selectbox("Method 2", methods_list, index=4)
    methods_to_run = [method1, method2]
else:
    method = st.sidebar.selectbox("Method", methods_list, index=0)
    methods_to_run = [method]

budget = st.sidebar.slider("Budget (k nodes)", min_value=1, max_value=500, value=50, step=1)

# Method specific parameters
st.sidebar.header("4. Advanced Parameters")
source_method = st.sidebar.selectbox("Source Inference", ["rumor", "deeptrace", "auto"])
k_sources = st.sidebar.slider("Top-K Sources for E-PPR", 1, 50, 10)
ppr_alpha = st.sidebar.slider("PPR Teleport (α)", 0.05, 0.5, 0.15, 0.05)
run_sis = st.sidebar.checkbox("Run SIS Evaluation (Slow but accurate)", value=True)

# --- Main Logic ---
if st.button("Run Simulation & Immunization"):
    # --- Step 1: Load Data ---
    with st.spinner(f"Loading {dataset_choice.upper()} dataset..."):
        try:
            G = load_dataset(dataset_choice)
            if G is None:
                st.error(f"Could not load dataset {dataset_choice}.")
                st.stop()
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.stop()
        
    n, m = G.number_of_nodes(), G.number_of_edges()
    rho0 = spectral_radius(G)
    
    st.info(f"**Network Loaded:** {n} nodes, {m} edges | **Initial ρ:** {rho0:.4f}")

    # --- Step 2: Initial Epidemic ---
    with st.spinner("Simulating initial epidemic spread (SI)..."):
        # Source is node with max in-degree
        source = max(G.nodes(), key=lambda v: G.in_degree(v))
        Gn, infection_order, _ = simulate_si(G, source=source, beta=beta, max_steps=max_steps)
        if Gn.number_of_nodes() < 2:
            Gn = G
    
    st.write(f"**Initial Infection (Gn):** {Gn.number_of_nodes()} nodes infected.")

    # --- Step 3: Run Immunization ---
    results = {}
    
    for m_name in methods_to_run:
        with st.spinner(f"Running {m_name.upper()} immunization..."):
            args = Namespace(
                method=m_name,
                source_method=source_method,
                K_sources=k_sources,
                ppr_alpha=ppr_alpha,
                adaptive=False,
                quiet=True,
                no_sis=not run_sis,
                sis_trials=10
            )
            t0 = time.time()
            L = run_method(G, Gn, budget, args)
            elapsed = time.time() - t0
            
            with st.spinner(f"Evaluating {m_name.upper()}..."):
                metrics = evaluate_method(G, L, method_name=m_name, run_sis=run_sis, verbose=False)
                metrics['runtime'] = elapsed
                metrics['nodes_removed'] = L
            
            results[m_name] = metrics

    # --- Step 4: Display Results ---
    if compare_mode:
        cols = st.columns(2)
        for i, m_name in enumerate(methods_to_run):
            with cols[i]:
                st.subheader(f"Method: {m_name.upper()}")
                res = results[m_name]
                c1, c2 = st.columns(2)
                c1.metric("Δρ (Spectral Drop)", f"{res['delta_rho']:.4f}")
                c2.metric("Runtime", f"{res['runtime']:.2f}s")
                
                if run_sis:
                    c1.metric("Final Infected (I_T)", f"{res['I_T']:.1f}")
                    c2.metric("Containment Time", f"{res['T_contain']:.1f}")
                
                st.write(f"**Final ρ:** {res['rho_after']:.4f}")

        # Comparison Charts
        st.divider()
        st.subheader("Performance Comparison")
        
        comp_data = []
        for m_name in methods_to_run:
            res = results[m_name]
            comp_data.append({
                "Method": m_name,
                "Δρ": res['delta_rho'],
                "Runtime (s)": res['runtime'],
                "Final Infected": res.get('I_T', 0) if run_sis else 0
            })
        df_comp = pd.DataFrame(comp_data)
        
        fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        sns.barplot(data=df_comp, x="Method", y="Δρ", ax=ax1, palette="viridis")
        ax1.set_title("Spectral Radius Decrease (Higher is better)")
        
        if run_sis:
            sns.barplot(data=df_comp, x="Method", y="Final Infected", ax=ax2, palette="magma")
            ax2.set_title("Final Infected Count (Lower is better)")
        else:
            sns.barplot(data=df_comp, x="Method", y="Runtime (s)", ax=ax2, palette="magma")
            ax2.set_title("Runtime (Seconds)")
        
        st.pyplot(fig_comp)

    else:
        m_name = methods_to_run[0]
        res = results[m_name]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nodes Removed", len(res['nodes_removed']))
        col2.metric("Δρ", f"{res['delta_rho']:.4f}")
        col3.metric("Runtime", f"{res['runtime']:.2f}s")
        if run_sis:
            col4.metric("I_T", f"{res['I_T']:.1f}")

        st.write(f"**Spectral Radius:** {res['rho_before']:.4f} → {res['rho_after']:.4f}")

    # --- Additional Visualizations ---
    st.divider()
    tabs = st.tabs(["Infection Curves", "Network Analysis", "Degree Distribution"])
    
    with tabs[0]:
        if run_sis:
            st.subheader("Epidemic Spread Over Time (SIS)")
            fig_sis, ax_sis = plt.subplots(figsize=(10, 5))
            for m_name, res in results.items():
                if 'curve' in res:
                    ax_sis.plot(res['curve'], label=m_name.upper(), linewidth=2)
            
            ax_sis.set_xlabel("Time Steps")
            ax_sis.set_ylabel("Number of Infected Nodes")
            ax_sis.legend()
            ax_sis.grid(True, alpha=0.3)
            st.pyplot(fig_sis)
        else:
            st.write("SIS evaluation was skipped. Enable it in the sidebar to see infection curves.")

    with tabs[1]:
        st.subheader("Immunization Impact Visualization")
        # Just show the first method for visualization to avoid clutter
        m_viz = methods_to_run[0]
        L = results[m_viz]['nodes_removed']
        
        if len(L) > 0:
            with st.spinner("Generating network plot..."):
                # Get 1-hop neighborhood of removed nodes
                subgraph_nodes = set(L)
                for node in L:
                    if G.has_node(node):
                        subgraph_nodes.update(G.successors(node))
                        subgraph_nodes.update(G.predecessors(node))
                
                # Limit size for visualization
                subgraph_nodes = list(subgraph_nodes)[:200]
                H = G.subgraph(subgraph_nodes)
                
                pos = nx.spring_layout(H, seed=42)
                fig_net, ax_net = plt.subplots(figsize=(10, 8))
                
                color_map = ["red" if node in L else "lightblue" for node in H.nodes()]
                nx.draw(H, pos, node_color=color_map, node_size=50, 
                        edge_color="gray", alpha=0.5, ax=ax_net, with_labels=False)
                st.pyplot(fig_net)
                st.caption(f"Visualization of removed nodes (red) and their neighbors in {m_viz.upper()}.")

    with tabs[2]:
        st.subheader("Network Degree Distribution")
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        
        fig_deg, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(in_degrees, bins=30, ax=ax_in, color="skyblue", kde=True)
        ax_in.set_title("In-Degree Distribution")
        ax_in.set_yscale('log')
        
        sns.histplot(out_degrees, bins=30, ax=ax_out, color="salmon", kde=True)
        ax_out.set_title("Out-Degree Distribution")
        ax_out.set_yscale('log')
        
        st.pyplot(fig_deg)

st.sidebar.markdown("---")
st.sidebar.info("""
**Methods Info:**
- **SONIC**: Source-aware + Structural (SPP)
- **SPP**: Pure Spectral Path-Product
- **DINO**: Pure Structural (He et al. 2025)
- **Katz**: High spreading power
- **Degree**: Classic connectivity baseline
""")
