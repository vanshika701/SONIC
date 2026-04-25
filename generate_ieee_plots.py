import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import networkx as nx

# IEEE Paper Styling
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("tab10")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

import sys

# Default paths if not provided
OUT_DIR = Path("plots")
INPUT_FILE = "results/reddit_results.json"

if len(sys.argv) > 1:
    INPUT_FILE = sys.argv[1]
if len(sys.argv) > 2:
    OUT_DIR = Path(sys.argv[2])

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors and markers identical to presentation style
COLORS = {
    "SONIC": "#d62728",      # Red
    "SPP": "#d62728",        # Red
    "DINO": "#2ca02c",       # Green
    "Degree": "#1f77b4",     # Blue
    "SourceOnly": "#ff7f0e", # Orange
    "HITS-Authority": "#9467bd", # Purple
    "HITS-Hub": "#8c564b",   # Brown
    "Acquaintance": "#e377c2",# Pink
    "Katz": "#7f7f7f",       # Gray
    "Random": "#bcbd22",     # Yellow-green
}

MARKERS = {
    "SONIC": "D", "SPP": "D", "DINO": "s", "Degree": "o",
    "SourceOnly": "^", "HITS-Authority": "v", "HITS-Hub": "p",
    "Acquaintance": "X", "Katz": "d", "Random": "h"
}

def load_data(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()
            end_idx = content.rfind(']')
            if end_idx != -1:
                content = content[:end_idx+1]
            return json.loads(content)
    except FileNotFoundError:
        return []

def plot_sis_curve():
    """Generates the infected people vs time SIS graph."""
    print("Generating SIS infection curves...")
    import sys
    sys.path.insert(0, ".")
    from data.loaders import load_dataset
    from algorithms.spp import spp_selection
    from experiments.baselines import degree_immunization, random_immunization, dino_immunization
    from evaluation.metrics import sis_metrics
    from algorithms.source_inference import rumor_centrality
    from algorithms.eppr import source_risk
    from benchmark_datasets import simulate_epidemic

    # Load small graph for fast SIS simulation
    G = load_dataset("hiv")
    if G is None:
        print("Dataset not found. Skipping SIS curve.")
        return
    
    # 1. Epidemic Setup
    Gn, _ = simulate_epidemic(G, seed=42)
    pi = rumor_centrality(Gn)
    tau = source_risk(G, pi, K=10, alpha=0.15)
    
    k = 100
    methods_to_plot = {
        "Random": random_immunization(G, k, seed=42),
        "Degree": degree_immunization(G, k),
        "DINO": dino_immunization(G, k),
        "SONIC": spp_selection(G, k, tau, return_delta_rho=False, verbose=False)
    }

    plt.figure(figsize=(8, 5))
    
    # Run SIS simulation over 200 time steps and plot
    for name, L in methods_to_plot.items():
        print(f"  Simulating {name}...")
        res = sis_metrics(G, L, beta=0.03, delta=0.1, I0=0.95, T=100, n_trials=5)
        curve = res['curve']
        
        plt.plot(range(len(curve)), curve, label=name, 
                 color=COLORS.get(name, 'black'), 
                 marker=MARKERS.get(name, 'o'), markevery=10,
                 linewidth=2, markersize=8, alpha=0.9)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Time (steps)')
    plt.ylabel('Infected people')
    plt.title('SIS Epidemic Spread (HIV Dataset, $I_0=0.95, \\beta=0.03$)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    outpath = OUT_DIR / "sis_curve.pdf"
    plt.savefig(outpath)
    print(f"Saved {outpath}")
    plt.close()

def plot_bar_chart(data, metric, title, ylabel, filename, k_filter=100):
    """Bar chart comparing methods at a specific budget k."""
    if not data: return
    df = pd.DataFrame(data)
    if 'k' in df.columns:
        df = df[df['k'] == k_filter]
    
    if df.empty or metric not in df.columns:
        return
        
    df = df.sort_values(by=metric, ascending=False)
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df['method'], df[metric], 
            color=[COLORS.get(m, 'gray') for m in df['method']],
            edgecolor='black', linewidth=1.2)
            
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylabel(ylabel)
    plt.title(f'{title} (Budget $k={k_filter}$)')
    plt.xticks(rotation=45, ha='right')
    
    # Add values on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.02), 
                 f'{yval:.1f}' if yval > 10 else f'{yval:.2f}', 
                 ha='center', va='bottom', fontsize=10)
                 
    outpath = OUT_DIR / filename
    plt.savefig(outpath)
    print(f"Saved {outpath}")
    plt.close()

def plot_line_chart(data, metric, title, ylabel, filename):
    """Line chart showing scaling across budgets (k)."""
    if not data: return
    df = pd.DataFrame(data)
    if 'k' not in df.columns or metric not in df.columns: return
    
    plt.figure(figsize=(8, 5))
    
    for method in df['method'].unique():
        m_df = df[df['method'] == method].sort_values('k')
        if len(m_df) < 2: continue
        plt.plot(m_df['k'], m_df[metric], 
                 label=method, color=COLORS.get(method, 'gray'),
                 marker=MARKERS.get(method, 'o'),
                 linewidth=2.5, markersize=8)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Immunization Budget ($k$)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(df['k'].unique())
    
    outpath = OUT_DIR / filename
    plt.savefig(outpath)
    print(f"Saved {outpath}")
    plt.close()

if __name__ == "__main__":
    print(f"Starting IEEE Plot Generation for {INPUT_FILE} -> {OUT_DIR}...")
    
    # 1. Bar charts from data
    data = load_data(INPUT_FILE)
    if data:
        plot_bar_chart(data, 'delta_rho', 'Spectral Radius Decrease ($\Delta\\rho$)', '$\Delta\\rho$', 'delta_rho_bar.pdf', k_filter=100)
        
        # Only plot runtime if logged
        if 'runtime_s' in data[0]:
            plot_bar_chart(data, 'runtime_s', 'Computational Time', 'Runtime (seconds)', 'runtime_bar.pdf', k_filter=100)
        
        # 3. Line chart across budgets
        plot_line_chart(data, 'delta_rho', '$\Delta\\rho$ vs. Immunization Budget', '$\Delta\\rho$', 'delta_rho_line.pdf')
        plot_line_chart(data, 'I_T', 'Total Infections ($I_T$) vs. Budget', 'Total Infected Over Time ($I_T$)', 'infections_line.pdf')

    # 4. Run full SIS Curve explicitly and plot
    plot_sis_curve()
    
    print("Done! All plots saved to the 'plots/' directory.")
