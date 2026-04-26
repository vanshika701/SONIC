import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def load_json_data(filepath):
    """
    Loads JSON data with fault tolerance for any trailing text 
    (like the markdown tables at the end of reddit_results.json).
    """
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Find the last closing bracket to ignore the trailing text table
    end_idx = content.rfind(']')
    if end_idx != -1:
        content = content[:end_idx+1]
        
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []

def plot_budget_sweep(data, output_file):
    """
    Takes the parsed JSON list of dictionaries and plots the budget sweep.
    Plots both Spectral Drop (Δρ) and Total Infections (I_T) vs Budget (k).
    """
    if not data:
        print("No data found!")
        return

    df = pd.DataFrame(data)
    
    # We only care about entries that have a valid 'method', 'k', and 'delta_rho'
    df = df.dropna(subset=['method', 'k', 'delta_rho'])

    sns.set_theme(style="whitegrid")
    
    # Colors for consistency
    palette = {
        "SONIC": "#d62728",      # Red
        "DINO": "#2ca02c",       # Green
        "Degree": "#1f77b4",     # Blue
        "HITS-Authority": "#9467bd", 
        "HITS-Hub": "#8c564b",   
        "SourceOnly": "#ff7f0e", 
        "Acquaintance": "#e377c2",
        "Random": "#7f7f7f"
    }
    
    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # -----------------------------
    # Plot 1: Spectral Drop (Δρ)
    # -----------------------------
    sns.lineplot(
        data=df, 
        x='k', y='delta_rho', hue='method', 
        marker='o', linewidth=2.5, markersize=8, 
        palette=palette, ax=ax1
    )
    
    ax1.set_title("Spectral Radius Reduction ($\Delta\\rho$) vs. Budget ($k$)", fontsize=14, pad=15)
    ax1.set_xlabel("Immunization Budget ($k$)", fontsize=12)
    ax1.set_ylabel("Spectral Drop ($\Delta\\rho$)", fontsize=12)
    ax1.set_xticks(sorted(df['k'].unique()))
    ax1.legend(title="Method", title_fontsize='11', fontsize='10', loc="upper left")
    
    # -----------------------------
    # Plot 2: Total Infections (I_T)
    # -----------------------------
    if 'I_T' in df.columns:
        sns.lineplot(
            data=df, 
            x='k', y='I_T', hue='method', 
            marker='s', linewidth=2.5, markersize=8, 
            palette=palette, ax=ax2
        )
        
        ax2.set_title("Total Epidemic Infections ($I_T$) vs. Budget ($k$)", fontsize=14, pad=15)
        ax2.set_xlabel("Immunization Budget ($k$)", fontsize=12)
        ax2.set_ylabel("Total Infected Over Time ($I_T$)", fontsize=12)
        ax2.set_xticks(sorted(df['k'].unique()))
        
        # We don't need a duplicate legend on the second plot if it clutters it
        ax2.get_legend().remove()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Successfully saved budget sweep plot to {output_file}")

if __name__ == "__main__":
    input_file = "results/reddit_results.json"
    output_image = "results/reddit_budget_sweep.png"
    
    print(f"Reading data from {input_file}...")
    data = load_json_data(input_file)
    
    # Filter methods if we have too many useless ones that clog the graph
    methods_to_keep = ["SONIC", "DINO", "Degree", "Random", "HITS-Authority", "Acquaintance"]
    filtered_data = [d for d in data if d.get('method') in methods_to_keep]
    
    print("Generating plot...")
    plot_budget_sweep(filtered_data, output_image)
