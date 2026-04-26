import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def load_json_data(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        content = f.read()
    end_idx = content.rfind(']')
    if end_idx != -1:
        content = content[:end_idx+1]
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        return []

palette = {
    "SONIC": "#d62728", "DINO": "#2ca02c", "Degree": "#1f77b4",
    "HITS-Authority": "#9467bd", "HITS-Hub": "#8c564b",
    "SourceOnly": "#ff7f0e", "Acquaintance": "#e377c2", "Random": "#7f7f7f"
}

def plot_bar_chart(data, metric, title, ylabel, output_file, k_val=100):
    df = pd.DataFrame(data)
    if df.empty or 'k' not in df.columns or metric not in df.columns: return
    df = df[df['k'] == k_val].sort_values(by=metric, ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    bars = sns.barplot(
        data=df, x='method', y=metric, 
        palette=[palette.get(m, '#333333') for m in df['method']],
        hue='method', legend=False
    )
    plt.title(f"{title} (Budget $k={k_val}$)", fontsize=14, pad=15)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=45, ha='right')
    
    # Add values on top of bars
    for bar in bars.patches:
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{bar.get_height():.2f}',
                 ha='center', va='bottom', fontsize=10)
                 
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_tradeoff_scatter(data, output_file, k_val=100):
    df = pd.DataFrame(data)
    if df.empty or 'I_T' not in df.columns or 'delta_rho' not in df.columns: return
    df = df[df['k'] == k_val]
    
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    
    sns.scatterplot(
        data=df, x='I_T', y='delta_rho',
        hue='method', palette=palette, s=200, edgecolor='black', zorder=5
    )
    
    plt.title(f"Trade-off: Spectral Drop vs Total Infections ($k={k_val}$)", fontsize=14, pad=15)
    plt.xlabel("Total Infections ($I_T$) -> Lower is Better", fontsize=12)
    plt.ylabel("Spectral Drop ($\Delta\\rho$) -> Higher is Better", fontsize=12)
    
    # Annotate points
    for i in range(df.shape[0]):
        plt.text(df['I_T'].iloc[i], df['delta_rho'].iloc[i] + 0.5, 
                 df['method'].iloc[i], horizontalalignment='center', size='10', color='black', weight='semibold')
    
    plt.legend([],[], frameon=False) # Hide legend as labels are on points
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

if __name__ == "__main__":
    reddit_data = load_json_data("results/reddit_results.json")
    gnutella_data = load_json_data("results/gnutella_results.json")
    
    methods_keep = ["SONIC", "DINO", "Degree", "Random", "HITS-Authority", "Acquaintance", "SourceOnly", "HITS-Hub"]
    red_f = [d for d in reddit_data if d.get('method') in methods_keep]
    gnu_f = [d for d in gnutella_data if d.get('method') in methods_keep]
    
    plot_bar_chart(gnu_f, 'delta_rho', "Gnutella: Spectral Radius Reduction", "$\Delta\\rho$", "results/gnutella_bar_chart.png", 100)
    plot_tradeoff_scatter(red_f, "results/reddit_scatter.png", 100)
    print("Graphs generated in results/")
