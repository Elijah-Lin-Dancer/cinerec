"""
Evaluation Visualization — Generate charts from *measured* results.

Produces (all in docs/):
1. Model comparison bar chart (all models × 3 metrics at K=10)
2. Training time comparison
3. Ablation study bar chart (from ablation_results.json)

Every chart is driven by an artefact on disk. When an artefact is missing the
corresponding chart is skipped rather than drawn from placeholder numbers — a
chart is indistinguishable from a real one once it is in the README.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

# Dark cinema theme
plt.style.use('dark_background')
COLORS = {
    'bg': '#0a0a0f',
    'text': '#e0e0e0',
    'gold': '#d4a843',
    'blue': '#4a9eff',
    'green': '#4ade80',
    'purple': '#a78bfa',
    'red': '#f87171',
    'orange': '#fb923c',
    'cyan': '#22d3ee',
}
MODEL_COLORS = ['#4a9eff', '#4ade80', '#d4a843', '#a78bfa', '#22d3ee', '#f87171']
MODEL_NAMES_ORDER = ['UserCF', 'ItemCF', 'SVD', 'NeuMF', 'LightGCN', 'MultiModalNCF']


def load_json(filename):
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_model_comparison(results):
    """Bar chart: all models × 3 metrics at K=10."""
    if not results:
        return

    metrics = ['HR@10', 'NDCG@10', 'Recall@10']
    metric_labels = ['HR@10', 'NDCG@10', 'Recall@10']
    n_models = len(MODEL_NAMES_ORDER)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    x = np.arange(n_models)
    width = 0.25

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = []
        for model in MODEL_NAMES_ORDER:
            if model in results and metric in results[model]:
                values.append(results[model][metric])
            else:
                values.append(0)
        bars = ax.bar(x + i * width, values, width, label=label,
                      color=MODEL_COLORS[i], edgecolor='white', linewidth=0.5, alpha=0.85)
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8, color=COLORS['text'])

    ax.set_xlabel('Models', color=COLORS['text'], fontsize=12)
    ax.set_ylabel('Score', color=COLORS['text'], fontsize=12)
    ax.set_title('CineRec — Model Comparison (MovieLens 100K, leave-last-5-out)',
                color=COLORS['gold'], fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(MODEL_NAMES_ORDER, color=COLORS['text'], fontsize=10)
    ax.legend(loc='upper left', facecolor='#1a1a2e', edgecolor=COLORS['gold'],
             labelcolor=COLORS['text'])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.grid(axis='y', alpha=0.2, color='#333')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'model_comparison.png'), dpi=150,
               facecolor=COLORS['bg'], bbox_inches='tight')
    plt.close()
    print("Saved docs/model_comparison.png")


def plot_training_time(results):
    """Bar chart: training time per model."""
    if not results:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    times = []
    labels = []
    colors = []
    for i, model in enumerate(MODEL_NAMES_ORDER):
        if model in results and 'train_time' in results[model]:
            times.append(results[model]['train_time'])
            labels.append(model)
            colors.append(MODEL_COLORS[i])

    if not times:
        return

    bars = ax.barh(labels, times, color=colors, edgecolor='white', linewidth=0.5, alpha=0.85)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2.,
               f'{t:.1f}s', ha='left', va='center', fontsize=10, color=COLORS['text'])

    ax.set_xlabel('Training Time (seconds)', color=COLORS['text'], fontsize=12)
    ax.set_title('CineRec — Training Time Comparison',
                color=COLORS['gold'], fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2, color='#333')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'training_time.png'), dpi=150,
               facecolor=COLORS['bg'], bbox_inches='tight')
    plt.close()
    print("Saved docs/training_time.png")


def plot_ablation(ablation):
    """Ablation study bar chart for MultiModalNCF variants (measured values only)."""
    if not ablation or ablation.get("available") is False:
        print("Skipping ablation chart: ablation_results.json not available.")
        return

    variants = [k for k in ablation if not k.startswith("_") and k != "available"]
    if not variants:
        return
    hr10 = [ablation[v].get('HR@10', 0) for v in variants]
    ndcg10 = [ablation[v].get('NDCG@10', 0) for v in variants]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    x = np.arange(len(variants))
    width = 0.35

    bars1 = ax.bar(x - width/2, hr10, width, label='HR@10',
                   color=COLORS['gold'], edgecolor='white', linewidth=0.5, alpha=0.85)
    bars2 = ax.bar(x + width/2, ndcg10, width, label='NDCG@10',
                   color=COLORS['cyan'], edgecolor='white', linewidth=0.5, alpha=0.85)

    for bar, val in zip(bars1, hr10):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=8, color=COLORS['text'])
    for bar, val in zip(bars2, ndcg10):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=8, color=COLORS['text'])

    ax.set_xlabel('Variant', color=COLORS['text'], fontsize=12)
    ax.set_ylabel('Score', color=COLORS['text'], fontsize=12)
    ax.set_title('CineRec — Ablation Study (MultiModalNCF)',
                color=COLORS['gold'], fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variants, color=COLORS['text'], fontsize=9, rotation=15)
    ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor=COLORS['gold'],
             labelcolor=COLORS['text'])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.grid(axis='y', alpha=0.2, color='#333')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'ablation_study.png'), dpi=150,
               facecolor=COLORS['bg'], bbox_inches='tight')
    plt.close()
    print("Saved docs/ablation_study.png")


def plot_coldstart(data):
    """Cold-start bar chart: content-aware vs content-blind on held-out new items."""
    if not data:
        print("Skipping cold-start chart: coldstart_results.json not available.")
        return
    metrics = data.get("metrics")
    if not metrics:
        return

    names = [k for k in metrics if not k.startswith("_")]
    hr10 = [metrics[n].get('HR@10', 0) for n in names]
    ndcg10 = [metrics[n].get('NDCG@10', 0) for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, hr10, width, label='HR@10',
                   color=COLORS['gold'], edgecolor='white', linewidth=0.5, alpha=0.85)
    bars2 = ax.bar(x + width/2, ndcg10, width, label='NDCG@10',
                   color=COLORS['green'], edgecolor='white', linewidth=0.5, alpha=0.85)

    for bar, val in zip(bars1, hr10):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9, color=COLORS['text'])
    for bar, val in zip(bars2, ndcg10):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9, color=COLORS['text'])

    n_cold = data.get("n_cold_items", "?")
    ax.set_ylabel('Score', color=COLORS['text'], fontsize=12)
    ax.set_title(f'CineRec — Cold-Start Study ({n_cold} items held out entirely)',
                 color=COLORS['gold'], fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, color=COLORS['text'], fontsize=10)
    ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor=COLORS['gold'],
             labelcolor=COLORS['text'])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.grid(axis='y', alpha=0.2, color='#333')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'coldstart.png'), dpi=150,
               facecolor=COLORS['bg'], bbox_inches='tight')
    plt.close()
    print("Saved docs/coldstart.png")


def generate_all():
    """Generate all visualization charts from measured artefacts."""
    print("Generating evaluation visualizations...")

    results = load_json("eval_results.json")
    if results:
        plot_model_comparison(results)
        plot_training_time(results)
    else:
        print("No eval_results.json found — skipping model comparison charts.")

    plot_ablation(load_json("ablation_results.json"))
    plot_coldstart(load_json("coldstart_results.json"))

    print("\nVisualizations generated in docs/ (only measured data).")


if __name__ == "__main__":
    generate_all()
