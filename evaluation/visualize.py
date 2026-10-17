"""
Evaluation Visualization — Generate charts from *measured* results.

Produces (all in docs/):
1. Model comparison bar chart (all models × 3 metrics at K=10)
2. Training time comparison
3. Ablation study bar chart (from ablation_results.json)

Every chart is driven by an artefact on disk. When an artefact is missing the
corresponding chart is skipped rather than drawn from placeholder numbers — a
chart is indistinguishable from a real one once it is in the README.

Two themes are supported. The dark cinema theme (default) backs the README and
the Streamlit dashboard; the ``paper`` theme renders the same charts on a white
background at 300 dpi, as required by the ChinaXiv submission spec, and writes
them to ``docs/paper_<name>.png`` so the README assets are left untouched.
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

from config import PROCESSED_DIR  # noqa: E402  (path set up above)

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

# Dark cinema theme (README / Streamlit) and the white print theme (paper).
DARK = {
    'style': 'dark_background',
    'bg': '#0a0a0f',
    'text': '#e0e0e0',
    'gold': '#d4a843',
    'cyan': '#22d3ee',
    'green': '#4ade80',
    'legend_bg': '#1a1a2e',
    'grid': '#333333',
    'grid_alpha': 0.2,
    'bar_alpha': 0.85,
    'edge': '#ffffff',
}
PAPER = {
    'style': 'default',
    'bg': '#ffffff',
    'text': '#111111',
    'gold': '#7a5c12',
    'cyan': '#17becf',
    'green': '#2ca02c',
    'legend_bg': '#ffffff',
    'grid': '#cccccc',
    'grid_alpha': 0.35,
    'bar_alpha': 0.9,
    'edge': '#555555',
}
DARK_MODEL_COLORS = ['#4a9eff', '#4ade80', '#d4a843', '#f472b6',
                     '#a78bfa', '#22d3ee', '#fb923c', '#f87171']
PAPER_MODEL_COLORS = ['#1f77b4', '#2ca02c', '#d4a843', '#e377c2',
                      '#9467bd', '#17becf', '#ff7f0e', '#d62728']

COLORS = DARK
MODEL_COLORS = DARK_MODEL_COLORS
DPI = 150
PREFIX = ''
PAPER_MODE = False
plt.style.use(DARK['style'])

# Paper theme: figures are typeset at their final printed width (120 mm) so the
# flat 8 pt in-figure text required by the ChinaXiv submission spec prints at
# 8 pt. The dark README/Streamlit charts keep their original on-screen size.
PRINT_WIDTH_IN = 120 / 25.4  # 4.7244 in == the paper's 120 mm figure width
FONT_PT = 8                  # spec: figure/table text 8 pt


def _figsize(dark_wh, paper_wh):
    """Figure size for the active theme (paper renders at final print size)."""
    return paper_wh if PAPER_MODE else dark_wh


def _fs(dark_pt):
    """In-figure text size: the paper theme uses the spec's flat 8 pt."""
    return FONT_PT if PAPER_MODE else dark_pt

# The six servable ladder models plus the two offline SOTA baselines (BPR-MF,
# VBPR), in ascending order of sophistication.
MODEL_NAMES_ORDER = ['UserCF', 'ItemCF', 'SVD', 'BPRMF', 'NeuMF', 'LightGCN', 'VBPR', 'MultiModalNCF']


def _use_theme(paper):
    """Select the module-wide chart theme, output dpi and filename prefix.

    Called at the top of every plot function so a single code path serves both
    the dark README charts and the white paper charts.
    """
    global COLORS, MODEL_COLORS, DPI, PREFIX, PAPER_MODE
    COLORS = PAPER if paper else DARK
    MODEL_COLORS = PAPER_MODEL_COLORS if paper else DARK_MODEL_COLORS
    DPI = 300 if paper else 150
    PREFIX = 'paper_' if paper else ''
    PAPER_MODE = paper
    plt.style.use(COLORS['style'])
    # The ChinaXiv spec requires in-figure text to be Times New Roman at 8 pt.
    # Liberation Serif is metrically compatible with TNR; the README charts keep
    # matplotlib's default sans-serif so their look is unchanged.
    if paper:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Liberation Serif', 'DejaVu Serif']
    else:
        plt.rcParams['font.family'] = 'sans-serif'


def load_json(filename):
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _unwrap(value):
    """Read a metric that is either a scalar or a ``{"mean","std"}`` aggregate.

    Multi-seed artefacts store ``{"mean","std"}`` per metric; the charts plot the
    mean. Single-seed artefacts store a plain float, which passes through.
    """
    if isinstance(value, dict):
        return value.get("mean", 0.0)
    return value


def _save(fig, name):
    """Write ``fig`` to docs/, honouring the active theme's dpi and prefix.

    The paper theme saves the full figure rather than a tight crop, so the PNG's
    physical width equals the 120 mm insert width and the 8 pt text prints 1:1.
    """
    path = os.path.join(DOCS_DIR, PREFIX + name + '.png')
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, facecolor=COLORS['bg'],
                bbox_inches=(None if PAPER_MODE else 'tight'))
    plt.close()
    print("Saved " + os.path.relpath(path, os.path.join(DOCS_DIR, "..")))


def plot_model_comparison(results, paper=False):
    """Bar chart: all models × 3 metrics at K=10."""
    _use_theme(paper)
    if not results:
        return

    metrics = ['HR@10', 'NDCG@10', 'Recall@10']
    metric_labels = ['HR@10', 'NDCG@10', 'Recall@10']
    n_models = len(MODEL_NAMES_ORDER)

    fig, ax = plt.subplots(figsize=_figsize((12, 6), (PRINT_WIDTH_IN, 2.9)))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    x = np.arange(n_models)
    width = 0.25

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = []
        for model in MODEL_NAMES_ORDER:
            if model in results and metric in results[model]:
                values.append(_unwrap(results[model][metric]))
            else:
                values.append(0)
        bars = ax.bar(x + i * width, values, width, label=label,
                      color=MODEL_COLORS[i], edgecolor=COLORS['edge'], linewidth=0.5, alpha=COLORS['bar_alpha'])
        # Add value labels (README only: 24 bars at 120 mm cannot hold 8 pt text)
        if not PAPER_MODE:
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=_fs(8), color=COLORS['text'])

    ax.set_xlabel('Models', color=COLORS['text'], fontsize=_fs(12))
    ax.set_ylabel('Score', color=COLORS['text'], fontsize=_fs(12))
    ax.set_title('CineRec — Model Comparison (MovieLens 100K, leave-last-5-out)',
                color=COLORS['gold'], fontsize=_fs(13), fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(MODEL_NAMES_ORDER, color=COLORS['text'], fontsize=_fs(10),
                       rotation=(45 if PAPER_MODE else 0),
                       ha=('right' if PAPER_MODE else 'center'))
    ax.legend(loc='upper left', facecolor=COLORS['legend_bg'], edgecolor=COLORS['gold'],
             labelcolor=COLORS['text'], fontsize=_fs(10))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', labelsize=_fs(10))
    ax.grid(axis='y', alpha=COLORS['grid_alpha'], color=COLORS['grid'])
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _save(fig, 'model_comparison')


def plot_training_time(results, paper=False):
    """Bar chart: training time per model."""
    _use_theme(paper)
    if not results:
        return

    fig, ax = plt.subplots(figsize=_figsize((10, 5), (PRINT_WIDTH_IN, 2.5)))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    times = []
    labels = []
    colors = []
    for i, model in enumerate(MODEL_NAMES_ORDER):
        if model in results and 'train_time' in results[model]:
            times.append(_unwrap(results[model]['train_time']))
            labels.append(model)
            colors.append(MODEL_COLORS[i])

    if not times:
        return

    bars = ax.barh(labels, times, color=colors, edgecolor=COLORS['edge'], linewidth=0.5, alpha=COLORS['bar_alpha'])
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2.,
               f'{t:.1f}s', ha='left', va='center', fontsize=_fs(10), color=COLORS['text'])

    ax.set_xlabel('Training Time (seconds)', color=COLORS['text'], fontsize=_fs(12))
    ax.set_title('CineRec — Training Time Comparison',
                color=COLORS['gold'], fontsize=_fs(14), fontweight='bold')
    ax.tick_params(axis='both', labelsize=_fs(10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=COLORS['grid_alpha'], color=COLORS['grid'])
    ax.set_axisbelow(True)

    _save(fig, 'training_time')


def plot_ablation(ablation, paper=False):
    """Ablation study bar chart for MultiModalNCF variants (measured values only)."""
    _use_theme(paper)
    if not ablation or ablation.get("available") is False:
        print("Skipping ablation chart: ablation_results.json not available.")
        return

    variants = [k for k in ablation if not k.startswith("_") and k != "available"]
    if not variants:
        return
    hr10 = [_unwrap(ablation[v].get('HR@10', 0)) for v in variants]
    ndcg10 = [_unwrap(ablation[v].get('NDCG@10', 0)) for v in variants]

    fig, ax = plt.subplots(figsize=_figsize((12, 5), (PRINT_WIDTH_IN, 2.6)))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    x = np.arange(len(variants))
    width = 0.35

    bars1 = ax.bar(x - width/2, hr10, width, label='HR@10',
                   color=COLORS['gold'], edgecolor=COLORS['edge'], linewidth=0.5, alpha=COLORS['bar_alpha'])
    bars2 = ax.bar(x + width/2, ndcg10, width, label='NDCG@10',
                   color=COLORS['cyan'], edgecolor=COLORS['edge'], linewidth=0.5, alpha=COLORS['bar_alpha'])

    for bar, val in zip(bars1, hr10):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=_fs(8), color=COLORS['text'])
    for bar, val in zip(bars2, ndcg10):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=_fs(8), color=COLORS['text'])

    ax.set_xlabel('Variant', color=COLORS['text'], fontsize=_fs(12))
    ax.set_ylabel('Score', color=COLORS['text'], fontsize=_fs(12))
    ax.set_title('CineRec — Ablation Study (MultiModalNCF)',
                color=COLORS['gold'], fontsize=_fs(14), fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variants, color=COLORS['text'], fontsize=_fs(9),
                       rotation=(30 if PAPER_MODE else 15),
                       ha=('right' if PAPER_MODE else 'center'))
    ax.legend(loc='upper right', facecolor=COLORS['legend_bg'], edgecolor=COLORS['gold'],
             labelcolor=COLORS['text'], fontsize=_fs(10))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.tick_params(axis='y', labelsize=_fs(10))
    ax.grid(axis='y', alpha=COLORS['grid_alpha'], color=COLORS['grid'])
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _save(fig, 'ablation_study')


def plot_coldstart(data, paper=False):
    """Cold-start bar chart: content-aware vs content-blind on held-out new items."""
    _use_theme(paper)
    if not data:
        print("Skipping cold-start chart: coldstart_results.json not available.")
        return
    metrics = data.get("metrics")
    if not metrics:
        return

    names = [k for k in metrics if not k.startswith("_")]
    hr10 = [_unwrap(metrics[n].get('HR@10', 0)) for n in names]
    ndcg10 = [_unwrap(metrics[n].get('NDCG@10', 0)) for n in names]

    fig, ax = plt.subplots(figsize=_figsize((10, 5), (PRINT_WIDTH_IN, 2.7)))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, hr10, width, label='HR@10',
                   color=COLORS['gold'], edgecolor=COLORS['edge'], linewidth=0.5, alpha=COLORS['bar_alpha'])
    bars2 = ax.bar(x + width/2, ndcg10, width, label='NDCG@10',
                   color=COLORS['green'], edgecolor=COLORS['edge'], linewidth=0.5, alpha=COLORS['bar_alpha'])

    for bar, val in zip(bars1, hr10):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=_fs(9), color=COLORS['text'])
    for bar, val in zip(bars2, ndcg10):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=_fs(9), color=COLORS['text'])

    n_cold = data.get("n_cold_items", "?")
    ax.set_ylabel('Score', color=COLORS['text'], fontsize=_fs(12))
    ax.set_title(f'CineRec — Cold-Start Study ({n_cold} items held out entirely)',
                 color=COLORS['gold'], fontsize=_fs(13), fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, color=COLORS['text'], fontsize=_fs(10))
    ax.legend(loc='upper right', facecolor=COLORS['legend_bg'], edgecolor=COLORS['gold'],
             labelcolor=COLORS['text'], fontsize=_fs(10))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.tick_params(axis='y', labelsize=_fs(10))
    ax.grid(axis='y', alpha=COLORS['grid_alpha'], color=COLORS['grid'])
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Paper theme compresses the figure; leave head room so the longest bar's
    # 8 pt value label does not touch the 8 pt title.
    if PAPER_MODE:
        ax.set_ylim(top=max(max(hr10, default=0), max(ndcg10, default=0)) * 1.20)

    _save(fig, 'coldstart')


def _generate(paper):
    """Render all charts in the requested theme from the measured artefacts."""
    # Prefer the multi-seed artefacts (mean±std) when present; fall back to the
    # single-seed ones so charts match whichever run produced the README table.
    results = load_json("eval_multiseed.json") or load_json("eval_results.json")
    if results:
        plot_model_comparison(results, paper=paper)
        plot_training_time(results, paper=paper)
    else:
        print("No eval_results.json found — skipping model comparison charts.")

    plot_ablation(load_json("ablation_multiseed.json") or load_json("ablation_results.json"),
                  paper=paper)
    plot_coldstart(load_json("coldstart_multiseed.json") or load_json("coldstart_results.json"),
                   paper=paper)


def generate_all():
    """Generate the dark README/dashboard charts from measured artefacts."""
    print("Generating evaluation visualizations...")
    _generate(paper=False)
    print("\nVisualizations generated in docs/ (only measured data).")


def generate_paper():
    """Generate the white 300 dpi paper charts (docs/paper_*.png)."""
    print("Generating paper visualizations (white background, 300 dpi)...")
    _generate(paper=True)
    print("\nPaper visualizations generated in docs/ as paper_*.png.")


if __name__ == "__main__":
    generate_all()
