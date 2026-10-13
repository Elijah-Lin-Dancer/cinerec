"""CineRec — live-inference companion (Streamlit Community Cloud).

The main deployment serves the bilingual product UI in ``lite`` mode: every
recommendation comes from a precomputed cache so the app fits a small free tier.
This app is the complementary half — it loads the *trained PyTorch models* and
runs them live, which is what the product's `full` mode does. It exists so the
six-model ladder, the multi-modal content tower and the cold-start path can be
exercised interactively without a GPU host.

Everything is reused from the application itself — ``models.registry`` for model
loading, ``models.explain`` for the reasons and the committed artefacts under
``data/processed`` for metadata. No logic is duplicated, so what you see here is
what the API serves.

Run locally::

    pip install -r deploy/streamlit/requirements.txt
    streamlit run streamlit_app.py

Deploy on Streamlit Community Cloud from the companion directory — main file
path ``deploy/streamlit/streamlit_app.py`` — which sits next to the CPU-torch
requirements the live models need. See ``deploy/streamlit/README.md``.
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from config import PROCESSED_DIR, SECOND_PROCESSED_DIR
from models.explain import RecommenderExplainer
from models.registry import ALGORITHMS, AlgorithmUnavailable, load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RATINGS_CSV = os.path.join(BASE_DIR, "data", "raw", "ratings.csv")
MOVIES_JSON = os.path.join(PROCESSED_DIR, "movies_enriched.json")

#: Fixed depth for the side-by-side comparison so columns are comparable.
COMPARE_K = 10

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  --cr-bg: #08080f;
  --cr-accent: #d4a843;
  --cr-accent-light: #e8c36a;
  --cr-border: rgba(255,255,255,.09);
  --cr-muted: #8b8ba6;
  --cr-text: #e8e8ed;
  --cr-blue: #4a9eff;
  --cr-purple: #a78bfa;
  --cr-green: #4ade80;
  --cr-red: #f87171;
}

/* --- Typography (mirrors the product's front end) --- */
html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {
  font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
}
code, pre, .cr-score, .cr-mono {
  font-family: 'JetBrains Mono', monospace;
}

/* --- Layered cinematic background (base wash + drifting aurora) --- */
.stApp {
  background-color: var(--cr-bg);
  background-image:
    radial-gradient(120% 85% at 50% -15%, rgba(74,158,255,.10), transparent 60%),
    radial-gradient(85% 65% at 88% 12%, rgba(167,139,250,.09), transparent 58%),
    radial-gradient(90% 70% at 8% 100%, rgba(212,168,67,.07), transparent 60%),
    linear-gradient(180deg,#0b0b18 0%,#08080f 55%,#0b0812 100%);
  background-attachment: fixed;
}
.stApp::before {
  content: '';
  position: fixed;
  inset: -25%;
  z-index: 0;
  pointer-events: none;
  background:
    radial-gradient(38% 42% at 20% 25%, rgba(212,168,67,.20), transparent 62%),
    radial-gradient(42% 46% at 82% 18%, rgba(74,158,255,.18), transparent 60%),
    radial-gradient(48% 50% at 72% 82%, rgba(167,139,250,.16), transparent 64%),
    radial-gradient(36% 40% at 15% 85%, rgba(74,222,128,.07), transparent 60%);
  animation: cr-aurora 26s ease-in-out infinite alternate;
  will-change: transform;
}
@keyframes cr-aurora {
  0%   { transform: translate3d(-3%,-2%,0) scale(1.08) rotate(0deg); }
  50%  { transform: translate3d(2%,3%,0) scale(1.16) rotate(4deg); }
  100% { transform: translate3d(4%,-3%,0) scale(1.10) rotate(-3deg); }
}

/* Keep Streamlit chrome transparent; lift real content above the aurora. */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stMain"], section.main { position: relative; z-index: 1; }

/* --- Layout --- */
.block-container, [data-testid="stMainBlockContainer"] {
  padding-top: 2.2rem;
  max-width: 1180px;
}

/* --- Hero with film-strip logo --- */
.cr-hero-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: .15rem;
}
.cr-film-logo {
  width: 54px;
  height: 54px;
  flex-shrink: 0;
  border-radius: 10px;
  background:
    linear-gradient(135deg, rgba(212,168,67,.18), rgba(74,158,255,.12)),
    repeating-linear-gradient(0deg, #1a1a2e 0 6px, #0d0d1a 6px 8px);
  border: 1px solid rgba(212,168,67,.45);
  box-shadow: 0 0 22px rgba(212,168,67,.25), inset 0 0 0 2px rgba(0,0,0,.25);
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--cr-accent-light);
  font-size: 1.5rem;
  font-weight: 900;
}
.cr-film-logo::before, .cr-film-logo::after {
  content: '';
  position: absolute;
  left: -3px; right: -3px;
  height: 5px;
  background:
    repeating-linear-gradient(90deg, transparent 0 4px, rgba(212,168,67,.5) 4px 6px, transparent 6px 10px);
}
.cr-film-logo::before { top: -3px; }
.cr-film-logo::after { bottom: -3px; }
.cr-hero {
  font-size: 2.4rem;
  font-weight: 800;
  letter-spacing: -.02em;
  line-height: 1.1;
  background: linear-gradient(120deg, var(--cr-accent-light), var(--cr-accent) 45%, #f6e6b8 70%, var(--cr-accent));
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  filter: drop-shadow(0 0 26px rgba(212,168,67,.28));
}
.cr-sub { color: var(--cr-muted); margin-bottom: 1rem; font-size: .95rem; }

/* --- KPI strip --- */
.cr-kpi-row {
  display: flex;
  gap: .7rem;
  margin-bottom: 1.4rem;
  flex-wrap: wrap;
}
.cr-kpi {
  flex: 1 1 0;
  min-width: 130px;
  border: 1px solid var(--cr-border);
  border-radius: 12px;
  padding: .65rem .9rem;
  background: linear-gradient(160deg, rgba(30,30,56,.55), rgba(15,15,30,.45));
  backdrop-filter: blur(8px);
  position: relative;
  overflow: hidden;
}
.cr-kpi::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--cr-accent), transparent);
  opacity: .6;
}
.cr-kpi-label { color: var(--cr-muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; }
.cr-kpi-value { color: var(--cr-text); font-size: 1.5rem; font-weight: 800; line-height: 1.2; margin-top: .12rem; }
.cr-kpi-value .cr-accent { color: var(--cr-accent-light); }
.cr-kpi-sub { color: var(--cr-muted); font-size: .68rem; margin-top: .1rem; }

/* --- Recommendation cards (poster + text fused into one) --- */
.cr-card {
  border: 1px solid rgba(148,163,184,.22);
  border-radius: 14px;
  overflow: hidden;
  height: 100%;
  background: linear-gradient(160deg, rgba(30,30,56,.72), rgba(15,15,30,.62));
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
  transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
  position: relative;
}
.cr-card:hover {
  transform: translateY(-3px);
  border-color: rgba(212,168,67,.38);
  box-shadow: 0 12px 30px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.07);
}
.cr-rank {
  position: absolute;
  top: 8px; left: 8px;
  z-index: 3;
  width: 28px; height: 28px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--cr-accent-light), var(--cr-accent));
  color: #0a0a0f;
  font-weight: 800;
  font-size: .8rem;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 2px 10px rgba(0,0,0,.5);
}
.cr-poster-wrap { position: relative; width: 100%; aspect-ratio: 2/3; background: linear-gradient(135deg, #1a1a2e, #0d0d1a); overflow: hidden; }
.cr-poster-wrap img { width: 100%; height: 100%; object-fit: cover; display: block; }
.cr-poster-grad {
  position: absolute; left: 0; right: 0; bottom: 0; height: 55%;
  background: linear-gradient(180deg, transparent, rgba(8,8,15,.95));
}
.cr-poster-ph {
  width: 100%; height: 100%;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, rgba(212,168,67,.12), rgba(74,158,255,.10));
  color: var(--cr-accent-light);
  font-size: 2.2rem; font-weight: 800;
}
.cr-card-body { padding: .55rem .8rem .7rem .8rem; }
.cr-title { font-weight: 600; font-size: .92rem; line-height: 1.28; color: var(--cr-text); margin: 0 0 .15rem 0; }
.cr-meta { color: var(--cr-muted); font-size: .73rem; }
.cr-score {
  display: inline-block;
  margin-top: .35rem;
  color: #0a0a0f;
  background: linear-gradient(135deg, var(--cr-accent-light), var(--cr-accent));
  font-weight: 700;
  font-size: .72rem;
  padding: .1rem .5rem;
  border-radius: 20px;
}
.cr-reason {
  font-size: .76rem;
  color: #cbd5e1;
  margin-top: .4rem;
  border-top: 1px solid var(--cr-border);
  padding-top: .35rem;
}

[data-testid="stImage"] img { border-radius: 12px; }

/* --- CSS bar chart (no plotly dependency) --- */
.cr-chart { display: flex; flex-direction: column; gap: .55rem; margin: .5rem 0; }
.cr-bar-row { display: grid; grid-template-columns: 110px 1fr 60px; align-items: center; gap: .6rem; }
.cr-bar-label { color: var(--cr-text); font-size: .82rem; font-weight: 600; text-align: right; }
.cr-bar-track { height: 22px; background: rgba(255,255,255,.05); border-radius: 6px; overflow: hidden; border: 1px solid var(--cr-border); }
.cr-bar-fill { height: 100%; border-radius: 5px; transition: width .8s cubic-bezier(.2,.8,.2,1); }
.cr-bar-val { color: var(--cr-muted); font-size: .76rem; font-family: 'JetBrains Mono', monospace; }

/* --- Heatmap (Jaccard) --- */
.cr-heatmap { display: grid; gap: 2px; margin: .5rem 0; }
.cr-heat-cell { display: flex; align-items: center; justify-content: center; font-size: .68rem; font-family: 'JetBrains Mono', monospace; border-radius: 3px; }
.cr-heat-label { color: var(--cr-muted); font-size: .68rem; font-weight: 600; padding: 0 4px; }

/* --- Tabs as chips --- */
.stTabs [data-baseweb="tab-list"] {
  gap: .35rem;
  background: rgba(255,255,255,.04);
  border: 1px solid var(--cr-border);
  border-radius: 12px;
  padding: .3rem;
}
.stTabs [data-baseweb="tab"] {
  height: auto;
  padding: .45rem 1.05rem;
  border-radius: 9px;
  color: var(--cr-muted);
  font-weight: 600;
}
.stTabs [aria-selected="true"] {
  background: rgba(212,168,67,.12) !important;
  color: var(--cr-accent) !important;
  box-shadow: 0 0 12px rgba(212,168,67,.15);
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none; }

/* --- Sidebar --- */
[data-testid="stSidebar"] {
  background: rgba(13,13,26,.72);
  border-right: 1px solid var(--cr-border);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
}
[data-testid="stSidebar"] h2 { color: var(--cr-accent-light); font-weight: 700; }
[data-testid="stSidebar"] h3 { color: var(--cr-text); font-size: .9rem; font-weight: 700; }

/* Algorithm ladder — six dots on a line, current one highlighted. */
.cr-ladder { display: flex; align-items: center; gap: 0; margin: .6rem 0 .2rem 0; padding: 0 4px; }
.cr-ladder-node { flex: 1; display: flex; flex-direction: column; align-items: center; position: relative; }
.cr-ladder-dot { width: 12px; height: 12px; border-radius: 50%; background: rgba(255,255,255,.18); border: 2px solid transparent; transition: all .2s ease; z-index: 1; }
.cr-ladder-node.active .cr-ladder-dot { background: var(--cr-accent); border-color: var(--cr-accent-light); box-shadow: 0 0 10px rgba(212,168,67,.6); transform: scale(1.35); }
.cr-ladder-name { font-size: .58rem; color: var(--cr-muted); margin-top: .25rem; white-space: nowrap; }
.cr-ladder-node.active .cr-ladder-name { color: var(--cr-accent-light); font-weight: 700; }
.cr-ladder-line { position: absolute; top: 5px; left: 50%; right: -50%; height: 2px; background: rgba(255,255,255,.10); z-index: 0; }
.cr-ladder-node:last-child .cr-ladder-line { display: none; }

/* User profile mini-card in the sidebar. */
.cr-profile { border: 1px solid var(--cr-border); border-radius: 10px; padding: .55rem .7rem; background: rgba(18,18,34,.45); margin-top: .35rem; }
.cr-profile-row { display: flex; justify-content: space-between; font-size: .74rem; color: var(--cr-muted); margin: .12rem 0; }
.cr-profile-row b { color: var(--cr-text); font-weight: 600; }
.cr-profile-genres { margin-top: .3rem; display: flex; flex-wrap: wrap; gap: .25rem; }
.cr-profile-genre { font-size: .62rem; padding: .08rem .4rem; border-radius: 8px; background: rgba(212,168,67,.12); color: var(--cr-accent-light); border: 1px solid rgba(212,168,67,.25); }

/* --- Headings, expanders, tables --- */
h1, h2, h3, h4 { color: var(--cr-text); letter-spacing: -.01em; }
h3 { font-weight: 700; }
[data-testid="stExpander"] {
  border: 1px solid var(--cr-border);
  border-radius: 12px;
  background: rgba(18,18,34,.5);
  overflow: hidden;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--cr-border);
  border-radius: 12px;
  overflow: hidden;
}

/* --- Controls --- */
.stSlider [role="slider"] {
  background: var(--cr-accent) !important;
  border-color: var(--cr-accent) !important;
  box-shadow: 0 0 10px rgba(212,168,67,.45);
}
.stButton > button, .stDownloadButton > button {
  border-radius: 9px;
  border: 1px solid rgba(212,168,67,.45);
  background: linear-gradient(135deg, var(--cr-accent), var(--cr-accent-light));
  color: #0a0a0f;
  font-weight: 700;
  transition: transform .2s ease, box-shadow .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(212,168,67,.35);
  border-color: var(--cr-accent-light);
}
[data-testid="stAlert"] { border-radius: 12px; }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--cr-muted); }

/* --- Scrollbar --- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(212,168,67,.25); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(212,168,67,.45); }

@media (prefers-reduced-motion: reduce) {
  .stApp::before { animation: none; }
  .cr-bar-fill, .cr-card { transition: none; }
}
</style>
"""


# --------------------------------------------------------------------------- #
# Data (all committed under data/, so a fresh clone works)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_movies():
    """Movie metadata keyed by id."""
    with open(MOVIES_JSON, encoding="utf-8") as f:
        return {int(m["id"]): m for m in json.load(f)}


@st.cache_data(show_spinner=False)
def load_ratings():
    """The MovieLens-100K seed interactions."""
    return pd.read_csv(RATINGS_CSV)[["user_id", "item_id", "rating"]]


@st.cache_data(show_spinner=False)
def load_artifact(filename, directory=None):
    """One of the committed evaluation JSON artefacts (empty dict when absent)."""
    path = os.path.join(directory or PROCESSED_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ms_value(art, key, metric):
    """``(mean, std)`` from a multiseed artefact, falling back to a flat value.

    Multiseed JSONs store ``{model: {metric: {"mean": m, "std": s}}}``; the
    single-seed ones store plain numbers. Every chart below works with either.
    """
    v = art.get(key, {}).get(metric)
    if isinstance(v, dict):
        return float(v.get("mean") or 0.0), float(v.get("std") or 0.0)
    return float(v or 0.0), 0.0


def ms_label(art, key, metric, fmt="{:.4f}"):
    """``mean ± std`` label, or the plain value when there is no spread."""
    mean, std = ms_value(art, key, metric)
    return fmt.format(mean) + (f" ± {std:.4f}" if std else "")


@st.cache_data(show_spinner=False)
def all_user_ids():
    return [int(u) for u in sorted(load_ratings()["user_id"].unique())]


@st.cache_data(show_spinner=False)
def user_history(user_id):
    """``{item_id: rating}`` for one user."""
    df = load_ratings()
    sub = df[df["user_id"] == int(user_id)]
    return {int(i): float(r) for i, r in zip(sub["item_id"], sub["rating"])}


# --------------------------------------------------------------------------- #
# Models (loaded through the application's registry, never re-implemented)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def get_model(name):
    """Load one algorithm. Raises AlgorithmUnavailable when it cannot be served."""
    return load_model(name)


@st.cache_resource(show_spinner="Loading the multi-modal model...")
def get_multimodal_variant(disabled):
    """A MultiModalNCF with the given content modalities blanked.

    Uses the model's own ``disabled_features`` hook (the same switch the
    ablation study flips), so the content tower receives no signal from those
    inputs. One checkpoint is shared; only the inputs change.
    """
    from models.multimodal_ncf import MultiModalNCF

    model = MultiModalNCF(embedding_dim=32, mlp_dims=(128, 64, 32), disabled_features=list(disabled))
    model.load(os.path.join(PROCESSED_DIR, "model_multimodalncf.pt"))
    return model


@st.cache_resource(show_spinner="Preparing explanations...")
def get_explainer():
    """The product's explainer, fed from the same seed ratings."""
    explainer = RecommenderExplainer()
    explainer.load_data()
    df = load_ratings()
    explainer.load_user_ratings({
        "user_id": df["user_id"].to_numpy(),
        "item_id": df["item_id"].to_numpy(),
        "rating": df["rating"].to_numpy(),
    })
    return explainer


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def top_items(model, user_id, top_k, exclude):
    """Return ``([(item_id, score)], error)`` for one model."""
    try:
        recs = model.recommend(int(user_id), top_k=int(top_k),
                               exclude_items={int(i) for i in exclude})
    except Exception as exc:  # a broken artefact must not take the page down
        return [], f"{type(exc).__name__}: {exc}"
    return [(int(i), float(s)) for i, s in recs], ""


def cold_rank(model, user_id, candidate_ids, top_k):
    """Rank a candidate pool through the content tower only.

    Calls ``forward_cold`` directly so the item's collaborative embedding is
    bypassed — exactly the path the cold-start study measures — instead of
    ranking a new item against the full warm catalogue.
    """
    import torch

    idx = np.asarray(sorted({int(i) for i in candidate_ids}), dtype=int)
    if idx.size == 0:
        return []
    net = model.net
    net.eval()
    with torch.no_grad():
        users = torch.full((idx.size,), int(user_id), dtype=torch.long).to(model.device)
        text = torch.FloatTensor(model.text_emb[idx]).to(model.device)
        image = torch.FloatTensor(model.image_emb[idx]).to(model.device)
        genre = torch.FloatTensor(model.genre_vec[idx]).to(model.device)
        scores = net.forward_cold(users, text, image, genre).cpu().numpy()
    order = np.argsort(scores)[::-1][:top_k]
    return [(int(idx[j]), float(scores[j])) for j in order]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def reason_text(reason, lang):
    """Pick the requested language, falling back on whatever the reason carries."""
    if lang == "中文":
        return reason.get("reason_zh") or reason.get("reason_en") or reason.get("reason", "")
    return reason.get("reason_en") or reason.get("reason_zh") or reason.get("reason", "")


def movie_title(movies, item_id):
    return movies.get(int(item_id), {}).get("title", f"Item {item_id}")


def render_cards(recs, movies, reasons_by_item=None, lang="中文", columns=4):
    """Grid of recommendation cards: poster fused with title/score/reasons.

    The card is one fused HTML block (poster + gradient + body), so it reads as
    a single object rather than a detached ``st.image`` plus a ``st.markdown``.
    Server-provided strings are HTML-escaped before interpolation; nothing is
    ever ``innerHTML``-ed raw from the model output.
    """
    reasons_by_item = reasons_by_item or {}
    for start in range(0, len(recs), columns):
        cols = st.columns(columns)
        for idx, col in enumerate(cols):
            rank = start + idx
            if rank >= len(recs):
                break
            item_id, score = recs[rank]
            movie = movies.get(int(item_id), {})
            with col:
                st.markdown(_card_html(movie, item_id, score, rank, reasons_by_item, lang),
                            unsafe_allow_html=True)


def _esc(s):
    """HTML-escape a server-provided string so it can be interpolated safely."""
    if not s:
        return ""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _card_html(movie, item_id, score, rank, reasons_by_item, lang):
    title = _esc(movie.get("title", f"Item {item_id}"))
    genres = _esc(movie.get("genres", ""))
    year = movie.get("release_year") or "—"
    poster = movie.get("poster_url") or ""
    initial = _esc(movie.get("title", "?"))[:1].upper() or "?"
    poster_block = (
        f"<img src='{_esc(poster)}' alt='{title}' loading='lazy'>"
        f"<div class='cr-poster-grad'></div>"
        if poster else
        f"<div class='cr-poster-ph'>{initial}</div>"
    )
    reasons_html = _reasons_html(reasons_by_item.get(int(item_id), []), lang)
    return (
        f"<div class='cr-card'>"
        f"<div class='cr-rank'>#{rank + 1}</div>"
        f"<div class='cr-poster-wrap'>{poster_block}</div>"
        f"<div class='cr-card-body'>"
        f"<div class='cr-title'>{title}</div>"
        f"<div class='cr-meta'>{genres} · {year}</div>"
        f"<div class='cr-score'>score {score:.3f}</div>"
        f"{reasons_html}"
        f"</div></div>"
    )


def _reasons_html(reasons, lang):
    lines = [reason_text(r, lang) for r in reasons]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    body = "<br>".join(_esc(line) for line in lines[:3])
    return f"<div class='cr-reason'>{body}</div>"


# --------------------------------------------------------------------------- #
# Chart helpers (pure CSS/HTML, no plotly dependency)
# --------------------------------------------------------------------------- #
_ALGO_COLORS = {
    "UserCF": "#8b8ba6",
    "ItemCF": "#4a9eff",
    "SVD": "#4ade80",
    "BPRMF": "#f87171",
    "NeuMF": "#a78bfa",
    "LightGCN": "#2dd4bf",
    "VBPR": "#fb923c",
    "MultiModalNCF": "#d4a843",
}

#: Display names for the offline SOTA baselines that appear in the evaluation
#: artefacts but are not servable ladder members (not in ``ALGORITHMS``).
_EVAL_DISPLAY = {
    "BPRMF": "BPR-MF (Rendle 2009)",
    "VBPR": "VBPR (He & McAuley 2016)",
}


def _bar_chart_html(rows, value_fmt="{:.4f}", baseline=None, baseline_label="baseline",
                    labels=None):
    """Render a horizontal CSS bar chart.

    ``rows`` is a list of ``(label, value, color)`` tuples; ``value`` is scaled
    against the maximum so the longest bar fills the track. An optional
    ``baseline`` (a value in the same units) draws a dashed reference line on
    each track, used for ablation/cold-start studies. ``labels`` optionally
    overrides the numeric value shown at the end of each bar (e.g. ``m ± s``).
    """
    if not rows:
        return "<div class='cr-muted' style='color:var(--cr-muted)'>No data.</div>"
    max_val = max(v for _, v, _ in rows)
    if max_val <= 0:
        max_val = 1.0
    out = ["<div class='cr-chart'>"]
    for i, (label, val, color) in enumerate(rows):
        pct = max(2, (val / max_val) * 100)
        bl_pct = (baseline / max_val) * 100 if baseline else None
        bl_marker = (
            f"<div style='position:absolute;top:0;bottom:0;left:{bl_pct:.1f}%;"
            f"width:2px;background:var(--cr-red);opacity:.8'></div>"
            if bl_pct is not None else ""
        )
        shown = labels[i] if labels else value_fmt.format(val)
        out.append(
            f"<div class='cr-bar-row'>"
            f"<div class='cr-bar-label'>{_esc(label)}</div>"
            f"<div class='cr-bar-track' style='position:relative'>{bl_marker}"
            f"<div class='cr-bar-fill' style='width:{pct:.1f}%;background:linear-gradient(90deg,{color},{color}cc)'></div>"
            f"</div>"
            f"<div class='cr-bar-val'>{shown}</div>"
            f"</div>"
        )
    if baseline is not None:
        out.append(
            f"<div class='cr-bar-val' style='text-align:right'>"
            f"<span style='color:var(--cr-red)'>━</span> {baseline_label}: {value_fmt.format(baseline)}</div>"
        )
    return "".join(out) + "</div>"


def _heatmap_html(matrix, labels):
    """Render a square Jaccard/overlap matrix as a CSS grid heatmap."""
    n = len(labels)
    # Build an (n+1) x (n+1) grid: header row + header col + cells.
    cells = ["<div class='cr-heat-label'></div>"]
    for lab in labels:
        cells.append(f"<div class='cr-heat-label'>{_esc(lab)}</div>")
    for i, ri in enumerate(labels):
        cells.append(f"<div class='cr-heat-label'>{_esc(ri)}</div>")
        for j, _ in enumerate(labels):
            v = matrix[i][j]
            # Gold gradient: 0 = dark, 1 = bright gold.
            alpha = 0.08 + 0.82 * v
            cells.append(
                f"<div class='cr-heat-cell' style='aspect-ratio:1;"
                f"background:rgba(212,168,67,{alpha:.2f});"
                f"color:{'#0a0a0f' if v > 0.45 else '#e8e8ed'}'>{v:.2f}</div>"
            )
    return (
        f"<div class='cr-heatmap' style='grid-template-columns:repeat({n + 1}, 1fr);"
        f"max-width:{(n + 1) * 60}px'>"
        + "".join(cells) + "</div>"
    )


# --------------------------------------------------------------------------- #
# Ladder, KPIs and user profile (sidebar visuals)
# --------------------------------------------------------------------------- #
_ALGO_SHORT = {
    "UserCF": "UserCF",
    "ItemCF": "ItemCF",
    "SVD": "SVD",
    "NeuMF": "NeuMF",
    "LightGCN": "LightGCN",
    "MultiModalNCF": "Multi-Modal",
}


def _ladder_html(active):
    """Six dots on a line — the algorithm ladder; the active one lights up."""
    nodes = []
    for name in ALGORITHMS:
        short = _ALGO_SHORT.get(name, name)
        cls = "cr-ladder-node active" if name == active else "cr-ladder-node"
        nodes.append(
            f"<div class='{cls}'>"
            f"<div class='cr-ladder-dot'></div>"
            f"<div class='cr-ladder-name'>{short}</div>"
            f"<div class='cr-ladder-line'></div>"
            f"</div>"
        )
    return f"<div class='cr-ladder'>{''.join(nodes)}</div>"


def _kpi_row_html(movies, ratings_df, n_algos):
    """Four KPI cards under the hero: models / films / ratings / GPU."""
    n_movies = len(movies)
    n_ratings = len(ratings_df)
    n_users = int(ratings_df["user_id"].nunique())
    return (
        "<div class='cr-kpi-row'>"
        f"<div class='cr-kpi'><div class='cr-kpi-label'>Models</div>"
        f"<div class='cr-kpi-value'><span class='cr-accent'>{n_algos}</span></div>"
        f"<div class='cr-kpi-sub'>UserCF → Multi-Modal</div></div>"
        f"<div class='cr-kpi'><div class='cr-kpi-label'>Films</div>"
        f"<div class='cr-kpi-value'>{n_movies:,}</div>"
        f"<div class='cr-kpi-sub'>MovieLens 100K</div></div>"
        f"<div class='cr-kpi'><div class='cr-kpi-label'>Ratings</div>"
        f"<div class='cr-kpi-value'>{n_ratings:,}</div>"
        f"<div class='cr-kpi-sub'>{n_users:,} users</div></div>"
        f"<div class='cr-kpi'><div class='cr-kpi-label'>GPU</div>"
        f"<div class='cr-kpi-value'><span class='cr-accent'>0</span></div>"
        f"<div class='cr-kpi-sub'>CPU-only inference</div></div>"
        "</div>"
    )


def _profile_html(movies, ratings_df, user_id):
    """Sidebar mini-card: how many ratings, average, top genres for this user."""
    sub = ratings_df[ratings_df["user_id"] == int(user_id)]
    count = len(sub)
    avg = float(sub["rating"].mean()) if count else 0.0
    # Top genres from the user's rated items.
    genre_counts = {}
    for item_id in sub["item_id"]:
        genres = movies.get(int(item_id), {}).get("genres", "")
        for g in str(genres).split("|"):
            g = g.strip()
            if g:
                genre_counts[g] = genre_counts.get(g, 0) + 1
    top_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:3]
    genre_chips = "".join(f"<span class='cr-profile-genre'>{_esc(g)}</span>" for g in top_genres)
    fallback = "<span class='cr-profile-row'>—</span>"
    return (
        "<div class='cr-profile'>"
        f"<div class='cr-profile-row'><span>Ratings</span><b>{count}</b></div>"
        f"<div class='cr-profile-row'><span>Avg score</span><b>{avg:.2f} / 5</b></div>"
        f"<div class='cr-profile-row'><span>Top genres</span></div>"
        f"<div class='cr-profile-genres'>{genre_chips or fallback}</div>"
        "</div>"
    )


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_live(movies, user_id, algorithm, top_k, lang):
    st.subheader("Live recommendation")
    history = user_history(user_id)
    top_rated = sorted(history.items(), key=lambda kv: kv[1], reverse=True)[:5]
    with st.expander(f"Your top-rated films ({len(history)} ratings in the training set)", expanded=True):
        for item_id, rating in top_rated:
            st.markdown(f"- {movie_title(movies, item_id)} — your rating {rating:.0f}")

    try:
        model = get_model(algorithm)
    except AlgorithmUnavailable as exc:
        st.error(f"`{algorithm}` is not available here: {exc}")
        return
    if model is None:
        st.error(f"Unknown algorithm '{algorithm}'.")
        return

    recs, error = top_items(model, user_id, top_k, set(history))
    if error:
        st.error(f"Inference failed: {error}")
        return
    if not recs:
        st.warning("No recommendations — the user may be outside the model's training range.")
        return

    reasons_by_item = {}
    try:
        explainer = get_explainer()
        reasons_by_item = {item_id: explainer.explain(user_id, item_id).get("reasons", [])
                           for item_id, _ in recs}
    except Exception as exc:  # explanations are decoration; never fatal
        st.info(f"Explanations unavailable: {exc}")

    render_cards(recs, movies, reasons_by_item, lang)
    st.caption(
        f"{algorithm} scored {model.num_items} candidates live in-process (full mode). "
        "The hosted product runs the same call, or a cached equivalent in lite mode."
    )


def tab_compare(movies, user_id):
    st.subheader("Six-model comparison")
    st.caption(f"Top-{COMPARE_K} for the same user, computed live for every rung of the ladder.")

    history = set(user_history(user_id))
    columns, ids_by_algo, unavailable = {}, {}, []
    for name in ALGORITHMS:
        try:
            model = get_model(name)
        except AlgorithmUnavailable as exc:
            unavailable.append(f"{name} ({exc})")
            columns[name] = [f"— ({name} unavailable)"] * COMPARE_K
            continue
        recs, _ = top_items(model, user_id, COMPARE_K, history)
        titles = [movie_title(movies, item_id) for item_id, _ in recs]
        columns[name] = titles + ["—"] * (COMPARE_K - len(titles))
        ids_by_algo[name] = {item_id for item_id, _ in recs}

    table = pd.DataFrame(columns, index=[f"#{i + 1}" for i in range(COMPARE_K)])
    st.dataframe(table, use_container_width=True, height=380)
    if unavailable:
        st.warning("Unavailable in this deployment: " + "; ".join(unavailable))

    # Prefer the 5-seed multiseed artefact (8 models incl. the offline SOTA
    # baselines); fall back to the single-seed one when only that ships.
    eval_results = load_artifact("eval_multiseed.json") or load_artifact("eval_results.json")
    if eval_results:
        eval_names = [n for n in ALGORITHMS if n in eval_results] + [
            n for n in ("BPRMF", "VBPR") if n in eval_results
        ]

        def metric_rows(metric):
            rows, labels = [], []
            for name in eval_names:
                mean, _ = ms_value(eval_results, name, metric)
                rows.append((name, mean, _ALGO_COLORS.get(name, "#8b8ba6")))
                labels.append(ms_label(eval_results, name, metric))
            return rows, labels

        def display_name(name):
            return _EVAL_DISPLAY.get(name, name)

        st.markdown("**HR@10 — leave-last-5-out, mean ± std over 5 seeds (42–46)**")
        st.caption(
            "Hit-rate at 10: the fraction of users whose held-out film appears in the top 10. "
            "BPR-MF / VBPR are the published SOTA baselines — offline references, not servable."
        )
        hr_rows, hr_labels = metric_rows("HR@10")
        st.markdown(_bar_chart_html(hr_rows, labels=hr_labels), unsafe_allow_html=True)

        st.markdown("**NDCG@10 — mean ± std over 5 seeds**")
        st.caption("Discounted cumulative gain at 10: rewards ranking the held-out film higher.")
        ndcg_rows, ndcg_labels = metric_rows("NDCG@10")
        st.markdown(_bar_chart_html(ndcg_rows, labels=ndcg_labels), unsafe_allow_html=True)

        st.markdown("**Training time (seconds, CPU)**")
        st.caption("Why the classic models win on cost; why the neural ones win on hit-rate.")
        tt_rows, tt_labels = metric_rows("train_time")
        st.markdown(
            _bar_chart_html(tt_rows, value_fmt="{:.1f}s", labels=tt_labels),
            unsafe_allow_html=True,
        )

        with st.expander("Raw metrics table (mean ± std over seeds)"):
            metrics = ["HR@10", "NDCG@10", "Recall@10", "train_time"]
            frame = pd.DataFrame(
                {
                    display_name(name): {
                        m: ms_label(eval_results, name, m,
                                    "{:.2f}s" if m == "train_time" else "{:.4f}")
                        for m in metrics
                    }
                    for name in eval_names
                }
            ).T
            st.dataframe(frame, use_container_width=True)
        st.caption("All numbers are reproduced with `make eval-seeds` — mean ± std over seeds "
                   "42–46; deterministic models have exactly zero spread.")
    else:
        st.info("eval_results.json not found.")

    st.markdown(f"**Top-{COMPARE_K} overlap between algorithms (Jaccard)**")
    st.caption("0 = disjoint recommendation lists, 1 = identical. Gold = high agreement.")
    if len(ids_by_algo) >= 2:
        names = list(ids_by_algo)
        matrix = [[0.0] * len(names) for _ in names]
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                union = ids_by_algo[a] | ids_by_algo[b]
                matrix[i][j] = (len(ids_by_algo[a] & ids_by_algo[b]) / len(union)) if union else 0.0
        st.markdown(_heatmap_html(matrix, names), unsafe_allow_html=True)
        with st.expander("Raw overlap table"):
            overlap = pd.DataFrame(matrix, index=names, columns=names)
            st.dataframe(overlap.style.format("{:.2f}"), use_container_width=True)


def tab_multimodal(movies, user_id, lang):
    st.subheader("Multi-modal content tower")
    st.caption(
        "The core innovation: text (Sentence-BERT), image (ResNet-50) and genre vectors are fused "
        "into the MLP path. Blank a modality and watch the same checkpoint re-rank."
    )

    disabled = st.multiselect("Modalities to blank out", ["text", "image", "genre"], default=[])
    history = set(user_history(user_id))
    try:
        model = get_multimodal_variant(tuple(disabled))
    except AlgorithmUnavailable as exc:
        st.error(f"The multi-modal model is unavailable here: {exc}")
        return

    recs, error = top_items(model, user_id, COMPARE_K, history)
    if error:
        st.error(f"Inference failed: {error}")
        return
    label = "full fusion" if not disabled else "w/o " + ", ".join(disabled)
    st.markdown(f"**Live top-{COMPARE_K} — {label}**")
    render_cards(recs, movies, lang=lang, columns=5)

    st.markdown("**Retrained ablation (rigorous — each configuration is retrained, not just masked)**")
    st.caption(
        "Drop a modality at the input, retrain the full model, measure — mean ± std over 5 seeds. "
        "Behavior-only is the floor; single-modality deltas sit inside the seed noise."
    )
    ablation = load_artifact("ablation_multiseed.json") or load_artifact("ablation_results.json")
    if ablation:
        # HR@10 horizontal bars with a behaviour-only baseline line.
        baseline = ms_value(ablation, "Behavior only (no content)", "HR@10")[0]
        rows, labels = [], []
        for name, values in ablation.items():
            if not isinstance(values, dict) or "HR@10" not in values:
                continue
            rows.append((name, ms_value(ablation, name, "HR@10")[0],
                         _ALGO_COLORS.get("MultiModalNCF", "#d4a843")
                         if name == "Full (Text+Image+Genre)" else "#8b8ba6"))
            labels.append(ms_label(ablation, name, "HR@10"))
        st.markdown(_bar_chart_html(rows, labels=labels,
                                    baseline=baseline,
                                    baseline_label="behavior-only floor"),
                    unsafe_allow_html=True)
        with st.expander("Raw ablation table (mean ± std over 5 seeds)"):
            metrics = ["HR@10", "NDCG@10", "Recall@10"]
            variant_names = [r[0] for r in rows]
            frame = pd.DataFrame(
                {name: {m: ms_label(ablation, name, m) for m in metrics}
                 for name in variant_names}
            ).T
            st.dataframe(frame, use_container_width=True)
        st.caption("Inference-time masking above is a sensitivity probe; the chart is the measured study.")
    else:
        st.info("ablation_results.json not found.")


def tab_coldstart(movies, user_id, lang):
    st.subheader("Cold start — scoring items with no collaborative signal")
    st.caption(
        "Candidates are ranked through the content tower alone (`forward_cold`): the item embedding "
        "is bypassed, so nothing about user-item history can help. This is the path a brand-new film takes."
    )

    try:
        model = get_multimodal_variant(())
    except AlgorithmUnavailable as exc:
        st.error(f"The multi-modal model is unavailable here: {exc}")
        return

    history = set(user_history(user_id))
    catalogue = sorted(set(movies) - history)
    pool_size = st.slider("Candidate pool size", 50, min(1000, len(catalogue)),
                          min(300, len(catalogue)), step=50)
    rng = np.random.default_rng(42)  # fixed seed: the demo is reproducible
    pool = rng.choice(catalogue, size=pool_size, replace=False)
    recs = cold_rank(model, user_id, pool, COMPARE_K)

    st.markdown(f"**Content-only top-{COMPARE_K} out of {pool_size} unseen films**")
    render_cards(recs, movies, lang=lang, columns=5)

    st.markdown("**Cold-start study (100 genuinely cold items, candidate pool = cold items only)**")
    cold_art = (load_artifact("coldstart_multiseed.json")
                or load_artifact("coldstart_results.json"))
    study = cold_art.get("metrics", {}) if isinstance(cold_art, dict) else {}
    if study:
        # HR@10 bars with the random floor as a red dashed baseline.
        random_floor = ms_value(study, "Random", "HR@10")[0] or None
        rows, labels = [], []
        for name in study:
            raw = study[name]
            # Values are {metric: {mean, std}} dicts; "Random" is a plain float.
            if not ((isinstance(raw, dict) and "HR@10" in raw)
                    or isinstance(raw, (int, float))):
                continue
            color = (_ALGO_COLORS.get("MultiModalNCF", "#d4a843")
                     if "MultiModal" in name else
                     _ALGO_COLORS.get("NeuMF", "#a78bfa") if "NeuMF" in name else "#8b8ba6")
            rows.append((name, ms_value(study, name, "HR@10")[0], color))
            labels.append(ms_label(study, name, "HR@10"))
        st.markdown(_bar_chart_html(rows, labels=labels,
                                    baseline=random_floor,
                                    baseline_label="random floor"),
                    unsafe_allow_html=True)
        n_seeds = cold_art.get("n_seeds")
        seed_note = f"mean ± std over {n_seeds} seeds" if n_seeds else "values over seeds"
        st.caption(f"Content tower beats the content-blind control and the random floor on "
                   f"unseen items ({seed_note}).")
        with st.expander(f"Raw cold-start table ({seed_note})"):
            metrics = ["HR@10", "NDCG@10", "Recall@10", "HR@20", "NDCG@20", "Recall@20"]
            model_names = [r[0] for r in rows]
            frame = pd.DataFrame(
                {name: {m: ms_label(study, name, m) for m in metrics}
                 for name in model_names}
            ).T
            st.dataframe(frame, use_container_width=True)
    else:
        st.info("coldstart_results.json not found.")


# --------------------------------------------------------------------------- #
# Cross-dataset validation (RQ5)
# --------------------------------------------------------------------------- #
def _eval_names(art):
    """Chart order: the servable ladder first, then the offline SOTA baselines."""
    return [n for n in ALGORITHMS if n in art] + [n for n in ("BPRMF", "VBPR") if n in art]


def _metric_bars(art, metric, names):
    """``(rows, labels)`` for ``_bar_chart_html`` from a multiseed artefact."""
    rows = [(n, ms_value(art, n, metric)[0], _ALGO_COLORS.get(n, "#8b8ba6")) for n in names]
    labels = [ms_label(art, n, metric) for n in names]
    return rows, labels


def _domain_column(art, metric, names, title):
    st.markdown(f"<div class='cr-bar-label' style='text-align:center'>{_esc(title)}</div>",
                unsafe_allow_html=True)
    rows, labels = _metric_bars(art, metric, names)
    st.markdown(_bar_chart_html(rows, labels=labels), unsafe_allow_html=True)


def tab_crossdataset():
    st.subheader("Cross-dataset validation (RQ5)")
    st.caption(
        "The whole protocol — same leave-last-5-out split, same candidate filtering, eight models, "
        "seed list 42–46 — re-run unchanged on Amazon Reviews-2023 Video_Games (ADR 0005): a "
        "2,288-user × 4,753-item heavy-user slice built with MovieLens-100K's own population rule. "
        "The ~2.8× larger catalogue mechanically depresses absolute HR@K — read the orderings, "
        "not the levels."
    )

    ml_eval = load_artifact("eval_multiseed.json") or load_artifact("eval_results.json")
    amz_eval = load_artifact("eval_multiseed.json", directory=SECOND_PROCESSED_DIR)
    if not (ml_eval and amz_eval):
        st.info(
            "Cross-dataset artefacts not found. Build the second dataset with `make amazon-data`, "
            "then re-run the evaluation with `CINEREC_DATASET=amazon-video-games make eval-seeds`."
        )
        return

    names = [n for n in _eval_names(ml_eval) if n in amz_eval]

    for metric, blurb in [
        ("HR@10", "Hit-rate at 10: the fraction of users whose held-out item appears in the top 10."),
        ("NDCG@10", "NDCG at 10: rewards ranking the held-out item higher."),
    ]:
        st.markdown(f"**{metric} — MovieLens-100K vs Amazon Video_Games (mean ± std, 5 seeds)**")
        st.caption(blurb)
        left, right = st.columns(2)
        with left:
            _domain_column(ml_eval, metric, names, "MovieLens-100K (movies)")
        with right:
            _domain_column(amz_eval, metric, names, "Amazon Video_Games (games)")

    # --- Ablation on both domains ---
    ml_abl = (load_artifact("ablation_multiseed.json")
              or load_artifact("ablation_results.json"))
    amz_abl = (load_artifact("ablation_multiseed.json", directory=SECOND_PROCESSED_DIR)
               or load_artifact("ablation_results.json", directory=SECOND_PROCESSED_DIR))
    if ml_abl and amz_abl:
        st.markdown("**Ablation on both domains (HR@10, mean ± std over 5 seeds)**")
        st.caption(
            "Single-modality deltas sit inside the seed noise on *both* domains; the "
            "behavior-only drop that is ~1 seed std on MovieLens vanishes on Amazon."
        )
        abl_rows = []
        for variant in ml_abl:
            if not isinstance(ml_abl[variant], dict) or "HR@10" not in ml_abl[variant]:
                continue
            abl_rows.append({
                "Variant": variant,
                "ML-100K": ms_label(ml_abl, variant, "HR@10"),
                "Amazon": ms_label(amz_abl, variant, "HR@10"),
            })
        st.dataframe(pd.DataFrame(abl_rows).set_index("Variant"), use_container_width=True)

    # --- Cold start on both domains ---
    ml_cold = (load_artifact("coldstart_multiseed.json")
               or load_artifact("coldstart_results.json"))
    amz_cold = (load_artifact("coldstart_multiseed.json", directory=SECOND_PROCESSED_DIR)
                or load_artifact("coldstart_results.json", directory=SECOND_PROCESSED_DIR))
    if ml_cold.get("metrics") and amz_cold.get("metrics"):
        ml_m, amz_m = ml_cold["metrics"], amz_cold["metrics"]
        st.markdown("**Cold start on both domains (HR@10, mean ± std over 5 seeds)**")
        st.caption(
            "Content-aware ranking of items with no collaborative history. The content tower's "
            "edge over the content-blind control *amplifies* on the sparser domain — the "
            "strongest cross-domain transfer in the study."
        )
        cold_rows = []
        for model in ml_m:
            if not isinstance(ml_m[model], dict) and not isinstance(ml_m[model], (int, float)):
                continue
            cold_rows.append({
                "Model": model,
                "ML-100K": ms_label(ml_m, model, "HR@10"),
                "Amazon": ms_label(amz_m, model, "HR@10"),
            })
        c_ml = (ms_value(ml_m, "MultiModalNCF (content)", "HR@10")[0]
                / max(ms_value(ml_m, "NeuMF (content-blind)", "HR@10")[0], 1e-9))
        c_amz = (ms_value(amz_m, "MultiModalNCF (content)", "HR@10")[0]
                 / max(ms_value(amz_m, "NeuMF (content-blind)", "HR@10")[0], 1e-9))
        cold_rows.append({"Model": "content / blind ratio",
                          "ML-100K": f"{c_ml:.2f}×", "Amazon": f"{c_amz:.2f}×"})
        st.dataframe(pd.DataFrame(cold_rows).set_index("Model"), use_container_width=True)

    with st.expander("What transfers and what doesn't (RQ5 verdicts)"):
        st.markdown(
            "**Transfers** — agreement strengthens the MovieLens conclusions:\n"
            "- The ladder ordering **UserCF < ItemCF < SVD** holds in both domains, and the "
            "neural lead over SVD is *larger* on Amazon.\n"
            "- **BPR-MF ≈ NeuMF**: the behaviour-only pair is a statistical tie in both domains.\n"
            "- Single-modality ablation deltas stay **inside seed noise** in both domains.\n"
            "- **Cold start amplifies**: the content/blind edge grows 1.17× → 1.96× — the "
            "sparser the collaborative signal, the more the content tower is worth.\n\n"
            "**Does not transfer** — reported, not hidden:\n"
            "- *\"MultiModalNCF is the best ladder model\"*: on Amazon **LightGCN** takes the top "
            "HR@10 point estimate (0.1187 vs 0.1164) — a statistical tie won at 1/7 of the "
            "training cost. The neural band (0.105–0.119) is the same; the pick inside it is "
            "domain-dependent.\n"
            "- *\"Removing all content hurts\"*: on Amazon Behavior only ≈ Full (0.1163 vs 0.1164)."
        )

    with st.expander("Raw cross-domain metrics"):
        eval_tbl = {
            _EVAL_DISPLAY.get(n, n): {
                "ML HR@10": ms_label(ml_eval, n, "HR@10"),
                "Amazon HR@10": ms_label(amz_eval, n, "HR@10"),
                "ML NDCG@10": ms_label(ml_eval, n, "NDCG@10"),
                "Amazon NDCG@10": ms_label(amz_eval, n, "NDCG@10"),
            }
            for n in names
        }
        st.dataframe(pd.DataFrame(eval_tbl).T, use_container_width=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="CineRec — Live Inference", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    # Hero with film-strip logo.
    st.markdown(
        "<div class='cr-hero-row'>"
        "<div class='cr-film-logo'>▶</div>"
        "<div><div class='cr-hero'>CineRec — Live Inference</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='cr-sub'>The six trained recommenders running in-process on MovieLens-100K, "
        "with the 5-seed evaluation, the published SOTA baselines and the RQ5 cross-dataset "
        "study (Amazon Video_Games) — the <code>full</code> counterpart to the cached product "
        "deployment.</div>",
        unsafe_allow_html=True,
    )

    movies = load_movies()
    ratings_df = load_ratings()

    # KPI strip.
    st.markdown(_kpi_row_html(movies, ratings_df, len(ALGORITHMS)), unsafe_allow_html=True)

    with st.sidebar:
        st.header("Controls")
        st.markdown(_ladder_html("SVD"), unsafe_allow_html=True)
        algorithm = st.selectbox("Algorithm", ALGORITHMS, index=ALGORITHMS.index("SVD"))
        # Re-render the ladder with the now-selected algorithm highlighted.
        st.markdown(f"<div style='margin-top:-.4rem'>{_ladder_html(algorithm)}</div>",
                    unsafe_allow_html=True)
        user_id = st.selectbox("User", all_user_ids(), index=0, format_func=lambda u: f"user {u}")
        st.markdown(_profile_html(movies, ratings_df, user_id), unsafe_allow_html=True)
        top_k = st.slider("Top-K", 1, 20, 10)
        lang = st.radio("Explanation language", ["中文", "English"], horizontal=True)
        st.divider()
        st.caption(f"{len(movies)} films · {len(all_user_ids())} users · "
                   f"{len(ALGORITHMS)} servable + 2 baselines · 2 datasets")

    live, compare, multimodal, cold, cross = st.tabs(
        ["Live recommendation", "Six-model comparison", "Multi-modal ablation",
         "Cold start", "Cross-dataset (RQ5)"]
    )
    with live:
        tab_live(movies, user_id, algorithm, top_k, lang)
    with compare:
        tab_compare(movies, user_id)
    with multimodal:
        tab_multimodal(movies, user_id, lang)
    with cold:
        tab_coldstart(movies, user_id, lang)
    with cross:
        tab_crossdataset()


if __name__ == "__main__":
    main()
