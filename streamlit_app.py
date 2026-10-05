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

from config import PROCESSED_DIR
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
}

/* --- Typography (mirrors the product's front end) --- */
html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {
  font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
}
code, pre, .cr-score {
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
  padding-top: 2.6rem;
  max-width: 1180px;
}

/* --- Hero --- */
.cr-hero {
  font-size: 2.5rem;
  font-weight: 800;
  letter-spacing: -.02em;
  line-height: 1.1;
  margin-bottom: .15rem;
  background: linear-gradient(120deg, var(--cr-accent-light), var(--cr-accent) 45%, #f6e6b8 70%, var(--cr-accent));
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  filter: drop-shadow(0 0 26px rgba(212,168,67,.28));
}
.cr-sub { color: var(--cr-muted); margin-bottom: 1.3rem; font-size: .95rem; }

/* --- Recommendation cards --- */
.cr-card {
  border: 1px solid rgba(148,163,184,.22);
  border-radius: 14px;
  padding: 14px 14px 12px 14px;
  height: 100%;
  background: linear-gradient(160deg, rgba(30,30,56,.72), rgba(15,15,30,.62));
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
  transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
.cr-card:hover {
  transform: translateY(-3px);
  border-color: rgba(212,168,67,.38);
  box-shadow: 0 12px 30px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.07);
}
.cr-title { font-weight: 600; font-size: .95rem; margin: .35rem 0 .12rem 0; line-height: 1.28; color: var(--cr-text); }
.cr-meta { color: var(--cr-muted); font-size: .76rem; }
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
  font-size: .78rem;
  color: #cbd5e1;
  margin-top: .45rem;
  border-top: 1px solid var(--cr-border);
  padding-top: .4rem;
}

[data-testid="stImage"] img { border-radius: 12px; }

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
def load_artifact(filename):
    """One of the committed evaluation JSON artefacts (empty dict when absent)."""
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
    """Grid of recommendation cards: poster, title, score and reasons."""
    reasons_by_item = reasons_by_item or {}
    for start in range(0, len(recs), columns):
        cols = st.columns(columns)
        for col, (item_id, score) in zip(cols, recs[start:start + columns]):
            movie = movies.get(int(item_id), {})
            with col:
                poster = movie.get("poster_url") or ""
                if poster:
                    st.image(poster, width=260)
                st.markdown(
                    f"<div class='cr-card'>"
                    f"<div class='cr-title'>{movie.get('title', f'Item {item_id}')}</div>"
                    f"<div class='cr-meta'>{movie.get('genres', '')} · {movie.get('release_year') or '—'}</div>"
                    f"<div class='cr-score'>score {score:.3f}</div>"
                    f"{_reasons_html(reasons_by_item.get(int(item_id), []), lang)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )


def _reasons_html(reasons, lang):
    lines = [reason_text(r, lang) for r in reasons]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    body = "<br>".join(line.replace("<", "&lt;") for line in lines[:3])
    return f"<div class='cr-reason'>{body}</div>"


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
    st.dataframe(table)
    if unavailable:
        st.warning("Unavailable in this deployment: " + "; ".join(unavailable))

    st.markdown("**Offline quality (leave-last-5-out on MovieLens-100K)**")
    eval_results = load_artifact("eval_results.json")
    if eval_results:
        metrics = ["HR@10", "NDCG@10", "Recall@10", "train_time"]
        frame = pd.DataFrame(
            {name: {m: eval_results.get(name, {}).get(m) for m in metrics} for name in ALGORITHMS}
        ).T
        st.dataframe(
            frame.style.format({"HR@10": "{:.4f}", "NDCG@10": "{:.4f}",
                                "Recall@10": "{:.4f}", "train_time": "{:.2f}s"})
        )
    else:
        st.info("eval_results.json not found.")

    st.markdown(f"**Top-{COMPARE_K} overlap between algorithms (Jaccard)**")
    st.caption("How much the rungs actually agree — 0 means disjoint lists, 1 means identical.")
    if len(ids_by_algo) >= 2:
        names = list(ids_by_algo)
        overlap = pd.DataFrame(index=names, columns=names, dtype=float)
        for a in names:
            for b in names:
                union = ids_by_algo[a] | ids_by_algo[b]
                overlap.loc[a, b] = len(ids_by_algo[a] & ids_by_algo[b]) / len(union) if union else 0.0
        st.dataframe(overlap.style.format("{:.2f}"))


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
    ablation = load_artifact("ablation_results.json")
    if ablation:
        metrics = ["HR@10", "NDCG@10", "Recall@10"]
        frame = pd.DataFrame(
            {name: {m: values.get(m) for m in metrics} for name, values in ablation.items()}
        ).T
        st.dataframe(frame.style.format("{:.4f}"))
        st.caption("Inference-time masking above is a sensitivity probe; the table is the measured study.")
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
    study = load_artifact("coldstart_results.json").get("metrics", {})
    if study:
        metrics = ["HR@10", "NDCG@10", "Recall@10", "HR@20", "NDCG@20", "Recall@20"]
        frame = pd.DataFrame(
            {name: {m: values.get(m) for m in metrics} for name, values in study.items()}
        ).T
        st.dataframe(frame.style.format("{:.4f}"))
        st.caption("Content tower beats the content-blind control and the random floor on unseen items.")
    else:
        st.info("coldstart_results.json not found.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="CineRec — Live Inference", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("<div class='cr-hero'>CineRec — Live Inference</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='cr-sub'>The six trained recommenders running in-process on MovieLens-100K — "
        "the <code>full</code> counterpart to the cached product deployment.</div>",
        unsafe_allow_html=True,
    )

    movies = load_movies()
    with st.sidebar:
        st.header("Controls")
        algorithm = st.selectbox("Algorithm", ALGORITHMS, index=ALGORITHMS.index("SVD"))
        user_id = st.selectbox("User", all_user_ids(), index=0, format_func=lambda u: f"user {u}")
        top_k = st.slider("Top-K", 1, 20, 10)
        lang = st.radio("Explanation language", ["中文", "English"], horizontal=True)
        st.divider()
        st.caption(f"{len(movies)} films · {len(all_user_ids())} users · 6 algorithms")

    live, compare, multimodal, cold = st.tabs(
        ["Live recommendation", "Six-model comparison", "Multi-modal ablation", "Cold start"]
    )
    with live:
        tab_live(movies, user_id, algorithm, top_k, lang)
    with compare:
        tab_compare(movies, user_id)
    with multimodal:
        tab_multimodal(movies, user_id, lang)
    with cold:
        tab_coldstart(movies, user_id, lang)


if __name__ == "__main__":
    main()
