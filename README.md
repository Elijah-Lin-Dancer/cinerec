<div align="center">

<img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-2.13-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/FastAPI-0.100-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI">
<img src="https://img.shields.io/badge/License-MIT-F5C518?style=flat" alt="License">

<br><br>

# 🎬 CineRec

**Multi-Modal Movie Recommendation System**

*Six servable recommenders on one codebase — from UserCF to a multi-modal NCF — plus
two published SOTA baselines (BPR-MF, VBPR), all on a single leave-last-5-out split,
with the neural models reported as mean ± std over five seeds.*

English · [简体中文](README.zh-CN.md)

📄 **Companion paper**: *CineRec: A Practical Multi-Modal Recommender Study with
Cross-Dataset Validation and SOTA Baseline Comparison* — [`docs/paper.pdf`](docs/paper.pdf)

[Live Demo](#-live-demo) · [Demo](#-demo-preview) · [Evaluation](#-evaluation) · [Quick Start](#-quick-start)

</div>

---

## 🔗 Live Demo

One codebase, two runtime modes (`APP_MODE`), each hosted on the free tier it fits
([ADR 0001](docs/adr/0001-dual-mode-deployment.md)):

| Mode | What it shows | Link |
|------|---------------|------|
| **`lite`** — product | The bilingual FastAPI web app. Recommendations come from a precomputed cache, so it fits a 512 MB instance. No account needed: click **Try as Guest**. | **https://cinerec-rfew.onrender.com** |
| **`full`** — companion | A Streamlit app that loads the trained PyTorch models and runs them **live**: live inference, the six-model comparison, the modality ablation, the cold-start study. | **https://cinerec-movie-recommend.streamlit.app** |

> Both run on free tiers and sleep when idle — the first visit can take a few
> seconds to wake. The companion lives in [`deploy/streamlit/`](deploy/streamlit/).

---

## 📋 Table of Contents

- [Live Demo](#-live-demo)
- [Project Introduction](#-project-introduction)
- [Key Features](#-key-features)
- [Demo Preview](#-demo-preview)
- [Architecture](#-architecture)
- [Algorithms](#-algorithms)
- [Evaluation](#-evaluation)
- [Quick Start](#-quick-start)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [FAQ](#-faq)
- [License](#-license)
- [Changelog](#-changelog)

---

## 📖 Project Introduction

CineRec is a full-stack movie recommender that implements six models in ascending
order of complexity and scores all of them against the same candidate set. It ships
a bilingual dark-cinema web frontend, a REST API, and a Docker build that serves
either runtime mode from the same image.

> 💡 **"Don't just use models — understand them."**
> Every layer of CineRec's algorithm ladder shows one step of the evolution from
> classical to frontier recommender systems.

---

## ✨ Key Features

- **6 models, one source of truth** — UserCF → ItemCF → SVD → NeuMF → LightGCN →
  Multi-Modal NCF are all registered in [`models/registry.py`](models/registry.py);
  the frontend, the API and the precompute scripts all read from that single place.
- **Multi-modal fusion (core innovation)** — Multi-Modal NCF fuses three content
  streams on the MLP path through a Content Tower: Sentence-BERT text (384d),
  ResNet-50 image (2048d), and genre multi-hot (18d).
- **Honest offline evaluation** — a single leave-last-5-out split, candidate sets
  stripped of training-seen items; the main, ablation and cold-start results are all
  reported as mean ± std over 5 seeds, benchmarked against the two published SOTA
  baselines (BPR-MF / VBPR) with no "best model" claims; the same protocol is
  re-validated end-to-end on a second dataset (Amazon Video_Games).
- **Explainable · bilingual · dual-mode** — every recommendation ships a reason;
  EN/ZH frontend; `lite` (precomputed cache) vs `full` (live inference) switched by
  `APP_MODE`. The movie detail modal and the Streamlit companion are in the
  [Demo Preview](#-demo-preview).
- **Observable** — inference LRU cache, `/api/metrics` in-process latency/throughput,
  plus one measured Locust load test → [`reports/loadtest.md`](reports/loadtest.md).

---

## 🖼 Demo Preview

### Recommendation Engine
> Real-time personalized recommendations with algorithm switching and explainable reasons.

<img src="screenshots/recommend.png" width="800" alt="CineRec Recommendation Page">

### Movie Library
> Browse 1,682 movies with real posters, search, genre filtering, and pagination. Click any card to open a detail popup with synopsis, rating, and IMDb link.

<img src="screenshots/movies.png" width="800" alt="CineRec Movie Library">

### Movie Detail Modal
> Click any movie card (or press Enter on a focused card) to open an accessible detail popup — poster, synopsis, genres, average rating, and actions to rate or view on IMDb. Closes with Escape, close button, or backdrop click; re-renders on language switch.

<img src="screenshots/detail_modal.png" width="800" alt="CineRec Movie Detail Modal">

### Evaluation Dashboard
> Interactive model comparison with real leave-last-5-out evaluation metrics — all six models side by side.

<img src="screenshots/dashboard.png" width="800" alt="CineRec Evaluation Dashboard">

### Ablation Charts
> A scrolled view of the evaluation page: the ablation study (one modality blanked at a time). Rendered from the committed JSON results — no hard-coded numbers.

<img src="screenshots/eval_charts.png" width="800" alt="Ablation study charts">

> Cold-start is covered separately with its own measured chart in the **Measured Results** section below.

### Streamlit Companion (Full Mode)
> Live inference companion with a film-strip logo, KPI strip, CSS bar charts, a Jaccard overlap heatmap, an algorithm ladder, and a user profile mini-card.

<table>
<tr>
<td><img src="screenshots/streamlit_live.png" width="400" alt="Streamlit Live Recommendation"></td>
<td><img src="screenshots/streamlit_comparison.png" width="400" alt="Streamlit Six-Model Comparison"></td>
</tr>
<tr>
<td align="center"><sub>Live recommendation + KPI strip</sub></td>
<td align="center"><sub>Six-model comparison + heatmap</sub></td>
</tr>
</table>

---

## 🏗 Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                        User Browser                          │
│     Dark/Light Theme · GSAP Animations · ECharts · i18n       │
├───────────────────────────────────────────────────────────────┤
│                     FastAPI REST API                          │
│                                                              │
│   /api/auth ───── /api/movies ───── /api/recommend           │
│        │              │                  │                   │
│        │              │         ┌────────┴────────┐           │
│        │              │         │  Algorithm Switch │           │
│        │              │         └────────┬────────┘           │
├────────┼──────────────┼──────────────────┼───────────────────┤
│        │       ┌──────┴──────┐    ┌───────┴───────┐          │
│  Auth Service   Movie Service  Recommend Service            │
│  (demo login)   (CRUD + posters) (6 Models + Explainer)       │
│        │              │                  │                   │
├────────┼──────────────┼──────────────────┼───────────────────┤
│        │       Model Layer (6 Models)                      │
│        │                                                   │
│        │  ┌─────────┐  ┌─────────┐                        │
│        │  │  UserCF  │  │  ItemCF  │                       │
│        │  │ (Pearson)│  │(Adj Cos)│                         │
│        │  └─────────┘  └─────────┘                        │
│        │  ┌─────────┐  ┌─────────┐                        │
│        │  │   SVD    │  │  NeuMF   │                       │
│        │  │(SVD,k=64)│  │(GMF+MLP) │                        │
│        │  └─────────┘  └─────────┘                        │
│        │  ┌───────────────────────┐                      │
│        │  │       LightGCN         │                      │
│        │  │  Graph Convolution (BPR)│                     │
│        │  └───────────────────────┘                      │
│        │  ┌───────────────────────┐                      │
│        │  │   Multi-Modal NCF ⭐   │                      │
│        │  │ Text+Image+Genre Fusion│                      │
│        │  └───────────────────────┘                      │
├────────┼─────────────────────────────────────────────────────┤
│        │       Feature Engineering (Pre-computed)            │
│  ┌─────┴─────┐  ┌──────────────┐  ┌──────────────┐        │
│  │Sentence-BERT│  │  ResNet-50   │  │Genre Encoding│        │
│  │  Text 384d │  │  Image 2048d │  │  Multi-hot   │        │
│  └───────────┘  └──────────────┘  └──────────────┘        │
├─────────────────────────────────────────────────────────────┤
│               Data Layer (SQLite + MovieLens 100K)          │
│  100,000 ratings · 943 users · 1,682 movies · enriched metadata │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧮 Algorithms

| Level | Model | Method | Description |
|:-----:|-------|--------|-------------|
| 1 | **UserCF** | Pearson Correlation | Find K similar users, aggregate their preferences. Vectorized computation. |
| 2 | **ItemCF** | Adjusted Cosine | Recommend items similar to user's rated history. Vectorized similarity. |
| 3 | **SVD** | Matrix Factorization | Decompose the user-item matrix into latent factors (k=64) via truncated SVD (`scipy.sparse.linalg.svds`). |
| 4 | **NeuMF** | GMF + MLP (PyTorch) | Dual-path architecture: element-wise product + deep MLP. BCE loss with negative sampling. |
| 5 | **LightGCN** | Graph Convolution (PyTorch) | Simplified GCN on the user–item bipartite graph: neighbourhood aggregation averaged over layers, trained with BPR. |
| 6 | **Multi-Modal NCF** ⭐ | Text+Image+Genre Fusion | Core innovation. Replaces item embedding in MLP path with a Content Tower fusing multi-modal features. |

> **Offline SOTA baselines** — BPR-MF and VBPR are implemented
> ([`models/bpr_mf.py`](models/bpr_mf.py) / [`models/vbpr.py`](models/vbpr.py)) purely
> for the evaluation comparison; they are **not** part of this servable ladder and
> never ship to the API.

### Multi-Modal NCF Architecture (Core Innovation)

```
                    ┌──────────────────┐
                    │   User Tower     │
                    │ user_id → Emb(64)│
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌──────────────┐
     │  GMF Path  │  │  MLP Path    │
     │ user⊙item  │  │              │
     │ (behavior) │  │  Content Tower (⭐)
     └─────┬──────┘  │  ┌────────────────────┐
           │         │  │ Text(384)→FC(64)    │
           │         │  │ Image(2048)→FC(64)   │
           │         │  │ Genre(18)→FC(64)    │
           │         │  │ Concat(192)→FC(64)  │
           │         │  └──────────┬─────────┘
           │         │             │
           │         │  Concat(user, content)
           │         │      FC(128→256→128→64)
           │         └──────┬──────┘
           │                │
           └─────── Concat(GMF, MLP) ──→ FC(128→1) ──→ Sigmoid ──→ Prediction
```

**Cold Start Support**: New items use `content_emb` only, GMF path uses zero vector.

---

## 📊 Evaluation

> **Scope** — CineRec is a full-stack engineering and trade-off study, not
> a leaderboard entry. All models share one fixed split; neural numbers are reported
> as **mean ± std over seeds**, so every claim is read against the measured noise.
> The whole protocol is re-run unchanged on a second domain (Amazon Video_Games)
> to check which conclusions travel.

> 📄 **Companion paper** — *CineRec: A Practical Multi-Modal Recommender Study with
> Cross-Dataset Validation and SOTA Baseline Comparison* ships in this repo as
> [`docs/paper.pdf`](docs/paper.pdf); the protocol, configurations and raw results
> behind it are documented in [`docs/tech-report.md`](docs/tech-report.md).

### Research Questions

| # | Question | Answered by |
|:--|:---------|:------------|
| **RQ1** | How much do content features (text/image/genre) improve the same ranking task over behaviour-only signals? | [Ablation](#ablation-study) |
| **RQ2** | Is that improvement larger than random-seed noise? | The **mean ± std** in the main and ablation tables |
| **RQ3** | Do content features transfer to cold-start items with zero interaction history? | [Cold-start](#cold-start-study) |
| **RQ4** | Where does CineRec stand against the published SOTA baselines (BPR-MF / VBPR)? | [Baseline comparison](#sota-baselines) |
| **RQ5** | Are the conclusions above MovieLens-specific? Do they hold in another domain? | [Cross-dataset](#cross-dataset-validation) |

### Contributions

1. **A reproducible unified evaluation protocol** — a single leave-last-5-out split,
   candidate sets aligned across models (training-seen items removed), and multi-seed
   evaluation covering the whole random-seed chain; main results are reported as
   mean ± std over 5 seeds (`make eval-seeds`).
2. **Multi-Modal NCF (core innovation)** — fuses Sentence-BERT text, ResNet-50 image
   and genre multi-hot features through a Content Tower on the NCF MLP path.
3. **Honest ablation and cold-start studies** — both reported over multiple seeds,
   explicitly separating real effects from seed noise.
4. **Published-baseline comparison** — BPR-MF (Rendle et al., 2009) and
   VBPR (He & McAuley, 2016) reproduced on the same split as the behaviour-only and
   content-aware external references.
5. **Cross-dataset validation** — the same protocol and seed list re-run end-to-end
   on a second domain (Amazon Video_Games, 2,288 heavy users × 4,753 items) to test
   which conclusions travel ([ADR 0005](docs/adr/0005-cross-dataset-validation.md)).

### Model Comparison on MovieLens 100K (leave-last-5-out)

Protocol: the 5 most recent interactions of each user are held out; every model is
scored against the same candidate set (all items minus the user's training items),
so re-recommending already-seen titles cannot inflate the numbers. Numbers are
**mean ± std over 5 random seeds (42–46)**, reproduced with `make eval-seeds`
(writes `data/processed/eval_multiseed.json`). The Train Time column is the same
5-seed mean from that file; a single-seed run's wall-clock lives in
`eval_results.json` (the server-side artefact). The three heuristics/MF models are
deterministic, so their spread is exactly zero.

| Model | HR@10 | NDCG@10 | HR@20 | NDCG@20 | Train Time |
|:-----:|:-----:|:-------:|:-----:|:-------:|:-----------:|
| UserCF | 0.0191 ± 0.0000 | 0.0035 ± 0.0000 | 0.0477 ± 0.0000 | 0.0060 ± 0.0000 | 0.1s |
| ItemCF | 0.0721 ± 0.0000 | 0.0142 ± 0.0000 | 0.1283 ± 0.0000 | 0.0199 ± 0.0000 | 0.3s |
| SVD | 0.2534 ± 0.0000 | 0.0532 ± 0.0000 | 0.3542 ± 0.0000 | 0.0664 ± 0.0000 | 0.3s |
| NeuMF | 0.3516 ± 0.0036 | 0.0814 ± 0.0016 | 0.4978 ± 0.0073 | 0.1083 ± 0.0017 | 31.4s |
| LightGCN | 0.2850 ± 0.0041 | 0.0614 ± 0.0004 | 0.4250 ± 0.0047 | 0.0816 ± 0.0007 | 10.7s |
| **MultiModalNCF** ⭐ | **0.3650 ± 0.0077** | **0.0859 ± 0.0017** | **0.5073 ± 0.0148** | **0.1144 ± 0.0015** | 73.2s |

> **What the numbers say (six ladder models)**: averaged over 5 seeds,
> **MultiModalNCF** leads all four shown metrics *within the ladder* — HR@10 0.3650 /
> NDCG@10 0.0859 / HR@20 0.5073 / NDCG@20 0.1144 — and **NeuMF** is the runner-up
> (0.3516 / 0.0814). Within the ladder the HR@10 gap (0.0134) is larger than either
> model's seed spread, while the HR@20 gap (0.0095) sits inside the MultiModalNCF std
> (0.0148). **LightGCN** lands between SVD and NeuMF at a fraction of NeuMF's
> training time. The neural models pay for their gains with a much longer training
> time, making the accuracy/cost trade-off explicit. Once the *published*
> baselines join the comparison (next section), the top margins close further, so
> no strong "best model" significance claim is made.

### Key Findings

- UserCF → ItemCF → SVD keeps the expected ordering — neighbourhood heuristics give
  way to matrix factorization — and SVD is the strongest accuracy-per-second point on
  the ladder (HR@10 0.2534, and deterministic, so seed-free).
- NeuMF: HR@10 0.3516 ± 0.0036 / HR@20 0.4978 ± 0.0073. Non-linear interaction
  modelling edges out plain factorization once candidates are filtered to unseen
  items, with a small but non-zero seed spread.
- LightGCN: HR@10 0.2850 ± 0.0041, between SVD and NeuMF, and the best accuracy per
  second among the neural models — competitive with no content features at all.
- MultiModalNCF: HR@10 0.3650 ± 0.0077 / NDCG@10 0.0859 ± 0.0017. Averaged over 5
  seeds it leads both the hit-rate and NDCG columns among the six ladder models, and
  the ablation study below quantifies each modality's share.

### SOTA Baselines

To place CineRec against the literature rather than only against itself, the same
split is used to reproduce two published baselines — **BPR-MF** (Rendle et al.,
UAI 2009) as the behaviour-only reference, and **VBPR** (He & McAuley, AAAI 2016)
as the content-aware one. They are offline references only: they are not part of
the servable ladder ([`models/registry.py`](models/registry.py)) and never ship to
the API.

| Model | Line | HR@10 | NDCG@10 | HR@20 | NDCG@20 | Train Time |
|:------|:-----|:-----:|:-------:|:-----:|:-------:|:----------:|
| BPR-MF (Rendle 2009) | behaviour-only | 0.3476 ± 0.0116 | 0.0785 ± 0.0018 | 0.4831 ± 0.0116 | 0.1034 ± 0.0018 | 10.6s |
| NeuMF (ours) | behaviour-only | 0.3516 ± 0.0036 | 0.0814 ± 0.0016 | 0.4978 ± 0.0073 | 0.1083 ± 0.0017 | 31.4s |
| VBPR (He & McAuley 2016) | content-aware | 0.3705 ± 0.0048 | 0.0855 ± 0.0009 | 0.5200 ± 0.0096 | 0.1130 ± 0.0008 | 41.9s |
| **MultiModalNCF** (ours) ⭐ | content-aware | 0.3650 ± 0.0077 | **0.0859 ± 0.0017** | 0.5073 ± 0.0148 | **0.1144 ± 0.0015** | 73.2s |

> **Reading the baseline comparison (RQ4)**: adding content helps *within each pair* —
> VBPR > BPR-MF and MultiModalNCF > NeuMF. Against the *published* content-aware
> baseline, MultiModalNCF is statistically tied: VBPR edges HR@10 (0.3705 vs 0.3650)
> and HR@20 (0.5200 vs 0.5073) by less than one seed std, while MultiModalNCF leads
> both NDCG columns (0.0859 / 0.1144 vs 0.0855 / 0.1130). Reaching the published
> baseline's level on its own terms — with a text+image+genre tower rather than a
> single visual term — is the honest reading; the differences are too small to claim
> an outright win. One caveat on the budget: MultiModalNCF runs only **10** epochs
> against VBPR's 30 and BPR-MF's 50, so the baseline numbers come from a longer
> schedule (though MultiModalNCF still costs the most wall-clock time, 73.2s vs
> 41.9s, because of its larger content tower).

### Ablation Study

Each row retrains MultiModalNCF on the same split with one content modality blanked
at the input, isolating its contribution. **Every row is mean ± std over 5 seeds
(42–46)** (`make ablation-seeds`), so each delta can be read against the seed noise
measured on the Full variant.

| Variant | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|:--------|:-----:|:-------:|:-----:|:-------:|
| Full (Text+Image+Genre) | 0.3650 ± 0.0077 | 0.0859 ± 0.0017 | 0.5073 ± 0.0148 | 0.1144 ± 0.0015 |
| w/o Text | 0.3642 ± 0.0100 | 0.0855 ± 0.0024 | 0.5075 ± 0.0108 | 0.1144 ± 0.0025 |
| w/o Image | 0.3631 ± 0.0085 | 0.0842 ± 0.0026 | 0.5080 ± 0.0029 | 0.1132 ± 0.0025 |
| w/o Genre | 0.3673 ± 0.0080 | 0.0857 ± 0.0015 | 0.5107 ± 0.0139 | 0.1144 ± 0.0016 |
| **Behavior only (no content)** | 0.3565 ± 0.0061 | 0.0844 ± 0.0024 | 0.4931 ± 0.0089 | 0.1123 ± 0.0033 |

> **Reading the ablation (RQ1/RQ2)**: dropping *all* content features (behavior only)
> lowers **every** metric — HR@10 0.3650→0.3565, NDCG@10 0.0859→0.0844, HR@20
> 0.5073→0.4931, NDCG@20 0.1144→0.1123 — so content has a consistent, same-signed
> effect. Its magnitude, though, is about one seed std (Full HR@10 std 0.0077 vs the
> 0.0085 drop), so it is a **small** effect, not a decisive one. The single-modality
> deltas sit inside the noise and one even flips sign (w/o Genre edges the Full
> HR@10), so **no per-modality ranking is claimed** — at this scale the three
> modalities are close to interchangeable.

<img src="docs/ablation_study.png" width="720" alt="CineRec Ablation Study">

### Cold-Start Study

100 items are removed from training entirely, then ranked within the cold pool
(the content-aware task). MultiModalNCF scores them through the content tower —
no trained embedding — while NeuMF is the content-blind control. **Reported as
mean ± std over 5 seeds** (`make coldstart-seeds`).

| Model | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|:------|:-----:|:-------:|:-----:|:-------:|
| **MultiModalNCF (content)** ⭐ | **0.1333 ± 0.0290** | **0.0542 ± 0.0151** | **0.2754 ± 0.0233** | **0.0882 ± 0.0139** |
| NeuMF (content-blind) | 0.1140 ± 0.0147 | 0.0467 ± 0.0069 | 0.2351 ± 0.0129 | 0.0739 ± 0.0038 |
| Random | 0.0702 | 0.0288 | 0.2018 | 0.0610 |

> **Reading the cold-start study (RQ3)**: the ordering is identical on every metric —
> content-aware > content-blind > random (e.g. HR@10 0.1333 > 0.1140 > 0.0702). The
> multi-modal tower therefore transfers to items with *no* collaborative history,
> which a behaviour-only model structurally cannot do (its cold embeddings are never
> trained). The content-aware margin over NeuMF is consistent in sign but its HR@10
> seed spread is wide (std 0.0290 vs the 0.0193 gap), so it is reported as a
> direction, not a precise magnitude.

<img src="docs/coldstart.png" width="720" alt="CineRec Cold-Start Study">

### Cross-dataset Validation

The whole protocol — same leave-last-5-out split rule, same candidate filtering,
same eight models, same positive-feedback threshold (≥ 4.0), same seed list
(42–46) — is re-run unchanged on **Amazon Reviews-2023 Video_Games**
([ADR 0005](docs/adr/0005-cross-dataset-validation.md)): a heavy-user slice of
2,288 users × 4,753 items × 75,980 ratings, built by applying *MovieLens 100K's
own population rule* (users with ≥ 20 ratings, item 5-core) to a different
domain. The candidate catalogue is ~2.8× larger than ML-100K's, which
mechanically depresses absolute HR@K — so the comparison reads **relative
orderings and delta directions**, not absolute values.

| Model | ML-100K HR@10 | Amazon HR@10 | ML-100K NDCG@10 | Amazon NDCG@10 |
|:------|:-------------:|:------------:|:---------------:|:--------------:|
| UserCF | 0.0191 ± 0.0000 | 0.0122 ± 0.0000 | 0.0035 ± 0.0000 | 0.0017 ± 0.0000 |
| ItemCF | 0.0721 ± 0.0000 | 0.0214 ± 0.0000 | 0.0142 ± 0.0000 | 0.0035 ± 0.0000 |
| SVD | 0.2534 ± 0.0000 | 0.0651 ± 0.0000 | 0.0532 ± 0.0000 | 0.0116 ± 0.0000 |
| BPR-MF (Rendle 2009) | 0.3476 ± 0.0116 | 0.1055 ± 0.0053 | 0.0785 ± 0.0018 | 0.0198 ± 0.0012 |
| NeuMF | 0.3516 ± 0.0036 | 0.1053 ± 0.0043 | 0.0814 ± 0.0016 | 0.0190 ± 0.0009 |
| LightGCN | 0.2850 ± 0.0041 | **0.1187 ± 0.0074** | 0.0614 ± 0.0004 | **0.0222 ± 0.0011** |
| VBPR (He & McAuley 2016) | **0.3705 ± 0.0048** | 0.1097 ± 0.0050 | 0.0855 ± 0.0009 | 0.0196 ± 0.0017 |
| **MultiModalNCF** ⭐ | 0.3650 ± 0.0077 | 0.1164 ± 0.0085 | **0.0859 ± 0.0017** | 0.0213 ± 0.0018 |

**Ablation on the second domain (HR@10, mean ± std over 5 seeds)**:

| Variant | ML-100K | Amazon |
|:--------|:-------:|:------:|
| Full (Text+Image+Genre) | 0.3650 ± 0.0077 | 0.1164 ± 0.0085 |
| w/o Text | 0.3642 ± 0.0100 | 0.1159 ± 0.0050 |
| w/o Image | 0.3631 ± 0.0085 | 0.1177 ± 0.0033 |
| w/o Genre | 0.3673 ± 0.0080 | 0.1194 ± 0.0017 |
| Behavior only (no content) | 0.3565 ± 0.0061 | 0.1163 ± 0.0011 |

**Cold-start on the second domain (HR@10)**:

| Model | ML-100K | Amazon |
|:------|:-------:|:------:|
| **MultiModalNCF (content)** ⭐ | 0.1333 ± 0.0290 | **0.2364 ± 0.0202** |
| NeuMF (content-blind) | 0.1140 ± 0.0147 | 0.1204 ± 0.0120 |
| Random | 0.0702 | 0.1092 |
| *content / blind ratio* | *1.17×* | *1.96×* |

> **Reading RQ5 — what travels and what doesn't**:
>
> **Transfers (agreement strengthens the MovieLens conclusions):**
> - The ladder ordering **UserCF < ItemCF < SVD** holds in both domains, and the
>   neural models' lead over SVD is *larger* on Amazon (0.105–0.119 vs 0.065).
> - **BPR-MF ≈ NeuMF**: the behaviour-only pair stays a statistical tie in both
>   domains, so the RQ4 reading is not a MovieLens artefact.
> - The ablation's single-modality deltas stay **inside seed noise** in both
>   domains (no per-modality ranking claimed anywhere).
> - **Cold-start is the strongest transfer of all — and it amplifies**: the
>   content tower's edge over the content-blind control grows from **1.17×**
>   on ML-100K to **1.96×** on Amazon (0.2364 vs 0.1204 HR@10, gap ≈ 5.6× the
>   combined std). The sparser the collaborative signal, the more the content
>   tower is worth — exactly what the modality story predicts.
>
> **Does not transfer (reported, not hidden):**
> - *"MultiModalNCF is the best model in the ladder."* On Amazon, **LightGCN**
>   takes the top HR@10 point estimate (0.1187 vs 0.1164) — a difference of
>   0.0023 against a combined seed spread of ~0.011, i.e. a statistical tie
>   that LightGCN wins at **1/7 of the training cost** (11.2s vs 77.4s). The
>   honest cross-domain conclusion: the neural band (0.105–0.119) is the same
>   in both domains, and the preferred model within it is domain-dependent.
> - *"Removing all content hurts."* On Amazon, Behavior only ≈ Full
>   (0.1163 vs 0.1164): with sparser implicit feedback, the NCF path re-learns
>   whatever the content tower added within the 10-epoch budget. On ML-100K
>   that drop was itself only ~1 seed std — the ablation already called it a
>   *small* effect, and on the second domain it vanishes entirely.

---

### Charts from Measured Results

<img src="docs/model_comparison.png" width="720" alt="Model comparison">
<img src="docs/training_time.png" width="600" alt="Training time comparison">

---

## 🚀 Quick Start

> **Zero-training start**: the trained models, content features and a MovieLens
> seed ship in the repo (see [ADR 0002](docs/adr/0002-artifact-storage.md)), and the
> database is seeded on first boot — so a fresh clone serves real recommendations
> without downloading or training anything.

### Docker (Recommended)

```bash
git clone https://github.com/ElijahZhao/cinerec.git
cd cinerec
docker compose up --build        # APP_MODE=full by default
# Visit http://localhost:8000
```

### Local

```bash
git clone https://github.com/ElijahZhao/cinerec.git
cd cinerec
make setup && make serve         # runtime deps only, no torch
# Visit http://localhost:8000
```

`APP_MODE=lite make serve` skips torch entirely and serves the precomputed
`recs_cache.json` — that is the mode used on the always-on free tier.

<details>
<summary><b>Reproduce from scratch (optional)</b></summary>

```bash
make setup-train                 # torch + vision stack
make data                        # MovieLens 100K + enriched metadata
make features                    # Sentence-BERT + ResNet-50 content features
make train && make eval          # train all models → eval_results.json
make eval-seeds                  # 5 seeds → eval_multiseed.json (mean±std)
make ablation-seeds              # 5-seed modality ablation → ablation_multiseed.json
make coldstart-seeds             # 5-seed cold-start study → coldstart_multiseed.json
make charts                      # regenerate docs/*.png from measured results
make test                        # pytest suite

# Cross-dataset (Amazon Video_Games, see ADR 0005) — same protocol, isolated
# data/datasets/amazon-video-games subtree; ~20 min of downloads on first run.
make amazon-data                         # build the heavy-user slice
make features DATASET=amazon-video-games # content features for the slice
make eval-seeds DATASET=amazon-video-games
make ablation-seeds DATASET=amazon-video-games
make coldstart-seeds DATASET=amazon-video-games
```
</details>

---

## 🔧 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML Models** | PyTorch, scikit-learn, `scipy.sparse.linalg`, Sentence-BERT, ResNet-50 |
| **Backend** | Python, FastAPI, SQLite, Uvicorn |
| **Frontend** | Vanilla JS (SPA), GSAP 3, Lenis, Canvas star field, ECharts |
| **Data** | MovieLens 100K + enriched metadata (plot summaries & posters) |
| **Deployment** | Docker / docker-compose · Render free tier (`lite`) · Streamlit Community Cloud (`full`) |
| **Quality** | ruff, black, pytest + coverage floor, pip-audit, pre-commit, GitHub Actions CI |

---

## 📁 Project Structure

```
cinerec/
├── api/                  # FastAPI REST endpoints
│   ├── main.py           # App entry, CORS, static files, /api/health, /api/metrics
│   ├── auth.py           # Session auth (PBKDF2 + signed demo tokens)
│   ├── movies.py         # Movie browsing, search, filtering
│   ├── recommend.py      # Model inference + explanation (APP_MODE aware, LRU cache)
│   ├── metrics.py        # In-process request counters + latency middleware
│   └── eval_api.py       # Evaluation results API
├── models/                # 6 servable models + 2 offline SOTA baselines
│   ├── base.py           # Base recommender class
│   ├── registry.py       # Single source of truth for the algorithm ladder
│   ├── user_cf.py        # User-based CF (Pearson)
│   ├── item_cf.py        # Item-based CF (Adjusted Cosine)
│   ├── svd_als.py        # SVD via truncated SVD (scipy; despite the name, not ALS)
│   ├── neumf.py          # Neural MF (GMF + MLP)
│   ├── lightgcn.py       # LightGCN graph convolution (BPR)
│   ├── multimodal_ncf.py # Multi-Modal NCF ⭐ (Core Innovation)
│   ├── bpr_mf.py         # BPR-MF baseline (Rendle 2009; offline reference only)
│   ├── vbpr.py           # VBPR baseline (He & McAuley 2016; offline reference only)
│   └── explain.py        # RecommenderExplainer (XAI)
├── evaluation/            # Offline evaluation framework
│   ├── metrics.py         # HR@K, NDCG@K, Recall@K
│   ├── runner.py          # leave-last-5-out evaluation runner (single + multi-seed)
│   └── visualize.py       # chart generation from measured results
├── data/                  # Data pipeline
│   ├── download.py        # MovieLens 100K download
│   ├── amazon_dataset.py  # Second dataset: Amazon Video_Games slice builder (ADR 0005)
│   ├── preprocess.py      # Sentence-BERT / ResNet-50 feature engineering
│   ├── enrich_tmdb.py     # TMDB poster/plot enrichment
│   ├── processed/         # Committed models, features and results
│   └── datasets/<name>/   # Per-dataset subtree for CINEREC_DATASET runs (raw + processed)
├── db/                    # SQLite layer (idempotent schema + first-boot seed)
├── frontend/              # Web UI
│   ├── index.html         # SPA shell
│   ├── css/               # Dark/Light theme styles
│   ├── js/                # App logic, animations, effects
│   └── assets/i18n/       # EN/ZH translations
├── scripts/               # precompute, ablation, cold-start, train_all, locustfile
├── tests/                 # pytest suite
├── deploy/                # Streamlit companion (own deps) + notes
│   ├── streamlit/         # Companion app: streamlit_app.py + its own requirements.txt
│   └── hf-space/          # Record of the rejected HF Spaces option (see ADR 0001)
├── docs/                  # Technical report + companion paper + architecture decision records
│   ├── adr/               # Architecture decision records
│   ├── tech-report.md     # Protocol, configurations and raw results (paper-facing)
│   ├── paper.pdf          # Companion paper of this study
│   └── paper.tex          # LaTeX source of the companion paper
├── reports/               # Measured load-test report + Locust HTML output
│   ├── loadtest.md        # Curated, measured load-test findings
│   ├── locust_report.html # Locust HTML report (rendered run)
│   └── locust_stdout.txt  # Raw Locust stdout captured during the run
├── config.py              # Paths, APP_MODE, cache switches
├── pyproject.toml         # Lint/format/test config + coverage floor
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── Makefile
├── requirements.txt / requirements-train.txt
├── CONTRIBUTING.md        # Dev setup, PR checklist
├── CODE_OF_CONDUCT.md     # Contributor Covenant 2.1
├── streamlit_app.py       # live-inference companion (entry: deploy/streamlit/)
├── README.md              # This file — English edition
└── README.zh-CN.md        # Chinese edition of this README
```

---

## ❓ FAQ

<details>
<summary><b>Why are there both a lite and a full mode?</b></summary>

`full` needs torch and serves live inference; `lite` trades live inference for a
precomputed `recs_cache.json` and a minimal memory footprint, so it can stay
resident on a 512 MB free instance. Both are the same codebase, switched by
`APP_MODE`. See [ADR 0001](docs/adr/0001-dual-mode-deployment.md).
</details>

<details>
<summary><b>Why does a brand-new user get popular items?</b></summary>

The precomputed cache only covers existing users; a new user is not in it, so the
API falls back to popular recommendations and **labels that fallback explicitly**
instead of erroring. See the "Consequences" of ADR 0001.
</details>

<details>
<summary><b>Why a single split? How does multi-seed evaluation work?</b></summary>

The split remains a deliberate single leave-last-5-out (the project's goal is to
demonstrate algorithm trade-offs and engineering rigour, not to chase leaderboards).
The neural models (NeuMF / LightGCN / MultiModalNCF) are seeded and support
multi-seed evaluation: `make eval-seeds` trains once per seed 42–46 and writes
mean ± std into `data/processed/eval_multiseed.json`. The default single-seed run
(`make eval`) still writes `eval_results.json` for the dashboard; the two never
overwrite each other. Ablation (`make ablation-seeds`) and cold-start
(`make coldstart-seeds`) support multi-seed the same way, writing
`ablation_multiseed.json` / `coldstart_multiseed.json`. All tables in this README
use 5-seed mean ± std, and no strict significance claims are made about the margins.
</details>

<details>
<summary><b>What are BPR-MF / VBPR? Do they serve live traffic?</b></summary>

They are two **published SOTA baselines** used as external references on the same
split: BPR-MF (Rendle et al., UAI 2009) is a purely behavioural BPR-optimized MF,
and VBPR (He & McAuley, AAAI 2016) is a content-aware visual BPR. Both are trained
and reported in the offline evaluation only (see
[Baseline comparison](#sota-baselines)); they are **not** registered in
[`models/registry.py`](models/registry.py) and **never deploy to the API** — the
serving ladder remains the 6 ladder models.
</details>

<details>
<summary><b>Why Amazon Video_Games as the second dataset? Why keep only heavy users?</b></summary>

The cross-dataset validation ([ADR 0005](docs/adr/0005-cross-dataset-validation.md))
needs an **out-of-domain** dataset to answer RQ5: are the conclusions
MovieLens-specific? The raw Video_Games 5-core benchmark (94.8k users × 25.6k items)
does not fit the dense neighbourhood models on a small-memory host, and randomly
sampled users have a median of ~6 interactions — leave-last-5-out would leave most
of them with empty training sets. So the rule is exactly MovieLens-100K's own
admission rule: keep users with ≥ 20 ratings and rebuild the item 5-core, yielding
2,288 users × 4,753 items × 75,980 ratings. There is no random sampling — the slice
is deterministic by construction; `ratings.csv` is committed to pin the slice
definition, and re-running `make amazon-data` reproduces it. The dataset is used
for offline evaluation only; the live service remains MovieLens 100K.
</details>

<details>
<summary><b>Does SVD use ALS?</b></summary>

No. The filename `models/svd_als.py` is a historical name; the implementation is
truncated SVD (`scipy.sparse.linalg.svds`), as documented in
[ADR 0003](docs/adr/0003-algorithm-ladder.md).
</details>

<details>
<summary><b>How do I run the lite mode with Docker?</b></summary>

```bash
docker build --build-arg APP_MODE=lite -t cinerec:lite .
docker run -p 8000:8000 cinerec:lite
```
Health check: `GET /api/health`; process metrics: `GET /api/metrics`.
</details>

<details>
<summary><b>Why are all the movies so old?</b></summary>

Because the dataset is **MovieLens 100K**, collected through the
MovieLens website over the seven months from **19 Sep 1997 to 22 Apr 1998**; its
catalogue spans **1922–1998**, so a 1998 title is the *end of the range*, not a
stale mirror. That is a property of the source data, not a pipeline bug.

It was a deliberate trade-off: only 100,000 ratings (943 users × 1,682 movies,
each user with ≥ 20 ratings) means all six models train in seconds-to-minutes on
CPU, which lets us report an honest accuracy/cost trade-off and commit the trained
artefacts so a fresh clone serves recommendations with **zero training**
([ADR 0002](docs/adr/0002-artifact-storage.md)). It is also the standard, citable
benchmark, so our numbers can be compared against the literature. Full reasoning
and the rejected alternatives are in [ADR 0004](docs/adr/0004-dataset-choice.md).
</details>

<details>
<summary><b>Where does the data come from, and what about commercial use & redistribution?</b></summary>

Three inputs, three licences, none of which covers the others:

1. **This project's code** — MIT (see [LICENSE](LICENSE)).
2. **The MovieLens 100K ratings** — provided by the GroupLens Research Project,
   University of Minnesota, for **research / non-commercial** use: it must **not be
   redistributed** without permission and must **not** be used for commercial or
   revenue-bearing purposes, and no endorsement by the University of Minnesota may
   be implied. This repository is a **non-commercial portfolio project** and uses it
   on exactly those terms. Publications must cite Harper & Konstan (2015),
   *The MovieLens Datasets: History and Context*, ACM TiiS 5(4). The raw data is
   fetched by [`data/download.py`](data/download.py) from `files.grouplens.org`
   (https-only, allow-listed host).
3. **Poster and overview metadata** — fetched from the TMDB API
   ([`data/enrich_tmdb.py`](data/enrich_tmdb.py)) and subject to TMDB's terms;
   1,400 / 1,682 rows have an overview and 1,366 / 1,682 a poster, and unmatched
   titles degrade gracefully.
</details>

---

## 📄 License

The **source code** is licensed under the MIT License — see [LICENSE](LICENSE).

This does **not** extend to the bundled data:

- **MovieLens 100K** (ratings, titles, genres, years) is © GroupLens Research,
  University of Minnesota, provided for **non-commercial research use**. It may
  not be redistributed or used commercially without permission. See the
  [FAQ entry above](#-faq) and [ADR 0004](docs/adr/0004-dataset-choice.md).
- **Poster and overview metadata** are fetched from the TMDB API
  ([`data/enrich_tmdb.py`](data/enrich_tmdb.py)); TMDB's terms apply.

---

## 📝 Changelog

<details>
<summary>Expand</summary>

- **2026-10-13** — README split into two single-language editions: `README.md`
  (English) and `README.zh-CN.md` (Chinese), with identical structure, tables and
  numbers.
- **2026-10-13** — Paper: the companion paper *CineRec: A Practical Multi-Modal
  Recommender Study with Cross-Dataset Validation and SOTA Baseline Comparison*
  ships as [`docs/paper.pdf`](docs/paper.pdf), referenced from the Evaluation
  section.
- **2026-10-13** — Released **v1.0.0** (first tag + GitHub Release). Governance:
  the three requirements files pin fixed version floors (`urllib3>=2.8.0` /
  `anyio>=4.14.2` / `setuptools>=83.0` / `torch>=2.13.0`) to zero out pip-audit,
  and CI gains a coverage gate (`fail_under = 45`). Consolidations & additions:
  second-dataset paths unified under the `config.SECOND_PROCESSED_DIR` single
  source, asset cache versions unified, plus new [CONTRIBUTING.md](CONTRIBUTING.md),
  [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and the technical report
  [`docs/tech-report.md`](docs/tech-report.md).
- **2026-10-12** — Substance: two published SOTA baselines, BPR-MF (Rendle et al.,
  2009) and VBPR (He & McAuley, 2016), added to the unified evaluation (offline
  comparison only, never served); ablation and cold-start re-run as 5-seed
  mean ± std (`make ablation-seeds` / `make coldstart-seeds`, writing
  `ablation_multiseed.json` / `coldstart_multiseed.json`); README gains research
  questions (RQ1–RQ4), contributions and the SOTA baseline comparison table.
- **2026-10-12** — Reproducibility: NeuMF / LightGCN / MultiModalNCF get a unified
  seeding utility (`set_seed`); `evaluation/runner.py` supports multi-seed
  evaluation (`make eval-seeds`, seeds 42–46); the main table switches to
  mean ± std written to `eval_multiseed.json`; single-seed behaviour is unchanged
  and still writes `eval_results.json` for the dashboard.
- **2026-10-06** — Tests/CI: Playwright E2E (guest login → recommendation,
  filtering → detail modal, rating → recommendation excludes rated) + axe-core
  accessibility assertions; torch-gated forward tests for the three neural models;
  tests for the recommend `full` branch and the explainer failure path; CI gains a
  `full` image build + `/api/health` smoke test; the load test joins a weekly
  schedule with a trend file (`reports/loadtest_trend.csv`); `/api/metrics`
  documents multi-replica aggregation.
- **2026-10-05** — UI: movie detail modal (open by click/keyboard, rating and IMDb
  entry points, re-renders on language switch); dataset trade-offs and licensing
  written into [ADR 0004](docs/adr/0004-dataset-choice.md) and the FAQ; Streamlit
  companion restyled (film-strip logo + KPI strip + fusion card + CSS bar
  charts/heatmap + algorithm ladder + user profile); all README screenshots
  refreshed.
- **2026-10-05** — UI: animated background + canvas star field + typography rework;
  Streamlit companion moves to its own directory and theme.
- **2026-10-05** — Deployment: Render (`lite`) + Streamlit (`full`) dual free-tier
  deployment finalized (ADR 0001).
- **2026-10-04** — Evaluation: unified 6-model evaluation + ablation + cold-start;
  MLOps: LRU cache + `/api/metrics` + Locust load test.
- **2026-07-19** — Initial version: UserCF / ItemCF / SVD / NeuMF + frontend + database.

</details>

---

<div align="center">

**Built with ❤️ by an AI undergraduate student**

*Classic CF → Matrix Factorization → Neural CF → Multi-Modal CF*

</div>
