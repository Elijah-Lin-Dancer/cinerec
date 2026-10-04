<div align="center">

<img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/FastAPI-0.100-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI">
<img src="https://img.shields.io/badge/License-MIT-F5C518?style=flat" alt="License">

<br><br>

# 🎬 CineRec

**Multi-Modal Movie Recommendation System**

*From classic collaborative filtering to cutting-edge multi-modal neural networks — a 5-level algorithm ladder with explainable AI.*

[中文文档](#-项目简介) · [Demo](#-demo-preview) · [Architecture](#-architecture) · [Quick Start](#--quick-start)

</div>

---

## 📋 Table of Contents

- [Project Introduction / 项目简介](#-project-introduction--项目简介)
- [Key Features / 核心特性](#-key-features--核心特性)
- [Demo Preview / 效果预览](#-demo-preview)
- [Architecture / 系统架构](#-architecture)
- [Algorithms / 算法说明](#-algorithms--算法说明)
- [Evaluation / 评测结果](#-evaluation--评测结果)
- [Quick Start / 快速开始](#--quick-start)
- [Tech Stack / 技术栈](#-tech-stack--技术栈)
- [Project Structure / 项目结构](#-project-structure--项目结构)
- [License](#-license)

---

## 📖 Project Introduction / 项目简介

**CineRec** is a full-stack AI recommendation system that implements **5 progressively advanced algorithms** — from classic collaborative filtering to a novel multi-modal neural collaborative filtering model. It features a rigorous offline evaluation framework, an interactive dark-cinema themed bilingual web frontend, and Docker-ready deployment.

**CineRec** 是一个全栈 AI 推荐系统，实现了 **5 个逐层递进的算法** —— 从经典协同过滤到创新的多模态神经协同过滤。具备严谨的离线评测框架、暗色电影院主题的双语交互式 Web 前端，以及开箱即用的 Docker 部署。

> 💡 **"Don't just use models — understand them."**
> 不要只是用模型，要理解它们。CineRec 的每一层算法都展示了推荐系统从传统到前沿的演进路径。

---

## ✨ Key Features / 核心特性

| Feature | Description |
|----------|-------------|
| 🔬 **5-Level Algorithm Ladder** | UserCF → ItemCF → SVD → NeuMF → Multi-Modal NCF. Five progressively advanced algorithms from classic to cutting-edge. |
| 🧠 **Multi-Modal Fusion** | Core innovation: fuses Sentence-BERT text (384d), ResNet-50 image (2048d), and genre (18d) features into the MLP path. |
| 📊 **Rigorous Evaluation** | Leave-last-5-out split (training items excluded from candidates) with HR@K, NDCG@K, Recall@K metrics + modality ablation on MovieLens 100K. |
| 🎯 **Explainable AI** | Content-based + collaborative recommendation reasons for each suggestion. |
| 🌙 **Dark/Light Theme** | Cinema-inspired dark theme with glassmorphism + clean light mode. Bilingual (EN/ZH). |
| 🐳 **Docker Ready** | One-click deployment with Docker Compose. |

---

## 🖼 Demo Preview / 效果预览

### Recommendation Engine / 推荐引擎
> Real-time personalized recommendations with algorithm switching and explainable reasons.

<img src="screenshots/recommend.png" width="800" alt="CineRec Recommendation Page">

### Movie Library / 电影库
> Browse 1,682 movies with real IMDb posters, search, genre filtering, and pagination.

<img src="screenshots/movies.png" width="800" alt="CineRec Movie Library">

### Evaluation Dashboard / 评测看板
> Interactive model comparison with real leave-last-5-out evaluation metrics.

<img src="screenshots/dashboard.png" width="800" alt="CineRec Evaluation Dashboard">

---

## 🏗 Architecture / 系统架构

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
│  (demo login)   (CRUD + posters) (5 Models + Explainer)       │
│        │              │                  │                   │
├────────┼──────────────┼──────────────────┼───────────────────┤
│        │       ┌──────┴──────────────────┴───────┐           │
│        │       │      Model Layer (5 Models)      │           │
│        │       │                                   │           │
│        │       │  ┌─────────┐  ┌─────────┐        │           │
│        │       │  │  UserCF  │  │  ItemCF  │        │           │
│        │       │  │ (Pearson)│  │(Adj Cos)│        │           │
│        │       │  └─────────┘  └─────────┘        │           │
│        │       │  ┌─────────┐  ┌─────────┐        │           │
│        │       │  │   SVD    │  │  NeuMF   │        │           │
│        │       │  │(SVD,k=64)│  │(GMF+MLP) │        │           │
│        │       │  └─────────┘  └─────────┘        │           │
│        │       │  ┌───────────────────────┐      │           │
│        │       │  │   Multi-Modal NCF ⭐   │      │           │
│        │       │  │ Text+Image+Genre Fusion│      │           │
│        │       │  └───────────────────────┘      │           │
│        │       └──────────────────────────────────┘           │
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

## 🧮 Algorithms / 算法说明

| Level | Model | Method | Description |
|:-----:|-------|--------|-------------|
| 1 | **UserCF** | Pearson Correlation | Find K similar users, aggregate their preferences. Vectorized computation. |
| 2 | **ItemCF** | Adjusted Cosine | Recommend items similar to user's rated history. Vectorized similarity. |
| 3 | **SVD** | Matrix Factorization | Decompose the user-item matrix into latent factors (k=64) via truncated SVD (`scipy.sparse.linalg.svds`). |
| 4 | **NeuMF** | GMF + MLP (PyTorch) | Dual-path architecture: element-wise product + deep MLP. BCE loss with negative sampling. |
| 5 | **Multi-Modal NCF** ⭐ | Text+Image+Genre Fusion | Core innovation. Replaces item embedding in MLP path with a Content Tower fusing multi-modal features. |

### Multi-Modal NCF Architecture (Core Innovation / 核心创新)

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

## 📊 Evaluation / 评测结果

### Model Comparison on MovieLens 100K (leave-last-5-out)

Protocol: the 5 most recent interactions of each user are held out; every model is
scored against the same candidate set (all items minus the user's training items),
so re-recommending already-seen titles cannot inflate the numbers.

| Model | HR@10 | NDCG@10 | HR@20 | NDCG@20 | Train Time |
|:-----:|:-----:|:-------:|:-----:|:-------:|:-----------:|
| UserCF | 0.0191 | 0.0035 | 0.0477 | 0.0060 | 0.13s |
| ItemCF | 0.0721 | 0.0142 | 0.1283 | 0.0199 | 0.24s |
| SVD | 0.2534 | 0.0532 | 0.3542 | 0.0664 | 0.2s |
| NeuMF | 0.3468 | 0.0764 | 0.4952 | 0.1025 | 49.7s |
| **MultiModalNCF** ⭐ | **0.3478** | **0.0784** | **0.5005** | **0.1060** | 96.9s |

> **Note**: MultiModalNCF leads on every metric at both cut-offs, narrowly ahead of
> NeuMF and clearly ahead of matrix factorization. The neural models pay for that
> gain with a much longer training time — the ladder makes the accuracy/cost
> trade-off explicit.

### Key Findings / 关键发现

- **UserCF → ItemCF → SVD** shows the expected jump from neighbourhood heuristics to
  matrix factorization; SVD is the best accuracy-per-second point on the ladder.
- **NeuMF** shows that learned non-linear interaction modelling beats plain
  factorization on this dataset once candidates are filtered to unseen items.
- **MultiModalNCF** edges out NeuMF by fusing Sentence-BERT text, ResNet-50 image and
  genre content — the ablation study quantifies each modality's contribution.

### Ablation Study / 消融实验

Each row retrains MultiModalNCF on the same split with one content modality blanked
at the input, isolating its contribution.

| Variant | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|:--------|:-----:|:-------:|:-----:|:-------:|
| Full (Text+Image+Genre) | 0.3606 | 0.0846 | 0.5037 | 0.1110 |
| w/o Text | 0.3552 | 0.0849 | 0.5027 | 0.1120 |
| w/o Image | 0.3648 | 0.0834 | 0.4984 | 0.1096 |
| w/o Genre | 0.3648 | 0.0860 | 0.5080 | 0.1122 |
| **Behavior only (no content)** | 0.3436 | 0.0795 | 0.4825 | 0.1069 |

> **Honest reading**: dropping *all* content features (behavior only) consistently
> hurts every metric — content is doing real work. The single-modality deltas,
> however, fall within single-seed noise (this is one seed, not a significance
> test), so no per-modality ranking is claimed.

<img src="docs/ablation_study.png" width="720" alt="CineRec Ablation Study">

### Cold-Start Study / 冷启动实验

100 items are removed from training entirely, then ranked within the cold pool
(the content-aware task). MultiModalNCF scores them through the content tower —
no trained embedding — while NeuMF is the content-blind control.

| Model | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|:------|:-----:|:-------:|:-----:|:-------:|
| **MultiModalNCF (content)** ⭐ | **0.1228** | **0.0382** | **0.2719** | **0.0743** |
| NeuMF (content-blind) | 0.0965 | 0.0344 | 0.2193 | 0.0630 |
| Random | 0.0702 | 0.0288 | 0.2018 | 0.0610 |

> Content-aware scoring beats the content-blind model, which in turn beats the
> random floor — the multi-modal tower genuinely transfers to items with no
> collaborative history. Single-seed study on MovieLens 100K; no significance claims.

<img src="docs/coldstart.png" width="720" alt="CineRec Cold-Start Study">

### Charts from Measured Results / 实测图表

<img src="docs/model_comparison.png" width="720" alt="Model comparison">
<img src="docs/training_time.png" width="600" alt="Training time comparison">

---

## 🚀 Quick Start / 快速开始

> **Zero-training start**: the trained models, content features and a MovieLens
> seed ship in the repo (see [ADR 0002](docs/adr/0002-artifact-storage.md)), and the
> database is seeded on first boot — so a fresh clone serves real recommendations
> without downloading or training anything.

### Docker (Recommended / 推荐)

```bash
git clone https://github.com/ElijahZhao/cinerec.git
cd cinerec
docker compose up --build        # APP_MODE=full by default
# Visit http://localhost:8000
```

### Local / 本地

```bash
git clone https://github.com/ElijahZhao/cinerec.git
cd cinerec
make setup && make serve         # runtime deps only, no torch
# Visit http://localhost:8000
```

`APP_MODE=lite make serve` skips torch entirely and serves the precomputed
`recs_cache.json` — that is the mode used on the always-on free tier.

### Reproduce from scratch / 一键复现（可选）

```bash
make setup-train                 # torch + vision stack
make data                        # MovieLens 100K + enriched metadata
make features                    # Sentence-BERT + ResNet-50 content features
make train && make eval          # train all 5 models → eval_results.json
make ablation && make coldstart  # modality ablation + cold-start study
make charts                      # regenerate docs/*.png from measured results
make test                        # pytest suite
```

---

## 🔧 Tech Stack / 技术栈

| Layer | Technologies |
|-------|-------------|
| **ML Models** | PyTorch, scikit-learn, scipy.sparse.linalg, Sentence-BERT, ResNet-50 |
| **Backend** | Python, FastAPI, SQLite, Uvicorn |
| **Frontend** | Vanilla JS (SPA), GSAP 3, Lenis, tsParticles, ECharts |
| **Data** | MovieLens 100K + enriched metadata (plot summaries & posters) |
| **Deployment** | Docker, docker-compose |

---

## 📁 Project Structure / 项目结构

```
cinerec/
├── api/                  # FastAPI REST endpoints
│   ├── main.py           # App entry, CORS, static files, /api/health
│   ├── auth.py           # Session auth (PBKDF2 + signed demo tokens)
│   ├── movies.py         # Movie browsing, search, filtering
│   ├── recommend.py      # Model inference + explanation (APP_MODE aware)
│   └── eval_api.py       # Evaluation results API
├── models/                # 5 recommender models
│   ├── base.py           # Base recommender class
│   ├── registry.py       # Single source of truth for the algorithm ladder
│   ├── user_cf.py        # User-based CF (Pearson)
│   ├── item_cf.py        # Item-based CF (Adjusted Cosine)
│   ├── svd_als.py        # SVD via truncated SVD (scipy)
│   ├── neumf.py          # Neural MF (GMF + MLP)
│   ├── multimodal_ncf.py # Multi-Modal NCF ⭐ (Core Innovation)
│   └── explain.py        # RecommenderExplainer (XAI)
├── evaluation/            # Offline evaluation framework
│   ├── metrics.py         # HR@K, NDCG@K, Recall@K
│   ├── runner.py          # leave-last-5-out evaluation runner
│   └── visualize.py       # chart generation from measured results
├── data/                  # Data pipeline
│   ├── download.py        # MovieLens 100K download
│   ├── preprocess.py      # Sentence-BERT / ResNet-50 feature engineering
│   ├── enrich_tmdb.py     # TMDB poster/plot enrichment
│   └── processed/         # Committed models, features and results
├── db/                    # SQLite layer (idempotent schema + first-boot seed)
├── frontend/              # Web UI
│   ├── index.html         # SPA shell
│   ├── css/               # Dark/Light theme styles
│   ├── js/                # App logic, animations, effects
│   └── assets/i18n/       # EN/ZH translations
├── scripts/               # precompute, ablation, cold-start, train_all
├── tests/                 # pytest suite
├── deploy/                # HF Space / Northflank deployment notes
├── docs/adr/              # Architecture decision records
├── config.py              # Paths, APP_MODE, cache switches
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── Makefile
├── requirements.txt / requirements-train.txt
└── README.md
```

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

**Built with ❤️ by an AI undergraduate student**

*Classic CF → Matrix Factorization → Neural CF → Multi-Modal CF*

</div>
