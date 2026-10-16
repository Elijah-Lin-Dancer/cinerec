<div align="center">

<img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-2.13-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/FastAPI-0.100-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI">
<img src="https://img.shields.io/badge/License-MIT-F5C518?style=flat" alt="License">

<br><br>

# 🎬 CineRec

**多模态电影推荐系统**

*六个可服务的推荐器共用一套代码库 —— 从 UserCF 到多模态 NCF，外加两个已发表的
SOTA 基线（BPR-MF、VBPR），全部在同一 leave-last-5-out 划分上评测，神经模型以
5 个种子的均值±标准差报告。*

[English](README.md) · 简体中文

📄 **伴随论文**：*CineRec: A Practical Multi-Modal Recommender Study with
Cross-Dataset Validation and Published-Baseline Comparison* —— [`docs/paper.pdf`](docs/paper.pdf)

[在线演示](#-在线演示) · [效果预览](#-效果预览) · [评测结果](#-评测结果) · [快速开始](#-快速开始)

</div>

---

## 🔗 在线演示

一套代码库，两种运行模式（`APP_MODE`），各自部署在匹配其需求的免费托管平台上
（[ADR 0001](docs/adr/0001-dual-mode-deployment.md)）：

| 模式 | 展示内容 | 链接 |
|------|----------|------|
| **`lite`** — 产品 | 双语 FastAPI Web 应用。推荐结果来自预计算缓存，可运行在 512 MB 实例上。无需注册：点击 **游客体验** 即可。 | **https://cinerec-rfew.onrender.com** |
| **`full`** — 伴生 | Streamlit 应用，加载训练好的 PyTorch 模型**实时推理**：在线推荐、六模型对比、模态消融、冷启动研究。 | **https://cinerec-movie-recommend.streamlit.app** |

> 两者都跑在免费档上，闲置会休眠 —— 首次访问可能需要几秒钟唤醒。伴生应用位于
> [`deploy/streamlit/`](deploy/streamlit/)。

---

## 📋 目录

- [在线演示](#-在线演示)
- [项目简介](#-项目简介)
- [核心亮点](#-核心亮点)
- [效果预览](#-效果预览)
- [系统架构](#-系统架构)
- [算法说明](#-算法说明)
- [评测结果](#-评测结果)
- [快速开始](#-快速开始)
- [技术栈](#-技术栈)
- [项目结构](#-项目结构)
- [常见问题](#-常见问题)
- [许可协议](#-许可协议)
- [更新日志](#-更新日志)

---

## 📖 项目简介

CineRec 是一个全栈电影推荐系统，实现了 6 个复杂度递进的模型，并在同一候选集上统一
评测。它包含双语暗色 Web 前端、REST API，以及可用同一镜像服务两种运行模式的 Docker 构建。

> 💡 **"不要只是用模型，要理解它们。"**
> CineRec 的每一层算法都展示了推荐系统从传统到前沿的演进路径。

---

## ✨ 核心亮点

- **6 个模型，一个真相源** — UserCF → ItemCF → SVD → NeuMF → LightGCN →
  Multi-Modal NCF 全部登记在 [`models/registry.py`](models/registry.py)，前端、API
  与预计算脚本都从这一处取。
- **多模态融合（核心创新）** — Multi-Modal NCF 在 MLP 路径上用 Content Tower 融合
  三路内容特征：Sentence-BERT 文本(384d)、ResNet-50 图像(2048d)、类型多热(18d)。
- **诚实的离线评测** — 单一 leave-last-5-out 划分、候选集剔除训练已见条目；主评测、
  消融与冷启动均以 5 个种子的均值±标准差报告，对照 BPR-MF / VBPR 两个公开 SOTA
  基线且不作"最优模型"声明；同一协议在第二个数据集（Amazon Video_Games）上完整复验。
- **可解释 · 双语 · 双模式** — 每条推荐给出理由；EN/ZH 前端；`lite`（预计算缓存）
  与 `full`（实时推理）由 `APP_MODE` 切换。电影详情弹窗与 Streamlit 伴生应用见
  [效果预览](#-效果预览)。
- **可观测** — 推理 LRU 缓存、`/api/metrics` 进程内延迟/吞吐，外加一次实测 Locust
  压测 → [`reports/loadtest.md`](reports/loadtest.md)。

---

## 🖼 效果预览

### 推荐引擎
> 实时个性化推荐，支持算法切换与可解释推荐理由。

<img src="screenshots/recommend.png" width="800" alt="CineRec 推荐页">

### 电影库
> 浏览 1,682 部电影（其中 1,366 部含真实海报）、搜索、类型筛选与分页。点击卡片打开详情弹窗，含剧情简介、评分与 IMDb 链接。

<img src="screenshots/movies.png" width="800" alt="CineRec 电影库">

### 电影详情弹窗
> 点击任意电影卡片（或在焦点卡片上按 Enter）打开无障碍详情弹窗 —— 海报、简介、类型、平均评分，以及评分与跳转 IMDb 的入口。支持 Escape、关闭按钮与点击遮罩关闭；切换语言时随语言重绘。

<img src="screenshots/detail_modal.png" width="800" alt="CineRec 电影详情弹窗">

### 评测看板
> 基于真实 leave-last-5-out 评测指标的交互式模型对比 —— 六个模型同屏。

<img src="screenshots/dashboard.png" width="800" alt="CineRec 评测看板">

### 消融实验图表
> 评测页向下滚动：消融研究（每次屏蔽一种模态）。由入库的 JSON 结果渲染 —— 无硬编码数字。

<img src="screenshots/eval_charts.png" width="800" alt="消融实验图表">

> 冷启动研究在下方 **实测图表** 区有单独的实测图。

### Streamlit 伴生应用（Full 模式）
> 实时推理伴生应用：胶片 logo、KPI 条、CSS 柱状图、Jaccard 重叠热力图、算法阶梯与用户画像小卡。

<table>
<tr>
<td><img src="screenshots/streamlit_live.png" width="400" alt="Streamlit 实时推荐"></td>
<td><img src="screenshots/streamlit_comparison.png" width="400" alt="Streamlit 六模型对比"></td>
</tr>
<tr>
<td align="center"><sub>实时推荐 + KPI 条</sub></td>
<td align="center"><sub>六模型对比 + 热力图</sub></td>
</tr>
</table>

---

## 🏗 系统架构

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

## 🧮 算法说明

| 层级 | 模型 | 方法 | 说明 |
|:----:|------|------|------|
| 1 | **UserCF** | Pearson 相关 | 找出 K 个相似用户并聚合其偏好，向量化计算。 |
| 2 | **ItemCF** | 调整余弦 | 推荐与用户已评历史相似的物品，向量化相似度。 |
| 3 | **SVD** | 矩阵分解 | 用截断 SVD（`scipy.sparse.linalg.svds`）把用户-物品矩阵分解为 k=64 的隐因子。 |
| 4 | **NeuMF** | GMF + MLP（PyTorch） | 双路结构：逐元素积 + 深层 MLP，BCE 损失 + 负采样。 |
| 5 | **LightGCN** | 图卷积（PyTorch） | 用户-物品二部图上的简化 GCN：逐层邻域聚合取平均，BPR 训练。 |
| 6 | **Multi-Modal NCF** ⭐ | 文本+图像+类型融合 | 核心创新。MLP 路径的物品嵌入替换为融合多模态特征的 Content Tower。 |

> **离线 SOTA 基线** — BPR-MF 与 VBPR 仅为评测对照而实现
> （[`models/bpr_mf.py`](models/bpr_mf.py) / [`models/vbpr.py`](models/vbpr.py)），
> **不属于**可服务阶梯，从不进入 API。

### Multi-Modal NCF 架构（核心创新）

```
                    ┌──────────────────┐
                    │   User Tower     │
                    │ user_id → Emb(32)│
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌──────────────┐
     │  GMF Path  │  │  MLP Path    │
     │ user⊙item  │  │              │
     │ (behavior) │  │  Content Tower (⭐)
     └─────┬──────┘  │  ┌────────────────────┐
           │         │  │ Text(384)→FC(32)    │
           │         │  │ Image(2048)→FC(32)   │
           │         │  │ Genre(18)→FC(32)    │
           │         │  │ Concat(96)→FC(32)   │
           │         │  └──────────┬─────────┘
           │         │             │
           │         │  Concat(user, content)
           │         │      FC(128→64→32)
           │         └──────┬──────┘
           │                │
           └─────── Concat(GMF, MLP) ──→ FC(64→1) ──→ Sigmoid ──→ Prediction
```

**冷启动支持**：新物品仅使用 `content_emb`，GMF 路径补零向量。

---

## 📊 评测结果

> **定位说明** — CineRec 是一个工程与取舍研究项目，不是刷榜：所有模型共用同一划分，
> 神经结果以**多种子均值±标准差**报告，结论的强弱以实测噪声为界；同一协议在第二个域
> （Amazon Video_Games）上原样重跑，以检验哪些结论能跨域成立。

> 📄 **伴随论文** — *CineRec: A Practical Multi-Modal Recommender Study with
> Cross-Dataset Validation and Published-Baseline Comparison* 已随本仓库交付于
> [`docs/paper.pdf`](docs/paper.pdf)；其协议、配置与原始结果记录在
> [`docs/tech-report.md`](docs/tech-report.md)。

### 研究问题

| # | 问题 | 由谁回答 |
|:--|:-----|:---------|
| **RQ1** | 相比纯行为信号，内容特征（文本/图像/类型）在同一排序任务上带来多大提升？ | [消融实验](#消融实验) |
| **RQ2** | 这一提升是否大于随机种子的波动？ | 主表与消融表的**均值±标准差** |
| **RQ3** | 内容特征能否迁移到完全没有交互历史的冷启动条目？ | [冷启动实验](#冷启动实验) |
| **RQ4** | 与已发表的 SOTA 基线（BPR-MF / VBPR）相比处于什么位置？ | [SOTA 基线对比](#sota-基线对比) |
| **RQ5** | 上述结论是否是 MovieLens 特有的？换一个域还成立吗？ | [跨数据集验证](#跨数据集验证) |

### 主要贡献

1. **一个可复现的统一评测协议** —— 单一 leave-last-5-out 划分、跨模型对齐的候选集
   （剔除用户训练已见条目）、以及覆盖随机种子全链路的多种子评测，主结果以 5 个种子的
   均值±标准差报告（`make eval-seeds`）。
2. **Multi-Modal NCF（核心创新）** —— 在 NCF 的 MLP 路径上用 Content Tower 融合
   Sentence-BERT 文本、ResNet-50 图像与类型多热三路特征。
3. **诚实的消融与冷启动研究** —— 两者都以多种子报告，明确区分"真实效应"与"种子噪声"。
4. **与公开 SOTA 基线对照** —— 在同一划分上复现 BPR-MF（Rendle 2009）与
   VBPR（He & McAuley 2016），作为行为-only 与内容感知两条线的外部参考。
5. **跨数据集验证** —— 同一协议与种子列表在第二个域（Amazon Video_Games，
   2,288 重度用户 × 4,753 物品）上完整重跑主评测/消融/冷启动，检验结论是否
   随域迁移（[ADR 0005](docs/adr/0005-cross-dataset-validation.md)）。

### MovieLens 100K 模型对比（leave-last-5-out）

协议：每个用户最近 5 次交互被留出；每个模型都在同一候选集上打分（全物品减去用户训练
已见条目），因此重复推荐已看影片不可能虚高数字。数值为 **5 个随机种子（42–46）的
均值±标准差**，用 `make eval-seeds` 复现（写入 `data/processed/eval_multiseed.json`）。
训练耗时列取自同一文件的多 seed 均值；单 seed 运行的实际耗时在 `eval_results.json`
（服务端 artefact）。三个启发式/MF 模型是确定性的，波动恰为零。

| 模型 | HR@10 | NDCG@10 | HR@20 | NDCG@20 | 训练耗时 |
|:-----:|:-----:|:-------:|:-----:|:-------:|:-----------:|
| UserCF | 0.0191 ± 0.0000 | 0.0035 ± 0.0000 | 0.0477 ± 0.0000 | 0.0060 ± 0.0000 | 0.1s |
| ItemCF | 0.0721 ± 0.0000 | 0.0142 ± 0.0000 | 0.1283 ± 0.0000 | 0.0199 ± 0.0000 | 0.3s |
| SVD | 0.2534 ± 0.0000 | 0.0532 ± 0.0000 | 0.3542 ± 0.0000 | 0.0664 ± 0.0000 | 0.3s |
| NeuMF | 0.3516 ± 0.0036 | 0.0814 ± 0.0016 | 0.4978 ± 0.0073 | 0.1083 ± 0.0017 | 31.4s |
| LightGCN | 0.2850 ± 0.0041 | 0.0614 ± 0.0004 | 0.4250 ± 0.0047 | 0.0816 ± 0.0007 | 10.7s |
| **MultiModalNCF** ⭐ | **0.3650 ± 0.0077** | **0.0859 ± 0.0017** | **0.5073 ± 0.0148** | **0.1144 ± 0.0015** | 73.2s |

> **这组数字说明了什么（阶梯六模型）**：5 个种子平均后，**MultiModalNCF** 在阶梯内
> 领跑全部四项指标 —— HR@10 0.3650 / NDCG@10 0.0859 / HR@20 0.5073 / NDCG@20 0.1144
> —— **NeuMF** 居次（0.3516 / 0.0814）。阶梯内 HR@10 差距（0.0134）大于任一模型的
> 种子波动，而 HR@20 差距（0.0095）落在 MultiModalNCF 的标准差（0.0148）之内。
> **LightGCN** 落在 SVD 与 NeuMF 之间，训练耗时只有 NeuMF 的零头。神经模型用长得多的
> 训练时间换取精度，精度/成本取舍由此显式化。等**已发表**基线加入对比（下一节）后，
> 头部差距进一步收窄，因此不对"最优模型"做强显著性声明。

### 关键发现

- UserCF → ItemCF → SVD 保持预期顺序 —— 近邻启发式让位于矩阵分解 —— 且 SVD 是
  阶梯上精度/秒最优的点（HR@10 0.2534，且为确定性模型，无种子波动）。
- NeuMF：HR@10 0.3516 ± 0.0036 / HR@20 0.4978 ± 0.0073。候选集过滤已看条目后，
  非线性交互建模小幅胜出纯分解，种子波动小但非零。
- LightGCN：HR@10 0.2850 ± 0.0041，介于 SVD 与 NeuMF 之间，且是神经模型中精度/秒
  最优者 —— 完全不用内容特征也有竞争力。
- MultiModalNCF：HR@10 0.3650 ± 0.0077 / NDCG@10 0.0859 ± 0.0017。5 个种子平均后
  在阶梯六模型的命中率与 NDCG 两列均领先，每种模态的贡献占比由下方消融实验量化。

### SOTA 基线对比

为了把 CineRec 放到文献坐标系而不只是和自己比，同一划分复现了两个已发表基线 ——
**BPR-MF**（Rendle et al., UAI 2009）作为行为-only 参考，**VBPR**（He & McAuley,
AAAI 2016）作为内容感知参考。它们只是离线参考：不进可服务阶梯
（[`models/registry.py`](models/registry.py)），从不部署到 API。

| 模型 | 路线 | HR@10 | NDCG@10 | HR@20 | NDCG@20 | 训练耗时 |
|:------|:-----|:-----:|:-------:|:-----:|:-------:|:----------:|
| BPR-MF（Rendle 2009） | 行为-only | 0.3476 ± 0.0116 | 0.0785 ± 0.0018 | 0.4831 ± 0.0116 | 0.1034 ± 0.0018 | 10.6s |
| NeuMF（本项目） | 行为-only | 0.3516 ± 0.0036 | 0.0814 ± 0.0016 | 0.4978 ± 0.0073 | 0.1083 ± 0.0017 | 31.4s |
| VBPR（He & McAuley 2016） | 内容感知 | 0.3705 ± 0.0048 | 0.0855 ± 0.0009 | 0.5200 ± 0.0096 | 0.1130 ± 0.0008 | 41.9s |
| **MultiModalNCF**（本项目）⭐ | 内容感知 | 0.3650 ± 0.0077 | **0.0859 ± 0.0017** | 0.5073 ± 0.0148 | **0.1144 ± 0.0015** | 73.2s |

> **如何读基线对比（RQ4）**：在每一对内部加内容都有帮助 —— VBPR > BPR-MF，
> MultiModalNCF > NeuMF。对照**已发表**的内容感知基线，MultiModalNCF 是统计平局：
> VBPR 在 HR@10（0.3705 vs 0.3650）与 HR@20（0.5200 vs 0.5073）上以小于一个种子
> 标准差的优势领先，而 MultiModalNCF 在两列 NDCG 上领先（0.0859 / 0.1144 vs
> 0.0855 / 0.1130）。用文本+图像+类型的 Content Tower（而非单一视觉项）追平已发表
> 基线的水准，是诚实的读法；差距太小，不足以宣称完胜。一个预算上的说明：
> MultiModalNCF 只跑 **10** 个 epoch，VBPR 30 个、BPR-MF 50 个，基线数字来自更长的
> 训练日程（不过 MultiModalNCF 的墙钟时间仍最贵，73.2s vs 41.9s，因为它有更大的
> Content Tower）。

### 消融实验

每行在相同划分上重训 MultiModalNCF，但把一种内容模态在输入端置空，以隔离其贡献。
**每行都是 5 个种子（42–46）的均值±标准差**（`make ablation-seeds`），因此每个差值
都可以对照 Full 变体的种子噪声来读。

| 变体 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|:--------|:-----:|:-------:|:-----:|:-------:|
| Full（文本+图像+类型） | 0.3650 ± 0.0077 | 0.0859 ± 0.0017 | 0.5073 ± 0.0148 | 0.1144 ± 0.0015 |
| 去掉 Text | 0.3642 ± 0.0100 | 0.0855 ± 0.0024 | 0.5075 ± 0.0108 | 0.1144 ± 0.0025 |
| 去掉 Image | 0.3631 ± 0.0085 | 0.0842 ± 0.0026 | 0.5080 ± 0.0029 | 0.1132 ± 0.0025 |
| 去掉 Genre | 0.3673 ± 0.0080 | 0.0857 ± 0.0015 | 0.5107 ± 0.0139 | 0.1144 ± 0.0016 |
| **仅行为（无内容）** | 0.3565 ± 0.0061 | 0.0844 ± 0.0024 | 0.4931 ± 0.0089 | 0.1123 ± 0.0033 |

> **如何读消融（RQ1/RQ2）**：把全部内容特征拿掉（仅行为）会让**每一项**指标下降 ——
> HR@10 0.3650→0.3565，NDCG@10 0.0859→0.0844，HR@20 0.5073→0.4931，NDCG@20
> 0.1144→0.1123 —— 内容的效应方向一致。但其量级约为一个种子标准差（Full HR@10 的
> std 0.0077 vs 降幅 0.0085），所以是**小**效应，不是决定性的。单模态差值都落在噪声
> 之内，甚至有一个翻转了符号（去掉 Genre 的 HR@10 反而略高），因此**不声明任何单模态
> 排序** —— 在这个规模上三种模态接近可互换。

<img src="docs/ablation_study.png" width="720" alt="CineRec 消融实验">

### 冷启动实验

100 个物品从训练中完全移除，然后在冷池内排序（内容感知任务）。MultiModalNCF 通过
Content Tower 给它们打分 —— 没有训练过的嵌入 —— NeuMF 是内容盲对照。**以 5 个种子
的均值±标准差报告**（`make coldstart-seeds`）。

| 模型 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|:------|:-----:|:-------:|:-----:|:-------:|
| **MultiModalNCF（内容）** ⭐ | **0.1333 ± 0.0290** | **0.0542 ± 0.0151** | **0.2754 ± 0.0233** | **0.0882 ± 0.0139** |
| NeuMF（内容盲） | 0.1140 ± 0.0147 | 0.0467 ± 0.0069 | 0.2351 ± 0.0129 | 0.0739 ± 0.0038 |
| 随机 | 0.0702 | 0.0288 | 0.2018 | 0.0610 |

> **如何读冷启动（RQ3）**：每一项指标上的顺序都一致 —— 内容感知 > 内容盲 > 随机
> （如 HR@10 0.1333 > 0.1140 > 0.0702）。多模态塔因此能迁移到**完全没有**协作信号的
> 物品 —— 这是行为-only 模型在结构上做不到的（它的冷嵌入从未被训练）。内容感知相对
> NeuMF 的优势方向一致，但 HR@10 的种子波动很宽（std 0.0290 vs 差距 0.0193），所以
> 只报告方向，不报精确量级。

<img src="docs/coldstart.png" width="720" alt="CineRec 冷启动实验">

### 跨数据集验证

整套协议 —— 相同的 leave-last-5-out 划分规则、相同的候选集过滤、相同的八个模型、
相同的正反馈阈值（≥ 4.0）、相同的种子列表（42–46）—— 在
**Amazon Reviews-2023 Video_Games**（[ADR 0005](docs/adr/0005-cross-dataset-validation.md)）
上原样重跑：一个 2,288 用户 × 4,753 物品 × 75,980 评分的重度用户切片，用 *MovieLens
100K 自己的人群准入规则*（评分数 ≥ 20 的用户、物品 5-core）套在另一个域上构造。
候选目录约是 ML-100K 的 2.8 倍，机械性地压低绝对 HR@K —— 因此对比读的是**相对顺序
与差值方向**，不是绝对值。

| 模型 | ML-100K HR@10 | Amazon HR@10 | ML-100K NDCG@10 | Amazon NDCG@10 |
|:------|:-------------:|:------------:|:---------------:|:--------------:|
| UserCF | 0.0191 ± 0.0000 | 0.0122 ± 0.0000 | 0.0035 ± 0.0000 | 0.0017 ± 0.0000 |
| ItemCF | 0.0721 ± 0.0000 | 0.0214 ± 0.0000 | 0.0142 ± 0.0000 | 0.0035 ± 0.0000 |
| SVD | 0.2534 ± 0.0000 | 0.0651 ± 0.0000 | 0.0532 ± 0.0000 | 0.0116 ± 0.0000 |
| BPR-MF（Rendle 2009） | 0.3476 ± 0.0116 | 0.1055 ± 0.0053 | 0.0785 ± 0.0018 | 0.0198 ± 0.0012 |
| NeuMF | 0.3516 ± 0.0036 | 0.1053 ± 0.0043 | 0.0814 ± 0.0016 | 0.0190 ± 0.0009 |
| LightGCN | 0.2850 ± 0.0041 | **0.1187 ± 0.0074** | 0.0614 ± 0.0004 | **0.0222 ± 0.0011** |
| VBPR（He & McAuley 2016） | **0.3705 ± 0.0048** | 0.1097 ± 0.0050 | 0.0855 ± 0.0009 | 0.0196 ± 0.0017 |
| **MultiModalNCF** ⭐ | 0.3650 ± 0.0077 | 0.1164 ± 0.0085 | **0.0859 ± 0.0017** | 0.0213 ± 0.0018 |

**第二域上的消融（HR@10，5 个种子的均值±标准差）**：

| 变体 | ML-100K | Amazon |
|:--------|:-------:|:------:|
| Full（文本+图像+类型） | 0.3650 ± 0.0077 | 0.1164 ± 0.0085 |
| 去掉 Text | 0.3642 ± 0.0100 | 0.1159 ± 0.0050 |
| 去掉 Image | 0.3631 ± 0.0085 | 0.1177 ± 0.0033 |
| 去掉 Genre | 0.3673 ± 0.0080 | 0.1194 ± 0.0017 |
| 仅行为（无内容） | 0.3565 ± 0.0061 | 0.1163 ± 0.0011 |

**第二域上的冷启动（HR@10）**：

| 模型 | ML-100K | Amazon |
|:------|:-------:|:------:|
| **MultiModalNCF（内容）** ⭐ | 0.1333 ± 0.0290 | **0.2364 ± 0.0202** |
| NeuMF（内容盲） | 0.1140 ± 0.0147 | 0.1204 ± 0.0120 |
| 随机 | 0.0702 | 0.1092 |
| *内容/盲 比* | *1.17×* | *1.96×* |

> **如何读 RQ5 —— 什么迁移了，什么没有**：
>
> **迁移（一致强化了 MovieLens 的结论）：**
> - 阶梯顺序 **UserCF < ItemCF < SVD** 在两个域都成立，且神经模型对 SVD 的领先在
>   Amazon 上*更大*（0.105–0.119 vs 0.065）。
> - **BPR-MF ≈ NeuMF**：行为-only 对在两个域都是统计平局，RQ4 的读法不是 MovieLens
>   特产。
> - 消融的单模态差值在两个域都**落在种子噪声之内**（任何域都不声明单模态排序）。
> - **冷启动是最强的迁移 —— 且被放大**：内容塔相对内容盲的优势从 ML-100K 的
>   **1.17×** 增长到 Amazon 的 **1.96×**（HR@10 0.2364 vs 0.1204，差距 ≈ 组合标准差
>   的 4.9 倍）。协作信号越稀疏，内容塔越值钱 —— 与模态叙事的预测完全一致。
>
> **不迁移（如实报告，不隐藏）：**
> - *"MultiModalNCF 是阶梯最优模型。"* 在 Amazon 上，**LightGCN** 拿下 HR@10 的最高
>   点估计（0.1187 vs 0.1164）—— 差值 0.0023 对约 0.011 的组合种子波动，是统计平局，
>   而且 LightGCN 只用 **1/7 的训练成本**（11.2s vs 77.4s）获胜。诚实的跨域结论：
>   神经带（0.105–0.119）在两个域相同，带内偏好的模型随域而定。
> - *"去掉全部内容会变差。"* 在 Amazon 上，仅行为 ≈ Full（0.1163 vs 0.1164）：隐式
>   反馈更稀疏时，NCF 路径在 10-epoch 预算内把内容塔带来的东西重新学了回来。ML-100K
>   上那个降幅本身只有约 1 个种子标准差 —— 消融早已称其为*小*效应，到第二个域上它
>   完全消失。

---

### 实测图表

<img src="docs/model_comparison.png" width="720" alt="模型对比">
<img src="docs/training_time.png" width="600" alt="训练耗时对比">

---

## 🚀 快速开始

> **零训练启动**：训练好的模型、内容特征与 MovieLens 种子数据都随仓库交付
> （见 [ADR 0002](docs/adr/0002-artifact-storage.md)），数据库在首次启动时自动播种
> —— 新克隆无需下载、无需训练即可提供真实推荐。

### Docker（推荐）

```bash
git clone https://github.com/ElijahZhao/cinerec.git
cd cinerec
docker compose up --build        # 默认 APP_MODE=full
# 访问 http://localhost:8000
```

### 本地运行

```bash
git clone https://github.com/ElijahZhao/cinerec.git
cd cinerec
make setup && make serve         # 仅运行时依赖，不含 torch
# 访问 http://localhost:8000
```

`APP_MODE=lite make serve` 完全跳过 torch，直接服务预计算的 `recs_cache.json`
—— 这就是常驻免费实例所用的模式。

<details>
<summary><b>从零复现（可选）</b></summary>

```bash
make setup-train                 # torch + 视觉栈
make data                        # MovieLens 100K + 富化元数据
make features                    # Sentence-BERT + ResNet-50 内容特征
make train && make eval          # 训练全部模型 → eval_results.json
make eval-seeds                  # 5 seeds → eval_multiseed.json（均值±标准差）
make ablation-seeds              # 5-seed 模态消融 → ablation_multiseed.json
make coldstart-seeds             # 5-seed 冷启动研究 → coldstart_multiseed.json
make charts                      # 从实测结果重新生成 docs/*.png
make test                        # pytest 套件

# 跨数据集（Amazon Video_Games，见 ADR 0005）—— 同一协议，隔离于
# data/datasets/amazon-video-games 子树；首次运行约 20 分钟下载。
make amazon-data                         # 构建重度用户切片
make features DATASET=amazon-video-games # 切片的内容特征
make eval-seeds DATASET=amazon-video-games
make ablation-seeds DATASET=amazon-video-games
make coldstart-seeds DATASET=amazon-video-games
```
</details>

---

## 🔧 技术栈

| 层 | 技术 |
|-------|-------------|
| **机器学习模型** | PyTorch, scikit-learn, `scipy.sparse.linalg`, Sentence-BERT, ResNet-50 |
| **后端** | Python, FastAPI, SQLite, Uvicorn |
| **前端** | 原生 JS (SPA), GSAP 3, Lenis, Canvas 星空, ECharts |
| **数据** | MovieLens 100K + 富化元数据（剧情简介与海报） |
| **部署** | Docker / docker-compose · Render 免费档（`lite`）· Streamlit Community Cloud（`full`） |
| **质量** | ruff, black, pytest + 覆盖率门槛, pip-audit, pre-commit, GitHub Actions CI |

---

## 📁 项目结构

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
├── README.md              # English edition
└── README.zh-CN.md        # 本文件 —— 中文版 README
```

---

## ❓ 常见问题

<details>
<summary><b>为什么有 lite 和 full 两种模式？</b></summary>

`full` 需要 torch 并实时推理；`lite` 用预计算的 `recs_cache.json` 换取极小内存占用，
以便在 512 MB 免费实例上常驻。两者是同一代码库，由 `APP_MODE` 切换。详见
[ADR 0001](docs/adr/0001-dual-mode-deployment.md)。
</details>

<details>
<summary><b>新注册用户的推荐为什么是热门榜？</b></summary>

预计算缓存只覆盖已有用户；新用户不在缓存内，接口会回退到热门推荐并**显式标注**，
而不是报错。见 ADR 0001 的 "Consequences"。
</details>

<details>
<summary><b>评测为什么只有 single split？多 seed 是怎么跑的？</b></summary>

划分仍是刻意的单一 leave-last-5-out（项目目标是展示算法取舍与工程规范，而非刷 SOTA）。
神经模型（NeuMF / LightGCN / MultiModalNCF）已固定随机种子并支持多种子评测：`make eval-seeds`
会用种子 42–46 各训练一次，把均值±标准差写入 `data/processed/eval_multiseed.json`；
默认单 seed 运行（`make eval`）仍写 `eval_results.json` 供看板读取，二者互不覆盖。
消融（`make ablation-seeds`）与冷启动（`make coldstart-seeds`）同样支持多种子，
分别写 `ablation_multiseed.json` / `coldstart_multiseed.json`。正文主表、消融表、
冷启动表均采用 5 seed 均值±标准差，但对领先幅度不作严格显著性声明。
</details>

<details>
<summary><b>BPR-MF / VBPR 是什么？会进入线上服务吗？</b></summary>

它们是两个**公开的 SOTA 基线**，用于在同一划分上做外部对照：BPR-MF（Rendle et al.,
UAI 2009）是纯行为的 BPR 排序 MF，VBPR（He & McAuley, AAAI 2016）是内容感知的视觉
BPR。两者只在离线评测中训练与报告（见 [SOTA 基线对比](#sota-基线对比)），
**不登记进 [`models/registry.py`](models/registry.py)，不会部署到 API**——线上服务的
仍是 6 个阶梯模型。
</details>

<details>
<summary><b>第二数据集为什么选 Amazon Video_Games？为什么只保留重度用户？</b></summary>

跨数据集验证（[ADR 0005](docs/adr/0005-cross-dataset-validation.md)）需要一个
**域外**数据集来回答 RQ5：结论是否 MovieLens 特有。Video_Games 的原始 5-core
基准（94.8k 用户 × 25.6k 物品）在小内存主机上跑不动稠密邻域模型，而随机抽样
用户的中位交互数只有 ~6，leave-last-5-out 会把大多数用户划成空训练集——所以
采用与 MovieLens-100K **完全相同的准入规则**：只保留评分数 ≥ 20 的用户并对物品
重建 5-core，得到 2,288 用户 × 4,753 物品 × 75,980 评分。无随机抽样，切片由
构造保证确定性；`ratings.csv` 入库固定切片定义，重跑 `make amazon-data` 即可复现。
该数据集仅用于离线评测，线上服务仍是 MovieLens 100K。
</details>

<details>
<summary><b>SVD 用的是 ALS 吗？</b></summary>

不是。文件名 `models/svd_als.py` 是历史命名，实现是 truncated SVD
（`scipy.sparse.linalg.svds`），[ADR 0003](docs/adr/0003-algorithm-ladder.md) 已说明。
</details>

<details>
<summary><b>如何用 Docker 跑 lite 模式？</b></summary>

```bash
docker build --build-arg APP_MODE=lite -t cinerec:lite .
docker run -p 8000:8000 cinerec:lite
```
健康检查：`GET /api/health`；进程指标：`GET /api/metrics`。
</details>

<details>
<summary><b>为什么数据集里的电影都这么老？</b></summary>

因为数据集是 **MovieLens 100K**，它是在 **1997-09-19 到 1998-04-22** 这七个月里
从 MovieLens 网站收集的，片目覆盖 **1922–1998**，所以片库里最新的电影就是 1998 年的——
这是**原始数据的年代范围**，不是我们的数据管道出问题。

选它是一次明确的取舍：它只有 100,000 条评分（943 用户 × 1,682 部电影，每人都至少评过 20 部），
六模型全部可在 CPU 上秒级到分钟级训练完，因此我们能把"精度 vs 训练耗时"的真实权衡讲清楚，
并把训练好的模型与特征直接入库，让一个新克隆**零训练**即可跑起来（见 [ADR 0002](docs/adr/0002-artifact-storage.md)）。
它同时是学术界通行、可引用的基准，便于把 HR@K / NDCG@K 与文献对照。
完整的取舍与被否决的选项见 [ADR 0004](docs/adr/0004-dataset-choice.md)。
</details>

<details>
<summary><b>数据从哪里来？商用 / 再分发的许可如何？</b></summary>

三份材料，三套许可，互不覆盖：

1. **本项目代码** —— MIT（见 [LICENSE](LICENSE)）。
2. **MovieLens 100K 评分数据** —— 由明尼苏达大学 GroupLens Research Project 提供，**仅供研究/非商业**用途：
   未经许可**不得再分发**，**不得用于任何商业或盈利目的**，且不得暗示获得明尼苏达大学的背书。
   本项目是**非商业的个人作品集项目**，正是按此条件使用。若在出版物中使用，请引用
   Harper & Konstan (2015), *The MovieLens Datasets: History and Context*, ACM TiiS 5(4)。
   原始数据经 [`data/download.py`](data/download.py) 从 `files.grouplens.org` 获取（https-only + 白名单）。
3. **海报与剧情简介（元数据）** —— 通过 TMDB API 获取（[`data/enrich_tmdb.py`](data/enrich_tmdb.py)），
   受 TMDB 条款约束；1,682 部中有 1,400 部有简介、1,366 部有海报，未匹配的会优雅降级。
</details>

---

## 📄 许可协议

**源代码**以 MIT 许可发布 —— 见 [LICENSE](LICENSE)。

MIT 许可**不**覆盖随仓库交付的数据：

- **MovieLens 100K**（评分、片名、类型、年份）© GroupLens Research（明尼苏达大学），
  按**非商业研究用途**提供；未经许可不得再分发或商用。见上方
  [常见问题](#-常见问题)与 [ADR 0004](docs/adr/0004-dataset-choice.md)。
- **海报与剧情简介元数据**通过 TMDB API 获取
  （[`data/enrich_tmdb.py`](data/enrich_tmdb.py)），受 TMDB 条款约束。

---

## 📝 更新日志

<details>
<summary>展开查看</summary>

- **2026-10-13** — README 拆分为两个单语版本：`README.md`（英文）与 `README.zh-CN.md`
  （中文），结构、表格与数字完全一致。
- **2026-10-13** — 论文：伴随论文《CineRec: A Practical Multi-Modal Recommender
  Study with Cross-Dataset Validation and Published-Baseline Comparison》入库为
  [`docs/paper.pdf`](docs/paper.pdf)，README 评测区挂引用。
- **2026-10-13** — 发布 **v1.0.0**（首个 tag + GitHub Release）。治理面：三个 requirements 写入修复版本下限（`urllib3>=2.8.0` / `anyio>=4.14.2` / `setuptools>=83.0` / `torch>=2.13.0`）使 pip-audit 归零，CI 增加覆盖率门槛（`fail_under = 45`）。收敛与新增：第二数据集路径统一为 `config.SECOND_PROCESSED_DIR` 单一源，资源缓存版本号统一，新增 [CONTRIBUTING.md](CONTRIBUTING.md)、[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) 与技术文档 [`docs/tech-report.md`](docs/tech-report.md)。
- **2026-10-12** — 含金量：新增两个公开 SOTA 基线 BPR-MF（Rendle et al., 2009）与 VBPR（He & McAuley, 2016）并纳入统一评测（仅离线对照，不进线上服务）；消融与冷启动改为 5 seed 均值±标准差（`make ablation-seeds` / `make coldstart-seeds`，写 `ablation_multiseed.json` / `coldstart_multiseed.json`）；README 新增研究问题（RQ1–RQ4）、主要贡献与 SOTA 基线对比表。
- **2026-10-12** — 可复现性：NeuMF / LightGCN / MultiModalNCF 增加统一随机种子（`set_seed`）；`evaluation/runner.py` 支持多种子评测（`make eval-seeds`，种子 42–46），主评测表改为均值±标准差并写入 `eval_multiseed.json`；单 seed 运行行为不变，仍写 `eval_results.json` 供看板读取。
- **2026-10-06** — 测试/CI 补齐：Playwright E2E（游客登录→推荐、筛选→详情弹窗、评分→推荐排除已评）+ axe-core 无障碍断言；三个神经模型的 torch 门控 forward 测试；recommend `full` 分支与解释器失败路径测试；CI 新增 `full` 镜像构建 + `/api/health` 冒烟；压测纳入每周定时任务并留趋势（`reports/loadtest_trend.csv`）；`/api/metrics` 补充多副本聚合说明。
- **2026-10-05** — UI：电影详情弹窗（卡片点击/键盘打开、评分与 IMDb 入口、随语言切换重绘）；数据集取舍与许可写入 [ADR 0004](docs/adr/0004-dataset-choice.md) 与上方 FAQ；Streamlit 伴生应用美化（胶片 logo + KPI 条 + 融合推荐卡 + CSS 柱状图/热力图 + 算法阶梯 + 用户画像）；README 截图全部更新为最新版本。
- **2026-10-05** — UI：动画背景 + canvas 星空 + 字体改版；Streamlit 伴生应用独立目录与主题。
- **2026-10-05** — 部署：Render(`lite`) + Streamlit(`full`) 双档免费部署定稿（ADR 0001）。
- **2026-10-04** — 评测：6 模型统一评测 + 消融 + 冷启动实验；MLOps：LRU 缓存 + `/api/metrics` + Locust 压测。
- **2026-07-19** — 初始版本：UserCF / ItemCF / SVD / NeuMF + 前端 + 数据库。

</details>

---

<div align="center">

**Built with ❤️ by an AI undergraduate student**

*Classic CF → Matrix Factorization → Neural CF → Multi-Modal CF*

</div>
