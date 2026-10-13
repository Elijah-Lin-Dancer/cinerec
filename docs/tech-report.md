# CineRec Technical Report

Companion document to `README.md` for readers who need the experimental
configuration and raw numbers rather than a product overview. All figures here
are read from the committed result artefacts listed under
[Reproducibility](#6-reproducibility); they are the same values the README
tables show.

## 1. Scope

CineRec compares six recommendation models of increasing complexity — plus two
published baselines — on one fixed split, with modality-ablation and
cold-start studies, and repeats the whole protocol on a second domain. It is a
trade-off study (accuracy against training cost, content against behaviour),
not a leaderboard submission: no claim of statistical superiority is made
beyond what the reported spreads support.

## 2. Datasets

| Property | MovieLens 100K | Amazon Reviews-2023 Video_Games (slice) |
|---|---|---|
| Users | 943 | 2,288 |
| Items | 1,682 | 4,753 |
| Ratings | 100,000 | 75,980 |
| Rating scale | 1–5 (explicit) | 1–5 (explicit) |
| Population rule | users with ≥ 20 ratings (source property) | users with ≥ 20 ratings, items re-filtered to 5-core |
| Slice construction | — | deterministic; no random sampling |
| Catalogue enrichment | plot overview (1,400/1,682), poster (1,366/1,682) via TMDB | poster URLs from review metadata |

The MovieLens catalogue spans 1922–1998 (collection window 1997-09-19 to
1998-04-22), which explains the age of all titles served. The Amazon slice
exists to answer one question — do the MovieLens conclusions survive a domain
change — and is built by applying the same population rule to a domain whose
raw 5-core set (94.8k users × 25.6k items) exceeds the memory available on the
host. Only its slice definition (`raw/ratings.csv`) and result JSONs are
committed; rebuild with `make amazon-data`.

Content features are computed once and shared by every content-aware model:

| Modality | Encoder | Dim | Artefact |
|---|---|---|---|
| Text | Sentence-BERT | 384 | `text_embeddings.npy` |
| Image | ResNet-50 (pool5) | 2,048 | `image_embeddings.npy` |
| Genre | multi-hot | 18 | `genre_vectors.npy` |

Feature arrays are indexed by raw item id; rows of unused items are zero.

## 3. Evaluation protocol

- **Split**: leave-last-5-out — the five most recent interactions of each user
  are held out (`evaluation/runner.py`).
- **Candidates**: all items minus the user's training interactions
  (`evaluation/metrics.py`), so re-recommending a seen title cannot score.
- **Metrics**: HR@K, NDCG@K, Recall@K at K ∈ {5, 10, 20}.
- **Positive threshold**: neural models binarise `rating >= 4` as positive
  during training (`models/neumf.py:121`, `models/lightgcn.py:16`); the
  neighbourhood and factorisation models consume the explicit ratings.
- **Seeds**: deterministic models (UserCF, ItemCF, SVD) are seeded and show
  zero spread; neural models and both baselines are trained with seeds
  42–46 and reported as mean ± std. `set_seed` fixes Python, NumPy and
  torch RNGs including negative sampling and DataLoader shuffling.
- **Baselines**: BPR-MF (Rendle et al., UAI 2009) and VBPR (He & McAuley,
  AAAI 2016) are offline references only; they are not registered in
  `models/registry.py` and are not served.

## 4. Model configurations

The exact constructors used by the single- and multi-seed runners
(`evaluation/runner.py:125-129`):

| Model | Key configuration |
|---|---|
| UserCF | Pearson similarity, k = 50 |
| ItemCF | adjusted cosine, k = 50 |
| SVD | truncated SVD (`scipy.sparse.linalg.svds`), k = 64 |
| BPR-MF | dim 64, epochs 50, lr 0.001, batch 4096, patience 8 |
| NeuMF | dim 32, MLP (128, 64, 32), epochs 10, batch 512, 4 negatives, patience 3 |
| LightGCN | dim 64, 3 layers, epochs 50, batch 4096, BPR objective, patience 8 |
| VBPR | behaviour dim 64, visual dim 64, epochs 30, lr 0.005 |
| MultiModalNCF | dim 32, MLP (128, 64, 32), epochs 10, batch 512; content tower projects 384/2048/18 → 64 each, concatenated (192) → 64 |

Schedule caveat: MultiModalNCF runs 10 epochs against VBPR's 30 and BPR-MF's
50, so the baseline numbers come from longer schedules; wall-clock time is
still highest for MultiModalNCF because of the content tower.

## 5. Results

All values are mean ± std over seeds 42–46 from the artefacts in
`data/processed/` and `data/datasets/amazon-video-games/processed/`.

### 5.1 Model comparison (MovieLens 100K)

| Model | HR@10 | NDCG@10 | HR@20 | NDCG@20 | Train time (s) |
|---|---|---|---|---|---|
| UserCF | 0.0191 ± 0.0000 | 0.0035 ± 0.0000 | 0.0477 ± 0.0000 | 0.0060 ± 0.0000 | 0.1 |
| ItemCF | 0.0721 ± 0.0000 | 0.0142 ± 0.0000 | 0.1283 ± 0.0000 | 0.0199 ± 0.0000 | 0.3 |
| SVD | 0.2534 ± 0.0000 | 0.0532 ± 0.0000 | 0.3542 ± 0.0000 | 0.0664 ± 0.0000 | 0.3 |
| BPR-MF | 0.3476 ± 0.0116 | 0.0785 ± 0.0018 | 0.4831 ± 0.0116 | 0.1034 ± 0.0018 | 10.6 |
| NeuMF | 0.3516 ± 0.0036 | 0.0814 ± 0.0016 | 0.4978 ± 0.0073 | 0.1083 ± 0.0017 | 31.4 |
| LightGCN | 0.2850 ± 0.0041 | 0.0614 ± 0.0004 | 0.4250 ± 0.0047 | 0.0816 ± 0.0007 | 10.7 |
| VBPR | 0.3705 ± 0.0048 | 0.0855 ± 0.0009 | 0.5200 ± 0.0096 | 0.1130 ± 0.0008 | 41.9 |
| MultiModalNCF | 0.3650 ± 0.0077 | 0.0859 ± 0.0017 | 0.5073 ± 0.0148 | 0.1144 ± 0.0015 | 73.2 |

Observations: the ordering UserCF < ItemCF < SVD holds; the neural band
(0.285–0.371 HR@10) sits above SVD; BPR-MF and NeuMF are within one combined
std of each other; VBPR's HR@10 edge over MultiModalNCF (0.0055) is smaller
than one MultiModalNCF std, while MultiModalNCF leads both NDCG columns — the
pair is read as statistically tied.

### 5.2 Modality ablation (MultiModalNCF, MovieLens)

| Variant | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|
| Full | 0.3650 ± 0.0077 | 0.0859 ± 0.0017 | 0.5073 ± 0.0148 | 0.1144 ± 0.0015 |
| w/o Text | 0.3642 ± 0.0100 | 0.0855 ± 0.0024 | 0.5075 ± 0.0108 | 0.1144 ± 0.0025 |
| w/o Image | 0.3631 ± 0.0085 | 0.0842 ± 0.0026 | 0.5080 ± 0.0029 | 0.1132 ± 0.0025 |
| w/o Genre | 0.3673 ± 0.0080 | 0.0857 ± 0.0015 | 0.5107 ± 0.0139 | 0.1144 ± 0.0016 |
| Behaviour only | 0.3565 ± 0.0061 | 0.0844 ± 0.0024 | 0.4931 ± 0.0089 | 0.1123 ± 0.0033 |

Removing all content lowers every metric, but by roughly one seed std; the
single-modality deltas sit inside the noise (one even reverses sign), so no
per-modality ranking is claimed.

### 5.3 Cold-start (100 items removed from training)

| Model | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|
| MultiModalNCF (content) | 0.1333 ± 0.0290 | 0.0542 ± 0.0151 | 0.2754 ± 0.0233 | 0.0882 ± 0.0139 |
| NeuMF (content-blind) | 0.1140 ± 0.0147 | 0.0467 ± 0.0069 | 0.2351 ± 0.0129 | 0.0739 ± 0.0038 |
| Random | 0.0702 | 0.0288 | 0.2018 | 0.0610 |

The ordering content > blind > random holds on every metric; the HR@10 spread
of the content variant (0.0290) exceeds the gap to NeuMF (0.0193), so the
margin is reported as a direction rather than a point magnitude.

### 5.4 Cross-dataset (HR@10 / NDCG@10)

| Model | ML-100K HR@10 | Amazon HR@10 | ML-100K NDCG@10 | Amazon NDCG@10 |
|---|---|---|---|---|
| UserCF | 0.0191 ± 0.0000 | 0.0122 ± 0.0000 | 0.0035 ± 0.0000 | 0.0017 ± 0.0000 |
| ItemCF | 0.0721 ± 0.0000 | 0.0214 ± 0.0000 | 0.0142 ± 0.0000 | 0.0035 ± 0.0000 |
| SVD | 0.2534 ± 0.0000 | 0.0651 ± 0.0000 | 0.0532 ± 0.0000 | 0.0116 ± 0.0000 |
| BPR-MF | 0.3476 ± 0.0116 | 0.1055 ± 0.0053 | 0.0785 ± 0.0018 | 0.0198 ± 0.0012 |
| NeuMF | 0.3516 ± 0.0036 | 0.1053 ± 0.0043 | 0.0814 ± 0.0016 | 0.0190 ± 0.0009 |
| LightGCN | 0.2850 ± 0.0041 | 0.1187 ± 0.0074 | 0.0614 ± 0.0004 | 0.0222 ± 0.0011 |
| VBPR | 0.3705 ± 0.0048 | 0.1097 ± 0.0050 | 0.0855 ± 0.0009 | 0.0196 ± 0.0017 |
| MultiModalNCF | 0.3650 ± 0.0077 | 0.1164 ± 0.0085 | 0.0859 ± 0.0017 | 0.0213 ± 0.0018 |

Cold-start HR@10 on the second domain: MultiModalNCF (content) 0.2364 ±
0.0202 vs NeuMF 0.1204 ± 0.0120 vs Random 0.1092 — the content/blind ratio
grows from 1.17× (MovieLens) to 1.96× (Amazon). The Amazon candidate
catalogue is 2.8× larger, which mechanically depresses absolute HR@K;
cross-domain reading is therefore restricted to orderings and delta
directions.

What transfers: the ladder ordering, the BPR-MF ≈ NeuMF tie, and the
"single-modality deltas are inside seed noise" conclusion. What does not: on
Amazon the top HR@10 point estimate belongs to LightGCN (0.1187 vs 0.1164,
a difference well inside the combined spread), and Behaviour-only ≈ Full
(0.1163 vs 0.1164).

## 6. Reproducibility

```bash
make setup && make setup-train     # runtime + ML dependencies
make data && make features         # dataset + content features
make train && make eval-seeds      # single split, seeds 42–46
make ablation-seeds                # modality ablation, 5 seeds
make coldstart-seeds               # cold-start study, 5 seeds
make amazon-data                   # second dataset (isolated subtree)
make test                          # pytest suite
```

- Result artefacts: `eval_multiseed.json`, `ablation_multiseed.json`,
  `coldstart_multiseed.json` under `data/processed/`, plus the mirrored
  files under `data/datasets/amazon-video-games/processed/`. The single-seed
  `eval_results.json` is what the serving dashboard reads; it is never
  overwritten by multi-seed runs.
- Environment: Python 3.10, torch >= 2.13 (CPU wheels for deployments),
  dependency floors pinned to fixed releases for known advisories;
  `pip-audit -r requirements.txt -r requirements-train.txt` reports no known
  vulnerabilities.
- Quality gates: `ruff check .`; pytest with a coverage floor
  (`fail_under = 45` in `pyproject.toml`); CI runs lint/tests (with CPU
  torch so neural-model tests are not skipped), Playwright e2e/a11y journeys,
  and builds both Docker images.
- Test suite state at v1.0.0: 125 passed with torch installed, 110 passed
  plus 6 skipped in the torch-less configuration (coverage 50.9% of the
  configured sources).

## 7. Limitations

1. **Single split.** One leave-last-5-out partition by design; conclusions
   are read against seed spread, not split variance. No cross-validation.
2. **No significance testing.** Differences are compared against reported
   stds; formal tests are not run, and small deltas (e.g. VBPR vs
   MultiModalNCF HR@10) are explicitly left undecided.
3. **Epoch budgets are not equalised** across models (10 for both NCF models,
   30 for VBPR, 50 for BPR-MF/LightGCN), so wall-clock and accuracy
   comparisons carry that asymmetry.
4. **Training-time measurement conditions** (hardware, warm/cold cache) are
   not recorded alongside the reported seconds; values are 5-seed means from
   `eval_multiseed.json` but are not portable across machines.
5. **Second-domain slice** represents heavy users only (≥ 20 ratings) —
   chosen to match MovieLens 100K's population rule — and does not describe
   the full Amazon Video_Games population.
6. **Content ablation effect is small** at this scale: removing all content
   costs about one seed std on MovieLens and nothing measurable on Amazon.
7. **Free-tier deployments sleep** when idle; reported live behaviour may
   include cold-start latency on first request.
