# ADR 0005 — Cross-dataset validation: Amazon Video_Games as a second dataset

- **Status**: Accepted
- **Context**: ADR 0004 chose MovieLens 100K as the single evaluation dataset.
  That leaves the project's headline weakness: every number in the README could
  be a MovieLens artefact — an accident of one domain, one split, one catalogue.
  The honest fix is a second dataset from a *different domain*, run through the
  exact same protocol (same leave-last-5-out split, same candidate filtering,
  same eight models, same 5-seed mean ± std reporting).

  Three constraints shaped the decision:
  1. **No cross-contamination.** The served MovieLens artefacts
     (`data/processed/*`) are the product. A second dataset must not be able to
     overwrite them, ever.
  2. **Small host.** The evaluation host has 2 vCPU / 3 GB RAM. The full
     Video_Games 5-core split (94.8k users × 25.6k items) makes the dense
     neighbourhood models materialise 19–72 GB similarity matrices — impossible.
  3. **Protocol comparability.** MovieLens 100K only admits users with ≥ 20
     ratings. A second dataset whose users average 6 ratings cannot be compared
     under leave-last-5-out: most users would have an *empty* training set.

- **Decision**: Add **Amazon Reviews-2023 Video_Games** (the McAuley-Lab 5-core
  benchmark) as a second dataset, selected via `CINEREC_DATASET=amazon-video-games`,
  and restrict it with the *same population rule MovieLens 100K itself uses*:

  - keep only users with **≥ 20 ratings** (`AMAZON_MIN_USER_INTER`, default 20);
  - re-establish the item **5-core** on the remainder (every kept item has ≥ 5
    ratings in the slice);
  - no random sampling anywhere — the subset is a deterministic population
    filter of the benchmark.

  That yields **2,288 users × 4,753 items × 75,980 ratings** — comparable in
  volume to MovieLens 100K (943 × 1,682 × 100,000), with every user retaining
  ≥ 15 training interactions after the split. Items map to the existing schema:
  Amazon's title + features + description become the overview (Sentence-BERT
  text), the first image becomes the poster (ResNet-50), and the two category
  levels under the "Video Games" root become an 18-tag category vocabulary —
  the domain analogue of MovieLens genres. `data/amazon_dataset.py` builds the
  slice; `config.py` resolves every offline pipeline path per dataset, so the
  MovieLens tree is unreachable from a second-dataset run by construction.

- **Alternatives considered**:
  - **Full 5-core Video_Games** — the faithful benchmark, but dense UserCF/
    ItemCF/SVD alone need more RAM than the host has; the neural models would
    also train an order of magnitude longer for no additional methodological
    claim.
  - **Random 5k-user sample** — deterministic when seeded, but Amazon's light
    users (median ~6 ratings) leave 61 % of interactions in the test set and
    empty training sets; the protocol comparison breaks.
  - **MovieLens-1M fallback** — cheap (reuses the TMDB enrichment) but stays
    inside the MovieLens family: it would only demonstrate *scale* robustness,
    not cross-*domain* transfer, which is the claim under test.

- **Consequences**:
  - The README gains a cross-dataset section: the main table, the ablation and
    the cold-start study are each re-run on the slice with the identical
    protocol and seed list, and read *per domain* — agreement strengthens the
    MovieLens conclusions; disagreement is reported, not hidden.
  - The "single dataset" limitation is retired; the remaining caveats are now
    the honest ones: one *second* domain (games, not movies), a heavy-user
    population (selection matches ML-100K but is not the general Amazon
    population), and a candidate catalogue ~2.8× larger than ML-100K's, which
    mechanically depresses absolute HR@K.
  - Shipping policy (see [.gitignore](../../.gitignore)): the slice definition
    (`ratings.csv`) and the measured result JSONs are committed; the 46 MB +
    418 MB upstream downloads and the 25 MB feature arrays are not — they are
    rebuilt with the documented commands (`CINEREC_DATASET=… make data/features/…`).
  - Upstream drift: Amazon-Reviews-2023 is a living dataset. The committed
    `ratings.csv` pins the exact evaluated slice even if upstream is revised;
    rebuilding from a newer upstream produces a (documented, seeded) different
    slice and requires re-running the evaluation before republishing numbers.
