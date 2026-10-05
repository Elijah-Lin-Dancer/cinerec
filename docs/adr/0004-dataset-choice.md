# ADR 0004 — Dataset choice: MovieLens 100K, and why the catalogue is old

- **Status**: Accepted
- **Context**: Every model in the ladder needs a ratings dataset to train and
  evaluate on, and the product needs *item metadata* for two jobs: rendering the
  library (posters, plot summaries, genres, year) and feeding the Multi-Modal NCF
  content tower (text, image, genre features). Whatever we pick is visible in the
  UI, so the choice has to be defensible — including its flaws.

  The most obvious flaw is age: browse the library and the newest film is from
  **1998**. That is not a bug in our pipeline. MovieLens 100K was collected
  through the MovieLens website over the seven-month window **19 Sep 1997 –
  22 Apr 1998**; its catalogue spans **1,922–1,998**, so a title from 1998 is the
  end of the range, not a stale mirror.

- **Decision**: Ship **MovieLens 100K** as the ratings dataset, enriched with
  **TMDB** metadata for posters and plot summaries.

  Concretely, the shipped data is:
  - **100,000 ratings (1–5)** from **943 users** over **1,682 movies**, every user
    having rated at least 20 movies. The density matters: collaborative filtering
    (and HR@K / NDCG@K evaluation) is only meaningful when users share items, and
    MovieLens 100K filters out users below 20 ratings precisely to guarantee that.
  - Every item carries **genres and a release year**, which gives the content
    tower real genre signal and gives the UI something to filter and display.
  - **TMDB enrichment** (`data/enrich_tmdb.py`) attaches a plot summary and a
    poster where a match exists: **1,400 / 1,682** rows have an overview and
    **1,366 / 1,682** have a poster. Titles with no match degrade gracefully
    (initial-letter placeholder in the UI, "no synopsis" fallback in the detail
    modal) rather than breaking the grid.

  Reasons this is the right trade-off for *this* project:
  1. **It is the standard, citable benchmark.** Harper & Konstan (2015) is the
     canonical citation, so our HR@K / NDCG@K numbers sit next to a large body of
     literature instead of being uncomparable.
  2. **It is small enough to be honest about cost.** All six models train in
     seconds-to-minutes on CPU (0.1s UserCF → 93.5s Multi-Modal NCF, see the
     README table), which lets us report a real accuracy/cost trade-off and commit
     the trained artefacts plus content features so a fresh clone serves
     recommendations with **zero training** (ADR 0002).
  3. **It has explicit feedback and item side-info.** Explicit 1–5 ratings make
     the ranking metrics well-defined, and the genre/year columns plus TMDB text
     and images are exactly what the Multi-Modal NCF ablation needs.

- **Rejected**:
  - **MovieLens 1M / 10M / 25M** — more ratings and newer coverage, but training
    six models (especially the graph and neural ones) and committing their
    artefacts/features would balloon the repo and the reproducibility story. The
    *progression* from UserCF to Multi-Modal NCF is already fully visible at 100K.
  - **The Movies Dataset (Kaggle) / TMDB-derived sets** — a far more modern
    catalogue, but heavier, messier, and its rating signal is sparse or implicit,
    which makes clean HR@K / NDCG@K comparison worse, not better.
  - **Netflix Prize** — enormous, distribution-restricted, and redundant with
    MovieLens for teaching purposes.

- **Consequences**:
  - **The catalogue reads as "old" (1922–1998).** This is intrinsic to the source
    data and is the honest price of using a classic, well-understood benchmark.
    The UI does not hide it; the year filter and the detail modal simply report
    what the data says. If a modern catalogue is ever needed, it should come in as
    a **second** dataset behind the enrichment layer, not by silently swapping
    this one — the evaluation numbers are tied to this split.
  - **Licensing is non-commercial and non-redistributable.** MovieLens data is
    provided by the GroupLens Research Project (University of Minnesota) for
    research use: it must not be redistributed without permission and must not be
    used for commercial or revenue-bearing purposes. This repository is a
    non-commercial portfolio project. Publications using it must cite Harper &
    Konstan (2015) and must not imply endorsement by the University of Minnesota.
  - **Poster and overview metadata come from TMDB** (`data/enrich_tmdb.py`), so
    TMDB's terms apply to that metadata in addition to the MovieLens terms on the
    ratings. The two obligations are independent and both are stated in the
    README FAQ.