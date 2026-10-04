# ADR 0003 — Algorithm ladder: five models, honestly labelled

- **Status**: Accepted
- **Context**: The portfolio value comes from showing the *progression* from
  classic collaborative filtering to a content-aware neural model, and from
  being able to explain the trade-offs — not from chasing a state-of-the-art
  number on a small dataset.

- **Decision**: Keep five models, and name them for what they actually are:
  1. **UserCF** — user-based CF, vectorised Pearson correlation.
  2. **ItemCF** — item-based CF, vectorised adjusted cosine.
  3. **SVD** — matrix factorisation via **truncated SVD**
     (`scipy.sparse.linalg.svds`). *Not* ALS.
  4. **NeuMF** — GMF + MLP dual path (PyTorch), BCE loss with negative sampling.
  5. **Multi-Modal NCF** — fuses Sentence-BERT text (384-d), ResNet-50 image
     (2048-d) and genre (18-d) features into the MLP path via a Content Tower.

- **Rejected**: renaming SVD to "SVD/ALS" (the `lambda_reg` parameter is unused
  and the implementation is `svds`, not ALS); adding a LightGCN baseline
  (optional, and not needed to demonstrate the intended concepts).

- **Consequences**: Every claim in the README maps to a named implementation.
  The evaluation protocol is documented (leave-last-5-out, train items excluded)
  so the reported HR/NDCG/Recall are reproducible and Recall is not an alias
  of HR.