"""Learned representations of products and their text, turned into session features.

Two kinds of embedding, both fitted **only on events before the validation period** (no
look-ahead into the weeks used to tune or test models):

* **Behavioural product embeddings ("item2vec")**: products that are viewed close
  together in the same sessions get similar vectors. Built as the truncated SVD of the
  positive pointwise mutual information (PPMI) matrix of product co-occurrences within a
  small window, which is equivalent in spirit to skip-gram word2vec with negative sampling
  (Levy & Goldberg, 2014) but deterministic and fast.
* **Text embeddings from a language model**: the category path and brand of each product
  (e.g. "electronics smartphone apple") embedded with ``nomic-embed-text`` through Ollama,
  then reduced with PCA. They let two products the model has rarely seen share
  information through their names.

Session features (first K events only): the mean vector of the visited products, the share
of products with a known vector, and *coherence*, the mean cosine similarity between
consecutive products (does the visitor stay on one kind of product or wander?).
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD


def cooccurrence(events: pl.LazyFrame, min_count: int = 20, window: int = 2) -> tuple[list[int], sparse.csr_matrix]:
    """Symmetric product co-occurrence counts within ``window`` steps in the same session."""
    ev = events.select("user_session", "event_time", "product_id")
    counts = ev.group_by("product_id").agg(n=pl.len()).filter(pl.col("n") >= min_count).collect()
    vocab = counts["product_id"].sort().to_list()
    index = pl.DataFrame({"product_id": vocab, "idx": np.arange(len(vocab), dtype=np.int32)})
    seq = (ev.join(index.lazy(), on="product_id", how="inner").sort("user_session", "event_time")
           .select("user_session", "idx"))
    parts = []
    for lag in range(1, window + 1):
        parts.append(seq.with_columns(nxt=pl.col("idx").shift(-lag).over("user_session"))
                     .filter(pl.col("nxt").is_not_null() & (pl.col("nxt") != pl.col("idx")))
                     .group_by("idx", "nxt").agg(c=pl.len()))
    pairs = pl.concat(parts).group_by("idx", "nxt").agg(pl.col("c").sum()).collect()
    a, b, c = (pairs[x].to_numpy() for x in ("idx", "nxt", "c"))
    n = len(vocab)
    m = sparse.coo_matrix((np.concatenate([c, c]).astype(np.float64), (np.concatenate([a, b]), np.concatenate([b, a]))),
                          shape=(n, n)).tocsr()
    m.sum_duplicates()
    return vocab, m


def ppmi_svd(m: sparse.csr_matrix, dim: int = 16, seed: int = 0, shift: float = 1.0) -> np.ndarray:
    """L2-normalised SVD embedding of the (shifted) PPMI matrix."""
    total = m.sum()
    row = np.asarray(m.sum(axis=1)).ravel()
    col = row ** 0.75                              # context smoothing as in word2vec
    col = col / col.sum()
    coo = m.tocoo()
    pmi = np.log(coo.data * 1.0 / total) - np.log(row[coo.row] / total) - np.log(col[coo.col]) - np.log(shift)
    keep = pmi > 0
    ppmi = sparse.csr_matrix((pmi[keep], (coo.row[keep], coo.col[keep])), shape=m.shape)
    dim = max(1, min(dim, m.shape[0] - 1))         # small vocabularies (tests) cannot support many dimensions
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    u = svd.fit_transform(ppmi)                    # U * S
    v = u / np.sqrt(np.clip(svd.singular_values_, 1e-12, None))   # U * sqrt(S): symmetric weighting
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def item2vec(events: pl.LazyFrame, before: datetime, dim: int = 16, min_count: int = 20, window: int = 2,
             seed: int = 0) -> pl.DataFrame:
    """``product_id`` + ``i2v_0..`` vectors from events strictly before ``before``."""
    ev = events.filter(pl.col("event_time") < before)
    vocab, m = cooccurrence(ev, min_count=min_count, window=window)
    vec = ppmi_svd(m, dim=dim, seed=seed)
    return pl.DataFrame({"product_id": vocab}).with_columns(
        [pl.Series(f"i2v_{j}", vec[:, j].astype(np.float32)) for j in range(dim)])


def product_texts(events: pl.LazyFrame, before: datetime) -> pl.DataFrame:
    """One text per (category_code, brand) seen before ``before``: "electronics smartphone · apple"."""
    return (events.filter(pl.col("event_time") < before)
            .select(pl.col("category_code").fill_null("unknown"), pl.col("brand").fill_null("unknown"))
            .unique().collect()
            .with_columns(text=pl.col("category_code").str.replace_all(r"[._]", " ") + pl.lit(" · ") + pl.col("brand")))


def text_vectors(texts: pl.DataFrame, embedder, dim: int = 8, seed: int = 0) -> pl.DataFrame:
    """PCA-reduced language-model embeddings of the product texts."""
    full = embedder.embed(texts["text"].to_list(), kind="document")
    red = PCA(n_components=min(dim, full.shape[1], len(texts)), random_state=seed).fit_transform(full)
    return texts.select("category_code", "brand").with_columns(
        [pl.Series(f"txt_{j}", red[:, j].astype(np.float32)) for j in range(red.shape[1])])


def session_features(prefix: pl.DataFrame, vectors: pl.DataFrame, on: Sequence[str], prefix_name: str) -> pl.DataFrame:
    """Mean vector, coverage and coherence of each session's first K products.

    ``prefix``: one row per prefix event (``user_session``, ``pos`` and the ``on`` columns).
    ``vectors``: the ``on`` columns + vector columns starting with ``prefix_name``.
    """
    cols = [c for c in vectors.columns if c.startswith(prefix_name)]
    p = prefix.select("user_session", "pos", *on).join(vectors, on=list(on), how="left").sort("user_session", "pos")
    mean = p.group_by("user_session").agg([pl.col(c).mean() for c in cols],
                                          **{f"{prefix_name}known": pl.col(cols[0]).is_not_null().mean()})
    x = p.select(cols).fill_null(0.0).to_numpy()
    norm = np.linalg.norm(x, axis=1)
    nx = x / np.clip(norm, 1e-12, None)[:, None]
    same = (p["user_session"].to_numpy()[1:] == p["user_session"].to_numpy()[:-1])
    both = (norm[1:] > 0) & (norm[:-1] > 0) & same
    cos = np.where(both, (nx[1:] * nx[:-1]).sum(axis=1), np.nan)
    coh = (pl.DataFrame({"user_session": p["user_session"][1:], "cos": cos})
           .with_columns(pl.col("cos").fill_nan(None))                # NaN = pair across two sessions: skip it
           .group_by("user_session").agg(pl.col("cos").mean().alias(f"{prefix_name}coherence")))
    return mean.join(coh, on="user_session", how="left")
