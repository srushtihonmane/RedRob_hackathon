"""bm25.py — Goal 3 D3: narrative-only BM25 index with controlled phrase-joining.

Corpus = headline + summary + role titles (x2 light weighting) + role descriptions.
**skills[] array EXCLUDED** (uniform ~12k x/skill noise — the keyword-stuffer trap).
Phrase-joining preserves multi-word concepts ("information retrieval", "vector database",
"learning to rank", "sentence transformers") as single tokens on both corpus and query side.

We precompute the full BM25 doc-term WEIGHT matrix W[d,t] offline, so the runtime score for a
fixed JD query set Q is a single sparse mat-vec: scores = W @ q_indicator (Goal 8 D3).
Artifacts: bm25_index/{matrix.npz (W, CSR), idf.npy, doclen.npy, vocab.json, params.json}.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .common import read_json

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
K1 = 1.5
B = 0.75


def build_phrase_list(jd_query_path: str = "jd/jd_query.json",
                      concept_registry_path: str = "reference/concept_registry.json") -> list[str]:
    phrases: set[str] = set()
    jd = read_json(jd_query_path)
    for grp in jd.get("lexical_concept_groups", {}).values():
        for term in grp.get("terms", []) + grp.get("synonyms", []):
            if " " in term:
                phrases.add(term.lower())
    cr = read_json(concept_registry_path)
    for cdef in cr["concepts"].values():
        for kw in cdef["keywords"]:
            if " " in kw:
                phrases.add(kw.lower())
    # longest first so "vector database" joins before "vector"
    return sorted(phrases, key=lambda s: -len(s))


def tokenize(text: str, phrases: list[str]) -> list[str]:
    t = (text or "").lower()
    for ph in phrases:
        if ph in t:
            t = t.replace(ph, ph.replace(" ", "_"))
    return _TOKEN_RE.findall(t)


def bm25_tokens(rec: dict, phrases: list[str]) -> list[str]:
    """Narrative tokens only; titles counted twice (light field weighting). No skills[]."""
    p = rec["profile"]
    toks = tokenize(p.get("headline", ""), phrases) + tokenize(p.get("summary", ""), phrases)
    for role in rec["career_history"]:
        title_toks = tokenize(role.get("title", ""), phrases)
        toks += title_toks + title_toks          # x2 multiplicity
        toks += tokenize(role.get("description", ""), phrases)
    return toks


def build_index(records, out_dir: str, jd_query_path: str = "jd/jd_query.json",
                concept_registry_path: str = "reference/concept_registry.json",
                k1: float = K1, b: float = B) -> dict:
    phrases = build_phrase_list(jd_query_path, concept_registry_path)
    vocab: dict[str, int] = {}
    rows, cols, tf_vals = [], [], []
    doclen: list[int] = []
    for d, rec in enumerate(records):
        toks = bm25_tokens(rec, phrases)
        doclen.append(len(toks))
        counts: dict[int, int] = {}
        for tok in toks:
            j = vocab.setdefault(tok, len(vocab))
            counts[j] = counts.get(j, 0) + 1
        for j, c in counts.items():
            rows.append(d); cols.append(j); tf_vals.append(c)
    n_docs = len(doclen)
    n_terms = len(vocab)
    doclen_arr = np.asarray(doclen, dtype=np.float32)
    avgdl = float(doclen_arr.mean()) if n_docs else 0.0

    tf = sp.csr_matrix((np.asarray(tf_vals, dtype=np.float32), (rows, cols)),
                       shape=(n_docs, max(n_terms, 1)))
    # idf (BM25 smoothed): ln(1 + (N - df + 0.5)/(df + 0.5))
    df = np.asarray((tf > 0).sum(axis=0)).ravel().astype(np.float32)
    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

    # Precompute BM25 weight matrix W[d,t] = idf_t * tf*(k1+1) / (tf + k1*(1-b+b*dl_d/avgdl)).
    # Vectorized over CSR.data (no per-nnz Python loop) — memory-frugal + fast for ~8M nnz.
    denom_doc = (k1 * (1 - b + b * (doclen_arr / (avgdl or 1.0)))).astype(np.float32)
    rows_of_nnz = np.repeat(np.arange(n_docs, dtype=np.int64), np.diff(tf.indptr))
    t = tf.data
    w_data = (idf[tf.indices] * (t * (k1 + 1.0)) / (t + denom_doc[rows_of_nnz])).astype(np.float32)
    Wm = sp.csr_matrix((w_data, tf.indices.copy(), tf.indptr.copy()), shape=tf.shape)

    out = Path(out_dir) / "bm25_index"
    out.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out / "matrix.npz", Wm)
    np.save(out / "idf.npy", idf)
    np.save(out / "doclen.npy", doclen_arr)
    with open(out / "vocab.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(vocab, f, ensure_ascii=False, sort_keys=True)
    with open(out / "params.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"k1": k1, "b": b, "avgdl": avgdl, "n_docs": n_docs, "n_terms": n_terms},
                  f, indent=2)
    return {"n_docs": n_docs, "n_terms": n_terms, "avgdl": avgdl}


def query_terms_from_jd(jd_query_path: str = "jd/jd_query.json",
                        concept_registry_path: str = "reference/concept_registry.json") -> list[str]:
    """The JD lexical query: phrase-joined tokens from all concept-group terms/synonyms."""
    phrases = build_phrase_list(jd_query_path, concept_registry_path)
    jd = read_json(jd_query_path)
    terms: set[str] = set()
    for grp in jd.get("lexical_concept_groups", {}).values():
        for term in grp.get("terms", []) + grp.get("synonyms", []):
            terms.update(tokenize(term, phrases))
    return sorted(terms)


def score_query(out_dir: str, terms: list[str]) -> np.ndarray:
    """Runtime-style BM25 scoring: load W + vocab, score each doc against the term set."""
    out = Path(out_dir) / "bm25_index"
    W = sp.load_npz(out / "matrix.npz")
    vocab = json.load(open(out / "vocab.json", encoding="utf-8"))
    q = np.zeros(W.shape[1], dtype=np.float32)
    for t in terms:
        if t in vocab:
            q[vocab[t]] = 1.0
    return np.asarray(W @ q).ravel()
