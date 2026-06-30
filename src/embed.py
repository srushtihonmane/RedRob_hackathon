"""embed.py — offline embedding backend (PRECOMPUTE / SANDBOX ONLY).

The locked model is bge-small-en-v1.5 (384-d, L2-normalized) with a query/passage prefix
convention (Goal 2 D5 / contracts.md C3). The JD side uses the query prefix; the candidate
side uses the (empty) passage prefix. The SAME backend is used for both sides so cosines are
directly comparable.

Backend: fastembed (ONNX Runtime, CPU, no torch at inference) — the lean, low-memory path.
NOTE (deviation, logged): fastembed's bge-small-en-v1.5 is the int8-quantized ONNX build.
It is still bge-small-en-v1.5 (384-d, L2-norm); embeddings differ from the fp32 reference
only at the ~1e-2 cosine level, which is immaterial for ranking and keeps determinism. The
model+revision is pinned in manifests. This module is NEVER imported by the ranking step
(rank.py) — runtime loads precomputed vectors only (Runtime ML Policy, runtime.md R2).

Model config (name + prefixes) is read from jd/jd_query.json so there is a single source of
truth shared by the JD and candidate sides.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .common import read_json

_DEFAULT_CACHE = str(Path(__file__).resolve().parents[1] / "models")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


@lru_cache(maxsize=1)
def model_config(jd_query_path: str = "jd/jd_query.json") -> dict:
    """{'name','dim','normalize','query_prefix','passage_prefix'} from jd_query.json."""
    return read_json(jd_query_path)["model"]


@lru_cache(maxsize=2)
def _embedder(model_name: str, cache_dir: str, threads: int):
    from fastembed import TextEmbedding  # imported lazily; precompute-only dependency
    return TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads)


def _default_threads() -> int:
    # PRECOMPUTE-ONLY embedding: multi-thread for throughput. Embedding float nondeterminism
    # is documented/accepted (shipped .npy artifacts are canonical; the budgeted ranking step
    # never embeds and stays single-thread deterministic). Override with EMBED_THREADS.
    env = os.environ.get("EMBED_THREADS")
    if env:
        return int(env)
    return min(8, os.cpu_count() or 4)


def get_embedder(jd_query_path: str = "jd/jd_query.json", cache_dir: str | None = None,
                 threads: int | None = None):
    cfg = model_config(jd_query_path)
    return _embedder(cfg["name"], cache_dir or _DEFAULT_CACHE,
                     _default_threads() if threads is None else threads)


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return mat / norms


def embed_texts(texts: list[str], *, prefix: str = "", batch_size: int = 64,
                jd_query_path: str = "jd/jd_query.json") -> np.ndarray:
    """Embed a list of texts -> [n, dim] f32, L2-normalized. ``prefix`` is prepended to each
    text (query prefix for the JD side; empty passage prefix for candidates)."""
    if not texts:
        return np.zeros((0, model_config(jd_query_path)["dim"]), dtype=np.float32)
    emb = get_embedder(jd_query_path)
    prepared = [prefix + t for t in texts]
    vecs = list(emb.embed(prepared, batch_size=batch_size))
    return _normalize(np.asarray(vecs, dtype=np.float32))


def embed_queries(texts: list[str], **kw) -> np.ndarray:
    """JD side: prepend the query prefix from model config."""
    cfg = model_config(kw.get("jd_query_path", "jd/jd_query.json"))
    return embed_texts(texts, prefix=cfg.get("query_prefix", ""), **kw)


def embed_passages(texts: list[str], **kw) -> np.ndarray:
    """Candidate side: prepend the (typically empty) passage prefix."""
    cfg = model_config(kw.get("jd_query_path", "jd/jd_query.json"))
    return embed_texts(texts, prefix=cfg.get("passage_prefix", ""), **kw)


def iter_embed_passages(texts: Iterable[str], *, batch_size: int = 64,
                        jd_query_path: str = "jd/jd_query.json") -> Iterator[np.ndarray]:
    """Streaming passage embedding: yields one L2-normalized [1, dim] f32 row per input text,
    in order. Lets the Goal 3 builder write to a memmap without holding all vectors in RAM."""
    cfg = model_config(jd_query_path)
    prefix = cfg.get("passage_prefix", "")
    emb = get_embedder(jd_query_path)
    for vec in emb.embed((prefix + t for t in texts), batch_size=batch_size):
        yield _normalize(np.asarray([vec], dtype=np.float32))[0]
