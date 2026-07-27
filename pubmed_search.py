#!/usr/bin/env python3
"""
Phase 1c — Hybrid search: semantic (qdrant) + keyword (FTS5), fused.

Usage:
    python pubmed_search.py "PARP inhibitor maintenance after platinum resistance"
    python pubmed_search.py "HIPEC peritoneal carcinomatosis outcomes" --k 10
"""
import argparse
import os
import re
import sqlite3

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

DB_PATH = os.environ.get("PUBMED_DB", "pubmed.db")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "NeuML/pubmedbert-base-embeddings")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "pubmed_ovarian")


def semantic(query, k):
    model = SentenceTransformer(EMBED_MODEL, device="cuda")
    vec = model.encode(query, normalize_embeddings=True).tolist()
    qc = QdrantClient(url=QDRANT_URL)
    hits = qc.query_points(COLLECTION, query=vec, limit=k, with_payload=True).points
    return [(h.payload["pmid"], h.payload) for h in hits]


def keyword(query, k, db):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    # FTS5 needs a sanitised query; OR the terms so partial matches still rank.
    terms = [t for t in re.findall(r"[A-Za-z0-9-]{3,}", query)]
    if not terms:
        return []
    fts_query = " OR ".join(terms)
    rows = conn.execute(
        "SELECT a.pmid, a.title, a.abstract, a.journal, a.year, a.doi "
        "FROM articles_fts f JOIN articles a ON a.rowid = f.rowid "
        "WHERE articles_fts MATCH ? ORDER BY rank LIMIT ?",
        (fts_query, k),
    ).fetchall()
    conn.close()
    return [(r["pmid"], dict(r)) for r in rows]


def fuse(*ranked_lists, k=60):
    """Reciprocal rank fusion — combines rankings without tuning score scales."""
    scores, payloads = {}, {}
    for lst in ranked_lists:
        for rank, (pmid, payload) in enumerate(lst):
            scores[pmid] = scores.get(pmid, 0) + 1 / (k + rank + 1)
            payloads.setdefault(pmid, payload)
    order = sorted(scores, key=scores.get, reverse=True)
    return [(pmid, scores[pmid], payloads[pmid]) for pmid in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    results = fuse(
        semantic(args.query, args.k * 3),
        keyword(args.query, args.k * 3, args.db),
    )[: args.k]

    for i, (pmid, score, p) in enumerate(results, 1):
        print(f"\n[{i}] {p['title']}")
        print(f"    {p.get('journal')} {p.get('year')}  |  PMID {pmid}  |  score {score:.4f}")
        print(f"    https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        print(f"    {p['abstract'][:220].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
