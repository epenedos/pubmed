#!/usr/bin/env python3
"""
Phase 1b — Embed abstracts and store vectors in qdrant.

Usage:
    python pubmed_embed.py                # embeds everything not yet embedded
    python pubmed_embed.py --batch 64

Env:
    QDRANT_URL     default http://localhost:6333
    EMBED_MODEL    default NeuML/pubmedbert-base-embeddings  (biomedical, 768d)
                   alternative: BAAI/bge-m3 (1024d, multilingual, longer context)
"""
import argparse
import os
import sqlite3

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

DB_PATH = os.environ.get("PUBMED_DB", "pubmed.db")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "NeuML/pubmedbert-base-embeddings")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "pubmed_ovarian")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    model = SentenceTransformer(EMBED_MODEL, device="cuda")
    dim = model.get_sentence_embedding_dimension()
    print(f"model: {EMBED_MODEL} ({dim}d) on {model.device}")

    qc = QdrantClient(url=QDRANT_URL)
    if not qc.collection_exists(COLLECTION):
        qc.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"created collection '{COLLECTION}'")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    todo = conn.execute(
        "SELECT rowid, pmid, title, abstract, journal, year, doi "
        "FROM articles WHERE embedded = 0"
    ).fetchall()
    print(f"to embed: {len(todo):,}")

    for i in range(0, len(todo), args.batch):
        chunk = todo[i : i + args.batch]
        # Title + abstract as one unit: abstracts are short enough to embed whole.
        texts = [f"{r['title']}\n\n{r['abstract']}" for r in chunk]
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        qc.upsert(
            COLLECTION,
            points=[
                PointStruct(
                    id=int(r["rowid"]),
                    vector=v.tolist(),
                    payload={
                        "pmid": r["pmid"],
                        "title": r["title"],
                        "abstract": r["abstract"],
                        "journal": r["journal"],
                        "year": r["year"],
                        "doi": r["doi"],
                    },
                )
                for r, v in zip(chunk, vecs)
            ],
        )
        conn.executemany(
            "UPDATE articles SET embedded = 1 WHERE rowid = ?",
            [(r["rowid"],) for r in chunk],
        )
        conn.commit()
        print(f"  {min(i + args.batch, len(todo)):>6,} / {len(todo):,}", flush=True)

    print(f"\nDone. collection '{COLLECTION}': {qc.count(COLLECTION).count:,} vectors")
    conn.close()


if __name__ == "__main__":
    main()
