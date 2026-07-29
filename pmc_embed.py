#!/usr/bin/env python3
"""
Phase 4b — Embed full-text chunks into qdrant.

Mirrors pubmed_embed.py but reads the `chunks` table and writes to a separate
collection, so abstracts and full text can be searched independently or together.

Usage:
    python pmc_embed.py
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
COLLECTION = os.environ.get("QDRANT_CHUNK_COLLECTION", "pubmed_ovarian_chunks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    model = SentenceTransformer(EMBED_MODEL, device=os.environ.get("EMBED_DEVICE", "cuda"))
    dim = len(model.encode("probe"))
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
        "SELECT c.id, c.pmid, c.pmcid, c.section, c.text, "
        "       a.title, a.journal, a.year, a.doi "
        "FROM chunks c LEFT JOIN articles a ON a.pmid = c.pmid "
        "WHERE c.embedded = 0"
    ).fetchall()
    print(f"chunks to embed: {len(todo):,}")

    for i in range(0, len(todo), args.batch):
        batch = todo[i : i + args.batch]
        # Prefix the section label: it gives the encoder useful context about
        # whether this text is methods, results, or discussion.
        texts = [f"{r['section']}: {r['text']}" for r in batch]
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        qc.upsert(
            COLLECTION,
            points=[
                PointStruct(
                    id=int(r["id"]),
                    vector=v.tolist(),
                    payload={
                        "pmid": r["pmid"],
                        "pmcid": r["pmcid"],
                        "section": r["section"],
                        "text": r["text"],
                        "title": r["title"],
                        "journal": r["journal"],
                        "year": r["year"],
                        "doi": r["doi"],
                    },
                )
                for r, v in zip(batch, vecs)
            ],
        )
        conn.executemany(
            "UPDATE chunks SET embedded = 1 WHERE id = ?", [(r["id"],) for r in batch]
        )
        conn.commit()
        print(f"  {min(i + args.batch, len(todo)):>7,} / {len(todo):,}", flush=True)

    print(f"\nDone. '{COLLECTION}': {qc.count(COLLECTION).count:,} vectors")
    conn.close()


if __name__ == "__main__":
    main()
