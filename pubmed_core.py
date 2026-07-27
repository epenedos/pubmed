#!/usr/bin/env python3
"""
Phase 2 core — retrieval + grounded answering.

Shared by the CLI (pubmed_ask.py) and the API server (pubmed_api.py).
Heavy objects (embedding model, clients) are loaded once and reused.
"""
import os
import re
import sqlite3
from functools import lru_cache

import requests

DB_PATH = os.environ.get("PUBMED_DB", "pubmed.db")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "pubmed_ovarian")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "NeuML/pubmedbert-base-embeddings")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "nemotron-3-nano")
TOP_K = int(os.environ.get("TOP_K", "8"))

SYSTEM_PROMPT = """You are a biomedical literature assistant. You answer \
questions using ONLY the PubMed abstracts provided in the context below.

Rules:
- Every factual claim must cite its source as [PMID: 12345678]. Multiple \
sources for one claim: [PMID: 111, PMID: 222].
- If the abstracts do not answer the question, say so plainly. Do not fill \
gaps from your own knowledge, and never invent a PMID.
- Distinguish study types when it matters: a randomised trial, a retrospective \
cohort, and a case report carry different weight. Note sample sizes when given.
- Where the abstracts disagree, present the disagreement rather than picking a side.
- Be concise and factual. No hedging boilerplate.
- You summarise published research. You do not give medical advice or \
treatment recommendations for any individual."""


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    device = os.environ.get("EMBED_DEVICE", "cuda")
    return SentenceTransformer(EMBED_MODEL, device=device)


@lru_cache(maxsize=1)
def _qdrant():
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL)


def semantic_search(query, k):
    vec = _embedder().encode(query, normalize_embeddings=True).tolist()
    hits = _qdrant().query_points(COLLECTION, query=vec, limit=k, with_payload=True).points
    return [(h.payload["pmid"], h.payload) for h in hits]


def keyword_search(query, k, db=DB_PATH):
    terms = re.findall(r"[A-Za-z0-9-]{3,}", query)
    if not terms:
        return []
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT a.pmid, a.title, a.abstract, a.journal, a.year, a.doi "
            "FROM articles_fts f JOIN articles a ON a.rowid = f.rowid "
            "WHERE articles_fts MATCH ? ORDER BY rank LIMIT ?",
            (" OR ".join(terms), k),
        ).fetchall()
    finally:
        conn.close()
    return [(r["pmid"], dict(r)) for r in rows]


def fuse(*ranked_lists, k=60):
    """Reciprocal rank fusion of independent ranked lists."""
    scores, payloads = {}, {}
    for lst in ranked_lists:
        for rank, (pmid, payload) in enumerate(lst):
            scores[pmid] = scores.get(pmid, 0) + 1 / (k + rank + 1)
            payloads.setdefault(pmid, payload)
    return [
        (pmid, scores[pmid], payloads[pmid])
        for pmid in sorted(scores, key=scores.get, reverse=True)
    ]


def retrieve(question, top_k=TOP_K, db=DB_PATH):
    return fuse(
        semantic_search(question, top_k * 3),
        keyword_search(question, top_k * 3, db),
    )[:top_k]


def build_context(results, max_chars=1400):
    """Format retrieved abstracts as the model's evidence block."""
    blocks = []
    for _, _, p in results:
        abstract = p["abstract"]
        if len(abstract) > max_chars:
            abstract = abstract[:max_chars].rsplit(" ", 1)[0] + " [...]"
        blocks.append(
            f"[PMID: {p['pmid']}] {p['title']}\n"
            f"{p.get('journal') or 'n/a'}, {p.get('year') or 'n/a'}\n"
            f"{abstract}"
        )
    return "\n\n---\n\n".join(blocks)


def build_messages(question, results, history=None):
    context = build_context(results)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"Context abstracts:\n\n{context}\n\n---\n\nQuestion: {question}",
        }
    )
    return messages


def sources_markdown(results):
    lines = ["\n\n---\n**Sources**\n"]
    for _, _, p in results:
        year = p.get("year") or "n.d."
        journal = p.get("journal") or ""
        lines.append(
            f"- [{p['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/) — "
            f"{p['title']} *({journal} {year})*"
        )
    return "\n".join(lines)


def chat_ollama(messages, stream=False, model=OLLAMA_MODEL):
    """Call Ollama. Returns a string, or a generator of chunks when streaming."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": 0.2, "num_ctx": 32768},
    }
    if not stream:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def gen():
        import json

        with requests.post(
            f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=600
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break

    return gen()


def answer(question, top_k=TOP_K, db=DB_PATH, model=OLLAMA_MODEL):
    """Non-streaming convenience wrapper: returns (text, results)."""
    results = retrieve(question, top_k, db)
    if not results:
        return "No matching abstracts in the corpus for that question.", []
    text = chat_ollama(build_messages(question, results), model=model)
    return text + sources_markdown(results), results
