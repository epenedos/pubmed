#!/usr/bin/env python3
"""
RAG core — retrieval over abstracts + PMC full-text chunks, and grounded answering.

Replaces the earlier pubmed_core.py. Includes the fixes made along the way:
Ollama errors surfaced with their body, FTS5 terms quoted, thinking disabled,
and the embedding model preloaded behind a lock.
"""
import os
import re
import sqlite3
import threading
from functools import lru_cache

import requests

DB_PATH = os.environ.get("PUBMED_DB", "pubmed.db")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "pubmed_ovarian")
CHUNK_COLLECTION = os.environ.get("QDRANT_CHUNK_COLLECTION", "pubmed_ovarian_chunks")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "NeuML/pubmedbert-base-embeddings")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "nemotron-3-nano")
TOP_K = int(os.environ.get("TOP_K", "8"))
MAX_PER_PAPER = int(os.environ.get("MAX_PER_PAPER", "2"))

SYSTEM_PROMPT = """You are a biomedical literature assistant. You answer \
questions using ONLY the sources provided in the context below.

Rules:
- Every factual claim must cite its source as [PMID: 12345678]. Multiple \
sources for one claim: [PMID: 111, PMID: 222].
- If the sources do not answer the question, say so plainly. Do not fill gaps \
from your own knowledge, and never invent a PMID.
- Sources marked FULL TEXT are excerpts from the paper itself; those marked \
ABSTRACT are summaries only. Prefer full-text detail for methods and numbers.
- Distinguish study types when it matters: a randomised trial, a retrospective \
cohort, and a case report carry different weight. Note sample sizes when given.
- Where sources disagree, present the disagreement rather than picking a side.
- Be concise and factual. No hedging boilerplate.
- You summarise published research. You do not give medical advice or \
treatment recommendations for any individual."""

_embed_lock = threading.Lock()


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, device=os.environ.get("EMBED_DEVICE", "cuda"))


@lru_cache(maxsize=1)
def _qdrant():
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL)


def preload():
    """Load the embedding model at startup so the first request isn't slow,
    and so two concurrent requests don't each load their own copy."""
    _embedder()
    _qdrant()


def embed(text):
    with _embed_lock:
        return _embedder().encode(text, normalize_embeddings=True).tolist()


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------

def _fts_query(query, max_terms=40):
    """Build a safe FTS5 MATCH string.

    Terms must start alphanumeric and are quoted, otherwise FTS5 reads a
    leading '-' as a column filter and raises 'no such column'.
    """
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{2,}", query)
    if not terms:
        return None
    return " OR ".join('"' + t.replace('"', "") + '"' for t in terms[:max_terms])


def semantic_abstracts(query, k):
    hits = _qdrant().query_points(
        COLLECTION, query=embed(query), limit=k, with_payload=True
    ).points
    return [(f"a:{h.payload['pmid']}", {**h.payload, "kind": "abstract"}) for h in hits]


def semantic_chunks(query, k):
    try:
        hits = _qdrant().query_points(
            CHUNK_COLLECTION, query=embed(query), limit=k, with_payload=True
        ).points
    except Exception:
        return []                       # collection may not exist yet
    return [(f"c:{h.id}", {**h.payload, "kind": "fulltext"}) for h in hits]


def keyword_abstracts(query, k, db=DB_PATH):
    fts = _fts_query(query)
    if not fts:
        return []
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT a.pmid, a.title, a.abstract, a.journal, a.year, a.doi "
            "FROM articles_fts f JOIN articles a ON a.rowid = f.rowid "
            "WHERE articles_fts MATCH ? ORDER BY rank LIMIT ?",
            (fts, k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    return [(f"a:{r['pmid']}", {**dict(r), "kind": "abstract"}) for r in rows]


def keyword_chunks(query, k, db=DB_PATH):
    fts = _fts_query(query)
    if not fts:
        return []
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT c.id, c.pmid, c.pmcid, c.section, c.text, "
            "       a.title, a.journal, a.year, a.doi "
            "FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
            "LEFT JOIN articles a ON a.pmid = c.pmid "
            "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (fts, k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []                       # chunks table may not exist yet
    finally:
        conn.close()
    return [(f"c:{r['id']}", {**dict(r), "kind": "fulltext"}) for r in rows]


def fuse(*ranked_lists, k=60):
    """Reciprocal rank fusion over lists of (key, payload)."""
    scores, payloads = {}, {}
    for lst in ranked_lists:
        for rank, (key, payload) in enumerate(lst):
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            payloads.setdefault(key, payload)
    return [
        (key, scores[key], payloads[key])
        for key in sorted(scores, key=scores.get, reverse=True)
    ]


def retrieve(question, top_k=TOP_K, db=DB_PATH, max_per_paper=MAX_PER_PAPER):
    wide = top_k * 3
    ranked = fuse(
        semantic_abstracts(question, wide),
        semantic_chunks(question, wide),
        keyword_abstracts(question, wide, db),
        keyword_chunks(question, wide, db),
    )

    # Stop one heavily-chunked paper from crowding out everything else.
    seen, out = {}, []
    for key, score, payload in ranked:
        pmid = payload.get("pmid")
        if seen.get(pmid, 0) >= max_per_paper:
            continue
        seen[pmid] = seen.get(pmid, 0) + 1
        out.append((key, score, payload))
        if len(out) >= top_k:
            break
    return out


# --------------------------------------------------------------------------
# prompting
# --------------------------------------------------------------------------

def build_context(results, max_chars=1400):
    blocks = []
    for _, _, p in results:
        body = p.get("text") or p.get("abstract") or ""
        if len(body) > max_chars:
            body = body[:max_chars].rsplit(" ", 1)[0] + " [...]"
        kind = "FULL TEXT" if p.get("kind") == "fulltext" else "ABSTRACT"
        section = f" — {p['section']}" if p.get("section") else ""
        blocks.append(
            f"[PMID: {p.get('pmid')}] ({kind}{section}) {p.get('title')}\n"
            f"{p.get('journal') or 'n/a'}, {p.get('year') or 'n/a'}\n{body}"
        )
    return "\n\n---\n\n".join(blocks)


def build_messages(question, results, history=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"Context sources:\n\n{build_context(results)}\n\n"
                       f"---\n\nQuestion: {question}",
        }
    )
    return messages


def sources_markdown(results):
    lines, seen = ["\n\n---\n**Sources**\n"], set()
    for _, _, p in results:
        pmid = p.get("pmid")
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        tag = " *(full text)*" if p.get("kind") == "fulltext" else ""
        lines.append(
            f"- [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/) — "
            f"{p.get('title')} *({p.get('journal') or ''} {p.get('year') or 'n.d.'})*{tag}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def chat_ollama(messages, stream=False, model=OLLAMA_MODEL):
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "think": False,
        "options": {"temperature": 0.2, "num_ctx": 32768},
    }
    if not stream:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama {r.status_code}: {r.text[:300]}")
        return r.json()["message"]["content"]

    def gen():
        import json

        with requests.post(
            f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=600
        ) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Ollama {r.status_code}: {r.text[:300]}")
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = (data.get("message") or {}).get("content") or ""
                if chunk:
                    yield chunk
                if data.get("done"):
                    break

    return gen()


def answer(question, top_k=TOP_K, db=DB_PATH, model=OLLAMA_MODEL):
    results = retrieve(question, top_k, db)
    if not results:
        return "No matching sources in the corpus for that question.", []
    text = chat_ollama(build_messages(question, results), model=model)
    return text + sources_markdown(results), results
