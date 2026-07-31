# CLAUDE.md

Guidance for working in this repository.

## What this is

A local, GPU-hosted **RAG (retrieval-augmented generation) pipeline over PubMed
literature**, scoped to a single corpus: metastatic / advanced-stage ovarian
cancer (~29k abstracts). It ingests abstracts and PMC open-access full text into
SQLite, embeds them into Qdrant, retrieves with hybrid (semantic + keyword)
search, and answers questions with a local Ollama model that is forced to cite
PMIDs. It is exposed both as a CLI and as an OpenAI-compatible API that plugs
into Open WebUI.

It is a **literature-mapping tool, not a medical-advice tool.** Keep the
grounding constraints (cite-or-decline) intact in any change.

## Architecture at a glance

```
NCBI E-utilities ──► SQLite (pubmed.db) ──► Qdrant ──► retrieve+fuse ──► Ollama ──► answer
  esearch/efetch      articles / chunks     2 collections   RRF          local LLM   (cited)
```

- **SQLite (`pubmed.db`) is the source of truth.** It holds article metadata,
  abstracts, full-text chunks, the PMID→PMCID map, and FTS5 keyword indexes. If
  Qdrant is lost, reset the `embedded` flags and re-embed — never re-ingest.
- **Qdrant holds only vectors** in two collections: `pubmed_ovarian` (abstracts)
  and `pubmed_ovarian_chunks` (full-text chunks). It is a rebuildable cache.
- Embeddings: `NeuML/pubmedbert-base-embeddings` (768d, biomedical) via
  sentence-transformers on CUDA.
- Generation: local Ollama (`nemotron-3-nano` default).

## The pipeline (run in order)

| Phase | Script | Reads | Writes |
|---|---|---|---|
| 1a ingest abstracts | `pubmed_ingest.py` | NCBI esearch/efetch | `articles`, `articles_fts` |
| 1b embed abstracts | `pubmed_embed.py` | `articles` where `embedded=0` | Qdrant `pubmed_ovarian` |
| 1c search (CLI) | `pubmed_search.py` | Qdrant + FTS5 | stdout |
| 4a ingest full text | `pmc_ingest.py` | NCBI idconv + PMC efetch | `pmc_map`, `chunks`, `chunks_fts` |
| 4b embed chunks | `pmc_embed.py` | `chunks` where `embedded=0` | Qdrant `pubmed_ovarian_chunks` |
| 2 answer (CLI) | `pubmed_ask.py` → core | Qdrant + FTS5 + Ollama | stdout |
| 3 answer (API) | `pubmed_api.py` → core | same | OpenAI-compatible HTTP |

`pubmed_core.py` is the single shared retrieval/answer library — it searches
abstracts **and** full-text chunks, and both `pubmed_ask.py` and `pubmed_api.py`
`import pubmed_core`. (There is no longer a `pubmed_core_v2.py` or a file-swap
step; the full-text core is the only core.) An optional cross-encoder reranker
sits inside `retrieve()`, enabled with `RERANK=1`.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install requests qdrant-client sentence-transformers fastapi uvicorn
export NCBI_API_KEY=...        # optional, 3/s -> 10/s
python pubmed_ingest.py --max 29000
python pubmed_embed.py
python pmc_ingest.py           # full-text (slow, ~2-4h for full corpus)
python pmc_embed.py
python pubmed_ask.py "Does HIPEC improve survival in stage III disease?"
uvicorn pubmed_api:app --host 0.0.0.0 --port 8100     # for Open WebUI
```

Ingest and embed are both **incremental and resumable**: `INSERT OR IGNORE` on
PMID skips duplicates, and only `embedded=0` rows are re-embedded.

## Configuration (environment variables)

| Var | Default | Used by |
|---|---|---|
| `PUBMED_DB` | `pubmed.db` | all |
| `NCBI_API_KEY` | – | ingest scripts (rate limit) |
| `QDRANT_URL` | `http://localhost:6333` | embed + core |
| `QDRANT_COLLECTION` | `pubmed_ovarian` | embed + core |
| `QDRANT_CHUNK_COLLECTION` | `pubmed_ovarian_chunks` | pmc_embed + core_v2 |
| `EMBED_MODEL` | `NeuML/pubmedbert-base-embeddings` | embed + core |
| `EMBED_DEVICE` | `cuda` | core, pubmed_embed, pmc_embed |
| `OLLAMA_URL` / `OLLAMA_MODEL` | `localhost:11434` / `nemotron-3-nano` | core |
| `TOP_K` / `MAX_PER_PAPER` | `8` / `2` | core |
| `RERANK` | `0` | core (`1` = cross-encoder rerank) |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | core (when `RERANK=1`) |
| `RERANK_CANDIDATES` | `50` | core (pool size rescored) |

## Conventions

- Single-file scripts, stdlib + a few deps, no framework. Each script has a
  docstring with usage and is runnable standalone.
- All DB access is raw `sqlite3`; no ORM. Schema lives inline in each ingest
  script as `CREATE TABLE IF NOT EXISTS`.
- Be polite to NCBI: keep `polite_sleep()` between requests and honor the batch
  sizes (esearch never pages past retstart 10000 — it slices by year/month).
- Retrieval uses **reciprocal rank fusion** to merge ranked lists without tuning
  score scales. Preserve this when touching retrieval.
- The system prompt is the safety boundary: cite every claim, decline when the
  context doesn't answer, never invent a PMID. Don't weaken it.

## Known gotchas / sharp edges (read before debugging)

1. **`pubmed_ingest.py` never touches Qdrant.** It only writes SQLite. No script
   "cleans" Qdrant on any run — the embed scripts `create_collection` only if
   absent and otherwise `upsert` (update-in-place). Stale vectors from a previous
   run therefore persist; if the corpus definition changes, drop the Qdrant
   collections manually.
2. **Reranking is opt-in and self-contained.** `retrieve()` runs a cross-encoder
   pass only when `RERANK=1`; otherwise behaviour is the plain fused ranking. The
   reranker model loads lazily (and is preloaded by `pubmed_api.py`), and if it
   fails to load, retrieval falls back to the fused order rather than erroring —
   so a bad `RERANK_MODEL` degrades quality but never breaks answering. (There is
   no longer a v1/v2 core split or a file-swap step to forget.)
3. **`pmc_ingest.py` full text hinges on the PMCID key.** `fetch_fulltext()`
   now reads the PMCID from `article-id` under `pub-id-type="pmc"` (PMC JATS's
   actual tag), with `"pmcid"`/`"pmcaid"` kept only as fallbacks. Historically
   it read the fallbacks alone, so every article was silently skipped and **zero
   chunks were written** while `pmc_map` still recorded the PMID (so re-runs
   never retried it). If you have old `pmc_map` rows from before the fix, clear
   the stale ones (`DELETE FROM pmc_map WHERE pmid NOT IN (SELECT pmid FROM
   chunks);`) and re-run. Sanity-check full text with `SELECT COUNT(*) FROM
   chunks;` — if it's 0, no full text was ingested.
4. **FTS5 query sanitization differs between files.** `pubmed_core_v2.py` quotes
   terms and requires an alphanumeric first char (`_fts_query`); the older
   `pubmed_core.py` and `pubmed_search.py` do not, so a query token with a
   leading `-` can raise `no such column` from FTS5.
5. **`pubmed_ingest.py` `efetch()` retries transient NCBI drops.** NCBI
   intermittently ends a response early (`ChunkedEncodingError`); `efetch()`
   retries with exponential backoff, and the ingest loop skips a batch that
   fails all retries rather than aborting. The run is resumable regardless —
   each batch is committed — so re-running always makes forward progress.
6. Very recent papers have empty `mesh_terms` (NCBI indexes them weeks later) —
   expected, not a bug.

## Debugging answer quality

Retrieval quality is the ceiling on answer quality. When an answer is thin or
"nothing visible":

1. Check the corpus is populated: `SELECT COUNT(*) FROM articles;` and
   `SELECT COUNT(*) FROM chunks;` (chunks == 0 means full text never ingested —
   see gotcha 3).
2. Check vectors exist: Qdrant collection counts should match the DB row counts.
3. Print the retrieved sources (the CLI marks full-text hits with `*`) before
   blaming the model — thin answers are usually a retrieval miss, not the LLM.
4. If precise passages aren't surfacing, try `RERANK=1` (cross-encoder) and/or
   a larger `OLLAMA_MODEL`; retrieval precision and model size are the two
   ceilings on answer quality.

## Git / workflow

- Default branch: `main`. `pubmed.db` and `.venv/` are git-ignored — the DB is
  large and rebuildable; never commit it.
- READMEs are phase-staged: `README.md` (phase 1), `README_phase2_3.md`,
  `README_phase4_pmc.md`.
