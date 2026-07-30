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

`pubmed_core.py` (abstracts only) and `pubmed_core_v2.py` (abstracts **and**
chunks) are the shared retrieval/answer libraries. `pubmed_ask.py` and
`pubmed_api.py` both `import pubmed_core`. **To use full text you must replace
`pubmed_core.py` with `pubmed_core_v2.py`** — see the gotcha below.

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
| `EMBED_DEVICE` | `cuda` | core, pmc_embed (**not** pubmed_embed) |
| `OLLAMA_URL` / `OLLAMA_MODEL` | `localhost:11434` / `nemotron-3-nano` | core |
| `TOP_K` / `MAX_PER_PAPER` | `8` / `2` | core |

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
2. **Full text requires swapping the core file.** `ask`/`api` import
   `pubmed_core`, which is the abstracts-only v1. If you ingested and embedded
   PMC chunks but answers still only cite abstracts, it's almost always because
   `pubmed_core.py` was never replaced with `pubmed_core_v2.py`.
3. **`pmc_ingest.py` PMCID parsing is the prime suspect for "no full text."**
   `fetch_fulltext()` reads the PMCID from `article-id` using keys `"pmcid"`/
   `"pmcaid"`, but PMC JATS tags it `pub-id-type="pmc"`. When the key doesn't
   match, the article is silently skipped and **zero chunks are written** — while
   `pmc_map` still records the PMID, so re-runs never retry it. Symptom: PMC
   coverage percentage prints fine but "chunks so far" stays 0. Verify with
   `SELECT COUNT(*) FROM chunks;` before assuming full text is present.
4. **FTS5 query sanitization differs between files.** `pubmed_core_v2.py` quotes
   terms and requires an alphanumeric first char (`_fts_query`); the older
   `pubmed_core.py` and `pubmed_search.py` do not, so a query token with a
   leading `-` can raise `no such column` from FTS5.
5. **`pubmed_embed.py` hardcodes `device="cuda"`** (ignores `EMBED_DEVICE`),
   unlike the other embed/core files. It will crash on a CPU-only box.
6. Very recent papers have empty `mesh_terms` (NCBI indexes them weeks later) —
   expected, not a bug.

## Debugging answer quality

Retrieval quality is the ceiling on answer quality. When an answer is thin or
"nothing visible":

1. Check the corpus is populated: `SELECT COUNT(*) FROM articles;` and
   `SELECT COUNT(*) FROM chunks;` (chunks == 0 means full text never ingested —
   see gotcha 3).
2. Check vectors exist: Qdrant collection counts should match the DB row counts.
3. Check which core is imported (gotcha 2) — chunks are only searched by v2.
4. Print the retrieved PMIDs (the CLI does this) before blaming the model.

## Git / workflow

- Default branch: `main`. `pubmed.db` and `.venv/` are git-ignored — the DB is
  large and rebuildable; never commit it.
- READMEs are phase-staged: `README.md` (phase 1), `README_phase2_3.md`,
  `README_phase4_pmc.md`.
