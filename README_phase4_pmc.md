# Phase 4 — Adding PMC full text

Abstracts tell you what a study concluded. Full text tells you how many
patients, what the exclusion criteria were, and what the authors admit didn't
work. This adds the PMC open-access full text for papers already in your DB.

## What changes

| File | Action |
|---|---|
| `pmc_ingest.py` | new — resolves PMIDs to PMCIDs, fetches JATS XML, chunks into SQLite |
| `pmc_embed.py` | new — embeds chunks into a second qdrant collection |
| `pubmed_core_v2.py` | replaces `pubmed_core.py` — searches abstracts *and* chunks |

`pubmed_ask.py` and `pubmed_api.py` need no changes beyond the import.

## Run

```bash
cd ~/pubmed && source .venv/bin/activate

# 1. try a small batch first to see your PMC coverage
python pmc_ingest.py --limit 500

# 2. if it looks good, do the rest (slow: ~2-4 hours for ~29k articles)
python pmc_ingest.py

# 3. embed the chunks
python pmc_embed.py
```

`pubmed_core.py` already searches abstracts **and** full-text chunks — there is
no core file to swap and `pubmed_api.py` already preloads models at startup.
(The old `pubmed_core_v2.py` / `cp`-over-`pubmed_core.py` step is gone; the
full-text core is now the only core.)

Restart uvicorn and ask a methods-heavy question — answers should now cite
full-text excerpts with section labels rather than abstract summaries.

Expect roughly 30-40% of oncology papers to be in PMC open access; older and
paywalled ones simply aren't there. The script records which so it never
retries them. Both ingest steps are resumable — safe to interrupt.

## How it works

- **Chunking is section-aware.** Each `<sec>` keeps its title path
  (`Methods > Statistical analysis`), split into ~1200-character windows on
  sentence boundaries with overlap. The section label is prefixed to the text
  at embedding time, so the encoder knows whether it's reading methods or
  discussion. References, funding, and ethics sections are dropped.
- **Two collections, one search.** Abstracts stay in `pubmed_ovarian`, chunks
  go to `pubmed_ovarian_chunks`. Retrieval queries both semantically and both
  keyword indexes — four ranked lists, fused.
- **Per-paper cap.** A 60-chunk paper would otherwise fill every slot, so at
  most `MAX_PER_PAPER` (default 2) passages per PMID reach the model.
- **Sources are deduplicated** by PMID and flagged when full text was used.

## Tuning

| Variable | Default | Effect |
|---|---|---|
| `TOP_K` | 8 | passages sent to the model |
| `MAX_PER_PAPER` | 2 | diversity vs depth on a single paper |
| `CHUNK_SIZE` | 1200 | in `pmc_ingest.py`; larger = more context, worse precision |
| `RERANK` | 0 | `1` enables cross-encoder reranking (below) |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | cross-encoder; `ncbi/MedCPT-Cross-Encoder` for biomedical |
| `RERANK_CANDIDATES` | 50 | fused hits rescored before the per-paper cap |

If answers get vaguer after adding full text, the usual cause is chunk-level
noise crowding out abstracts. Lower `MAX_PER_PAPER` to 1, or raise `TOP_K` to
12 so both kinds have room.

## Reranking (optional, recommended for precise questions)

With ~300k chunks, the one methods sentence that answers a question ("184
patients were randomised…") often doesn't rank in the top few by bi-encoder
similarity alone. Set `RERANK=1` to add a **cross-encoder** pass: retrieval
pulls a wide candidate pool (`RERANK_CANDIDATES`), a cross-encoder rescores each
`(question, passage)` pair jointly — far more precise than the retrieval
embedding — and the best passages surface before the per-paper cap and `TOP_K`
cut are applied.

```bash
RERANK=1 python pubmed_ask.py "What grade 3-4 toxicities were reported in HIPEC trials?"
```

The reranker model downloads on first use (~1GB) and adds latency per query but
runs on GPU. If it can't load, retrieval silently falls back to the fused order,
so enabling it never breaks answering. Enable it in the API service the same
way — add `Environment="RERANK=1"` to the systemd unit.

## Storage

~10 chunks per paper, ~3KB per vector — around 300MB of vectors for the full
ovarian corpus, plus a few GB of text in SQLite. Trivial for the 4TB drive.
Embedding runs at a few thousand chunks per minute on the GB10.

## Licensing

PMC open access permits text mining; individual articles carry varying CC
licences. Fine for private research use. If you ever republish excerpts, check
the per-article licence in the JATS `<permissions>` element.
