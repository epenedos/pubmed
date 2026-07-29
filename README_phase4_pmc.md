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

# 4. swap in the new core
cp pubmed_core.py pubmed_core_backup.py
cp pubmed_core_v2.py pubmed_core.py

# 5. add the preload call at the top of pubmed_api.py, after `app = FastAPI(...)`:
#       core.preload()
```

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

If answers get vaguer after adding full text, the usual cause is chunk-level
noise crowding out abstracts. Lower `MAX_PER_PAPER` to 1, or raise `TOP_K` to
12 so both kinds have room.

## Storage

~10 chunks per paper, ~3KB per vector — around 300MB of vectors for the full
ovarian corpus, plus a few GB of text in SQLite. Trivial for the 4TB drive.
Embedding runs at a few thousand chunks per minute on the GB10.

## Licensing

PMC open access permits text mining; individual articles carry varying CC
licences. Fine for private research use. If you ever republish excerpts, check
the per-article licence in the JATS `<permissions>` element.
