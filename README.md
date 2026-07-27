# PubMed research assistant — Phase 1

Corpus: metastatic / advanced-stage ovarian cancer (~29,000 abstracts available).

## Setup on the Spark

```bash
mkdir -p ~/pubmed-rag && cd ~/pubmed-rag
python3 -m venv .venv && source .venv/bin/activate
pip install requests qdrant-client sentence-transformers
```

Check the GPU is actually being used (DGX OS ships CUDA-ready PyTorch; if this
prints False, install torch from NVIDIA's index rather than plain PyPI):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Optional but recommended — a free NCBI API key raises the rate limit from
3 to 10 requests/second, roughly 3x faster ingestion:
https://account.ncbi.nlm.nih.gov/settings/

```bash
export NCBI_API_KEY=your_key_here
```

## Run

```bash
# 1. ingest — start small to sanity-check, then go wide
python pubmed_ingest.py --max 200
python pubmed_ingest.py --max 29000        # full corpus, ~15-25 min

# 2. embed into qdrant (uses the container you already have on :6333)
python pubmed_embed.py

# 3. search
python pubmed_search.py "PARP inhibitor maintenance in platinum-resistant disease"
```

Re-running ingest is safe and incremental: `INSERT OR IGNORE` on PMID skips
duplicates, and only new rows have `embedded = 0`, so step 2 picks up just the
new ones. A weekly cron of steps 1+2 keeps the corpus current.

## Design notes

- **SQLite holds metadata + FTS5 keyword index; qdrant holds vectors.** SQLite
  is the source of truth — if qdrant is ever lost, reset `embedded = 0` and
  re-run step 2.
- **Embedding model** defaults to PubMedBERT, trained on biomedical text, so it
  understands that "platinum-refractory" and "chemoresistant" are related.
  Swap via `EMBED_MODEL` — `BAAI/bge-m3` is the alternative if you ever want
  multilingual or longer inputs.
- **Hybrid retrieval**: semantic search finds concepts, FTS5 finds exact terms
  (drug names, gene symbols like BRCA1, trial acronyms). Reciprocal rank fusion
  merges them — this consistently beats either method alone in medical search.
- **Structured abstracts** keep their section labels (BACKGROUND/METHODS/
  RESULTS/CONCLUSIONS), which improves both retrieval and later LLM answers.
- Very recent papers have empty `mesh_terms` — NCBI indexes them weeks later.
  Expected, not a bug.

## Query cookbook

Narrow the corpus by editing `DEFAULT_QUERY` or passing `--query`:

```bash
# only clinical trials
--query '"Ovarian Neoplasms"[MeSH] AND metasta*[tiab] AND clinical trial[pt] AND hasabstract'

# only the last 3 years
--query '"Ovarian Neoplasms"[MeSH] AND metasta*[tiab] AND 2023:2026[dp] AND hasabstract'

# a specific angle
--query '"Ovarian Neoplasms"[MeSH] AND ("PARP Inhibitors"[MeSH] OR olaparib[tiab] OR niraparib[tiab]) AND hasabstract'
```

## Next (Phase 2)

Answering layer: retrieve top-k with the above, feed abstracts to a local model
via Ollama with a strict "cite the PMID for every claim, say so if the abstracts
don't answer it" prompt. Then expose it inside Open WebUI so it lives in the
interface you already use.
