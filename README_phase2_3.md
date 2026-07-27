# PubMed research assistant — Phase 2 & 3

Phase 2 adds the answering layer; Phase 3 puts it inside Open WebUI as a
selectable model. Both share `pubmed_core.py`.

## Install

```bash
cd ~/pubmed && source .venv/bin/activate
pip install fastapi uvicorn
```

## Phase 2 — ask from the terminal

```bash
python pubmed_ask.py "Does HIPEC improve survival in stage III disease?"
python pubmed_ask.py            # interactive loop
```

It retrieves the top abstracts, prints which PMIDs it pulled (so you can see
what the model was actually shown), then streams a cited answer.

Pick the answering model with `OLLAMA_MODEL`:

```bash
export OLLAMA_MODEL=nemotron-3-nano     # fast, good default
export OLLAMA_MODEL=gpt-oss:120b        # slower, stronger synthesis
```

Bigger models are noticeably better at weighing conflicting studies. Worth
testing both on the same question — the difference is instructive.

## Phase 3 — inside Open WebUI

Start the API server on the Spark:

```bash
uvicorn pubmed_api:app --host 0.0.0.0 --port 8100
```

Check it: `curl http://localhost:8100/health`

Then in Open WebUI: **Admin Panel → Settings → Connections → OpenAI API → Add**

| Field | Value |
|---|---|
| URL | `http://host.docker.internal:8100/v1` (or `http://<spark-ip>:8100/v1`) |
| Key | anything non-empty, e.g. `local` |

Save, refresh, and **pubmed-ovarian** appears in the model dropdown. Selecting
it gives you the RAG pipeline with your normal chat history, and every answer
ends with clickable PubMed links.

### Run it as a service

So it survives reboots:

```bash
sudo tee /etc/systemd/system/pubmed-api.service > /dev/null <<'EOF'
[Unit]
Description=PubMed RAG API
After=network.target ollama.service

[Service]
User=epenedos
WorkingDirectory=/home/epenedos/pubmed
Environment="PUBMED_DB=/home/epenedos/pubmed/pubmed.db"
Environment="OLLAMA_MODEL=nemotron-3-nano"
ExecStart=/home/epenedos/pubmed/.venv/bin/uvicorn pubmed_api:app --host 0.0.0.0 --port 8100
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now pubmed-api
```

## How the answering works

1. **Retrieve** — hybrid search (semantic + FTS5, rank-fused), top 8 by default
2. **Ground** — abstracts injected as an evidence block, each tagged with its PMID
3. **Constrain** — the system prompt forbids outside knowledge and invented
   PMIDs, requires a citation per claim, asks it to weigh study types and
   surface disagreement rather than smoothing it over
4. **Cite** — retrieved sources are appended as real links, independent of what
   the model wrote, so you can always check the primary source

Tuning knobs: `TOP_K` (more abstracts = broader but noisier), `num_ctx` in
`chat_ollama` (raise if you increase TOP_K), and `max_chars` in `build_context`
(abstract truncation).

## Sanity checks worth running

- Ask something the corpus can't answer ("what's the capital of Portugal?") —
  it should decline rather than answer from general knowledge.
- Ask a question you already know the literature on, and click through two or
  three of the cited PMIDs to confirm the claims match the abstracts.
- Compare a small and a large model on the same question.

Retrieval quality is the ceiling on answer quality: if the right paper isn't
retrieved, no model can cite it. When an answer looks thin, check the retrieved
PMID list first — that's usually where the problem is, not the model.

## Scope

This summarises published abstracts. It is a literature tool, not a source of
medical advice, and abstracts omit much of what matters in a full paper —
methods detail, limitations, effect sizes. Treat its output as a map of the
literature that points you to papers worth reading properly.
