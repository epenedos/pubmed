#!/usr/bin/env python3
"""
Phase 3 — OpenAI-compatible endpoint, so the assistant appears as a model
inside Open WebUI (or any OpenAI-compatible client).

Run:
    uvicorn pubmed_api:app --host 0.0.0.0 --port 8100

Then in Open WebUI: Admin Panel -> Settings -> Connections -> add an
OpenAI API connection:
    URL:  http://host.docker.internal:8100/v1     (or http://<spark-ip>:8100/v1)
    Key:  anything non-empty
The model "pubmed-ovarian" then appears in the model dropdown.
"""
import json
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import pubmed_core as core

MODEL_ID = "pubmed-ovarian"

app = FastAPI(title="PubMed RAG")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[Message]
    stream: bool = False


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.get("/health")
def health():
    return {"ok": True, "model": core.OLLAMA_MODEL, "collection": core.COLLECTION}


def _last_question(messages):
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _history(messages, limit=6):
    """Prior turns, excluding the current question, trimmed for context budget."""
    prior = [m for m in messages if m.role in ("user", "assistant")][:-1]
    return [{"role": m.role, "content": m.content} for m in prior[-limit:]]


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    question = _last_question(req.messages)
    results = core.retrieve(question)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if not results:
        text = "No matching abstracts in the corpus for that question."
        if not req.stream:
            return _completion(completion_id, created, text)
        return StreamingResponse(
            _sse(completion_id, created, iter([text])), media_type="text/event-stream"
        )

    messages = core.build_messages(question, results, _history(req.messages))
    tail = core.sources_markdown(results)

    if not req.stream:
        text = core.chat_ollama(messages) + tail
        return _completion(completion_id, created, text)

    def chunks():
        yield from core.chat_ollama(messages, stream=True)
        yield tail

    return StreamingResponse(
        _sse(completion_id, created, chunks()), media_type="text/event-stream"
    )


def _completion(cid, created, text):
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


def _sse(cid, created, chunk_iter):
    def frame(delta, finish=None):
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield frame({"role": "assistant", "content": ""})
    for chunk in chunk_iter:
        yield frame({"content": chunk})
    yield frame({}, finish="stop")
    yield "data: [DONE]\n\n"
