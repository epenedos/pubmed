#!/usr/bin/env python3
"""
Phase 2 CLI — ask the corpus a question, get a cited answer.

    python pubmed_ask.py "Is HIPEC beneficial in stage III disease?"
    python pubmed_ask.py "PARP maintenance after platinum resistance" --k 12
    python pubmed_ask.py            # interactive mode
"""
import argparse
import sys

import pubmed_core as core


def ask(question, k, model):
    results = core.retrieve(question, top_k=k)
    if not results:
        print("No matching abstracts in the corpus.")
        return
    print(f"\n[retrieved {len(results)} abstracts: "
          f"{', '.join(p['pmid'] for _, _, p in results)}]\n")
    for chunk in core.chat_ollama(core.build_messages(question, results),
                                  stream=True, model=model):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    print(core.sources_markdown(results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--k", type=int, default=core.TOP_K)
    ap.add_argument("--model", default=core.OLLAMA_MODEL)
    args = ap.parse_args()

    if args.question:
        ask(args.question, args.k, args.model)
        return

    print("Ask a question (Ctrl-C to exit)")
    while True:
        try:
            q = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if q:
            ask(q, args.k, args.model)


if __name__ == "__main__":
    main()
