#!/usr/bin/env python3
"""
Phase 4a — Ingest PMC full text for articles already in pubmed.db.

Only a subset of PubMed articles are in PMC's open-access collection; the rest
are silently skipped and recorded so they aren't retried every run.

Usage:
    python pmc_ingest.py                 # all articles not yet resolved
    python pmc_ingest.py --limit 500     # try a batch first
"""
import argparse
import os
import re
import sqlite3
import time
import xml.etree.ElementTree as ET

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
DB_PATH = os.environ.get("PUBMED_DB", "pubmed.db")
API_KEY = os.environ.get("NCBI_API_KEY")

ID_BATCH = 200       # idconv accepts up to 200 ids
FETCH_BATCH = 20     # full text is large; keep responses manageable
CHUNK_SIZE = 1200    # chars — PubMedBERT truncates around 512 tokens
CHUNK_OVERLAP = 150

SKIP_SECTIONS = re.compile(
    r"reference|acknowledg|competing|conflict|funding|supplementary|"
    r"author contribution|abbreviation|ethics|consent|availability",
    re.I,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS pmc_map (
    pmid    TEXT PRIMARY KEY,
    pmcid   TEXT,
    status  TEXT,            -- 'ok' | 'no_pmc' | 'no_body'
    checked TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (
    id       INTEGER PRIMARY KEY,
    pmid     TEXT,
    pmcid    TEXT,
    section  TEXT,
    idx      INTEGER,
    text     TEXT,
    embedded INTEGER DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE INDEX IF NOT EXISTS idx_chunks_pmid ON chunks(pmid);
CREATE INDEX IF NOT EXISTS idx_chunks_embedded ON chunks(embedded);
"""


def polite_sleep():
    time.sleep(0.11 if API_KEY else 0.35)


def params(**kw):
    p = {"tool": "spark-pubmed-rag", "email": "user@example.com"}
    if API_KEY:
        p["api_key"] = API_KEY
    p.update(kw)
    return p


def resolve_pmcids(pmids):
    """PMID -> PMCID via the NCBI ID Converter. Returns {pmid: pmcid_or_None}."""
    out = {}
    r = requests.get(
        IDCONV, params=params(ids=",".join(pmids), format="json"), timeout=60
    )
    r.raise_for_status()
    for rec in r.json().get("records", []):
        pmid = str(rec.get("pmid"))
        pmcid = rec.get("pmcid")
        out[pmid] = pmcid.replace("PMC", "") if pmcid else None
    for p in pmids:                       # ids the API omitted entirely
        out.setdefault(p, None)
    return out


def iter_sections(node, path=()):
    """Yield (section_label, text) walking nested <sec> elements."""
    title_el = node.find("title")
    title = "".join(title_el.itertext()).strip() if title_el is not None else ""
    if title and SKIP_SECTIONS.search(title):
        return
    new_path = path + (title,) if title else path

    paras = ["".join(p.itertext()).strip() for p in node.findall("./p")]
    text = "\n".join(t for t in paras if t)
    if text:
        yield " > ".join(new_path) or "Body", text

    for sub in node.findall("./sec"):
        yield from iter_sections(sub, new_path)


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split on sentence boundaries, packing into ~size-char windows."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= size:
            current = f"{current} {s}".strip()
            continue
        if current:
            chunks.append(current)
        if len(s) > size:                      # a single huge sentence
            for i in range(0, len(s), size - overlap):
                chunks.append(s[i : i + size])
            current = ""
        else:
            tail = current[-overlap:] if current else ""
            current = f"{tail} {s}".strip()
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) > 120]  # drop scraps


def parse_fulltext(article):
    """Return [(section, chunk_text), ...] for one <article> element."""
    body = article.find(".//body")
    if body is None:
        return []
    out = []
    for sec in body.findall("./sec"):
        for section, text in iter_sections(sec):
            for c in chunk_text(text):
                out.append((section, c))
    if not out:                                # some articles have bare <p>
        loose = "\n".join("".join(p.itertext()) for p in body.findall("./p"))
        out = [("Body", c) for c in chunk_text(loose)]
    return out


def fetch_fulltext(pmcids):
    """Fetch a batch of PMC articles. Returns {(pmid, pmcid): [(section, chunk), ...]}."""
    r = requests.post(
        f"{EUTILS}/efetch.fcgi",
        data=params(db="pmc", id=",".join(pmcids), retmode="xml"),
        timeout=300,
    )
    r.raise_for_status()
    root = ET.fromstring(r.text)
    result = {}
    for art in root.findall(".//article"):
        ids = {a.get("pub-id-type"): (a.text or "") for a in art.findall(".//article-id")}
        # PMC JATS tags the PMCID as pub-id-type="pmc". Keep the legacy keys as
        # defensive fallbacks. Missing this key silently drops every article.
        pmcid = (ids.get("pmc") or ids.get("pmcid") or ids.get("pmcaid") or "").replace("PMC", "")
        pmid = ids.get("pmid") or None
        if pmcid:
            result[(pmid, pmcid)] = parse_fulltext(art)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--limit", type=int, default=0, help="0 = all unresolved")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.executescript(SCHEMA)

    todo = [
        r[0]
        for r in conn.execute(
            "SELECT a.pmid FROM articles a "
            "LEFT JOIN pmc_map m ON m.pmid = a.pmid WHERE m.pmid IS NULL"
        )
    ]
    if args.limit:
        todo = todo[: args.limit]
    print(f"articles without PMC status: {len(todo):,}")

    # --- 1. resolve PMIDs to PMCIDs -------------------------------------
    mapping = {}
    for i in range(0, len(todo), ID_BATCH):
        batch = todo[i : i + ID_BATCH]
        mapping.update(resolve_pmcids(batch))
        polite_sleep()
        print(f"  resolved {min(i + ID_BATCH, len(todo)):>6,} / {len(todo):,}", flush=True)

    conn.executemany(
        "INSERT OR REPLACE INTO pmc_map (pmid, pmcid, status) VALUES (?,?,?)",
        [(p, c, "ok" if c else "no_pmc") for p, c in mapping.items()],
    )
    conn.commit()

    have_pmc = {p: c for p, c in mapping.items() if c}
    print(f"\nin PMC: {len(have_pmc):,} of {len(todo):,} "
          f"({100 * len(have_pmc) / max(len(todo), 1):.0f}%)")

    # --- 2. fetch full text and chunk -----------------------------------
    pmc_to_pmid = {c: p for p, c in have_pmc.items()}
    pmcids = list(pmc_to_pmid)
    total_chunks, no_body = 0, 0

    for i in range(0, len(pmcids), FETCH_BATCH):
        batch = pmcids[i : i + FETCH_BATCH]
        try:
            fetched = fetch_fulltext(batch)
        except Exception as e:
            print(f"  batch failed ({e}); skipping", flush=True)
            polite_sleep()
            continue

        rows = []
        for (xml_pmid, pmcid), chunks in fetched.items():
            pmid = xml_pmid or pmc_to_pmid.get(pmcid)
            if not chunks:
                no_body += 1
                conn.execute(
                    "UPDATE pmc_map SET status='no_body' WHERE pmid=?", (pmid,)
                )
                continue
            for idx, (section, text) in enumerate(chunks):
                rows.append((pmid, pmcid, section, idx, text))

        if rows:
            conn.executemany(
                "INSERT INTO chunks (pmid, pmcid, section, idx, text) VALUES (?,?,?,?,?)",
                rows,
            )
            total_chunks += len(rows)
        conn.commit()
        print(f"  fetched {min(i + FETCH_BATCH, len(pmcids)):>6,} / {len(pmcids):,} "
              f"| chunks so far {total_chunks:,}", flush=True)
        polite_sleep()

    n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    papers = conn.execute("SELECT COUNT(DISTINCT pmid) FROM chunks").fetchone()[0]
    print(f"\nDone. {total_chunks:,} new chunks this run "
          f"({no_body:,} had no usable body).")
    print(f"Total: {n:,} chunks from {papers:,} papers.")
    conn.close()


if __name__ == "__main__":
    main()
