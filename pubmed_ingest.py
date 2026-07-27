#!/usr/bin/env python3
"""
Phase 1a — Ingest PubMed abstracts into SQLite.

Usage:
    python pubmed_ingest.py --max 5000
    python pubmed_ingest.py --query "your custom pubmed query" --max 500

Env:
    NCBI_API_KEY   optional, raises rate limit 3/s -> 10/s
                   get one free at https://account.ncbi.nlm.nih.gov/settings/
"""
import argparse
import os
import sqlite3
import time
from datetime import date
import xml.etree.ElementTree as ET

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DB_PATH = os.environ.get("PUBMED_DB", "pubmed.db")
API_KEY = os.environ.get("NCBI_API_KEY")
BATCH = 200

# Metastatic / advanced-stage ovarian cancer.
# MeSH terms catch indexed papers, [tiab] catches recent ones not yet indexed.
DEFAULT_QUERY = (
    '('
    '  "Ovarian Neoplasms"[MeSH] OR "ovarian cancer"[tiab] OR '
    '  "ovarian carcinoma"[tiab] OR "ovarian neoplasm"[tiab]'
    ') AND ('
    '  "Neoplasm Metastasis"[MeSH] OR metasta*[tiab] OR '
    '  "peritoneal carcinomatosis"[tiab] OR "advanced stage"[tiab] OR '
    '  "stage III"[tiab] OR "stage IV"[tiab] OR "platinum-resistant"[tiab] OR '
    '  "platinum resistant"[tiab] OR recurrent[tiab]'
    ') AND hasabstract AND english[lang]'
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    pmid        TEXT PRIMARY KEY,
    title       TEXT,
    abstract    TEXT,
    journal     TEXT,
    year        INTEGER,
    authors     TEXT,
    doi         TEXT,
    pub_types   TEXT,
    mesh_terms  TEXT,
    fetched_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    embedded    INTEGER DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
    title, abstract, content='articles', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS articles_ai AFTER INSERT ON articles BEGIN
    INSERT INTO articles_fts(rowid, title, abstract)
    VALUES (new.rowid, new.title, new.abstract);
END;
CREATE INDEX IF NOT EXISTS idx_year ON articles(year);
CREATE INDEX IF NOT EXISTS idx_embedded ON articles(embedded);
"""


def polite_sleep():
    time.sleep(0.11 if API_KEY else 0.35)


def params(**kw):
    p = {"db": "pubmed", "tool": "spark-pubmed-rag", "email": "user@example.com"}
    if API_KEY:
        p["api_key"] = API_KEY
    p.update(kw)
    return p


def esearch_slice(query, mindate, maxdate, retmax=9999):
    """Return (total_count, pmids) for one date window.

    NCBI refuses retstart >= 10000, so we never page: we slice the query into
    windows small enough that each returns fewer than 10k hits.
    """
    r = requests.get(
        f"{EUTILS}/esearch.fcgi",
        params=params(
            term=query, retmax=retmax, datetype="pdat",
            mindate=mindate, maxdate=maxdate,
        ),
        timeout=60,
    )
    r.raise_for_status()
    root = ET.fromstring(r.text)
    count = int(root.findtext("Count", "0"))
    pmids = [e.text for e in root.findall("./IdList/Id") if e.text]
    return count, pmids


def collect_pmids(query, start_year, end_year, limit):
    """Walk year by year, splitting into months if a year overflows the cap."""
    pmids = []
    for year in range(end_year, start_year - 1, -1):
        count, ids = esearch_slice(query, f"{year}/01/01", f"{year}/12/31")
        polite_sleep()

        if count >= 9999:
            ids = []
            for month in range(1, 13):
                _, mids = esearch_slice(
                    query, f"{year}/{month:02d}/01", f"{year}/{month:02d}/31"
                )
                ids.extend(mids)
                polite_sleep()
            print(f"  {year}: {count:,} hits (split by month, got {len(ids):,})")
        elif count:
            print(f"  {year}: {count:,}")

        pmids.extend(ids)
        if len(pmids) >= limit:
            return pmids[:limit]
    return pmids


def text_of(node):
    """Join an element's text including nested tags (abstracts have <i>, <sup>...)."""
    return "".join(node.itertext()).strip() if node is not None else ""


def parse_article(art):
    medline = art.find("MedlineCitation")
    if medline is None:
        return None
    pmid = medline.findtext("PMID")
    article = medline.find("Article")
    if article is None or pmid is None:
        return None

    title = text_of(article.find("ArticleTitle"))

    # Structured abstracts: keep the section labels, they help retrieval.
    parts = []
    for seg in article.findall("./Abstract/AbstractText"):
        label = seg.get("Label")
        body = text_of(seg)
        if not body:
            continue
        parts.append(f"{label}: {body}" if label else body)
    abstract = "\n".join(parts)
    if not abstract:
        return None

    journal = article.findtext("./Journal/ISOAbbreviation") or article.findtext(
        "./Journal/Title", ""
    )
    year = article.findtext("./Journal/JournalIssue/PubDate/Year")
    if not year:
        medline_date = article.findtext("./Journal/JournalIssue/PubDate/MedlineDate", "")
        year = medline_date[:4] if medline_date[:4].isdigit() else None

    authors = []
    for a in article.findall("./AuthorList/Author"):
        last, initials = a.findtext("LastName"), a.findtext("Initials")
        if last:
            authors.append(f"{last} {initials}" if initials else last)

    doi = None
    for aid in art.findall(".//ArticleId"):
        if aid.get("IdType") == "doi":
            doi = aid.text

    pub_types = [t.text for t in article.findall("./PublicationTypeList/PublicationType") if t.text]
    mesh = [m.text for m in medline.findall("./MeshHeadingList/MeshHeading/DescriptorName") if m.text]

    return (
        pmid, title, abstract, journal,
        int(year) if year and year.isdigit() else None,
        "; ".join(authors[:12]), doi,
        "; ".join(pub_types), "; ".join(mesh),
    )


def efetch(pmids):
    """Fetch a batch by explicit ID list. POST because the URL would be too long."""
    r = requests.post(
        f"{EUTILS}/efetch.fcgi",
        data=params(id=",".join(pmids), retmode="xml"),
        timeout=180,
    )
    r.raise_for_status()
    root = ET.fromstring(r.text)
    rows = []
    for art in root.findall("PubmedArticle"):
        parsed = parse_article(art)
        if parsed:
            rows.append(parsed)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--max", type=int, default=5000)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--from-year", type=int, default=1990)
    ap.add_argument("--to-year", type=int, default=date.today().year)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.executescript(SCHEMA)

    print("collecting PMIDs by year...")
    pmids = collect_pmids(args.query, args.from_year, args.to_year, args.max)
    print(f"\ncollected {len(pmids):,} PMIDs")

    # Skip anything already stored — makes re-runs cheap and resumable.
    have = {r[0] for r in conn.execute("SELECT pmid FROM articles")}
    pmids = [p for p in pmids if p not in have]
    print(f"new to fetch: {len(pmids):,} ({len(have):,} already in DB)\n")

    inserted = 0
    for start in range(0, len(pmids), BATCH):
        rows = efetch(pmids[start : start + BATCH])
        conn.executemany(
            "INSERT OR IGNORE INTO articles "
            "(pmid,title,abstract,journal,year,authors,doi,pub_types,mesh_terms) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        inserted += len(rows)
        print(f"  {min(start + BATCH, len(pmids)):>6,} / {len(pmids):,} fetched", flush=True)
        polite_sleep()

    total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    print(f"\nDone. {inserted:,} parsed this run — {total:,} articles in {args.db}")
    conn.close()


if __name__ == "__main__":
    main()
