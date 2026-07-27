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


def esearch(query, retmax):
    """Search, using the NCBI history server so we can page through results."""
    r = requests.get(
        f"{EUTILS}/esearch.fcgi",
        params=params(term=query, retmax=0, usehistory="y", sort="date"),
        timeout=60,
    )
    r.raise_for_status()
    root = ET.fromstring(r.text)
    count = int(root.findtext("Count", "0"))
    return {
        "count": min(count, retmax),
        "total": count,
        "webenv": root.findtext("WebEnv"),
        "query_key": root.findtext("QueryKey"),
    }


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


def efetch(webenv, query_key, retstart, retmax):
    r = requests.get(
        f"{EUTILS}/efetch.fcgi",
        params=params(
            WebEnv=webenv, query_key=query_key,
            retstart=retstart, retmax=retmax, retmode="xml",
        ),
        timeout=120,
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
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.executescript(SCHEMA)

    info = esearch(args.query, args.max)
    print(f"PubMed matches: {info['total']:,}  |  fetching: {info['count']:,}")
    polite_sleep()

    inserted = 0
    for start in range(0, info["count"], BATCH):
        take = min(BATCH, info["count"] - start)
        rows = efetch(info["webenv"], info["query_key"], start, take)
        conn.executemany(
            "INSERT OR IGNORE INTO articles "
            "(pmid,title,abstract,journal,year,authors,doi,pub_types,mesh_terms) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        inserted += len(rows)
        print(f"  {start + len(rows):>6,} / {info['count']:,} fetched", flush=True)
        polite_sleep()

    total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    print(f"\nDone. {inserted:,} parsed this run — {total:,} articles in {args.db}")
    conn.close()


if __name__ == "__main__":
    main()
