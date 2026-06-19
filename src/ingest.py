"""Ingest CVs as four embedded sections per candidate, with filter metadata.

Layout in ChromaDB: one record per (candidate, section). Each record's document is that
section's text; its embedding is that section's vector. The full extracted CV is
duplicated as JSON on every record so search can rebuild it without extra fetches.
"""

import os
import argparse

import fitz  # pymupdf

from src.schema import SECTIONS, CVExtract
from src.vectorstore import VectorStore
from src.llm import extract_cv
from src.embeddings import embed_texts


def load_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    text = "\n".join(page.get_text().strip() for page in doc)
    doc.close()
    return text.strip()


def _candidate_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _scalar_metadata(extract: CVExtract) -> dict:
    """Chroma metadata must be scalar — flatten lists/None to filterable values."""
    return {
        "name": extract.name,
        "role_title": extract.role_title,
        "location": extract.location or "",
        # -1 = unknown; search treats it as "not stated", not "0 years".
        "years_experience": extract.years_experience if extract.years_experience is not None else -1.0,
        "languages": ",".join(extract.languages).lower(),
        "availability": extract.availability or "",
        "extract_json": extract.model_dump_json(),
    }


def ingest_pdf(pdf_path: str, store: VectorStore) -> str:
    """Extract one CV, embed its four sections, upsert into the store."""
    cid = _candidate_id(pdf_path)
    raw = load_pdf_text(pdf_path)
    if not raw:
        print(f"[warn] No text extracted from {pdf_path}")
        return cid

    extract = extract_cv(raw)
    base_meta = _scalar_metadata(extract)

    section_texts = [extract.section_text(s) for s in SECTIONS]
    embeddings = embed_texts(section_texts)

    ids, docs, metas, embs = [], [], [], []
    for section, text, emb in zip(SECTIONS, section_texts, embeddings):
        ids.append(f"{cid}::{section}")
        docs.append(text)
        metas.append({**base_meta, "candidate_id": cid, "section": section})
        embs.append(emb)

    store.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    print(f"Ingested {extract.name} ({cid}) — {len(SECTIONS)} sections")
    return cid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest CV PDFs into ChromaDB")
    parser.add_argument("--pdf", default="data/raw", help="PDF file or directory")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max PDFs to ingest")
    args = parser.parse_args()

    store = VectorStore(args.db_path, args.collection)

    if os.path.isdir(args.pdf):
        pdfs = [
            os.path.join(args.pdf, f)
            for f in sorted(os.listdir(args.pdf))
            if f.lower().endswith(".pdf")
        ]
    else:
        pdfs = [args.pdf]

    if args.limit:
        pdfs = pdfs[: args.limit]

    for pdf in pdfs:
        ingest_pdf(pdf, store)

    print(f"\nDone — {len(pdfs)} CV(s) ingested")
