import os
import hashlib
import argparse
import fitz  # pymupdf
import chromadb
from src.embeddings import get_embedding_client, embed_texts


def load_pdf(path: str) -> list[dict]:
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"text": text, "page": i + 1, "source": os.path.basename(path)})
    doc.close()
    return pages


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def _chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{source}_{index}_{digest}"


def ingest_pdf(
    pdf_path: str,
    collection: chromadb.Collection,
    embedding_model,
    chunk_size: int = 500,
    overlap: int = 50,
) -> int:
    pages = load_pdf(pdf_path)
    chunks, metadatas, ids = [], [], []

    for page in pages:
        for chunk in chunk_text(page["text"], chunk_size, overlap):
            idx = len(chunks)
            chunks.append(chunk)
            metadatas.append({"source": page["source"], "page": page["page"]})
            ids.append(_chunk_id(page["source"], idx, chunk))

    if not chunks:
        print(f"[warn] No text extracted from {pdf_path}")
        return 0

    embeddings = embed_texts(embedding_model, chunks)
    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
    print(f"Ingested {len(chunks)} chunks from {os.path.basename(pdf_path)}")
    return len(chunks)


def get_collection(db_path: str = "./chroma_db", collection_name: str = "documents") -> chromadb.Collection:
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF files into ChromaDB")
    parser.add_argument("--pdf", required=True, help="PDF file or directory of PDFs")
    parser.add_argument("--sa", default="service_account.json", help="Service account JSON path")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--collection", default="documents")
    parser.add_argument("--db-path", default="./chroma_db")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    model = get_embedding_client(args.sa, args.project, args.location)
    collection = get_collection(args.db_path, args.collection)

    if os.path.isdir(args.pdf):
        pdf_files = [os.path.join(args.pdf, f) for f in os.listdir(args.pdf) if f.lower().endswith(".pdf")]
    else:
        pdf_files = [args.pdf]

    total = 0
    for pdf in pdf_files:
        total += ingest_pdf(pdf, collection, model, args.chunk_size, args.overlap)

    print(f"\nDone — {total} total chunks stored in '{args.collection}'")
