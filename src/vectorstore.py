"""ChromaDB client — connection + collection setup and low-level vector ops.

Embedded/persistent Chroma (no separate server). One record per (candidate, section);
cosine space. Used by ingestion (upsert) and the RAG client (query)."""

import chromadb

from src.config import get_settings


class VectorStore:
    def __init__(self, db_path: str | None = None, collection_name: str | None = None):
        s = get_settings()
        self._client = chromadb.PersistentClient(path=db_path or s.CHROMA_DB_PATH)
        self.collection = self._client.get_or_create_collection(
            name=collection_name or s.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, ids, documents, metadatas, embeddings) -> None:
        self.collection.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def query(self, embedding: list[float], where: dict | None = None, n_results: int | None = None):
        n = n_results or (self.collection.count() or 1)
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n,
            where=where,
            include=["metadatas", "distances"],
        )

    def fetch_meta(self, candidate_ids: list[str]) -> dict[str, dict]:
        """Return {candidate_id: metadata} for the given IDs by fetching the skills
        record for each (all sections share the same base metadata)."""
        ids = [f"{cid}::skills" for cid in candidate_ids]
        result = self.collection.get(ids=ids, include=["metadatas"])
        return {
            meta["candidate_id"]: meta
            for meta in result["metadatas"]
            if meta
        }

    def count(self) -> int:
        return self.collection.count()
