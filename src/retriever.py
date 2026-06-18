import chromadb
from src.embeddings import embed_query


def retrieve(
    query: str,
    collection: chromadb.Collection,
    embedding_model,
    top_k: int = 5,
) -> list[dict]:
    query_vec = embed_query(embedding_model, query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "text": doc,
                "source": meta.get("source", ""),
                "page": meta.get("page", ""),
                "score": round(1 - dist, 4),
            }
        )
    return chunks
