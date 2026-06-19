"""RAG client — natural-language semantic search over the vector store.

search() embeds the query and returns ranked candidate items. `where` scopes the
search (e.g. to one CV section); soft-pass filtering / blending happens upstream in
src/search.py so scoring stays transparent."""

from functools import lru_cache

from src.config import get_settings
from src.vectorstore import VectorStore
from src.embeddings import get_embedder


def _score(distance: float) -> int:
    """Cosine distance -> 0..100 similarity."""
    return round(max(0.0, 1.0 - distance) * 100)


class RagClient:
    def __init__(self, store: VectorStore | None = None, embedder=None):
        self.store = store or VectorStore()
        self.embedder = embedder or get_embedder()

    def search(self, query: str, where: dict | None = None, top_k: int | None = None) -> list[dict]:
        """Embed the query and run semantic search. Returns
        [{candidate_id, score (0..100), meta}], best first. Retrieval is bounded to
        top-N (Chroma has no score-threshold query); the relevance gate is applied
        downstream in src/search.py."""
        vec = self.embedder.embed_query(query)
        n = top_k or get_settings().TOP_N_PER_SECTION
        res = self.store.query(vec, where=where, n_results=n)
        items = []
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            items.append(
                {"candidate_id": meta["candidate_id"], "score": _score(dist), "meta": meta}
            )
        return items


@lru_cache(maxsize=1)
def get_rag_client() -> RagClient:
    """Process-wide singleton — embedder + Chroma client initialize once, reused across
    all requests/tools. The client is stateless, so sharing it is safe."""
    return RagClient()
