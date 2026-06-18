import chromadb
from src.retriever import retrieve


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] Source: {chunk['source']}, Page {chunk['page']}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def run_rag(
    query: str,
    collection: chromadb.Collection,
    embedding_model,
    llm_client=None,
    top_k: int = 5,
) -> dict:
    chunks = retrieve(query, collection, embedding_model, top_k)
    context = build_context(chunks)

    answer = None
    if llm_client is not None:
        from src.llm import generate_answer
        answer = generate_answer(llm_client, context, query)

    return {"query": query, "answer": answer, "context": context, "chunks": chunks}
