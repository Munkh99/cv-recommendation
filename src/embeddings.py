"""Embeddings via the LangChain wrapper (GoogleGenerativeAIEmbeddings on Vertex AI).

Separate client from the chat LLM. embed_query uses RETRIEVAL_QUERY and embed_documents
uses RETRIEVAL_DOCUMENT task types automatically; dimensionality comes from config."""

from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import get_settings
from src.llm import build_credentials


@lru_cache(maxsize=1)
def get_embedder() -> GoogleGenerativeAIEmbeddings:
    s = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=s.EMBEDDING_MODEL,
        vertexai=True,
        project=s.GOOGLE_CLOUD_PROJECT,
        location=s.GEMINI_LOCATION,
        credentials=build_credentials(),
        output_dimensionality=s.EMBEDDING_DIMENSIONS,
    )


def embed_query(text: str) -> list[float]:
    return get_embedder().embed_query(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    embedder = get_embedder()
    return [embedder.embed_documents([t])[0] for t in texts]
