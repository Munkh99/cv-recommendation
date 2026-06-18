import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.oauth2 import service_account

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_EMBEDDING_MODEL = "text-embedding-004"


def get_embedding_client(service_account_path: str, project_id: str, location: str = "us-central1") -> TextEmbeddingModel:
    credentials = service_account.Credentials.from_service_account_file(
        service_account_path, scopes=_SCOPES
    )
    vertexai.init(project=project_id, location=location, credentials=credentials)
    return TextEmbeddingModel.from_pretrained(_EMBEDDING_MODEL)


def embed_texts(model: TextEmbeddingModel, texts: list[str], batch_size: int = 20) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = [TextEmbeddingInput(t, "RETRIEVAL_DOCUMENT") for t in batch]
        results = model.get_embeddings(inputs)
        all_embeddings.extend(e.values for e in results)
    return all_embeddings


def embed_query(model: TextEmbeddingModel, query: str) -> list[float]:
    inputs = [TextEmbeddingInput(query, "RETRIEVAL_QUERY")]
    results = model.get_embeddings(inputs)
    return results[0].values
