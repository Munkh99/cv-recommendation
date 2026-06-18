from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings


def build_credentials() -> service_account.Credentials:
    settings = get_settings()
    return service_account.Credentials.from_service_account_file(
        settings.GOOGLE_SERVICE_ACCOUNT_JSON_PATH,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


def build_llm() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        temperature=0.0,
        max_output_tokens=settings.MAX_TOKENS,
        thinking_level=settings.THINKING_LEVEL,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        vertexai=True,
        project=settings.GOOGLE_CLOUD_PROJECT,
        location=settings.GEMINI_LOCATION,
        credentials=build_credentials(),
        labels={
            "env": settings.PROJECT_ENVIRONMENT,
            "project": settings.PROJECT_NAME,
        },
    )
