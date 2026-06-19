"""Gemini chat LLM via the LangChain wrapper (ChatGoogleGenerativeAI on Vertex AI).

This is the single LLM client used everywhere: the planner agent, CV extraction, and
the why-text. All calls pass the Langfuse callback so they're traced. Embeddings use a
separate client (see src/embeddings.py)."""

import logging
from functools import lru_cache
from typing import Any

from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings
from src.schema import CVExtract, WhyResult
from src.tracing import langchain_callbacks

logger = logging.getLogger(__name__)

_ALLOWED_ENVIRONMENTS = {"dev", "staging", "prod"}


@lru_cache(maxsize=1)
def build_credentials() -> service_account.Credentials:
    s = get_settings()
    path = s.GOOGLE_SERVICE_ACCOUNT_JSON_PATH or s.GOOGLE_APPLICATION_CREDENTIALS
    if not path:
        raise KeyError(
            "Set GOOGLE_SERVICE_ACCOUNT_JSON_PATH or GOOGLE_APPLICATION_CREDENTIALS "
            "to a service account JSON file path."
        )
    return service_account.Credentials.from_service_account_file(
        path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )


def build_google_genai_llm(
    *,
    model: str,
    temperature: float = 0.0,
    thinking_level: str | None = None,
    credentials=None,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """Build a ChatGoogleGenerativeAI instance on Vertex AI with service-account
    credentials and standardized cost-tracking labels. Extra kwargs (e.g.
    max_output_tokens, labels) are forwarded to the model."""
    s = get_settings()
    if credentials is None:
        credentials = build_credentials()

    env = s.PROJECT_ENVIRONMENT
    if env not in _ALLOWED_ENVIRONMENTS:
        raise ValueError(
            f"Invalid PROJECT_ENVIRONMENT '{env}'. Expected one of: {', '.join(sorted(_ALLOWED_ENVIRONMENTS))}."
        )

    labels: dict[str, str] = {"env": env, "project": s.PROJECT_NAME}
    labels.update(kwargs.pop("labels", {}))

    if thinking_level is not None:
        kwargs["thinking_level"] = thinking_level

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        vertexai=True,
        project=s.GOOGLE_CLOUD_PROJECT,
        location=s.GEMINI_LOCATION,
        credentials=credentials,
        labels=labels,
        **kwargs,
    )


_EXTRACT_INSTRUCTION = (
    "You parse a candidate CV into structured fields for a recruiter search engine.\n"
    "- role_title: normalize to a canonical title (e.g. 'SWE', 'Backend Developer' -> "
    "'Software Engineer'). Use the most senior/recent role.\n"
    "- experience: prose covering roles, responsibilities, and seniority signal.\n"
    "- skills: prose describing demonstrated skills, NOT a comma-separated keyword list.\n"
    "- education: degrees, institutions, fields.\n"
    "- years_experience: total professional years as a number; null if unclear.\n"
    "- location, languages, availability: extract only if stated; otherwise leave empty/null.\n"
    "Do not invent facts."
)


def extract_cv(raw_text: str) -> CVExtract:
    """LLM splits a raw CV into the four embeddable sections + filter metadata."""
    s = get_settings()
    llm = build_google_genai_llm(
        model=s.GEMINI_MODEL, thinking_level=s.THINKING_LEVEL, max_output_tokens=s.MAX_TOKENS
    )
    structured = llm.with_structured_output(CVExtract)
    return structured.invoke(
        [("system", _EXTRACT_INSTRUCTION), ("human", f"CV text:\n\n{raw_text}")],
        config={"callbacks": langchain_callbacks()},
    )


_WHY_INSTRUCTION = (
    "You explain why a candidate matched a recruiter query AND assess hard constraints.\n"
    "1) why: 1-2 sentences citing concrete signals (titles, years, domains). Be honest "
    "about gaps; mention any constraint met or missed. Do not repeat the numeric scores.\n"
    "2) filters: for EACH hard constraint stated in the query (location, minimum years, "
    "language(s), availability), return one check against the candidate's data — label "
    "(e.g. 'experience 5+'), passed (true/false), soft (true if missed but close/kept), "
    "and a short note (e.g. 'soft pass, 4.5 yrs'). If the query states no constraints, "
    "return an empty list. Judge only from the candidate fields provided."
)


def generate_why(query: str, extract: CVExtract, scores: dict[str, int]) -> WhyResult:
    """Structured: rationale prose + per-constraint filter assessment (covers filters in
    both the prose and the structured `filters` list)."""
    s = get_settings()
    llm = build_google_genai_llm(model=s.GEMINI_MODEL, thinking_level=s.THINKING_LEVEL)
    structured = llm.with_structured_output(WhyResult)
    context = (
        f"Query: {query}\n\n"
        f"Candidate: {extract.name} ({extract.role_title})\n"
        f"Experience: {extract.experience}\n"
        f"Skills: {extract.skills}\n"
        f"Education: {extract.education}\n"
        f"Location: {extract.location}\n"
        f"Years of experience: {extract.years_experience}\n"
        f"Languages: {extract.languages}\n"
        f"Availability: {extract.availability}\n"
        f"Per-vector scores: {scores}"
    )
    return structured.invoke(
        [("system", _WHY_INSTRUCTION), ("human", context)],
        config={"callbacks": langchain_callbacks()},
    )
