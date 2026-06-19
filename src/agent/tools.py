"""LangChain @tool definitions for the recruiter search agent — one per CV section.

Each tool runs a semantic search via the shared RAG client singleton and returns the raw
results to the agent. The agent's structured output contains the aggregated
sections/candidates data (no per-request accumulator needed)."""

from langchain.tools import tool

from src.rag import get_rag_client


@tool
def search_skills(sub_query: str) -> list:
    """Search candidate SKILLS. Pass only skill/technology terms."""
    return get_rag_client().search(sub_query, where={"section": "skills"})


@tool
def search_experience(sub_query: str) -> list:
    """Search candidate EXPERIENCE (roles, seniority, domains). Expand transferable terms."""
    return get_rag_client().search(sub_query, where={"section": "experience"})


@tool
def search_role(sub_query: str) -> list:
    """Search normalized ROLE TITLE(s). Pass canonical job titles."""
    return get_rag_client().search(sub_query, where={"section": "role_title"})


@tool
def search_education(sub_query: str) -> list:
    """Search EDUCATION (degree/field/institution). Only if the query mentions education."""
    return get_rag_client().search(sub_query, where={"section": "education"})


TOOLS = [search_skills, search_experience, search_role, search_education]
