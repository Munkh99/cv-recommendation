"""Recruiter search planner agent (LangChain create_agent).

Decomposes a messy free-text query into per-section sub-queries and calls the section
search tools. Returns the per-section score matrix for deterministic blending in
src/search.py. Traced via the Langfuse LangChain callback."""

from functools import lru_cache
from pathlib import Path

from langchain.agents import create_agent

from src.config import get_settings
from src.llm import build_google_genai_llm
from src.rag import get_rag_client
from src.schema import PlannerOutput
from src.tracing import langchain_callbacks
from src.agent.tools import TOOLS

_PROMPT = (Path(__file__).resolve().parent / "prompt.txt").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _agent():
    """Singleton compiled agent — stateless, safe to reuse across requests."""
    s = get_settings()
    model = build_google_genai_llm(model=s.GEMINI_MODEL, thinking_level=s.THINKING_LEVEL)
    return create_agent(model=model, tools=TOOLS, system_prompt=_PROMPT, response_format=PlannerOutput)


def agentic_retrieve(query: str) -> tuple[dict, set, dict]:
    """Run the planner agent.

    Returns (candidates, searched_sections, filters) where:
      candidates = {cid: {"meta": ..., "scores": {section: 0..100}}}
      searched_sections = set of section names the agent searched
      filters = {location, min_years, language, availability} — extracted from query,
                values are None when the query didn't mention that constraint
    """
    agent = _agent()

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"callbacks": langchain_callbacks(), "recursion_limit": 12},
    )

    out: PlannerOutput = result["structured_response"]

    # The LLM may key sections by tool name (search_skills) instead of section name (skills).
    _TOOL_TO_SECTION = {
        "search_skills": "skills",
        "search_experience": "experience",
        "search_role": "role_title",
        "search_education": "education",
    }
    out.sections = {
        _TOOL_TO_SECTION.get(k, k): v for k, v in out.sections.items()
    }

    # Guard: if the LLM returned 0-1 floats instead of 0-100 ints, rescale.
    all_scores = [s for sec in out.sections.values() for s in sec.values()]
    if all_scores and max(all_scores) <= 1.0:
        out.sections = {
            sec: {cid: round(s * 100) for cid, s in scores.items()}
            for sec, scores in out.sections.items()
        }

    all_cids = {cid for scores in out.sections.values() for cid in scores}
    meta_map = get_rag_client().store.fetch_meta(list(all_cids))

    candidates: dict = {}
    for section, scores in out.sections.items():
        for cid, score in scores.items():
            entry = candidates.setdefault(cid, {"meta": meta_map.get(cid, {}), "scores": {}})
            entry["scores"][section] = score

    filters = {
        "location": out.location,
        "min_years": out.min_years,
        "language": out.language,
        "availability": out.availability,
    }
    return candidates, set(out.sections.keys()), filters
