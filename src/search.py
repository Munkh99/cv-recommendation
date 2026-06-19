"""Recruiter search: agentic decomposition + deterministic semantic ranking.

A tool-calling agent (src/agent) splits the query into per-section sub-queries and
searches each section vector. Here we blend the per-section scores into a semantic score
(weights renormalized over searched sections), gate on a relevance threshold for clean
zero-results, rank, and for the top-K generate the rationale — which also assesses the
hard filters (location/years/language/availability) as structured output. Tracing is the
Langfuse LangChain callback inside the agent/LLM calls.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.schema import SECTIONS, CVExtract, CandidateResult, SearchRequest, SearchResponse
from src.config import get_settings
from src.rag import get_rag_client
from src.llm import generate_why
from src.agent import agentic_retrieve

log = logging.getLogger(__name__)

# Blend weights for the semantic score. Skills + experience dominate; normalized title
# and education are supporting signals. Renormalized over the sections actually searched.
SECTION_WEIGHTS = {
    "skills": 0.35,
    "experience": 0.35,
    "role_title": 0.20,
    "education": 0.10,
}


def _normalized_weights(searched: set) -> dict[str, float]:
    """Renormalize SECTION_WEIGHTS over the sections the agent actually searched, so
    skipping a section (e.g. education) redistributes its weight instead of penalizing."""
    present = {s: SECTION_WEIGHTS[s] for s in searched if s in SECTION_WEIGHTS}
    total = sum(present.values()) or 1.0
    return {s: w / total for s, w in present.items()}


def _section_floors(scored: dict, searched: set) -> dict[str, int]:
    """Per-section floor = lowest score returned in that section's top-N. A candidate
    absent from a section's top-N is below this, so it's a fair stand-in for the missing
    cell — better than 0, which would over-penalize."""
    floors: dict[str, int] = {}
    for s in searched:
        vals = [e["scores"][s] for e in scored.values() if s in e["scores"]]
        floors[s] = min(vals) if vals else 0
    return floors


def _log_score_spread(scored: dict, searched: set) -> None:
    """Log per-section min/median/max — data for calibrating RELEVANCE_THRESHOLD."""
    for s in sorted(searched):
        vals = sorted(e["scores"][s] for e in scored.values() if s in e["scores"])
        if vals:
            med = vals[len(vals) // 2]
            log.info("score spread [%s]: n=%d min=%d median=%d max=%d", s, len(vals), vals[0], med, vals[-1])


def _why_query(query: str, filters: dict) -> str:
    """Append agent-extracted hard constraints to the query so the why-LLM assesses them."""
    extras = []
    if filters.get("location"):
        extras.append(f"location: {filters['location']}")
    if filters.get("min_years") is not None:
        extras.append(f"minimum years of experience: {filters['min_years']:g}")
    if filters.get("language"):
        extras.append(f"language: {filters['language']}")
    if filters.get("availability"):
        extras.append(f"availability: {filters['availability']}")
    if extras:
        return f"{query}\nConstraints: " + ", ".join(extras)
    return query


def _fallback_scores(query: str, rag) -> dict:
    """Whole-query multi-vector search across all sections (used if the agent calls
    no tools). Same shape as agentic_retrieve's candidates."""
    candidates: dict = {}
    for section in SECTIONS:
        for it in rag.search(query, where={"section": section}):
            cid = it["candidate_id"]
            entry = candidates.setdefault(cid, {"meta": it["meta"], "scores": {}})
            entry["scores"][section] = it["score"]
    return candidates


def search(req: SearchRequest) -> SearchResponse:
    rag = get_rag_client()
    if rag.store.count() == 0:
        return SearchResponse(query=req.query, candidates=[])

    # Agent decomposes the query, searches sections, and extracts hard filters.
    scored, searched, filters = agentic_retrieve(req.query)

    # Fallback: agent called no search tools -> score the whole query across all sections.
    if not scored:
        scored = _fallback_scores(req.query, rag)
        searched = set(SECTIONS)
        filters = {}

    settings = get_settings()
    if settings.LOG_SCORE_SPREAD:
        _log_score_spread(scored, searched)

    weights = _normalized_weights(searched)
    floors = _section_floors(scored, searched)
    threshold = settings.RELEVANCE_THRESHOLD

    ranked = []
    for cid, entry in scored.items():
        scores = entry["scores"]
        # Semantic score = weighted blend over searched sections; missing cell -> floor.
        semantic = sum(scores.get(s, floors[s]) * w for s, w in weights.items())
        # Relevance gate: drop bad matches; if none clear it, the query returns zero cleanly.
        if semantic < threshold:
            continue
        ranked.append((round(semantic), cid, entry["meta"], scores))

    ranked.sort(key=lambda r: r[0], reverse=True)
    top = ranked[: req.top_k]

    why_query = _why_query(req.query, filters)

    def _why_task(item):
        overall, cid, meta, scores = item
        extract = CVExtract.model_validate_json(meta["extract_json"])
        wr = generate_why(why_query, extract, scores)
        return CandidateResult(
            candidate_id=cid,
            name=meta.get("name", cid),
            role_title=meta.get("role_title", ""),
            overall=overall,
            scores=scores,
            why=wr.why,
            filters=wr.filters,
        )

    results: dict[int, CandidateResult] = {}
    with ThreadPoolExecutor(max_workers=len(top) or 1) as pool:
        futures = {pool.submit(_why_task, item): i for i, item in enumerate(top)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    candidates = [results[i] for i in range(len(top))]

    return SearchResponse(query=req.query, filters=filters, candidates=candidates)
