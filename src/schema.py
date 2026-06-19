"""Shared data models for ingestion, search, and the API/UI boundary."""

from pydantic import BaseModel, Field

# The four embedded sections of a CV. Keys are used as Chroma metadata,
# scoring weights, and per-vector score labels in the UI.
SECTIONS = ("skills", "experience", "role_title", "education")


class PlannerOutput(BaseModel):
    """Structured output of the planner agent.

    sections:     {section_name: {candidate_id: score 0-100}}
    Hard filters extracted from the free-text query (None = not mentioned).
    """

    sections: dict[str, dict[str, float]] = Field(
        description="Per-section candidate scores. Scores MUST be integers in the range 0-100 "
                    "exactly as returned by the search tools — do NOT normalize to 0-1."
    )
    location: str | None = None
    min_years: float | None = None
    language: str | None = None
    availability: str | None = None


class CVExtract(BaseModel):
    """Structured CV produced by the LLM during ingestion."""

    name: str = Field(description="Candidate full name.")
    role_title: str = Field(
        description="Normalized current/target role title, e.g. 'Software Engineer'."
    )
    experience: str = Field(
        description="Roles, responsibilities, and seniority signal as prose."
    )
    education: str = Field(description="Degrees, institutions, and fields of study.")
    skills: str = Field(description="Extracted skills as prose, not a keyword list.")

    # Filterable metadata (best-effort; null when not stated in the CV).
    location: str | None = Field(default=None, description="City or region.")
    years_experience: float | None = Field(
        default=None, description="Total years of professional experience."
    )
    languages: list[str] = Field(
        default_factory=list, description="Spoken/written languages."
    )
    availability: str | None = Field(
        default=None, description="Availability/notice period if stated."
    )

    def section_text(self, section: str) -> str:
        return getattr(self, section)


class FilterCheck(BaseModel):
    """Outcome of one filter against one candidate. Never hidden in the UI."""

    label: str  # e.g. "experience 5+"
    passed: bool
    soft: bool = False  # failed the hard threshold but kept (ranked down)
    note: str | None = None  # e.g. "soft pass, 4.5 yrs"


class WhyResult(BaseModel):
    """Structured output of the why-generation step: the rationale prose plus the
    per-constraint filter assessment (the LLM evaluates filters from the query + CV)."""

    why: str = Field(description="1-2 sentence rationale; mention any constraint met/missed.")
    filters: list[FilterCheck] = Field(
        default_factory=list,
        description="One entry per hard constraint stated in the query (location, years, "
        "language, availability). Empty if the query states none.",
    )


class CandidateResult(BaseModel):
    candidate_id: str
    name: str
    role_title: str
    overall: int  # 0-100
    scores: dict[str, int]  # per-section, keys == SECTIONS
    why: str
    filters: list[FilterCheck] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class SearchResponse(BaseModel):
    query: str
    filters: dict = {}   # agent-extracted: {location, min_years, language, availability}
    candidates: list[CandidateResult]
