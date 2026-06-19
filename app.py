"""Recruiter query engine (thin client over the FastAPI /search endpoint).

Query in -> ranked list -> drill into why. Per-vector scores and soft-passed
filters stay visible, never hidden.
"""

import html
import requests
import streamlit as st

from src.config import get_settings

API = get_settings().API_BASE_URL

st.set_page_config(page_title="resume-search", layout="wide")

# --- dark theme + card styling (matches the recruiter mockup) ---
st.markdown(
    """
    <style>
      .stApp { background: #0f1419; }
      .block-container { max-width: 1100px; }
      .cv-card { background:#161b22; border:1px solid #232b36; border-radius:14px;
                 padding:20px 24px; margin-bottom:16px; }
      .cv-head { display:flex; justify-content:space-between; align-items:baseline; }
      .cv-name { font-size:1.25rem; font-weight:700; color:#e6edf3; }
      .cv-overall { font-family:ui-monospace,monospace; font-size:1.25rem; font-weight:700; }
      .cv-vectors { font-family:ui-monospace,monospace; color:#7d8590; margin:10px 0 4px; }
      .cv-vectors b { color:#e6edf3; }
      .cv-why { color:#c9d1d9; margin:10px 0; line-height:1.5; }
      .cv-why b { color:#e6edf3; }
      .cv-filters { font-family:ui-monospace,monospace; margin-top:6px; }
      .f-pass { color:#34d399; margin-right:18px; }
      .f-soft { color:#fbbf24; margin-right:18px; }
      .chips { display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 18px; }
      .chip { font-family:ui-monospace,monospace; font-size:0.8rem;
              padding:4px 12px; border-radius:999px; }
      .chip-active { background:#0d3a2a; color:#34d399; border:1px solid #34d399; }
      .chip-empty  { background:transparent; color:#7d8590; border:1px solid #30363d; }
    </style>
    """,
    unsafe_allow_html=True,
)


def score_color(v: int) -> str:
    return "#34d399" if v >= 85 else "#fbbf24" if v >= 70 else "#7d8590"


def render_card(c: dict) -> str:
    s = c["scores"]
    # Raw per-vector scores (never penalized). "—" = section the agent didn't search
    # (not-scored != scored-zero). The filter penalty lives only in the overall below.
    def cell(section: str) -> str:
        return f"<b>{s[section]}</b>" if section in s else "<b>—</b>"

    vectors = (
        f"skills {cell('skills')}&nbsp;&nbsp; "
        f"experience {cell('experience')}&nbsp;&nbsp; "
        f"role {cell('role_title')}&nbsp;&nbsp; "
        f"education {cell('education')}"
    )
    chips = []
    for f in c["filters"]:
        cls = "f-pass" if f["passed"] else "f-soft"
        mark = "✓" if f["passed"] else "✗"
        note = f" — {html.escape(f['note'])}" if f.get("note") else ""
        chips.append(f"<span class='{cls}'>{mark} {html.escape(f['label'])}{note}</span>")
    return f"""
    <div class="cv-card">
      <div class="cv-head">
        <span class="cv-name">{html.escape(c['name'])}</span>
        <span class="cv-overall" style="color:{score_color(c['overall'])}">{c['overall']} / 100</span>
      </div>
      <div class="cv-vectors">{vectors}</div>
      <div class="cv-why"><b>Why:</b> {html.escape(c['why'])}</div>
      <div class="cv-filters">{''.join(chips)}</div>
    </div>
    """


# --- sidebar: corpus controls ---
with st.sidebar:
    st.header("Corpus")
    try:
        stats = requests.get(f"{API}/stats", timeout=10).json()
        st.metric("Candidates indexed", stats.get("candidates", 0))
    except requests.RequestException:
        st.error("API unreachable.")
    if st.button("Ingest data/raw"):
        with st.spinner("Extracting + embedding CVs..."):
            r = requests.post(f"{API}/ingest", timeout=1800)
        st.success(f"Indexed {r.json().get('candidates', 0)} candidates")
    top_k = st.slider("Results", 1, 20, 10)

# --- query bar ---
st.markdown("#### resume-search")
query = st.text_input(
    "query", label_visibility="collapsed",
    placeholder="Senior backend engineer, Python, Ulaanbaatar, 5+ years",
)
if st.button("Search", type="primary") and query.strip():
    payload = {"query": query, "top_k": top_k}
    with st.spinner("Searching..."):
        try:
            resp = requests.post(f"{API}/search", json=payload, timeout=300).json()
        except requests.RequestException as e:
            st.error(f"Search failed: {e}")
            st.stop()

    # --- filter chips ---
    extracted = resp.get("filters", {})
    FILTER_LABELS = {
        "location": ("location", lambda v: f"location: {v}"),
        "min_years": ("exp", lambda v: f"exp: {v:g}+ yrs"),
        "language": ("language", lambda v: f"language: {v}"),
        "availability": ("availability", lambda v: f"availability: {v}"),
    }
    chips_html = ""
    for key, (_, fmt) in FILTER_LABELS.items():
        val = extracted.get(key)
        if val is not None:
            chips_html += f'<span class="chip chip-active">{html.escape(fmt(val))}</span>'
        else:
            label = key if key != "min_years" else "exp"
            chips_html += f'<span class="chip chip-empty">{label}</span>'
    st.markdown(f'<div class="chips">{chips_html}</div>', unsafe_allow_html=True)

    candidates = resp.get("candidates", [])
    if not candidates:
        st.warning("No candidates indexed yet — use 'Ingest data/raw' in the sidebar.")
    for c in candidates:
        st.markdown(render_card(c), unsafe_allow_html=True)
