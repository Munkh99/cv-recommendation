import os
import streamlit as st
from src.embeddings import get_embedding_client
from src.ingest import get_collection, ingest_pdf
from src.rag import run_rag

st.set_page_config(page_title="RAG Search", layout="wide")
st.title("RAG Document Search")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    sa_path = st.text_input("Service Account JSON", value="service_account.json")
    project_id = st.text_input("GCP Project ID")
    location = st.text_input("Location", value="us-central1")
    collection_name = st.text_input("Collection Name", value="documents")
    db_path = st.text_input("ChromaDB Path", value="./chroma_db")
    top_k = st.slider("Top K results", 1, 10, 5)

    st.divider()
    st.header("Ingest PDFs")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Ingest") and uploaded_files:
        if not project_id or not os.path.exists(sa_path):
            st.error("Set GCP Project ID and valid service account path first.")
        else:
            with st.spinner("Ingesting..."):
                model = get_embedding_client(sa_path, project_id, location)
                collection = get_collection(db_path, collection_name)
                os.makedirs("./uploads", exist_ok=True)
                for f in uploaded_files:
                    tmp = os.path.join("./uploads", f.name)
                    with open(tmp, "wb") as out:
                        out.write(f.read())
                    ingest_pdf(tmp, collection, model)
            st.success(f"Ingested {len(uploaded_files)} file(s)")


# --- Main area ---
ready = project_id and os.path.exists(sa_path)

if not ready:
    st.info("Configure your GCP Project ID and service account path in the sidebar to get started.")
    st.stop()


@st.cache_resource
def _init(sa, proj, loc, coll, db):
    model = get_embedding_client(sa, proj, loc)
    collection = get_collection(db, coll)
    return model, collection


embedding_model, collection = _init(sa_path, project_id, location, collection_name, db_path)

query = st.text_input("Ask a question about your documents:", placeholder="e.g. What is the main topic?")
search = st.button("Search", type="primary")

if search and query.strip():
    with st.spinner("Searching..."):
        result = run_rag(query, collection, embedding_model, top_k=top_k)

    if result["answer"]:
        st.subheader("Answer")
        st.write(result["answer"])
        st.divider()

    st.subheader(f"Top {top_k} Relevant Chunks")
    if not result["chunks"]:
        st.warning("No results found. Make sure you have ingested documents first.")
    for i, chunk in enumerate(result["chunks"], 1):
        with st.expander(
            f"[{i}] {chunk['source']} — Page {chunk['page']}  |  score: {chunk['score']}"
        ):
            st.text(chunk["text"])
