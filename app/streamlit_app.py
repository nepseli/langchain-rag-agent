"""
Streamlit UI — Document RAG Agent
Run with: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from config import settings
from core.agent.graph import build_graph
from core.ingestion.pipeline import delete_document, ingest_document
from core.retrieval.metadata_store import MetadataStore
from core.retrieval.vector_store import VectorStore
from models.schemas import ChatMessage, SourceRef

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Document RAG Agent", page_icon="📄", layout="wide")

# ─── CSS tweaks ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.source-chip {
    display: inline-block; background: #f0f2f6; border-radius: 12px;
    padding: 2px 10px; font-size: 0.78em; margin: 2px 3px; color: #444;
}
.calc-code {
    font-size: 0.82em; background: #f8f9fa;
    border-left: 3px solid #0066cc; padding: 8px 12px; border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─── Shared resource initialisation ─────────────────────────────────────────

@st.cache_resource
def get_stores():
    migrations_dir = Path(__file__).parent.parent / "db" / "migrations"
    meta = MetadataStore(settings.db_path, migrations_dir)
    vec = VectorStore(settings.chroma_path, settings.embedding_model, settings.openai_api_key)
    return meta, vec

@st.cache_resource
def get_graph(_meta, _vec):
    return build_graph(_meta, _vec, streaming=False)

meta_store, vector_store = get_stores()
rag_graph = get_graph(meta_store, vector_store)

def _init_session():
    if "session_id" not in st.session_state:
        sid = str(uuid.uuid4())
        st.session_state.session_id = sid
        meta_store.create_session(sid)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph_history" not in st.session_state:
        st.session_state.graph_history = []
    if "processed_uploads" not in st.session_state:
        st.session_state.processed_uploads = set()

_init_session()

# ─── Rendering helpers ───────────────────────────────────────────────────────

def _render_sources(sources: list[dict]):
    if not sources:
        return
    html = "".join(
        f'<span class="source-chip">📄 {s["doc_name"]}' +
        (f', p.{s["page_number"]}' if s.get("page_number") else "") + "</span>"
        for s in sources
    )
    st.markdown(f"**Sources:** {html}", unsafe_allow_html=True)

def _render_code(code: str | None):
    if not code:
        return
    with st.expander("🔢 Calculation code"):
        st.markdown(f'<div class="calc-code"><pre>{code}</pre></div>', unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Documents")

    uploaded = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "xlsx", "xls", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        import tempfile, os
        new_files = [f for f in uploaded if f.name not in st.session_state.processed_uploads]

        if new_files:
            for up_file in new_files:
                st.session_state.processed_uploads.add(up_file.name)

                existing = [
                    d for d in meta_store.list_documents()
                    if d.name == up_file.name and d.status == "indexed"
                ]
                if existing:
                    continue

                with st.status(f"Indexing {up_file.name}…", expanded=True) as status_box:
                    suffix = Path(up_file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(up_file.read())
                        tmp_path = Path(tmp.name)

                    def _progress(msg: str):
                        status_box.write(msg)

                    doc = ingest_document(
                        tmp_path, meta_store, vector_store, _progress,
                        original_name=up_file.name,
                    )
                    os.unlink(tmp_path)

                    if doc.status == "indexed":
                        status_box.update(label=f"✅ {up_file.name}", state="complete")
                    else:
                        status_box.update(label=f"❌ {up_file.name}: {doc.error_msg}", state="error")

            st.rerun()

    st.divider()
    st.subheader("Indexed Documents")

    docs = meta_store.list_documents()
    if not docs:
        st.caption("No documents yet. Upload one above.")
    else:
        for doc in docs:
            icon = {"indexed": "✅", "pending": "⏳", "error": "❌"}.get(doc.status, "❓")
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"{icon} **{doc.name}**")
                if doc.chunk_count:
                    st.caption(f"{doc.chunk_count} chunks · {doc.file_type.upper()}")
                if doc.status == "error":
                    st.caption(f"Error: {doc.error_msg}")
            with col2:
                if st.button("🗑", key=f"del_{doc.id}", help="Delete"):
                    delete_document(doc.id, meta_store, vector_store)
                    st.rerun()

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.graph_history = []
            sid = str(uuid.uuid4())
            st.session_state.session_id = sid
            meta_store.create_session(sid)
            st.rerun()
    with col_b:
        if st.button("🗑 Delete all docs", use_container_width=True, type="primary"):
            for d in meta_store.list_documents():
                delete_document(d.id, meta_store, vector_store)
            st.session_state.processed_uploads = set()
            st.rerun()

# ─── Main chat area ──────────────────────────────────────────────────────────

st.title("💬 Financial Document Q&A")
st.caption("Ask questions about your uploaded documents. Supports calculations, comparisons, and factual lookups.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_sources(msg.get("sources", []))
            _render_code(msg.get("generated_code"))

if query := st.chat_input("Ask a question about your documents…"):
    if not [d for d in meta_store.list_documents() if d.status == "indexed"]:
        st.warning("Please upload and index at least one document first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = rag_graph.invoke({
                    "query": query,
                    "session_id": st.session_state.session_id,
                    "conversation_history": st.session_state.graph_history,
                    "retrieved_chunks": [],
                    "table_context": [],
                    "sources": [],
                    "query_type": "",
                    "rewritten_query": "",
                    "generated_code": None,
                    "code_execution_result": None,
                    "final_answer": "",
                    "needs_clarification": False,
                    "clarification_question": "",
                })
                answer = result["final_answer"]
                sources = result.get("sources", [])
                generated_code = result.get("generated_code")
                st.session_state.graph_history = result.get("conversation_history", [])
            except Exception as exc:
                answer = f"An error occurred: {exc}"
                sources = []
                generated_code = None

        st.markdown(answer)
        sources_dicts = [s.model_dump() for s in sources] if sources else []
        _render_sources(sources_dicts)
        _render_code(generated_code)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources_dicts,
        "generated_code": generated_code,
    })

    meta_store.save_message(ChatMessage(
        session_id=st.session_state.session_id,
        role="user",
        content=query,
    ))
    meta_store.save_message(ChatMessage(
        session_id=st.session_state.session_id,
        role="assistant",
        content=answer,
        sources=[SourceRef(**s) for s in sources_dicts],
        generated_code=generated_code,
    ))
