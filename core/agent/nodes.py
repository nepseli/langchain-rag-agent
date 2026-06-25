"""
LangGraph node functions -- each takes RAGState and returns a partial state update.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from config import settings
from core.agent.prompts import (
    CLARITY_CHECK_PROMPT,
    CLASSIFY_PROMPT,
    CODE_GEN_PROMPT,
    GENERATE_PROMPT,
    HYDE_PROMPT,
)
from core.agent.state import RAGState
from core.retrieval.metadata_store import MetadataStore
from core.retrieval.vector_store import VectorStore
from models.schemas import Chunk, SourceRef


# --- Shared LLM factory ------------------------------------------------------

def _llm(temperature: float = 0.0, streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=temperature,
        openai_api_key=settings.openai_api_key,
        streaming=streaming,
    )


# --- Node: check_clarity ------------------------------------------------------

def make_check_clarity_node(meta_store: MetadataStore):
    """
    Detects ambiguous queries (e.g. 'this invoice' when multiple docs exist)
    and short-circuits the graph with a clarifying question instead of guessing.
    """
    llm = _llm(temperature=0.0)
    chain = CLARITY_CHECK_PROMPT | llm

    def check_clarity(state: RAGState) -> dict:
        docs = meta_store.list_documents()
        indexed = [d for d in docs if d.status == "indexed"]

        # Single document -- no ambiguity possible
        if len(indexed) <= 1:
            return {"needs_clarification": False, "clarification_question": ""}

        doc_list = "\n".join(f"- {d.name}" for d in indexed)

        history = state.get("conversation_history", [])
        if history:
            recent = history[-4:]
            history_summary = "\n".join(
                f"{m.type.upper()}: {m.content[:200]}" for m in recent
            )
        else:
            history_summary = "(no prior conversation)"

        try:
            result = chain.invoke({
                "query": state["query"],
                "doc_list": doc_list,
                "history_summary": history_summary,
            })
            parsed = json.loads(result.content.strip())
            needs = bool(parsed.get("needs_clarification", False))
            question = parsed.get("question", "") if needs else ""
        except Exception:
            # On any parse failure, let the query through rather than blocking
            needs = False
            question = ""

        return {"needs_clarification": needs, "clarification_question": question}

    return check_clarity


def make_clarify_node():
    """
    Terminal node when the query needs clarification.
    Sets final_answer to the clarifying question so the UI displays it naturally.
    """
    def clarify(state: RAGState) -> dict:
        question = state.get(
            "clarification_question",
            "Could you please clarify which document you are referring to?",
        )
        new_messages = [
            HumanMessage(content=state["query"]),
            AIMessage(content=question),
        ]
        return {
            "final_answer": question,
            "sources": [],
            "conversation_history": new_messages,
        }
    return clarify


def route_after_clarity(state: RAGState) -> str:
    return "clarify" if state.get("needs_clarification") else "classify_query"


# --- Node: classify_query -----------------------------------------------------

def make_classify_node():
    llm = _llm(temperature=0.0)
    chain = CLASSIFY_PROMPT | llm

    def classify_query(state: RAGState) -> dict:
        result = chain.invoke({"query": state["query"]})
        raw = result.content.strip().lower()
        valid = {"calculation", "numeric", "comparative", "factual", "general"}
        query_type = raw if raw in valid else "general"
        return {"query_type": query_type}

    return classify_query


# --- Node: rewrite_query (HyDE) -----------------------------------------------

def make_rewrite_node():
    llm = _llm(temperature=0.3)
    chain = HYDE_PROMPT | llm

    def rewrite_query(state: RAGState) -> dict:
        result = chain.invoke({"query": state["query"]})
        return {"rewritten_query": result.content.strip()}

    return rewrite_query


# --- Node: retrieve ------------------------------------------------------------

def make_retrieve_node(vector_store: VectorStore):

    def retrieve(state: RAGState) -> dict:
        query_text = state.get("rewritten_query") or state["query"]
        query_type = state.get("query_type", "general")

        if query_type == "comparative":
            chunks = vector_store.search_per_doc(
                query=query_text,
                n_per_doc=settings.top_k_per_doc_comparative,
            )
        else:
            chunks = vector_store.search(
                query=query_text,
                n_results=settings.top_k,
                max_per_doc=settings.max_chunks_per_doc,
            )

        return {"retrieved_chunks": chunks}

    return retrieve


# --- Node: fetch_table_data ---------------------------------------------------

def make_fetch_table_node(meta_store: MetadataStore):

    def fetch_table_data(state: RAGState) -> dict:
        chunks: list[Chunk] = state.get("retrieved_chunks", [])
        table_ids = list({
            c.table_id for c in chunks
            if c.chunk_type == "table" and c.table_id
        })

        if not table_ids:
            return {"table_context": []}

        tables = meta_store.get_tables_by_ids(table_ids)
        table_context = [
            {
                "table_id":    t.id,
                "header":      t.header,
                "markdown":    t.markdown,
                "json_data":   t.json_data,
                "doc_id":      t.doc_id,
                "page_number": t.page_number,
            }
            for t in tables
        ]
        return {"table_context": table_context}

    return fetch_table_data


# --- Node: execute_calculation ------------------------------------------------

def make_execute_calculation_node():
    from langchain_experimental.tools import PythonREPLTool

    llm = _llm(temperature=0.0)
    code_chain = CODE_GEN_PROMPT | llm
    repl = PythonREPLTool()

    def execute_calculation(state: RAGState) -> dict:
        table_context: list[dict] = state.get("table_context", [])
        if not table_context:
            return {
                "generated_code": None,
                "code_execution_result": "No table data available for calculation.",
            }

        if len(table_context) == 1:
            table_json_str = f"records = {table_context[0]['json_data']}"
        else:
            parts = []
            for i, t in enumerate(table_context):
                header = t.get("header", f"Table {i}")
                parts.append(f"# {header}\nrecords_{i} = {t['json_data']}")
            table_json_str = "\n".join(parts)

        code_result = code_chain.invoke({
            "query": state["query"],
            "table_json": table_json_str,
        })
        generated_code = code_result.content.strip()
        full_code = f"{table_json_str}\n\n{generated_code}"

        try:
            execution_result = repl.run(full_code)
        except Exception as exc:
            execution_result = f"Execution error: {exc}"

        return {
            "generated_code": generated_code,
            "code_execution_result": execution_result,
        }

    return execute_calculation


# --- Node: generate -----------------------------------------------------------

def make_generate_node(streaming: bool = False):
    llm = _llm(temperature=0.2, streaming=streaming)
    chain = GENERATE_PROMPT | llm

    def generate(state: RAGState) -> dict:
        chunks: list[Chunk] = state.get("retrieved_chunks", [])
        table_context: list[dict] = state.get("table_context", [])
        code_result = state.get("code_execution_result")

        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        text_context = _format_text_context(text_chunks)
        table_ctx_str = _format_table_context(table_context)

        if code_result:
            calc_section = (
                "--- CALCULATION RESULT ---\n"
                + code_result
                + "\n(computed by executing pandas code against the raw table data)"
            )
        else:
            calc_section = ""

        history = state.get("conversation_history", [])

        result = chain.invoke({
            "query": state["query"],
            "text_context": text_context or "(no text context retrieved)",
            "table_context": table_ctx_str or "(no table context retrieved)",
            "calculation_section": calc_section,
            "conversation_history": history,
        })

        answer = result.content.strip()
        sources = _build_sources(chunks)
        new_messages = [
            HumanMessage(content=state["query"]),
            AIMessage(content=answer),
        ]

        return {
            "final_answer": answer,
            "sources": sources,
            "conversation_history": new_messages,
        }

    return generate


# --- Conditional edge functions -----------------------------------------------

def route_after_retrieve(state: RAGState) -> str:
    query_type = state.get("query_type", "general")
    chunks: list[Chunk] = state.get("retrieved_chunks", [])
    has_table_chunks = any(c.chunk_type == "table" for c in chunks)

    if has_table_chunks and query_type in ("numeric", "calculation", "comparative"):
        return "fetch_table_data"
    return "generate"


def route_after_fetch(state: RAGState) -> str:
    if state.get("query_type") == "calculation" and state.get("table_context"):
        return "execute_calculation"
    return "generate"


# --- Formatting helpers -------------------------------------------------------

def _format_text_context(chunks: list[Chunk]) -> str:
    parts: list[str] = []
    for c in chunks:
        page = f", page {c.page_number}" if c.page_number else ""
        heading = f" [{c.section_heading}]" if c.section_heading else ""
        parts.append(f"[{c.doc_name}{page}{heading}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


def _format_table_context(table_context: list[dict]) -> str:
    parts: list[str] = []
    for t in table_context:
        header = t.get("header", "Table")
        markdown = t.get("markdown", "")
        parts.append(f"**{header}**\n{markdown}")
    return "\n\n".join(parts)


def _build_sources(chunks: list[Chunk]) -> list[SourceRef]:
    seen: set[str] = set()
    sources: list[SourceRef] = []
    for c in chunks:
        key = f"{c.doc_name}:{c.page_number}"
        if key not in seen:
            seen.add(key)
            sources.append(SourceRef(
                doc_name=c.doc_name,
                page_number=c.page_number,
                chunk_type=c.chunk_type,
                excerpt=c.text[:200],
            ))
    return sources
