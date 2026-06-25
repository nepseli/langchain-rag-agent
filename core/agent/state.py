"""
LangGraph state definition for the RAG workflow.
"""
from __future__ import annotations

from typing import Annotated, Optional
import operator

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from models.schemas import Chunk, SourceRef


class RAGState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    query: str
    session_id: str

    # ── Conversation history (LangGraph manages merging via add_messages) ─────
    conversation_history: Annotated[list[BaseMessage], add_messages]

    # ── Classification ────────────────────────────────────────────────────────
    query_type: str          # factual | numeric | calculation | comparative | general

    # ── Retrieval ─────────────────────────────────────────────────────────────
    rewritten_query: str     # HyDE-rewritten query for embedding
    retrieved_chunks: list[Chunk]

    # ── Table data (full JSON from SQLite) ────────────────────────────────────
    table_context: list[dict]  # {table_id, header, markdown, json_data}

    # ── Calculation ───────────────────────────────────────────────────────────
    generated_code: Optional[str]         # pandas code produced by LLM
    code_execution_result: Optional[str]  # stdout from PythonREPLTool

    # ── Clarity check ─────────────────────────────────────────────────────────
    needs_clarification: bool          # True → short-circuit to clarify node
    clarification_question: str        # the question to ask the user

    # ── Output ────────────────────────────────────────────────────────────────
    final_answer: str
    sources: list[SourceRef]
