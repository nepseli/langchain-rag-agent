"""
LangGraph StateGraph assembly for the RAG workflow.

Graph flow:
  check_clarity
    → [needs clarification?] → clarify → END
    → [clear] → classify_query → rewrite_query → retrieve
        → [route_after_retrieve]
          → fetch_table_data → [route_after_fetch]
              → execute_calculation → generate → END
              → generate → END
          → generate → END
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from core.agent.nodes import (
    make_check_clarity_node,
    make_clarify_node,
    make_classify_node,
    make_execute_calculation_node,
    make_fetch_table_node,
    make_generate_node,
    make_retrieve_node,
    make_rewrite_node,
    route_after_clarity,
    route_after_fetch,
    route_after_retrieve,
)
from core.agent.state import RAGState
from core.retrieval.metadata_store import MetadataStore
from core.retrieval.vector_store import VectorStore


def build_graph(
    meta_store: MetadataStore,
    vector_store: VectorStore,
    streaming: bool = False,
):
    """
    Build and compile the RAG StateGraph.

    New node: check_clarity — runs first, asks a clarifying question if the
    query is ambiguous (e.g. 'this invoice' when multiple docs are indexed).
    Only proceeds to the full RAG pipeline when the query is specific enough.
    """
    graph = StateGraph(RAGState)

    # ── Register nodes
    graph.add_node("check_clarity",         make_check_clarity_node(meta_store))
    graph.add_node("clarify",               make_clarify_node())
    graph.add_node("classify_query",        make_classify_node())
    graph.add_node("rewrite_query",         make_rewrite_node())
    graph.add_node("retrieve",              make_retrieve_node(vector_store))
    graph.add_node("fetch_table_data",      make_fetch_table_node(meta_store))
    graph.add_node("execute_calculation",   make_execute_calculation_node())
    graph.add_node("generate",              make_generate_node(streaming=streaming))

    # ── Entry point is now check_clarity
    graph.set_entry_point("check_clarity")

    # ── Conditional: after check_clarity
    graph.add_conditional_edges(
        "check_clarity",
        route_after_clarity,
        {
            "clarify":        "clarify",
            "classify_query": "classify_query",
        },
    )

    # ── clarify is a terminal node
    graph.add_edge("clarify", END)

    # ── Linear edges (normal RAG path)
    graph.add_edge("classify_query",   "rewrite_query")
    graph.add_edge("rewrite_query",    "retrieve")

    # ── Conditional: after retrieve
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "fetch_table_data": "fetch_table_data",
            "generate":         "generate",
        },
    )

    # ── Conditional: after fetch_table_data
    graph.add_conditional_edges(
        "fetch_table_data",
        route_after_fetch,
        {
            "execute_calculation": "execute_calculation",
            "generate":            "generate",
        },
    )

    graph.add_edge("execute_calculation", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
