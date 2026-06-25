"""
ChromaDB wrapper — add, search, delete document chunks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings

from models.schemas import Chunk


class VectorStore:
    COLLECTION_NAME = "document_chunks"

    def __init__(self, chroma_path: Path, embedding_model: str, openai_api_key: str):
        self._client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key,
        )

    # ─── Write ────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_documents(texts)
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "doc_id":          c.doc_id,
                    "doc_name":        c.doc_name,
                    "page_number":     c.page_number if c.page_number is not None else -1,
                    "chunk_type":      c.chunk_type,
                    "section_heading": c.section_heading,
                    "table_id":        c.table_id or "",
                }
                for c in chunks
            ],
        )

    def delete_by_doc_id(self, doc_id: str) -> None:
        results = self._collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])

    # ─── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 12,
        max_per_doc: int = 4,
        doc_ids: Optional[list[str]] = None,
    ) -> list[Chunk]:
        """
        Standard retrieval with a per-document cap to prevent any single
        document from dominating the results.
        """
        query_embedding = self._embedder.embed_query(query)
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None

        # Fetch more than needed so the per-doc cap can be applied
        fetch_n = min(n_results * 3, self._collection.count() or 1)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_n,
            where=where,
            include=["documents", "metadatas"],
        )

        chunks = _results_to_chunks(results)
        return _apply_per_doc_cap(chunks, max_per_doc, n_results)

    def search_per_doc(
        self,
        query: str,
        n_per_doc: int = 4,
    ) -> list[Chunk]:
        """
        Comparative retrieval: fetch top-n_per_doc chunks from EVERY indexed
        document. Used when query_type == 'comparative'.
        """
        # Collect all unique doc_ids
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        doc_ids = list({m["doc_id"] for m in all_meta})

        query_embedding = self._embedder.embed_query(query)
        all_chunks: list[Chunk] = []

        for doc_id in doc_ids:
            doc_count = self._collection.count()
            if doc_count == 0:
                continue
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_per_doc, doc_count),
                where={"doc_id": doc_id},
                include=["documents", "metadatas"],
            )
            all_chunks.extend(_results_to_chunks(results))

        return all_chunks

    def count(self) -> int:
        return self._collection.count()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _results_to_chunks(results: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    for cid, text, meta in zip(ids, docs, metas):
        page = meta.get("page_number", -1)
        chunks.append(
            Chunk(
                id=cid,
                doc_id=meta["doc_id"],
                doc_name=meta["doc_name"],
                page_number=page if page != -1 else None,
                chunk_type=meta["chunk_type"],
                text=text,
                section_heading=meta.get("section_heading", ""),
                table_id=meta.get("table_id") or None,
            )
        )
    return chunks


def _apply_per_doc_cap(
    chunks: list[Chunk], max_per_doc: int, total_limit: int
) -> list[Chunk]:
    seen: dict[str, int] = {}
    output: list[Chunk] = []
    for c in chunks:
        count = seen.get(c.doc_id, 0)
        if count < max_per_doc:
            output.append(c)
            seen[c.doc_id] = count + 1
        if len(output) >= total_limit:
            break
    return output
