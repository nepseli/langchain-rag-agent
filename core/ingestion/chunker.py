"""
Hybrid chunker — text sections use RecursiveCharacterTextSplitter;
tables remain as single chunks (no splitting), prepended with context.
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.schemas import Chunk, ExtractedTable


def chunk_text(
    text: str,
    doc_id: str,
    doc_name: str,
    page_number: int | None,
    section_heading: str = "",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Split plain text into overlapping chunks."""
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    texts = splitter.split_text(text)
    return [
        Chunk(
            doc_id=doc_id,
            doc_name=doc_name,
            page_number=page_number,
            chunk_type="text",
            text=t,
            section_heading=section_heading,
        )
        for t in texts
        if t.strip()
    ]


def chunk_table(table: ExtractedTable, doc_name: str) -> Chunk:
    """
    Represent a table as a single chunk using its markdown representation.
    Tables are never split — their column alignment would be destroyed.
    If a table is very large (>3000 chars of markdown), we still keep it
    as one chunk; GPT-4o's 128k context can handle it at demo scale.
    """
    return Chunk(
        doc_id=table.doc_id,
        doc_name=doc_name,
        page_number=table.page_number,
        chunk_type="table",
        text=table.markdown,
        section_heading=table.header,
        table_id=table.id,
    )
