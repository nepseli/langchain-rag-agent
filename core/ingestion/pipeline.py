"""
Ingestion pipeline — orchestrates: load -> extract tables -> chunk -> embed -> store.
Entry point: ingest_document()
"""
from __future__ import annotations

import traceback
from pathlib import Path

from config import settings
from core.ingestion.chunker import chunk_table, chunk_text
from core.ingestion.document_loader import load_csv, load_docx, load_excel, load_pdf
from core.ingestion.ocr import merge_page_text, needs_ocr, ocr_page
from core.ingestion.table_extractor import (
    dataframe_table,
    docx_tables,
    pdf_tables_from_page,
)
from core.retrieval.metadata_store import MetadataStore
from core.retrieval.vector_store import VectorStore
from models.schemas import Chunk, Document, ExtractedTable


def ingest_document(
    file_path: Path,
    store: MetadataStore,
    vector_store: VectorStore,
    progress_callback=None,
    original_name: str | None = None,
) -> Document:
    """
    Full ingestion pipeline for a single document.

    Returns the Document record (status='indexed' on success, 'error' on failure).
    progress_callback is called with status strings for Streamlit progress updates.
    original_name overrides file_path.name so temp-file paths don't leak into the UI.
    """
    file_type = _detect_type(file_path)
    doc = Document(
        name=original_name or file_path.name,
        file_type=file_type,
        file_path=str(file_path),
    )
    store.upsert_document(doc)

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    try:
        all_chunks: list[Chunk] = []
        all_tables: list[ExtractedTable] = []

        if file_type == "pdf":
            _log("Parsing PDF pages...")
            pages = list(load_pdf(file_path))
            doc.page_count = len(pages)

            for page_content in pages:
                page_index = page_content.page_number - 1  # 0-based for PyMuPDF

                # OCR supplement: run Tesseract on pages that have embedded
                # images (logos, stamps, stylised headers) or very sparse text.
                if needs_ocr(page_content.raw_page):
                    _log(f"  OCR on page {page_content.page_number} (image content detected)...")
                    ocr_text = ocr_page(file_path, page_index)
                    page_text = merge_page_text(page_content.text, ocr_text)
                else:
                    page_text = page_content.text

                preceding = page_text

                # Extract tables first
                tables = pdf_tables_from_page(
                    page_content.raw_page,
                    doc_id=doc.id,
                    page_number=page_content.page_number,
                    preceding_text=preceding,
                )
                all_tables.extend(tables)

                # Chunk the merged text (pdfplumber + OCR)
                text_chunks = chunk_text(
                    text=page_text,
                    doc_id=doc.id,
                    doc_name=doc.name,
                    page_number=page_content.page_number,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )
                all_chunks.extend(text_chunks)

                # One chunk per table (markdown representation)
                for t in tables:
                    all_chunks.append(chunk_table(t, doc.name))

        elif file_type == "docx":
            _log("Parsing Word document...")
            pages, docx_obj = load_docx(file_path)
            doc.page_count = 1

            for page_content in pages:
                text_chunks = chunk_text(
                    text=page_content.text,
                    doc_id=doc.id,
                    doc_name=doc.name,
                    page_number=page_content.page_number,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )
                all_chunks.extend(text_chunks)

            tables = docx_tables(docx_obj, doc_id=doc.id)
            all_tables.extend(tables)
            for t in tables:
                all_chunks.append(chunk_table(t, doc.name))

        elif file_type in ("xlsx", "csv"):
            _log(f"Parsing {'Excel' if file_type == 'xlsx' else 'CSV'} file...")
            loader = load_excel if file_type == "xlsx" else load_csv
            sheets = loader(file_path)
            doc.page_count = len(sheets)

            for idx, sheet in enumerate(sheets):
                table = dataframe_table(
                    df=sheet.df,
                    doc_id=doc.id,
                    sheet_name=sheet.sheet_name,
                    table_index=idx,
                )
                all_tables.append(table)
                all_chunks.append(chunk_table(table, doc.name))

                text_chunks = chunk_text(
                    text=table.markdown,
                    doc_id=doc.id,
                    doc_name=doc.name,
                    page_number=None,
                    section_heading=sheet.sheet_name,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )
                all_chunks.extend(text_chunks)

        # Persist tables to SQLite
        _log(f"Storing {len(all_tables)} extracted table(s)...")
        for t in all_tables:
            store.insert_table(t)

        # Embed and store chunks in ChromaDB
        _log(f"Embedding {len(all_chunks)} chunk(s)...")
        batch_size = 50
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            vector_store.add_chunks(batch)
            _log(f"  Embedded {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} chunks...")

        doc.chunk_count = len(all_chunks)
        doc.status = "indexed"
        store.upsert_document(doc)
        _log("Indexed successfully.")

    except Exception as exc:
        doc.status = "error"
        doc.error_msg = str(exc)
        store.upsert_document(doc)
        _log(f"Error: {exc}")
        traceback.print_exc()

    return doc


def delete_document(
    doc_id: str,
    store: MetadataStore,
    vector_store: VectorStore,
) -> None:
    """Remove a document and all its chunks/tables from all stores."""
    vector_store.delete_by_doc_id(doc_id)
    store.delete_tables_for_doc(doc_id)
    store.delete_document(doc_id)


# --- Helpers ------------------------------------------------------------------

_TYPE_MAP = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".csv": "csv",
}


def _detect_type(file_path: Path) -> str:
    ft = _TYPE_MAP.get(file_path.suffix.lower())
    if not ft:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    return ft
