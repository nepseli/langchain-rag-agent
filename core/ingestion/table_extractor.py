"""
Table extraction helpers for PDF, DOCX, and DataFrame sources.
Produces ExtractedTable instances with both markdown and JSON representations.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd

from models.schemas import ExtractedTable


def pdf_tables_from_page(
    page: Any,  # pdfplumber.Page
    doc_id: str,
    page_number: int,
    preceding_text: str = "",
) -> list[ExtractedTable]:
    """Extract all tables from a pdfplumber Page object."""
    raw_tables = page.extract_tables()
    if not raw_tables:
        return []

    tables: list[ExtractedTable] = []
    for idx, raw in enumerate(raw_tables):
        if not raw or not raw[0]:
            continue
        try:
            df = _raw_table_to_df(raw)
        except Exception:
            continue
        if df.empty:
            continue

        header = _extract_heading(preceding_text)
        tables.append(
            ExtractedTable(
                doc_id=doc_id,
                page_number=page_number,
                table_index=idx,
                header=header,
                markdown=_df_to_markdown(df, header),
                json_data=df.to_json(orient="records"),
                row_count=len(df),
                col_count=len(df.columns),
            )
        )
    return tables


def docx_tables(
    doc: Any,  # python-docx Document
    doc_id: str,
) -> list[ExtractedTable]:
    """Extract all tables from a python-docx Document."""
    tables: list[ExtractedTable] = []
    for idx, table in enumerate(doc.tables):
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if not rows:
            continue
        try:
            df = _raw_table_to_df(rows)
        except Exception:
            continue
        if df.empty:
            continue

        tables.append(
            ExtractedTable(
                doc_id=doc_id,
                page_number=None,
                table_index=idx,
                header="",
                markdown=_df_to_markdown(df, ""),
                json_data=df.to_json(orient="records"),
                row_count=len(df),
                col_count=len(df.columns),
            )
        )
    return tables


def dataframe_table(
    df: pd.DataFrame,
    doc_id: str,
    sheet_name: str,
    table_index: int = 0,
) -> ExtractedTable:
    """Wrap a pandas DataFrame (from Excel/CSV) as an ExtractedTable."""
    df = df.fillna("").astype(str)
    header = sheet_name
    return ExtractedTable(
        doc_id=doc_id,
        page_number=None,
        table_index=table_index,
        header=header,
        markdown=_df_to_markdown(df, header),
        json_data=df.to_json(orient="records"),
        row_count=len(df),
        col_count=len(df.columns),
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _raw_table_to_df(raw: list[list[Optional[str]]]) -> pd.DataFrame:
    """Convert a list-of-lists from pdfplumber into a cleaned DataFrame."""
    if not raw:
        return pd.DataFrame()

    # Replace None with empty string
    cleaned = [[cell if cell is not None else "" for cell in row] for row in raw]

    # Use first row as header if it looks like a header
    header = [str(c).strip() for c in cleaned[0]]
    # Deduplicate column names by appending index
    seen: dict[str, int] = {}
    unique_header: list[str] = []
    for col in header:
        if col in seen:
            seen[col] += 1
            unique_header.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            unique_header.append(col)

    data = cleaned[1:] if len(cleaned) > 1 else []
    df = pd.DataFrame(data, columns=unique_header)
    # Drop fully empty rows
    df = df[df.apply(lambda r: any(v.strip() for v in r), axis=1)].reset_index(drop=True)
    return df


def _df_to_markdown(df: pd.DataFrame, context_header: str) -> str:
    """Convert DataFrame to GitHub-flavored markdown table with optional context header."""
    lines: list[str] = []
    if context_header:
        lines.append(f"**{context_header}**\n")

    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")

    return "\n".join(lines)


def _extract_heading(text: str, max_chars: int = 120) -> str:
    """Pull the last non-empty line from preceding text as a context heading."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1][:max_chars] if lines else ""
