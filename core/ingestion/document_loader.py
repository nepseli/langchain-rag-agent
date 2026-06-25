"""
Document loaders — routes each file type to the appropriate parsing library
and yields (page_text, page_number, tables) tuples or DataFrame sheets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import pandas as pd


@dataclass
class PageContent:
    page_number: int
    text: str
    raw_page: object = field(default=None, repr=False)  # pdfplumber Page for table extraction


@dataclass
class SheetContent:
    sheet_name: str
    df: pd.DataFrame


def load_pdf(file_path: Path) -> Generator[PageContent, None, None]:
    """Yield PageContent for each page of a PDF.
    raw_page is the pdfplumber Page object (needed for table extraction)."""
    import pdfplumber

    with pdfplumber.open(str(file_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(layout=True) or ""
            yield PageContent(page_number=i, text=text, raw_page=page)


def load_docx(file_path: Path) -> tuple[list[PageContent], object]:
    """
    Return (pages, doc_object).
    DOCX has no native page breaks, so we treat each paragraph group as a
    single 'page' (page_number=1 for the whole document).
    The doc_object is the python-docx Document — needed for table extraction.
    """
    from docx import Document

    doc = Document(str(file_path))
    full_text = "\n".join(
        para.text for para in doc.paragraphs if para.text.strip()
    )
    pages = [PageContent(page_number=1, text=full_text, raw_page=None)]
    return pages, doc


def load_excel(file_path: Path) -> list[SheetContent]:
    """Return one SheetContent per worksheet."""
    xl = pd.ExcelFile(str(file_path))
    sheets: list[SheetContent] = []
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        if not df.empty:
            sheets.append(SheetContent(sheet_name=str(sheet_name), df=df))
    return sheets


def load_csv(file_path: Path) -> list[SheetContent]:
    """Return a single SheetContent from a CSV file."""
    df = pd.read_csv(str(file_path))
    return [SheetContent(sheet_name=file_path.stem, df=df)]
