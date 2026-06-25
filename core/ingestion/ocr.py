"""
OCR supplement for PDF pages that contain image-based content.

How it works:
  1. pdfplumber extracts machine-readable (vector) text as usual.
  2. If the page has embedded images (logos, stylised headers, stamps) OR the
     pdfplumber yield is very sparse, we also render the full page via PyMuPDF
     and run Tesseract OCR on the rendered bitmap.
  3. The two text sources are merged so both image-derived and vector-derived
     text end up in the same chunk / embedding.

Requirements:
  - PyMuPDF  (pip install pymupdf)          ← already in requirements.txt
  - pytesseract (pip install pytesseract)
  - Tesseract binary  ← must be installed separately:
      Windows: https://github.com/UB-Mannheim/tesseract/wiki
      macOS:   brew install tesseract
      Linux:   sudo apt-get install tesseract-ocr
"""
from __future__ import annotations

import io
from pathlib import Path

# ── Soft-import: OCR is optional. If unavailable, the pipeline degrades
# gracefully to pdfplumber-only extraction.
try:
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image as PILImage

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# If a page has fewer than this many chars from pdfplumber, treat it as
# image-heavy and always OCR regardless of whether images[] is populated.
SPARSE_THRESHOLD = 80

# Render resolution. 200 DPI is a good balance of accuracy vs speed.
# Bump to 300 for scanned documents with small fonts.
RENDER_DPI = 200


# ─────────────────────────────────────────────────────────────────────────────

def needs_ocr(pdfplumber_page) -> bool:
    """
    Decide whether to OCR a page.

    Returns True when:
    - The page has at least one embedded raster image (logo, stamp, photo…), OR
    - pdfplumber returned very little text (scanned / image-only pages).
    """
    if not OCR_AVAILABLE:
        return False

    plumber_text = (pdfplumber_page.extract_text() or "").strip()

    # Sparse page → likely image-only or scan
    if len(plumber_text) < SPARSE_THRESHOLD:
        return True

    # Page contains embedded images even though it also has vector text
    if pdfplumber_page.images:
        return True

    return False


def ocr_page(file_path: Path, page_index: int, dpi: int = RENDER_DPI) -> str:
    """
    Render one PDF page with PyMuPDF and extract text via Tesseract.

    Args:
        file_path:   Absolute path to the source PDF.
        page_index:  0-based page index (pdfplumber page_number - 1).
        dpi:         Render resolution.

    Returns:
        Extracted text string (may be empty on failure or if OCR unavailable).
    """
    if not OCR_AVAILABLE:
        return ""

    try:
        pdf = fitz.open(str(file_path))
        page = pdf[page_index]

        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_text = pytesseract.image_to_string(img, lang="eng")
        pdf.close()
        return ocr_text.strip()

    except Exception as exc:
        # Log but don't crash the ingestion pipeline
        import warnings
        warnings.warn(f"OCR failed for page {page_index} of {file_path}: {exc}")
        return ""


def merge_page_text(plumber_text: str, ocr_text: str) -> str:
    """
    Combine pdfplumber text and OCR text for a single page.

    Strategy:
    - If pdfplumber is sparse: OCR is primary (image-based page).
    - Otherwise: prepend OCR so image-derived tokens (e.g. "ABC TRADING"
      from a logo) appear in the text before the structured body.  The LLM
      handles any near-duplicate phrases without confusion.
    """
    plumber_clean = plumber_text.strip()
    ocr_clean = ocr_text.strip()

    if not ocr_clean:
        return plumber_clean

    if not plumber_clean or len(plumber_clean) < SPARSE_THRESHOLD:
        # Sparse pdfplumber output — OCR is the sole source
        return ocr_clean

    # Both sources have content — merge with a clear separator
    return f"{ocr_clean}\n\n---\n\n{plumber_clean}"
