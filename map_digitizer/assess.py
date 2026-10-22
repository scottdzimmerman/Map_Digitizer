"""
Stage 0 — Assess the source PDF.

Pure inspection, no judgment calls: count vector drawing paths, embedded
images, and extractable text, then classify the PDF so the pipeline knows
whether to run vector extraction, raster extraction, and/or OCR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF


@dataclass
class PDFAssessment:
    page_index: int
    page_width_pt: float
    page_height_pt: float
    n_vector_paths: int
    n_embedded_images: int
    extractable_text_chars: int
    likely_kind: str          # "vector" | "raster" | "ambiguous"
    needs_ocr: bool
    native_image_xref: Optional[int] = None
    native_image_size_px: Optional[tuple] = None


def assess_pdf(
    pdf_path: str,
    page_index: int = 0,
    vector_path_threshold: int = 50,
    text_char_threshold: int = 200,
) -> PDFAssessment:
    """
    Classify a map PDF page as vector-authored or a flattened raster export,
    and flag whether OCR will be needed for labels.

    Thresholds are deliberately conservative: a real vector map has hundreds
    to thousands of drawn paths; a flattened scan has ~0. Text-char count
    catches the common case where a "vector" PDF still has no usable
    ToUnicode map on its fonts, so get_text() returns nothing useful even
    though the artwork itself is vector.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_index]

    drawings = page.get_drawings()
    images = page.get_images(full=True)
    text = page.get_text()
    text_chars = len(text.strip())

    n_paths = len(drawings)
    n_images = len(images)

    if n_paths >= vector_path_threshold:
        likely_kind = "vector"
    elif n_paths == 0 and n_images >= 1:
        likely_kind = "raster"
    else:
        likely_kind = "ambiguous"

    needs_ocr = text_chars < text_char_threshold

    native_xref = None
    native_size = None
    if likely_kind == "raster" and n_images:
        native_xref = images[0][0]
        pix = fitz.Pixmap(doc, native_xref)
        native_size = (pix.width, pix.height)

    result = PDFAssessment(
        page_index=page_index,
        page_width_pt=page.rect.width,
        page_height_pt=page.rect.height,
        n_vector_paths=n_paths,
        n_embedded_images=n_images,
        extractable_text_chars=text_chars,
        likely_kind=likely_kind,
        needs_ocr=needs_ocr,
        native_image_xref=native_xref,
        native_image_size_px=native_size,
    )
    doc.close()
    return result


def extract_native_raster(pdf_path: str, page_index: int, xref: int, out_path: str) -> str:
    """Pull an embedded raster image at its native resolution (Stage 0 note:
    avoid re-rendering the page, which adds a second layer of resampling)."""
    doc = fitz.open(pdf_path)
    pix = fitz.Pixmap(doc, xref)
    if pix.n - pix.alpha >= 4:  # CMYK -> RGB
        pix = fitz.Pixmap(fitz.csRGB, pix)
    pix.save(out_path)
    doc.close()
    return out_path
