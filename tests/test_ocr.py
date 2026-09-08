"""Unit tests for OCR module."""

from PIL import Image
import pymupdf
import pytest
from modules.ocr import (
    WordBox,
    OCRResult,
    FallbackOCREngine,
    perform_ocr
)


def create_test_pdf_stream() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=500)
    page.insert_text((50, 50), "OCR TESTING INVOICE # INV-500", fontsize=14)
    b = doc.tobytes()
    doc.close()
    return b


def test_ocr_result_dataclass():
    res = OCRResult(
        text="Sample Text",
        confidence=94.5,
        page_count=1,
        page_results=[{"page": 1, "text": "Sample Text", "confidence": 94.5}],
        engine_name="tesseract"
    )
    d = res.to_dict()
    assert d["text"] == "Sample Text"
    assert d["confidence"] == 94.5
    assert d["pages"] == 1
    assert d["engine"] == "tesseract"


def test_word_box_dataclass():
    box = WordBox(word="Invoice", confidence=98.0, x=10, y=20, width=50, height=15)
    assert box.word == "Invoice"
    assert box.confidence == 98.0
    assert box.x == 10


def test_fallback_ocr_engine_with_pdf():
    pdf_bytes = create_test_pdf_stream()
    engine = FallbackOCREngine(raw_pdf_bytes=pdf_bytes)
    dummy_img = Image.new("RGB", (100, 100), color="white")
    result = engine.extract_text([dummy_img])
    assert "OCR TESTING INVOICE" in result.text
    assert result.confidence == 95.0
    assert result.engine_name == "pymupdf-direct-stream"


def test_perform_ocr_fallback():
    pdf_bytes = create_test_pdf_stream()
    dummy_img = Image.new("RGB", (100, 100), color="white")
    res = perform_ocr([dummy_img], raw_pdf_bytes=pdf_bytes, prefer_tesseract=False)
    assert isinstance(res, OCRResult)
    assert len(res.text) > 0
