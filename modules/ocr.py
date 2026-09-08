"""
Unified Optical Character Recognition (OCR) Module.
Supports Tesseract OCR with confidence extraction, multi-page handling,
and pluggable fallback architecture ready for PaddleOCR integration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

import numpy as np
from PIL import Image
import pytesseract
import pymupdf

from config.config import settings

logger = logging.getLogger(__name__)


@dataclass
class WordBox:
    """Bounding box and confidence for an individual extracted word."""
    word: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    page_num: int = 1


@dataclass
class OCRResult:
    """Standardized output structure for OCR operations across all engines."""
    text: str
    confidence: float  # Scale 0.0 to 100.0
    page_count: int
    page_results: List[Dict[str, Any]] = field(default_factory=list)
    word_boxes: List[WordBox] = field(default_factory=list)
    engine_name: str = "tesseract"
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 1),
            "pages": self.page_count,
            "engine": self.engine_name,
            "warning": self.warning
        }


class BaseOCREngine(ABC):
    """Abstract base class for all OCR engine implementations."""

    @abstractmethod
    def extract_text(self, images: List[Image.Image]) -> OCRResult:
        """Extract text and confidence scores from a list of PIL Images."""
        pass


class TesseractOCREngine(BaseOCREngine):
    """Tesseract OCR implementation using pytesseract with word-level confidence aggregation."""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.tesseract_cmd = tesseract_cmd or settings.tesseract_cmd
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    def extract_text(self, images: List[Image.Image]) -> OCRResult:
        all_text_parts: List[str] = []
        all_word_boxes: List[WordBox] = []
        page_results: List[Dict[str, Any]] = []
        total_word_confidences: List[float] = []

        for page_idx, img in enumerate(images, start=1):
            try:
                # Use image_to_data to obtain structured bounding boxes and confidence
                data = pytesseract.image_to_data(
                    img,
                    output_type=pytesseract.Output.DICT,
                    config="--oem 3 --psm 6"
                )

                page_words: List[str] = []
                page_confs: List[float] = []

                n_boxes = len(data["text"])
                for i in range(n_boxes):
                    word = data["text"][i].strip()
                    conf = float(data["conf"][i])

                    # Tesseract assigns -1 to empty blocks/layout elements
                    if word and conf >= 0:
                        page_words.append(word)
                        page_confs.append(conf)
                        total_word_confidences.append(conf)

                        all_word_boxes.append(WordBox(
                            word=word,
                            confidence=conf,
                            x=data["left"][i],
                            y=data["top"][i],
                            width=data["width"][i],
                            height=data["height"][i],
                            page_num=page_idx
                        ))

                # Also obtain clean preserved layout text
                plain_text = pytesseract.image_to_string(img, config="--psm 6").strip()
                if not plain_text:
                    plain_text = " ".join(page_words)

                all_text_parts.append(plain_text)
                avg_page_conf = float(np.mean(page_confs)) if page_confs else 0.0

                page_results.append({
                    "page": page_idx,
                    "text": plain_text,
                    "confidence": round(avg_page_conf, 1),
                    "word_count": len(page_words)
                })

            except Exception as e:
                logger.error(f"Tesseract OCR failed on page {page_idx}: {str(e)}")
                page_results.append({
                    "page": page_idx,
                    "text": "",
                    "confidence": 0.0,
                    "error": str(e)
                })

        overall_conf = float(np.mean(total_word_confidences)) if total_word_confidences else 0.0
        combined_text = "\n\n--- Page Break ---\n\n".join(all_text_parts)

        return OCRResult(
            text=combined_text,
            confidence=round(overall_conf, 1),
            page_count=len(images),
            page_results=page_results,
            word_boxes=all_word_boxes,
            engine_name="tesseract"
        )


class FallbackOCREngine(BaseOCREngine):
    """
    Intelligent fallback engine when Tesseract binary is not installed or for PDFs.
    Uses PyMuPDF direct text streaming if PDF bytes are provided, or simulated high-fidelity
    OCR analysis for test documents.
    """

    def __init__(self, raw_pdf_bytes: Optional[bytes] = None):
        self.raw_pdf_bytes = raw_pdf_bytes

    def extract_text(self, images: List[Image.Image]) -> OCRResult:
        # If we have original PDF bytes, PyMuPDF can directly extract digital text accurately
        if self.raw_pdf_bytes:
            try:
                doc = pymupdf.open(stream=self.raw_pdf_bytes, filetype="pdf")
                pages_text = []
                page_results = []
                for i in range(min(doc.page_count, len(images))):
                    page = doc.load_page(i)
                    p_text = page.get_text("text").strip()
                    pages_text.append(p_text)
                    page_results.append({
                        "page": i + 1,
                        "text": p_text,
                        "confidence": 95.0 if p_text else 0.0,
                        "word_count": len(p_text.split())
                    })
                doc.close()
                combined = "\n\n--- Page Break ---\n\n".join(pages_text)
                return OCRResult(
                    text=combined,
                    confidence=95.0 if combined else 0.0,
                    page_count=len(images),
                    page_results=page_results,
                    engine_name="pymupdf-direct-stream",
                    warning="Extracted using PyMuPDF direct text stream (Tesseract binary not installed)."
                )
            except Exception as e:
                logger.warning(f"PyMuPDF direct extraction fallback failed: {e}")

        # When only images are available without Tesseract binary
        return OCRResult(
            text="[Tesseract OCR binary not detected on system PATH. Install Tesseract OCR to perform image text recognition.]",
            confidence=0.0,
            page_count=len(images),
            page_results=[{"page": 1, "text": "", "confidence": 0.0}],
            engine_name="fallback-none",
            warning="Tesseract OCR binary not found. Please install Tesseract or configure TESSERACT_CMD in .env"
        )


def perform_ocr(
    images: List[Image.Image],
    raw_pdf_bytes: Optional[bytes] = None,
    prefer_tesseract: bool = True
) -> OCRResult:
    """
    Main entrypoint for document OCR.
    Automatically chooses Tesseract if available, otherwise falls back gracefully.
    """
    if prefer_tesseract and settings.tesseract_cmd:
        try:
            engine = TesseractOCREngine(settings.tesseract_cmd)
            result = engine.extract_text(images)
            # If tesseract succeeded and produced text, return it
            if result.text.strip():
                return result
        except Exception as e:
            logger.warning(f"Tesseract OCR encountered an issue: {e}. Attempting fallback.")

    # Fallback to PyMuPDF / Direct extractor
    fallback_engine = FallbackOCREngine(raw_pdf_bytes=raw_pdf_bytes)
    return fallback_engine.extract_text(images)
