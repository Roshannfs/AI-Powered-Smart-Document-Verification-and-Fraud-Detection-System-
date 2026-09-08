"""
Utility functions for file validation, PDF rendering, sensitive data masking,
and cryptographic/perceptual document hashing.
"""

from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional, Union
import hashlib

import numpy as np
from PIL import Image
import pymupdf
import imagehash

from config.config import settings


def validate_uploaded_file(file_bytes: bytes, filename: str) -> Tuple[bool, Optional[str]]:
    """
    Validates uploaded file for existence, extension, size limits, and non-corruption.
    Returns (is_valid, error_message).
    """
    if not file_bytes or len(file_bytes) == 0:
        return False, "The uploaded file is empty."

    # Size check
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        return False, f"File size exceeds maximum limit of {settings.max_file_size_mb} MB."

    # Extension check
    file_ext = Path(filename).suffix.lower()
    if file_ext not in settings.allowed_extensions:
        return False, f"Unsupported file type '{file_ext}'. Allowed types: {', '.join(settings.allowed_extensions)}"

    # Integrity / Corruption check
    if file_ext == ".pdf":
        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            if doc.page_count == 0:
                doc.close()
                return False, "The PDF file contains no pages."
            doc.close()
        except Exception as e:
            return False, f"The PDF file appears to be corrupted or password-protected: {str(e)}"
    else:
        try:
            img = Image.open(BytesIO(file_bytes))
            img.verify()
        except Exception as e:
            return False, f"The image file appears to be corrupted: {str(e)}"

    return True, None


def convert_pdf_to_images(pdf_bytes: bytes, max_pages: int = settings.max_pdf_pages, dpi: int = 200) -> List[Image.Image]:
    """
    Renders PDF document pages into high-resolution PIL RGB images using PyMuPDF.
    """
    images: List[Image.Image] = []
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    # Limit pages to avoid memory pressure
    total_pages = min(doc.page_count, max_pages)

    # Resolution zoom matrix: 72 DPI is base, dpi/72 is zoom factor
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)

    for page_idx in range(total_pages):
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)

    doc.close()
    return images


def load_document_as_images(file_bytes: bytes, filename: str) -> List[Image.Image]:
    """
    Universal document loader: returns a list of PIL Images (1 per page for PDF, 1 for image).
    """
    file_ext = Path(filename).suffix.lower()
    if file_ext == ".pdf":
        return convert_pdf_to_images(file_bytes)
    else:
        img = Image.open(BytesIO(file_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return [img]


def compute_sha256(data: bytes) -> str:
    """Computes SHA-256 cryptographic hex digest for exact duplicate detection."""
    return hashlib.sha256(data).hexdigest()


def compute_perceptual_hash(image: Union[Image.Image, np.ndarray]) -> str:
    """
    Computes pHash (perceptual hash) for visually similar / near-duplicate image detection.
    """
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    else:
        pil_img = image
    return str(imagehash.phash(pil_img))


def perceptual_hash_distance(hash1_str: str, hash2_str: str) -> int:
    """
    Calculates Hamming distance between two perceptual hashes.
    Distance <= 5 indicates highly similar or identical visual content.
    """
    try:
        h1 = imagehash.hex_to_hash(hash1_str)
        h2 = imagehash.hex_to_hash(hash2_str)
        return int(h1 - h2)
    except Exception:
        return 999


def mask_sensitive_identifier(identifier: Optional[str], keep_last: int = 4) -> str:
    """
    Masks sensitive personal data such as Account Numbers, Aadhaar, PAN, SSN for UI display.
    Example: '123456789012' -> 'XXXXXXXX9012'
    """
    if not identifier:
        return "N/A"
    clean_id = str(identifier).strip()
    if len(clean_id) <= keep_last:
        return "****"
    masked_part = "X" * (len(clean_id) - keep_last)
    visible_part = clean_id[-keep_last:]
    return f"{masked_part}{visible_part}"


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Converts PIL RGB image to OpenCV BGR NumPy array."""
    rgb_arr = np.array(pil_image.convert("RGB"))
    return rgb_arr[:, :, ::-1].copy()


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Converts OpenCV BGR or Grayscale NumPy array to PIL RGB Image."""
    if len(cv2_image.shape) == 2:
        return Image.fromarray(cv2_image).convert("RGB")
    rgb_arr = cv2_image[:, :, ::-1]
    return Image.fromarray(rgb_arr)
