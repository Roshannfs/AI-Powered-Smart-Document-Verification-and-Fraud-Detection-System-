"""Unit tests for utility module."""

from io import BytesIO
import numpy as np
from PIL import Image
import pymupdf
import pytest

from modules.utils import (
    validate_uploaded_file,
    compute_sha256,
    compute_perceptual_hash,
    perceptual_hash_distance,
    mask_sensitive_identifier,
    convert_pdf_to_images,
    load_document_as_images
)


def create_dummy_pdf_bytes():
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 100), "TEST INVOICE DOCUMENT", fontsize=18)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_dummy_image_bytes():
    img = Image.new("RGB", (200, 200), color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_validate_uploaded_file_valid_png():
    img_bytes = create_dummy_image_bytes()
    is_valid, err = validate_uploaded_file(img_bytes, "test.png")
    assert is_valid is True
    assert err is None


def test_validate_uploaded_file_valid_pdf():
    pdf_bytes = create_dummy_pdf_bytes()
    is_valid, err = validate_uploaded_file(pdf_bytes, "test.pdf")
    assert is_valid is True
    assert err is None


def test_validate_uploaded_file_invalid_extension():
    img_bytes = create_dummy_image_bytes()
    is_valid, err = validate_uploaded_file(img_bytes, "test.exe")
    assert is_valid is False
    assert "Unsupported file type" in err


def test_validate_uploaded_file_empty():
    is_valid, err = validate_uploaded_file(b"", "test.png")
    assert is_valid is False
    assert "empty" in err.lower()


def test_compute_sha256():
    data = b"Hello world fraud detection"
    h1 = compute_sha256(data)
    h2 = compute_sha256(data)
    assert h1 == h2
    assert len(h1) == 64


def test_perceptual_hash():
    img1 = Image.new("RGB", (100, 100), color="white")
    img2 = Image.new("RGB", (100, 100), color="white")
    h1 = compute_perceptual_hash(img1)
    h2 = compute_perceptual_hash(img2)
    dist = perceptual_hash_distance(h1, h2)
    assert dist == 0  # Identical images have distance 0


def test_mask_sensitive_identifier():
    assert mask_sensitive_identifier("123456789012") == "XXXXXXXX9012"
    assert mask_sensitive_identifier("ABCD1234E") == "XXXXX234E"
    assert mask_sensitive_identifier("123") == "****"
    assert mask_sensitive_identifier("") == "N/A"
    assert mask_sensitive_identifier(None) == "N/A"


def test_pdf_conversion():
    pdf_bytes = create_dummy_pdf_bytes()
    imgs = convert_pdf_to_images(pdf_bytes)
    assert len(imgs) == 1
    assert isinstance(imgs[0], Image.Image)
