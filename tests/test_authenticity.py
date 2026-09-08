"""Unit tests for authenticity verification checks."""

import pytest
from modules.extraction import ExtractedData
from modules.authenticity import verify_authenticity, parse_date_flexibly


def test_parse_date_flexibly():
    d1 = parse_date_flexibly("15/04/2025")
    assert d1 is not None
    assert d1.day == 15 and d1.month == 4 and d1.year == 2025

    d2 = parse_date_flexibly("2024-12-31")
    assert d2 is not None
    assert d2.year == 2024 and d2.month == 12 and d2.day == 31

    assert parse_date_flexibly("invalid_date") is None


def test_invoice_mathematical_consistency_valid():
    data = ExtractedData(
        document_type="invoice",
        raw_fields={
            "invoice_number": "INV-100",
            "invoice_date": "10/01/2025",
            "due_date": "25/01/2025",
            "subtotal": 1000.0,
            "tax_amount": 180.0,
            "total_amount": 1180.0
        },
        missing_required_fields=[]
    )
    report = verify_authenticity(data, ocr_confidence=92.0, file_sha256="abc", file_phash="def")
    assert report.math_consistent is True
    assert report.dates_consistent is True
    assert report.failed_count == 0


def test_invoice_mathematical_consistency_tampered():
    # Subtotal 1000 + tax 180 = 1180, but stated total is 3000!
    data = ExtractedData(
        document_type="invoice",
        raw_fields={
            "invoice_number": "INV-100",
            "invoice_date": "10/01/2025",
            "due_date": "25/01/2025",
            "subtotal": 1000.0,
            "tax_amount": 180.0,
            "total_amount": 3000.0
        },
        missing_required_fields=[]
    )
    report = verify_authenticity(data, ocr_confidence=92.0, file_sha256="abc", file_phash="def")
    assert report.math_consistent is False
    assert report.failed_count >= 1
    # Verify the check message mentions mismatch
    math_checks = [c for c in report.checks if c.name == "Mathematical Consistency"]
    assert len(math_checks) == 1
    assert math_checks[0].status == "FAIL"


def test_date_chronological_failure():
    # Invoice date after due date!
    data = ExtractedData(
        document_type="invoice",
        raw_fields={
            "invoice_number": "INV-100",
            "invoice_date": "25/01/2025",
            "due_date": "10/01/2025",  # Due before invoice date
            "subtotal": 100.0,
            "tax_amount": 10.0,
            "total_amount": 110.0
        },
        missing_required_fields=[]
    )
    report = verify_authenticity(data, ocr_confidence=90.0, file_sha256="abc", file_phash="def")
    assert report.dates_consistent is False
