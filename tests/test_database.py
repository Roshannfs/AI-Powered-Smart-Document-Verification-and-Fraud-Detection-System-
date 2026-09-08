"""Unit tests for SQLite database operations."""

import pytest
from modules.database import (
    insert_verification_record,
    get_all_verifications,
    get_verification_by_id,
    check_for_duplicate,
    delete_verification,
    clear_all_history
)


def test_database_lifecycle():
    # Insert test record
    rec_id = insert_verification_record(
        filename="test_invoice.pdf",
        sha256_hash="test_sha256_hash_12345",
        phash="ffff0000ffff0000",
        document_type="invoice",
        classification_confidence=95.0,
        ocr_confidence=92.0,
        risk_score=15,
        risk_level="LOW RISK",
        extracted_fields={"invoice_number": "INV-001", "total_amount": 500.0},
        verification_checks=[{"name": "Check1", "status": "PASS"}],
        reasons=["All checks passed"]
    )
    assert rec_id > 0

    # Retrieve record
    record = get_verification_by_id(rec_id)
    assert record is not None
    assert record["filename"] == "test_invoice.pdf"
    assert record["document_type"] == "invoice"
    assert record["extracted_fields"]["invoice_number"] == "INV-001"

    # Duplicate check
    dup = check_for_duplicate("test_sha256_hash_12345")
    assert dup is not None
    assert dup["id"] == rec_id

    # Non-duplicate check
    non_dup = check_for_duplicate("completely_unique_hash_99999")
    assert non_dup is None

    # Delete single record
    deleted = delete_verification(rec_id)
    assert deleted is True
    assert get_verification_by_id(rec_id) is None
