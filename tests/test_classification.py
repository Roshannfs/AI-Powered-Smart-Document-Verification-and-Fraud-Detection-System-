"""Unit tests for document classification module."""

import pytest
from modules.classification import classify_document, ClassificationResult


def test_classify_empty_text():
    res = classify_document("")
    assert isinstance(res, ClassificationResult)
    assert res.document_type == "unknown"
    assert res.confidence == 0.0
    assert not res.is_confident


def test_classify_invoice_sample():
    sample = """
    TAX INVOICE
    Seller: Apex Tech Solutions
    Invoice Number: INV-9842
    Invoice Date: 12/04/2025
    Subtotal: INR 5000.00
    GST: INR 900.00
    Total Amount Due: INR 5900.00
    """
    res = classify_document(sample)
    assert res.document_type == "invoice"
    assert res.confidence > 0.5
    assert res.is_confident


def test_classify_bank_statement_sample():
    sample = """
    METROPOLITAN TRUST BANK
    ACCOUNT STATEMENT - SAVINGS ACCOUNT
    Account Holder Name: Michael Chang
    Account Number: 987654321098
    Statement Period: 01/05/2025 to 31/05/2025
    Opening Balance: INR 15000.00
    Closing Balance: INR 17500.00
    """
    res = classify_document(sample)
    assert res.document_type == "bank_statement"
    assert res.confidence > 0.5


def test_classify_certificate_sample():
    sample = """
    STANFORD ACADEMIC UNIVERSITY
    PROVISIONAL DEGREE CERTIFICATE
    This is to certify that John Doe has completed Bachelor of Technology in Computer Science
    Certificate Serial Number: CERT-2024-9988
    Date of Issue: 18/06/2024
    """
    res = classify_document(sample)
    assert res.document_type == "certificate"
    assert res.confidence > 0.5
