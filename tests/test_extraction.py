"""Unit tests for information extraction module."""

import pytest
from modules.extraction import (
    extract_information,
    extract_invoice_fields,
    extract_bank_statement_fields,
    extract_certificate_fields,
    extract_id_card_fields,
    extract_application_form_fields,
    clean_amount
)


def test_clean_amount():
    assert clean_amount("INR 5,000.00") == 5000.00
    assert clean_amount("$120.50") == 120.50
    assert clean_amount("invalid") is None
    assert clean_amount(None) is None


def test_extract_invoice():
    text = """
    TAX INVOICE
    Seller: Acme Hardware Inc
    Billed To: Jane Doe
    Invoice Number: INV-2025-998
    Invoice Date: 15/03/2025
    Due Date: 30/03/2025
    Subtotal: INR 2000.00
    GST: INR 360.00
    Grand Total: INR 2360.00
    """
    data = extract_invoice_fields(text)
    assert data.document_type == "invoice"
    assert data.raw_fields["invoice_number"] == "INV-2025-998"
    assert data.raw_fields["invoice_date"] == "15/03/2025"
    assert data.raw_fields["subtotal"] == 2000.00
    assert data.raw_fields["tax_amount"] == 360.00
    assert data.raw_fields["total_amount"] == 2360.00
    assert len(data.missing_required_fields) == 0


def test_extract_bank_statement_with_masking():
    text = """
    STATE RESERVE BANK
    Account Holder Name: Aarav Sharma
    Account Number: 987654321012
    Opening Balance: INR 10000.00
    Closing Balance: INR 12000.00
    Total Transaction Count: 5
    """
    data = extract_bank_statement_fields(text)
    assert data.document_type == "bank_statement"
    assert data.raw_fields["account_holder"] == "Aarav Sharma"
    assert data.raw_fields["account_number"] == "987654321012"
    # Masked field should not expose full number
    assert data.masked_fields["account_number"] == "XXXXXXXX1012"
    assert data.raw_fields["opening_balance"] == 10000.00
    assert data.raw_fields["closing_balance"] == 12000.00


def test_extract_id_card():
    text = """
    GOVERNMENT IDENTITY CARD
    Cardholder Name: David Miller
    Identification Number: DL-9922-1100
    Date of Birth: 14/08/1990
    Date of Issue: 10/01/2020
    Date of Expiry: 10/01/2030
    """
    data = extract_id_card_fields(text)
    assert data.document_type == "id_card"
    assert data.raw_fields["name"] == "David Miller"
    assert data.masked_fields["id_number"].endswith("1100")
    assert data.raw_fields["date_of_birth"] == "14/08/1990"
