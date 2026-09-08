"""
Integration test for complete end-to-end document verification pipeline:
Upload -> File Validation -> Preprocessing -> OCR -> Classification
-> Information Extraction -> Authenticity Verification -> Image Forensics
-> Explainable Risk Scoring -> SQLite Storage -> PDF Report Generation.
"""

from io import BytesIO
from pathlib import Path
from PIL import Image
import pymupdf
import pytest

from modules.utils import validate_uploaded_file, load_document_as_images, compute_sha256, compute_perceptual_hash
from modules.preprocessing import preprocess_image
from modules.ocr import perform_ocr
from modules.classification import classify_document
from modules.extraction import extract_information
from modules.authenticity import verify_authenticity
from modules.manipulation import analyze_manipulation
from modules.risk_scoring import calculate_risk_score
from modules.database import insert_verification_record, get_verification_by_id, delete_verification, check_for_duplicate
from modules.reporting import generate_verification_pdf_report


def create_mock_invoice_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    text = """TAX INVOICE
Seller: Apex Tech Solutions
GSTIN: 27AABCT9845A1Z5
Invoice Number: INV-9901
Invoice Date: 12/04/2025
Due Date: 26/04/2025

Billed To: Aarav Sharma
Subtotal: INR 5000.00
Applicable GST (18%): INR 900.00
Grand Total Amount Due: INR 5900.00

Payment Terms: Net 15 Days.
Authorized Signatory: Apex Accounts"""
    page.insert_text((50, 80), text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_end_to_end_verification_pipeline():
    # 1. Simulate Upload
    pdf_bytes = create_mock_invoice_pdf()
    filename = "integration_test_invoice.pdf"

    # 2. File Validation
    is_valid, err = validate_uploaded_file(pdf_bytes, filename)
    assert is_valid is True
    assert err is None

    # 3. Document Loader
    images = load_document_as_images(pdf_bytes, filename)
    assert len(images) >= 1
    target_img = images[0]

    # 4. Computer Vision Preprocessing
    preprocessed_img, stages = preprocess_image(target_img)
    assert isinstance(preprocessed_img, Image.Image)
    assert "grayscale" in stages

    # 5. OCR Execution (using direct PDF stream fallback if Tesseract is not installed)
    ocr_result = perform_ocr(images, raw_pdf_bytes=pdf_bytes)
    assert len(ocr_result.text) > 20
    assert ocr_result.confidence > 0

    # 6. ML Document Classification
    cls_result = classify_document(ocr_result.text)
    assert cls_result.document_type == "invoice"
    assert cls_result.is_confident is True

    # 7. Key Field Extraction
    extracted_data = extract_information(ocr_result.text, cls_result.document_type)
    assert extracted_data.raw_fields.get("invoice_number") == "INV-9901"
    assert extracted_data.raw_fields.get("subtotal") == 5000.00
    assert extracted_data.raw_fields.get("total_amount") == 5900.00

    # 8. Cryptographic & Perceptual Hashing
    sha256_hash = compute_sha256(pdf_bytes)
    phash = compute_perceptual_hash(target_img)
    assert len(sha256_hash) == 64

    # 9. Authenticity & Consistency Checks
    auth_report = verify_authenticity(
        extracted_data,
        ocr_confidence=ocr_result.confidence,
        file_sha256=sha256_hash,
        file_phash=phash,
        check_duplicate_fn=check_for_duplicate
    )
    assert auth_report.math_consistent is True
    assert auth_report.dates_consistent is True

    # 10. Forensic Image Manipulation Analysis (ELA)
    manip_report = analyze_manipulation(target_img)
    assert isinstance(manip_report.manipulation_score, float)

    # 11. Explainable Risk Scoring
    risk_assessment = calculate_risk_score(
        classification=cls_result,
        authenticity=auth_report,
        manipulation=manip_report,
        ocr_confidence=ocr_result.confidence
    )
    assert risk_assessment.risk_level == "LOW RISK"
    assert risk_assessment.score <= 30

    # 12. SQLite Database Persistence
    rec_id = insert_verification_record(
        filename=filename,
        sha256_hash=sha256_hash,
        phash=phash,
        document_type=cls_result.document_type,
        classification_confidence=cls_result.confidence_percentage,
        ocr_confidence=ocr_result.confidence,
        risk_score=risk_assessment.score,
        risk_level=risk_assessment.risk_level,
        extracted_fields=extracted_data.masked_fields,
        verification_checks=[c.to_dict() for c in auth_report.checks],
        reasons=risk_assessment.reasons
    )
    assert rec_id > 0

    # 13. PDF Audit Report Generation
    report_path = generate_verification_pdf_report(
        filename=filename,
        document_type=cls_result.document_type,
        classification_confidence=cls_result.confidence_percentage,
        ocr_confidence=ocr_result.confidence,
        risk_score=risk_assessment.score,
        risk_level=risk_assessment.risk_level,
        extracted_fields=extracted_data.masked_fields,
        verification_checks=[c.to_dict() for c in auth_report.checks],
        reasons=risk_assessment.reasons,
        recommended_action=risk_assessment.recommended_action,
        sha256_hash=sha256_hash
    )
    assert Path(report_path).exists()
    assert Path(report_path).stat().st_size > 500

    # Cleanup test record
    delete_verification(rec_id)
