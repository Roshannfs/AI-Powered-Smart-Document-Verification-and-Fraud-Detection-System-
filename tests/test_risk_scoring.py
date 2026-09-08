"""Unit tests for risk scoring engine."""

import pytest
from PIL import Image
from modules.classification import ClassificationResult
from modules.authenticity import AuthenticityReport, CheckResult
from modules.manipulation import ManipulationReport
from modules.risk_scoring import calculate_risk_score, RiskAssessment


def test_calculate_risk_score_low_risk():
    cls_res = ClassificationResult(
        document_type="invoice",
        display_name="Invoice",
        confidence=0.95,
        confidence_percentage=95.0,
        probabilities={"invoice": 0.95},
        is_confident=True
    )
    auth_rep = AuthenticityReport(
        checks=[CheckResult(name="Required Fields Verification", status="PASS", message="OK")],
        passed_count=5,
        warning_count=0,
        failed_count=0,
        math_consistent=True,
        dates_consistent=True,
        has_duplicates=False
    )
    dummy_img = Image.new("RGB", (100, 100), color="white")
    manip_rep = ManipulationReport(
        has_suspicious_regions=False,
        manipulation_score=0.0,
        ela_image=dummy_img,
        annotated_image=dummy_img,
        regions=[],
        reasons=["Clean"]
    )

    assessment = calculate_risk_score(cls_res, auth_rep, manip_rep, ocr_confidence=92.0)
    assert isinstance(assessment, RiskAssessment)
    assert assessment.score <= 30
    assert assessment.risk_level == "LOW RISK"
    assert "Verified" in assessment.recommended_action


def test_calculate_risk_score_high_risk_on_tampering():
    cls_res = ClassificationResult(
        document_type="invoice",
        display_name="Invoice",
        confidence=0.92,
        confidence_percentage=92.0,
        probabilities={"invoice": 0.92},
        is_confident=True
    )
    auth_rep = AuthenticityReport(
        checks=[
            CheckResult(name="Required Fields Verification", status="FAIL", message="Missing total"),
            CheckResult(name="Mathematical Consistency", status="FAIL", message="Mismatch")
        ],
        passed_count=2,
        warning_count=0,
        failed_count=2,
        math_consistent=False,
        dates_consistent=False,
        has_duplicates=False
    )
    dummy_img = Image.new("RGB", (100, 100), color="white")
    manip_rep = ManipulationReport(
        has_suspicious_regions=True,
        manipulation_score=0.75,
        ela_image=dummy_img,
        annotated_image=dummy_img,
        regions=[None],  # 1 region
        reasons=["Tampered patch detected"]
    )

    assessment = calculate_risk_score(cls_res, auth_rep, manip_rep, ocr_confidence=45.0)
    # Missing fields (+15) + Math (+20) + Manip (+25) + Date (+15) + Low OCR (+10) = 85
    assert assessment.score >= 61
    assert assessment.risk_level == "HIGH RISK"
    assert "Manual Verification Required" in assessment.recommended_action
    assert len(assessment.reasons) >= 3
