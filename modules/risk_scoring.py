"""
Explainable Risk Scoring and Fraud Suspicion Engine.
Combines signals from classification confidence, OCR quality, authenticity checks,
mathematical arithmetic consistency, chronological logic, and image forensics
into an explainable 0-100 composite risk assessment score.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from config.config import settings, RiskWeights, RiskThresholds
from modules.classification import ClassificationResult
from modules.authenticity import AuthenticityReport
from modules.manipulation import ManipulationReport


@dataclass
class RiskBreakdownItem:
    """Individual rule penalty contributing to the composite risk score."""
    factor: str
    penalty_points: int
    reason: str
    severity: str  # "HIGH", "MEDIUM", "LOW"


@dataclass
class RiskAssessment:
    """Comprehensive, explainable risk score result."""
    score: int  # 0 to 100
    risk_level: str  # "LOW RISK", "MEDIUM RISK", "HIGH RISK"
    color_code: str  # Hex color for badges / gauges
    recommended_action: str
    reasons: List[str] = field(default_factory=list)
    breakdown: List[RiskBreakdownItem] = field(default_factory=list)
    disclaimer: str = (
        "This system provides an automated probabilistic risk assessment "
        "and does not constitute definitive proof of document fraud or authenticity."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "risk_level": self.risk_level,
            "recommended_action": self.recommended_action,
            "reasons": self.reasons,
            "breakdown": [
                {
                    "factor": b.factor,
                    "points": b.penalty_points,
                    "reason": b.reason,
                    "severity": b.severity
                }
                for b in self.breakdown
            ],
            "disclaimer": self.disclaimer
        }


def calculate_risk_score(
    classification: ClassificationResult,
    authenticity: AuthenticityReport,
    manipulation: ManipulationReport,
    ocr_confidence: float,
    custom_weights: Optional[RiskWeights] = None
) -> RiskAssessment:
    """
    Computes explainable composite risk score based on multi-factor penalties.
    """
    weights = custom_weights or settings.risk_weights
    thresholds = settings.risk_thresholds

    total_score = 0
    breakdown: List[RiskBreakdownItem] = []
    reasons: List[str] = []

    # 1. Missing Required Fields
    failed_req = any(c.name == "Required Fields Verification" and c.status == "FAIL" for c in authenticity.checks)
    if failed_req:
        total_score += weights.missing_required_fields
        breakdown.append(RiskBreakdownItem(
            factor="Missing Required Fields",
            penalty_points=weights.missing_required_fields,
            reason="One or more mandatory document fields could not be identified or extracted.",
            severity="MEDIUM"
        ))
        reasons.append("Missing required fields for identified document type.")

    # 2. Mathematical Inconsistency (Invoices / Balances)
    if not authenticity.math_consistent:
        total_score += weights.mathematical_mismatch
        breakdown.append(RiskBreakdownItem(
            factor="Mathematical Inconsistency",
            penalty_points=weights.mathematical_mismatch,
            reason="Significant mismatch between stated totals and calculated subtotals/tax.",
            severity="HIGH"
        ))
        reasons.append("Mathematical error: Total amount does not match sum of subtotal and tax.")

    # 3. Possible Image Manipulation (ELA & Compression Anomaly)
    if manipulation.has_suspicious_regions:
        total_score += weights.possible_image_manipulation
        breakdown.append(RiskBreakdownItem(
            factor="Image Manipulation Indicators",
            penalty_points=weights.possible_image_manipulation,
            reason=f"Detected {len(manipulation.regions)} region(s) with compression/edge artifacts in Error Level Analysis.",
            severity="HIGH"
        ))
        reasons.append(f"Visual anomaly: {len(manipulation.regions)} potential image manipulation patch(es) detected.")

    # 4. Duplicate Document Detection
    if authenticity.has_duplicates:
        total_score += weights.duplicate_document
        breakdown.append(RiskBreakdownItem(
            factor="Duplicate Document",
            penalty_points=weights.duplicate_document,
            reason="File hash or perceptual visual hash matches a previously analyzed document in the database.",
            severity="HIGH"
        ))
        reasons.append("Duplicate alert: Document matches an existing record in the verification database.")

    # 5. Chronological / Logical Inconsistency
    if not authenticity.dates_consistent:
        total_score += weights.date_inconsistency
        breakdown.append(RiskBreakdownItem(
            factor="Date Inconsistency",
            penalty_points=weights.date_inconsistency,
            reason="Chronological conflict detected between dates (e.g. issue date after due date / invalid DOB).",
            severity="MEDIUM"
        ))
        reasons.append("Logical date conflict: Sequence of dates does not follow temporal validity.")

    # 6. Low OCR Confidence
    if ocr_confidence < settings.ocr_min_confidence_warning:
        total_score += weights.low_ocr_confidence
        breakdown.append(RiskBreakdownItem(
            factor="Low OCR Clarity",
            penalty_points=weights.low_ocr_confidence,
            reason=f"Average OCR confidence ({ocr_confidence:.1f}%) is below optimal threshold ({settings.ocr_min_confidence_warning}%).",
            severity="LOW"
        ))
        reasons.append(f"OCR clarity warning: Low text recognition confidence ({ocr_confidence:.1f}%).")

    # 7. Low Classification Confidence
    if not classification.is_confident or classification.document_type == "unknown":
        total_score += weights.low_classification_confidence
        breakdown.append(RiskBreakdownItem(
            factor="Uncertain Document Category",
            penalty_points=weights.low_classification_confidence,
            reason="Document could not be classified into standard document categories with high confidence.",
            severity="LOW"
        ))
        reasons.append("Document classification uncertainty: Model could not conclusively verify document category.")

    # Cap score strictly between 0 and 100
    final_score = min(100, max(0, total_score))

    # Categorize Risk Level
    if final_score <= thresholds.LOW_MAX:
        risk_level = thresholds.LOW_LABEL
        color_code = "#2E7D32"  # Green
        recommended_action = "Document Verified — Low suspicion. Standard processing approved."
        if not reasons:
            reasons.append("All primary authenticity, mathematical, and forensic checks passed successfully.")
    elif final_score <= thresholds.MEDIUM_MAX:
        risk_level = thresholds.MEDIUM_LABEL
        color_code = "#F57F17"  # Amber
        recommended_action = "Secondary Review Advised — Moderate risk indicators detected. Manual inspection recommended."
    else:
        risk_level = thresholds.HIGH_LABEL
        color_code = "#C62828"  # Red
        recommended_action = "Manual Verification Required — High fraud suspicion. Document flags require immediate escalation."

    return RiskAssessment(
        score=final_score,
        risk_level=risk_level,
        color_code=color_code,
        recommended_action=recommended_action,
        reasons=reasons,
        breakdown=breakdown
    )
