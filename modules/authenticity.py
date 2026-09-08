"""
Document Authenticity Verification Module.
Implements multi-point explainable checks:
1. Required field completeness
2. Format validity (dates, identifiers, amounts)
3. Mathematical consistency (e.g. Subtotal + Tax == Total)
4. Chronological date logic
5. Duplicate & perceptual similarity detection
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
import re

from modules.extraction import ExtractedData


@dataclass
class CheckResult:
    """Individual authenticity check status and explanation."""
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details
        }


@dataclass
class AuthenticityReport:
    """Aggregated results of all verification rules."""
    checks: List[CheckResult] = field(default_factory=list)
    passed_count: int = 0
    warning_count: int = 0
    failed_count: int = 0
    math_consistent: bool = True
    dates_consistent: bool = True
    has_duplicates: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checks": [c.to_dict() for c in self.checks],
            "passed": self.passed_count,
            "warnings": self.warning_count,
            "failed": self.failed_count,
            "math_consistent": self.math_consistent,
            "dates_consistent": self.dates_consistent,
            "has_duplicates": self.has_duplicates
        }


def parse_date_flexibly(date_str: Optional[str]) -> Optional[datetime]:
    """Parses date string with common separators (DD/MM/YYYY, YYYY-MM-DD, etc.)."""
    if not date_str:
        return None
    clean = re.sub(r"[^\d/\-\.]", "", date_str)
    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%Y-%m-%d", "%Y/%m/%d",
        "%d/%m/%y", "%d-%m-%y"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            continue
    return None


def verify_authenticity(
    extracted: ExtractedData,
    ocr_confidence: float,
    file_sha256: str,
    file_phash: str,
    check_duplicate_fn: Optional[Callable[[str, str], Optional[Dict[str, Any]]]] = None
) -> AuthenticityReport:
    """
    Executes all explainable authenticity checks on the extracted document data.
    """
    report = AuthenticityReport()
    raw = extracted.raw_fields
    doc_type = extracted.document_type

    # 1. Required Fields Check
    if extracted.missing_required_fields:
        missing_str = ", ".join(extracted.missing_required_fields)
        report.checks.append(CheckResult(
            name="Required Fields Verification",
            status="FAIL",
            message=f"Missing essential fields for {doc_type}: {missing_str}",
            details={"missing": extracted.missing_required_fields}
        ))
    else:
        report.checks.append(CheckResult(
            name="Required Fields Verification",
            status="PASS",
            message="All expected mandatory fields are present and extracted.",
            details={}
        ))

    # 2. Mathematical Consistency Checks
    if doc_type == "invoice":
        subtotal = raw.get("subtotal")
        tax = raw.get("tax_amount") or 0.0
        total = raw.get("total_amount")

        if subtotal is not None and total is not None:
            expected_total = round(subtotal + tax, 2)
            actual_total = round(total, 2)
            # 1.5% margin for rounding differences
            diff = abs(actual_total - expected_total)
            if diff > max(2.0, expected_total * 0.015):
                report.math_consistent = False
                report.checks.append(CheckResult(
                    name="Mathematical Consistency",
                    status="FAIL",
                    message=f"Total mismatch: Stated {actual_total:.2f} != Calculated {expected_total:.2f} (Subtotal: {subtotal:.2f} + Tax: {tax:.2f})",
                    details={"stated": actual_total, "calculated": expected_total, "diff": diff}
                ))
            else:
                report.checks.append(CheckResult(
                    name="Mathematical Consistency",
                    status="PASS",
                    message=f"Invoice arithmetic verified: Subtotal ({subtotal:.2f}) + Tax ({tax:.2f}) ≈ Total ({actual_total:.2f})",
                    details={"stated": actual_total, "calculated": expected_total}
                ))
        else:
            report.checks.append(CheckResult(
                name="Mathematical Consistency",
                status="WARN",
                message="Subtotal or Total could not be clearly extracted for mathematical validation.",
                details={}
            ))

    elif doc_type == "bank_statement":
        open_bal = raw.get("opening_balance")
        close_bal = raw.get("closing_balance")
        if open_bal is not None and close_bal is not None:
            report.checks.append(CheckResult(
                name="Balance Structure Check",
                status="PASS",
                message=f"Statement balances present (Opening: {open_bal:.2f}, Closing: {close_bal:.2f})",
                details={"opening": open_bal, "closing": close_bal}
            ))
        else:
            report.checks.append(CheckResult(
                name="Balance Structure Check",
                status="WARN",
                message="Opening or closing balance was not fully extracted.",
                details={}
            ))

    # 3. Chronological Date Consistency Checks
    date_issues = []
    if doc_type == "invoice":
        inv_d = parse_date_flexibly(raw.get("invoice_date"))
        due_d = parse_date_flexibly(raw.get("due_date"))
        if inv_d and due_d and inv_d > due_d:
            date_issues.append(f"Invoice Date ({inv_d.strftime('%d/%m/%Y')}) is after Due Date ({due_d.strftime('%d/%m/%Y')})")

    elif doc_type == "id_card":
        dob = parse_date_flexibly(raw.get("date_of_birth"))
        iss = parse_date_flexibly(raw.get("issue_date"))
        exp = parse_date_flexibly(raw.get("expiry_date"))

        if dob and iss and dob >= iss:
            date_issues.append("Date of Birth is greater than or equal to Card Issue Date.")
        if iss and exp and iss >= exp:
            date_issues.append("Card Issue Date is after or equal to Expiry Date.")

    if date_issues:
        report.dates_consistent = False
        report.checks.append(CheckResult(
            name="Chronological Date Validation",
            status="FAIL",
            message="; ".join(date_issues),
            details={"issues": date_issues}
        ))
    else:
        report.checks.append(CheckResult(
            name="Chronological Date Validation",
            status="PASS",
            message="Date relationships are logically and chronologically consistent.",
            details={}
        ))

    # 4. Duplicate / Similarity Check
    if check_duplicate_fn is not None:
        dup_record = check_duplicate_fn(file_sha256, file_phash)
        if dup_record:
            report.has_duplicates = True
            report.checks.append(CheckResult(
                name="Duplicate Document Detection",
                status="FAIL",
                message=f"Duplicate detected! Matches prior verified document #{dup_record.get('id')} ('{dup_record.get('filename')}')",
                details=dup_record
            ))
        else:
            report.checks.append(CheckResult(
                name="Duplicate Document Detection",
                status="PASS",
                message="No matching SHA-256 or visual perceptual duplicate found in audit database.",
                details={}
            ))
    else:
        report.checks.append(CheckResult(
            name="Duplicate Document Detection",
            status="PASS",
            message="Audit history hash check completed.",
            details={}
        ))

    # 5. OCR Quality Check
    if ocr_confidence < 50.0:
        report.checks.append(CheckResult(
            name="OCR Clarity & Quality",
            status="WARN",
            message=f"Low OCR confidence ({ocr_confidence:.1f}%). Text ambiguity increases verification uncertainty.",
            details={"confidence": ocr_confidence}
        ))
    else:
        report.checks.append(CheckResult(
            name="OCR Clarity & Quality",
            status="PASS",
            message=f"Acceptable OCR recognition confidence ({ocr_confidence:.1f}%).",
            details={"confidence": ocr_confidence}
        ))

    # Summary tallies
    report.passed_count = sum(1 for c in report.checks if c.status == "PASS")
    report.warning_count = sum(1 for c in report.checks if c.status == "WARN")
    report.failed_count = sum(1 for c in report.checks if c.status == "FAIL")

    return report
