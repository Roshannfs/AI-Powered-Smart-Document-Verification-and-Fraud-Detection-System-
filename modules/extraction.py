"""
Key Information and Structured Data Extraction Module.
Uses regex patterns, keyword boundaries, and rule-based heuristics to extract
document-specific metadata with automated sensitive data masking.
"""

from dataclasses import dataclass, field
import re
from typing import Dict, Any, List, Optional

from modules.utils import mask_sensitive_identifier


@dataclass
class ExtractedData:
    """Standard container for extracted document information."""
    document_type: str
    raw_fields: Dict[str, Any] = field(default_factory=dict)
    masked_fields: Dict[str, Any] = field(default_factory=dict)
    missing_required_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "raw_fields": self.raw_fields,
            "masked_fields": self.masked_fields,
            "missing_required_fields": self.missing_required_fields
        }


# Generic regex helpers
DATE_PATTERN = r"\b(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})\b"
AMOUNT_PATTERN = r"(?:INR|Rs\.?|USD|\$|€|₹)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)"


def clean_amount(val_str: Optional[str]) -> Optional[float]:
    """Parses clean numeric float from amount strings."""
    if not val_str:
        return None
    try:
        cleaned = re.sub(r"[^\d.]", "", val_str)
        return float(cleaned)
    except Exception:
        return None


def extract_invoice_fields(text: str) -> ExtractedData:
    raw: Dict[str, Any] = {}
    masked: Dict[str, Any] = {}

    # Invoice Number: prefer explicit 'Invoice Number / No / #' or standalone INV-xxxx
    inv_no_match = re.search(r"(?:Invoice\s*(?:Number|No\.?|#)|Bill\s*(?:Number|No\.?))\s*[:\-]?\s*([A-Za-z0-9\-_]+)", text, re.IGNORECASE)
    if inv_no_match:
        raw["invoice_number"] = inv_no_match.group(1).strip()
    else:
        fallback_inv = re.search(r"\b(INV[-#][A-Za-z0-9\-_]+)\b", text, re.IGNORECASE)
        raw["invoice_number"] = fallback_inv.group(1).strip() if fallback_inv else None

    # Dates
    inv_date_match = re.search(r"(?:Invoice\s*Date|Bill\s*Date|Date)\s*[:\-]?\s*" + DATE_PATTERN, text, re.IGNORECASE)
    raw["invoice_date"] = inv_date_match.group(1).strip() if inv_date_match else None

    due_date_match = re.search(r"(?:Due\s*Date|Payment\s*Due)\s*[:\-]?\s*" + DATE_PATTERN, text, re.IGNORECASE)
    raw["due_date"] = due_date_match.group(1).strip() if due_date_match else None

    # Seller / Vendor
    seller_match = re.search(r"(?:Seller|Vendor|From|Billed By)\s*[:\-]?\s*([^\n\r]+)", text, re.IGNORECASE)
    raw["seller_name"] = seller_match.group(1).strip() if seller_match else None

    # Buyer / Customer
    buyer_match = re.search(r"(?:Billed\s*To|Buyer|Customer|Client Name)\s*[:\-]?\s*([^\n\r]+)", text, re.IGNORECASE)
    raw["buyer_name"] = buyer_match.group(1).strip() if buyer_match else None

    # Amounts
    subtotal_match = re.search(r"(?:Subtotal|Sub-Total|Net Amount)\s*[:\-]?\s*" + AMOUNT_PATTERN, text, re.IGNORECASE)
    raw["subtotal"] = clean_amount(subtotal_match.group(1)) if subtotal_match else None

    tax_match = re.search(r"(?:GST|Tax|VAT|Total Tax)\s*(?:\([^)]*\))?\s*[:\-]?\s*" + AMOUNT_PATTERN, text, re.IGNORECASE)
    raw["tax_amount"] = clean_amount(tax_match.group(1)) if tax_match else None

    total_match = re.search(r"(?:Grand\s*Total|Total\s*Amount|Amount\s*Due)\s*[:\-]?\s*" + AMOUNT_PATTERN, text, re.IGNORECASE)
    if not total_match:
        total_match = re.search(r"(?<!Sub)(?<!Sub-)\bTotal\s*[:\-]?\s*" + AMOUNT_PATTERN, text, re.IGNORECASE)
    raw["total_amount"] = clean_amount(total_match.group(1)) if total_match else None

    # Masked fields
    masked.update(raw)

    # Missing mandatory fields check
    required = ["invoice_number", "invoice_date", "total_amount"]
    missing = [f for f in required if raw.get(f) is None]

    return ExtractedData("invoice", raw, masked, missing)


def extract_bank_statement_fields(text: str) -> ExtractedData:
    raw: Dict[str, Any] = {}
    masked: Dict[str, Any] = {}

    # Account Holder
    name_match = re.search(r"(?:Account\s*Holder(?:\s*Name)?|Name)\s*[:\-]?\s*([^\n\r]+)", text, re.IGNORECASE)
    raw["account_holder"] = name_match.group(1).strip() if name_match else None

    # Account Number
    acc_match = re.search(r"(?:Account\s*(?:Number|No\.?|#))\s*[:\-]?\s*([0-9X*]{8,18})", text, re.IGNORECASE)
    raw["account_number"] = acc_match.group(1).strip() if acc_match else None

    # Statement Period
    period_match = re.search(r"(?:Statement\s*Period|Period)\s*[:\-]?\s*([^\n\r]+)", text, re.IGNORECASE)
    raw["statement_period"] = period_match.group(1).strip() if period_match else None

    # Balances
    open_bal_match = re.search(r"(?:Opening\s*Balance)\s*[:\-]?\s*" + AMOUNT_PATTERN, text, re.IGNORECASE)
    raw["opening_balance"] = clean_amount(open_bal_match.group(1)) if open_bal_match else None

    close_bal_match = re.search(r"(?:Closing(?:\s*Available)?\s*Balance)\s*[:\-]?\s*" + AMOUNT_PATTERN, text, re.IGNORECASE)
    raw["closing_balance"] = clean_amount(close_bal_match.group(1)) if close_bal_match else None

    # Transaction count
    tx_match = re.search(r"(?:Total\s*Transaction\s*Count|Transactions)\s*[:\-]?\s*(\d+)", text, re.IGNORECASE)
    raw["transaction_count"] = int(tx_match.group(1)) if tx_match else None

    # Mask sensitive account number
    masked.update(raw)
    masked["account_number"] = mask_sensitive_identifier(raw.get("account_number"))

    required = ["account_holder", "account_number", "closing_balance"]
    missing = [f for f in required if raw.get(f) is None]

    return ExtractedData("bank_statement", raw, masked, missing)


def extract_certificate_fields(text: str) -> ExtractedData:
    raw: Dict[str, Any] = {}
    masked: Dict[str, Any] = {}

    # Candidate Name
    name_match = re.search(r"(?:certify that|Name of Candidate)\s*[\r\n\:]*\s*([A-Za-z\s]+?)(?=\n|Roll|Registration|has|$)", text, re.IGNORECASE)
    raw["candidate_name"] = name_match.group(1).strip() if name_match else None

    # Institution
    inst_match = re.search(r"^(.*?)(?:PROVISIONAL|CERTIFICATE|UNIVERSITY|INSTITUTE)", text, re.IGNORECASE | re.MULTILINE)
    raw["institution"] = inst_match.group(1).strip() if inst_match and len(inst_match.group(1).strip()) > 3 else "Academic Institution"

    # Degree / Program
    deg_match = re.search(r"(?:award of|degree of|passed the examination for)\s*[:\-]?\s*([^\n\r]+)", text, re.IGNORECASE)
    raw["degree"] = deg_match.group(1).strip() if deg_match else None

    # Certificate / Roll No
    cert_no_match = re.search(r"(?:Certificate\s*(?:Serial\s*)?(?:Number|No\.?)|Roll\s*(?:Number|No\.?)|Registration\s*No\.?)\s*[:\-]?\s*([A-Za-z0-9\-_]+)", text, re.IGNORECASE)
    raw["certificate_number"] = cert_no_match.group(1).strip() if cert_no_match else None

    # Issue Date
    date_match = re.search(r"(?:Date\s*of\s*Issue|Issue\s*Date|Dated)\s*[:\-]?\s*" + DATE_PATTERN, text, re.IGNORECASE)
    raw["issue_date"] = date_match.group(1).strip() if date_match else None

    masked.update(raw)
    required = ["candidate_name", "degree", "certificate_number"]
    missing = [f for f in required if raw.get(f) is None]

    return ExtractedData("certificate", raw, masked, missing)


def extract_id_card_fields(text: str) -> ExtractedData:
    raw: Dict[str, Any] = {}
    masked: Dict[str, Any] = {}

    # Name
    name_match = re.search(r"(?:Cardholder\s*Name|Name)\s*[:\-]?\s*([^\n\r]+)", text, re.IGNORECASE)
    if name_match:
        clean_name = re.split(r"\b(?:Identification|ID|Date|DOB|Gender)\b", name_match.group(1).strip(), flags=re.IGNORECASE)[0].strip()
        raw["name"] = clean_name if clean_name else None
    else:
        raw["name"] = None

    # ID Number
    id_match = re.search(r"(?:Identification\s*Number|ID\s*(?:Number|No\.?)|DL\s*No\.?)\s*[:\-]?\s*([A-Za-z0-9\-]+)", text, re.IGNORECASE)
    raw["id_number"] = id_match.group(1).strip() if id_match else None

    # Date of Birth
    dob_match = re.search(r"(?:Date\s*of\s*Birth|DOB)\s*[:\-]?\s*" + DATE_PATTERN, text, re.IGNORECASE)
    raw["date_of_birth"] = dob_match.group(1).strip() if dob_match else None

    # Issue Date & Expiry Date
    issue_match = re.search(r"(?:Date\s*of\s*Issue|Issue\s*Date)\s*[:\-]?\s*" + DATE_PATTERN, text, re.IGNORECASE)
    raw["issue_date"] = issue_match.group(1).strip() if issue_match else None

    exp_match = re.search(r"(?:Date\s*of\s*Expiry|Expiry\s*Date|Valid\s*Till)\s*[:\-]?\s*" + DATE_PATTERN, text, re.IGNORECASE)
    raw["expiry_date"] = exp_match.group(1).strip() if exp_match else None

    masked.update(raw)
    masked["id_number"] = mask_sensitive_identifier(raw.get("id_number"))

    required = ["name", "id_number"]
    missing = [f for f in required if raw.get(f) is None]

    return ExtractedData("id_card", raw, masked, missing)


def extract_application_form_fields(text: str) -> ExtractedData:
    raw: Dict[str, Any] = {}
    masked: Dict[str, Any] = {}

    # Applicant Name
    name_match = re.search(r"(?:Full\s*Name|Applicant\s*Name|Name)\s*[:\-]?\s*([^\n\r]+)", text, re.IGNORECASE)
    if name_match:
        clean_name = re.split(r"\b(?:Email|Phone|Contact|Date|Ref)\b", name_match.group(1).strip(), flags=re.IGNORECASE)[0].strip()
        raw["applicant_name"] = clean_name if clean_name else None
    else:
        raw["applicant_name"] = None

    # Application Ref Number
    ref_match = re.search(r"(?:Reference\s*Number|Application\s*(?:No\.?|ID|#))\s*[:\-]?\s*([A-Za-z0-9\-_]+)", text, re.IGNORECASE)
    raw["reference_number"] = ref_match.group(1).strip() if ref_match else None

    # Date
    date_match = re.search(r"(?:Application\s*Date|Date\s*of\s*Submission|Date)\s*[:\-]?\s*" + DATE_PATTERN, text, re.IGNORECASE)
    raw["application_date"] = date_match.group(1).strip() if date_match else None

    # Contact Details
    email_match = re.search(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b", text)
    raw["email"] = email_match.group(1).strip() if email_match else None

    phone_match = re.search(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)
    raw["phone"] = phone_match.group(0).strip() if phone_match else None

    masked.update(raw)
    if raw.get("email"):
        # Mask email e.g. a***b@domain.com
        parts = raw["email"].split("@")
        if len(parts[0]) > 2:
            masked["email"] = f"{parts[0][0]}***{parts[0][-1]}@{parts[1]}"
    if raw.get("phone"):
        masked["phone"] = mask_sensitive_identifier(raw["phone"])

    required = ["applicant_name", "reference_number"]
    missing = [f for f in required if raw.get(f) is None]

    return ExtractedData("application_form", raw, masked, missing)


def extract_information(text: str, document_type: str) -> ExtractedData:
    """
    Main extraction dispatcher based on predicted or verified document type.
    """
    dispatchers = {
        "invoice": extract_invoice_fields,
        "bank_statement": extract_bank_statement_fields,
        "certificate": extract_certificate_fields,
        "id_card": extract_id_card_fields,
        "application_form": extract_application_form_fields,
    }

    extractor = dispatchers.get(document_type)
    if extractor:
        return extractor(text)

    # Unknown document fallback
    return ExtractedData(
        document_type="unknown",
        raw_fields={"text_snippet": text[:200]},
        masked_fields={"text_snippet": text[:200]},
        missing_required_fields=[]
    )
