"""
PDF Audit Verification Report Generator Module.
Produces an executive, tamper-evident PDF verification summary using ReportLab.
Includes metadata, extracted fields, check tables, risk score gauge badge, and legal disclaimer.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

from config.config import REPORTS_DIR


def generate_verification_pdf_report(
    filename: str,
    document_type: str,
    classification_confidence: float,
    ocr_confidence: float,
    risk_score: int,
    risk_level: str,
    extracted_fields: Dict[str, Any],
    verification_checks: List[Dict[str, Any]],
    reasons: List[str],
    recommended_action: str,
    sha256_hash: str
) -> str:
    """
    Builds a professional, multi-section PDF verification audit report.
    Returns the absolute path to the generated PDF file.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_stem = Path(filename).stem
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"Verification_Report_{clean_stem}_{timestamp_str}.pdf"
    report_path = REPORTS_DIR / report_filename

    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A237E"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "ReportSubTitle",
        parent=normal_style,
        fontSize=10,
        textColor=colors.HexColor("#616161"),
        spaceAfter=15
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#283593"),
        spaceBefore=12,
        spaceAfter=6
    )
    bold_cell = ParagraphStyle(
        "BoldCell",
        parent=normal_style,
        fontSize=9,
        leading=11,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#212121")
    )
    regular_cell = ParagraphStyle(
        "RegularCell",
        parent=normal_style,
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#424242")
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("AI-Powered Smart Document Verification System", title_style))
    story.append(Paragraph("Official Forensic Audit & Risk Assessment Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A237E"), spaceAfter=12))

    # 2. Executive Summary Block
    risk_color = "#2E7D32" if risk_level == "LOW RISK" else ("#F57F17" if risk_level == "MEDIUM RISK" else "#C62828")

    meta_data = [
        [
            Paragraph("<b>Target Document:</b>", bold_cell),
            Paragraph(filename, regular_cell),
            Paragraph("<b>Audit Date:</b>", bold_cell),
            Paragraph(datetime.now().strftime("%d %b %Y, %H:%M:%S"), regular_cell)
        ],
        [
            Paragraph("<b>Document Type:</b>", bold_cell),
            Paragraph(document_type.replace("_", " ").title(), regular_cell),
            Paragraph("<b>Classification Conf.:</b>", bold_cell),
            Paragraph(f"{classification_confidence:.1f}%", regular_cell)
        ],
        [
            Paragraph("<b>OCR Confidence:</b>", bold_cell),
            Paragraph(f"{ocr_confidence:.1f}%", regular_cell),
            Paragraph("<b>SHA-256 Digest:</b>", bold_cell),
            Paragraph(f"{sha256_hash[:16]}...", regular_cell)
        ],
        [
            Paragraph("<b>Assessed Risk Score:</b>", bold_cell),
            Paragraph(f"<b><font color='{risk_color}'>{risk_score} / 100 ({risk_level})</font></b>", bold_cell),
            Paragraph("<b>Recommended Action:</b>", bold_cell),
            Paragraph(recommended_action, regular_cell)
        ]
    ]

    meta_table = Table(meta_data, colWidths=[110, 150, 110, 160])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8F9FA")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 3. Key Extracted Information
    story.append(Paragraph("Extracted Document Information", section_heading))
    if extracted_fields:
        field_rows = [[Paragraph("<b>Field Name</b>", bold_cell), Paragraph("<b>Extracted Value</b>", bold_cell)]]
        for k, v in extracted_fields.items():
            field_name = k.replace("_", " ").title()
            val_str = str(v) if v is not None else "N/A"
            field_rows.append([Paragraph(field_name, bold_cell), Paragraph(val_str, regular_cell)])

        info_table = Table(field_rows, colWidths=[180, 350])
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EAF6")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(info_table)
    else:
        story.append(Paragraph("No structured fields were extracted.", regular_cell))

    story.append(Spacer(1, 10))

    # 4. Authenticity Checks
    story.append(Paragraph("Authenticity & Integrity Verification Checks", section_heading))
    if verification_checks:
        check_rows = [[
            Paragraph("<b>Verification Check</b>", bold_cell),
            Paragraph("<b>Status</b>", bold_cell),
            Paragraph("<b>Diagnostic Message</b>", bold_cell)
        ]]
        for chk in verification_checks:
            c_name = chk.get("name", "Check")
            c_status = chk.get("status", "PASS")
            c_msg = chk.get("message", "")

            status_color = "#2E7D32" if c_status == "PASS" else ("#F57F17" if c_status == "WARN" else "#C62828")
            status_p = Paragraph(f"<b><font color='{status_color}'>{c_status}</font></b>", bold_cell)

            check_rows.append([
                Paragraph(c_name, bold_cell),
                status_p,
                Paragraph(c_msg, regular_cell)
            ])

        chk_table = Table(check_rows, colWidths=[150, 70, 310])
        chk_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EAF6")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(chk_table)

    story.append(Spacer(1, 10))

    # 5. Detected Anomalies & Scoring Reasons
    story.append(Paragraph("Risk Factors & Detected Anomalies", section_heading))
    if reasons:
        for r in reasons:
            story.append(Paragraph(f"• {r}", regular_cell))
    else:
        story.append(Paragraph("• No critical anomalies or fraud-like characteristics identified.", regular_cell))

    story.append(Spacer(1, 15))

    # 6. Legal Academic Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#BDBDBD"), spaceAfter=8))
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=normal_style,
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#757575")
    )
    disclaimer_text = (
        "<b>DISCLAIMER:</b> This verification report was generated automatically using optical character recognition, "
        "statistical machine learning classification, and computer vision forensic filters. Fraud suspicion is "
        "presented as a risk probability metric for human decision support and does not constitute definitive legal proof "
        "of forgery or authenticity. Developed as an academic AI/ML project prototype."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    # Build PDF
    doc.build(story)
    return str(report_path)
