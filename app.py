"""
AI-Powered Smart Document Verification and Fraud Detection System
Using OCR, Computer Vision and Machine Learning.

Streamlit Interactive Dashboard & Web Interface.
"""

import io
import json
import os
from pathlib import Path
import sys
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from config.config import (
    settings,
    GENUINE_SAMPLES_DIR,
    SUSPICIOUS_SAMPLES_DIR,
    MODELS_DIR,
    DB_PATH
)
from modules.utils import (
    validate_uploaded_file,
    load_document_as_images,
    compute_sha256,
    compute_perceptual_hash
)
from modules.preprocessing import preprocess_image, PreprocessingOptions
from modules.ocr import perform_ocr, OCRResult
from modules.classification import classify_document, ClassificationResult
from modules.extraction import extract_information, ExtractedData
from modules.authenticity import verify_authenticity, AuthenticityReport
from modules.manipulation import analyze_manipulation, ManipulationReport
from modules.risk_scoring import calculate_risk_score, RiskAssessment
from modules.database import (
    insert_verification_record,
    get_all_verifications,
    get_verification_by_id,
    delete_verification,
    clear_all_history,
    check_for_duplicate
)
from modules.reporting import generate_verification_pdf_report

# Page Config
st.set_page_config(
    page_title="AI Document Verification & Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-header h1 {
        color: white;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #B0BEC5;
        font-size: 1rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(66, 165, 245, 0.5);
    }
    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #90A4AE;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ECEFF1;
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-low {
        background: rgba(46, 125, 50, 0.2);
        color: #4CAF50;
        border: 1px solid #2E7D32;
    }
    .badge-medium {
        background: rgba(245, 127, 23, 0.2);
        color: #FFB300;
        border: 1px solid #F57F17;
    }
    .badge-high {
        background: rgba(198, 40, 40, 0.2);
        color: #EF5350;
        border: 1px solid #C62828;
    }

    .check-card {
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 5px solid #9E9E9E;
        background: rgba(255, 255, 255, 0.02);
    }
    .check-pass {
        border-left-color: #2E7D32;
        background: rgba(46, 125, 50, 0.06);
    }
    .check-warn {
        border-left-color: #F57F17;
        background: rgba(245, 127, 23, 0.06);
    }
    .check-fail {
        border-left-color: #C62828;
        background: rgba(198, 40, 40, 0.06);
    }

    .disclaimer-box {
        background: rgba(33, 150, 243, 0.06);
        border: 1px solid rgba(33, 150, 243, 0.3);
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #90CAF9;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


def create_risk_gauge(score: int, level: str, color_code: str):
    """Generates an interactive Plotly gauge chart for the risk score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risk Assessment: <b>{level}</b>", 'font': {'size': 18, 'color': color_code}},
        number={'suffix': " / 100", 'font': {'size': 28, 'color': "#ECEFF1"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#78909C"},
            'bar': {'color': color_code, 'thickness': 0.28},
            'bgcolor': "rgba(255, 255, 255, 0.05)",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.15)",
            'steps': [
                {'range': [0, 30], 'color': "rgba(46, 125, 50, 0.25)"},
                {'range': [30, 60], 'color': "rgba(245, 127, 23, 0.25)"},
                {'range': [60, 100], 'color': "rgba(198, 40, 40, 0.25)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def render_sidebar():
    """Renders the persistent global sidebar with controls and environment diagnostics."""
    with st.sidebar:
        st.markdown("### 🛡️ DocVerify AI")
        st.caption("AI-Powered Smart Document Verification System")
        st.markdown("---")

        st.markdown("#### ⚙️ Engine Diagnostics")
        if settings.tesseract_cmd:
            st.success(f"✓ **Tesseract OCR Active**\n`{Path(settings.tesseract_cmd).name}`")
        else:
            st.info("ℹ️ **Fallback OCR Active**\n(PyMuPDF Stream Engine)")

        model_loaded = MODELS_DIR.joinpath("document_classifier.pkl").exists()
        if model_loaded:
            st.success("✓ **ML Classifier Model**: Loaded")
        else:
            st.warning("⚠️ **ML Classifier**: Not trained yet")

        st.markdown("---")
        st.markdown("#### ⚖️ Risk Scoring Weights")
        with st.expander("View Weight Allocations"):
            w = settings.risk_weights
            st.write(f"• Manipulation Artifacts: **+{w.possible_image_manipulation}**")
            st.write(f"• Mathematical Mismatch: **+{w.mathematical_mismatch}**")
            st.write(f"• Duplicate File Detected: **+{w.duplicate_document}**")
            st.write(f"• Date Logic Conflict: **+{w.date_inconsistency}**")
            st.write(f"• Missing Required Fields: **+{w.missing_required_fields}**")
            st.write(f"• Low OCR Quality: **+{w.low_ocr_confidence}**")
            st.write(f"• Low Class Confidence: **+{w.low_classification_confidence}**")

        st.markdown("---")
        st.caption("College Mini Project • Computer Vision & Machine Learning")


def process_document_pipeline(file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """
    Executes complete end-to-end verification pipeline on the supplied document bytes.
    """
    # 1. Validation
    is_valid, err_msg = validate_uploaded_file(file_bytes, filename)
    if not is_valid:
        st.error(f"File Validation Failed: {err_msg}")
        return None

    # 2. Rendering into images
    images = load_document_as_images(file_bytes, filename)
    if not images:
        st.error("Failed to render document into images.")
        return None

    primary_image = images[0]

    # 3. Preprocessing
    preproc_options = PreprocessingOptions(
        apply_deskew=True,
        apply_denoise=True,
        apply_clahe=True,
        apply_threshold=True,
        threshold_method="otsu"
    )
    preprocessed_img, stages = preprocess_image(primary_image, preproc_options)

    # 4. OCR
    ocr_result = perform_ocr(
        images,
        raw_pdf_bytes=file_bytes if filename.lower().endswith(".pdf") else None,
        filename=filename
    )

    # 5. ML Classification
    classification = classify_document(ocr_result.text)

    # 6. Structured Information Extraction
    extracted = extract_information(ocr_result.text, classification.document_type)

    # 7. Document Hashes & Duplicates
    sha256_hash = compute_sha256(file_bytes)
    phash = compute_perceptual_hash(primary_image)

    # 8. Authenticity Verification
    authenticity = verify_authenticity(
        extracted=extracted,
        ocr_confidence=ocr_result.confidence,
        file_sha256=sha256_hash,
        file_phash=phash,
        check_duplicate_fn=check_for_duplicate
    )

    # 9. Computer Vision Manipulation Analysis (ELA)
    manipulation = analyze_manipulation(primary_image)

    # 10. Explainable Risk Scoring
    risk = calculate_risk_score(
        classification=classification,
        authenticity=authenticity,
        manipulation=manipulation,
        ocr_confidence=ocr_result.confidence
    )

    # 11. PDF Report Generation
    report_path = generate_verification_pdf_report(
        filename=filename,
        document_type=classification.document_type,
        classification_confidence=classification.confidence_percentage,
        ocr_confidence=ocr_result.confidence,
        risk_score=risk.score,
        risk_level=risk.risk_level,
        extracted_fields=extracted.masked_fields,
        verification_checks=[c.to_dict() for c in authenticity.checks],
        reasons=risk.reasons,
        recommended_action=risk.recommended_action,
        sha256_hash=sha256_hash
    )

    # 12. Save Record to SQLite
    rec_id = insert_verification_record(
        filename=filename,
        sha256_hash=sha256_hash,
        phash=phash,
        document_type=classification.document_type,
        classification_confidence=classification.confidence_percentage,
        ocr_confidence=ocr_result.confidence,
        risk_score=risk.score,
        risk_level=risk.risk_level,
        extracted_fields=extracted.masked_fields,
        verification_checks=[c.to_dict() for c in authenticity.checks],
        reasons=risk.reasons,
        report_path=report_path
    )

    return {
        "filename": filename,
        "record_id": rec_id,
        "primary_image": primary_image,
        "preprocessed_image": preprocessed_img,
        "stages": stages,
        "ocr": ocr_result,
        "classification": classification,
        "extracted": extracted,
        "authenticity": authenticity,
        "manipulation": manipulation,
        "risk": risk,
        "sha256": sha256_hash,
        "report_path": report_path
    }


def render_analysis_results(res: Dict[str, Any], key_prefix: str = "main"):
    """Renders the comprehensive analysis dashboard for verified document."""
    st.markdown("---")

    # Header Summary Bar
    risk = res["risk"]
    cls = res["classification"]
    ocr = res["ocr"]

    badge_class = "badge-low" if risk.risk_level == "LOW RISK" else ("badge-medium" if risk.risk_level == "MEDIUM RISK" else "badge-high")

    col_meta, col_gauge = st.columns([1.8, 1.2])

    with col_meta:
        st.markdown(f"### Verification Audit: `{res['filename']}`")
        st.markdown(f"""
        <div style="margin-bottom: 12px;">
            <span class="badge-pill {badge_class}">{risk.risk_level}</span>
            <span style="margin-left: 10px; font-weight: 600; color: #B0BEC5;">Document ID: #{res['record_id']}</span>
        </div>
        """, unsafe_allow_html=True)

        st.info(f"**Recommended Action**: {risk.recommended_action}")

        # Top metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Doc Type</div>
                <div class="metric-value" style="font-size: 1.25rem;">{cls.display_name}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Class Conf.</div>
                <div class="metric-value">{cls.confidence_percentage:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">OCR Conf.</div>
                <div class="metric-value">{ocr.confidence:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Risk Score</div>
                <div class="metric-value" style="color: {risk.color_code};">{risk.score}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_gauge:
        gauge_fig = create_risk_gauge(risk.score, risk.risk_level, risk.color_code)
        st.plotly_chart(gauge_fig, use_container_width=True, key=f"{key_prefix}_risk_gauge_{res['record_id']}")

    # Download Report Button
    report_path = res.get("report_path")
    if report_path and Path(report_path).exists():
        with open(report_path, "rb") as f:
            pdf_data = f.read()
        st.download_button(
            label="📥 Download Official Verification PDF Report",
            data=pdf_data,
            file_name=Path(report_path).name,
            mime="application/pdf",
            use_container_width=True,
            key=f"{key_prefix}_download_pdf_{res['record_id']}"
        )

    # Detailed Analysis Sub-Tabs
    tab_summary, tab_extract, tab_auth, tab_forensics, tab_preproc, tab_ocr = st.tabs([
        "🔍 Risk & Reason Breakdown",
        "📋 Extracted Information",
        "⚖️ Authenticity Checklist",
        "🔬 Forensic Splicing / ELA",
        "🖼️ Preprocessing Comparison",
        "📝 Extracted OCR Text"
    ])

    with tab_summary:
        st.markdown("#### Explainable Risk Reasons")
        for r in risk.reasons:
            icon = "✓" if "passed" in r.lower() else "⚠️"
            st.markdown(f"• **{icon}** {r}")

        st.markdown("#### Penalty Score Breakdown")
        if risk.breakdown:
            breakdown_records = []
            for b in risk.breakdown:
                breakdown_records.append({
                    "Factor": b.factor,
                    "Penalty Points": f"+{b.penalty_points}",
                    "Severity": b.severity,
                    "Explanation": b.reason
                })
            st.dataframe(pd.DataFrame(breakdown_records), use_container_width=True, hide_index=True)
        else:
            st.success("Zero penalty additions applied. All verification rules completed without suspicion flags.")

    with tab_extract:
        st.markdown("#### Structured Key Document Fields")
        st.caption("Sensitive identifiers (Account Numbers, Tax IDs, Personal IDs) are masked automatically in the presentation layer.")
        extracted_data: ExtractedData = res["extracted"]

        if extracted_data.masked_fields:
            field_df = pd.DataFrame([
                {"Field Name": k.replace("_", " ").title(), "Extracted & Masked Value": str(v)}
                for k, v in extracted_data.masked_fields.items()
            ])
            st.dataframe(field_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No structured fields were extracted from this document.")

        if extracted_data.missing_required_fields:
            st.error(f"Missing mandatory fields for {cls.display_name}: {', '.join(extracted_data.missing_required_fields)}")

    with tab_auth:
        st.markdown("#### Authenticity and Integrity Checks")
        auth: AuthenticityReport = res["authenticity"]

        for chk in auth.checks:
            css_class = "check-pass" if chk.status == "PASS" else ("check-warn" if chk.status == "WARN" else "check-fail")
            icon = "✓" if chk.status == "PASS" else ("⚠️" if chk.status == "WARN" else "❌")
            st.markdown(f"""
            <div class="check-card {css_class}">
                <b>{icon} {chk.name}</b> — <code>{chk.status}</code><br/>
                <span style="font-size: 0.9rem; color: #CFD8DC;">{chk.message}</span>
            </div>
            """, unsafe_allow_html=True)

    with tab_forensics:
        st.markdown("#### Image Manipulation & Error Level Analysis (ELA)")
        st.markdown("""
        **Error Level Analysis** works by re-compressing the document at a calibrated JPEG quality and analyzing 
        the variance in compression errors. Digitally modified, copy-pasted, or spliced regions typically exhibit 
        divergent error levels compared to original document areas.
        """)

        manip: ManipulationReport = res["manipulation"]
        fcol1, fcol2 = st.columns(2)

        with fcol1:
            st.image(manip.annotated_image, caption="Annotated Image (Suspicious Regions Highlighted)", use_container_width=True)
        with fcol2:
            st.image(manip.ela_image, caption="Error Level Analysis (ELA) Compression Variance Map", use_container_width=True)

        if manip.has_suspicious_regions:
            st.warning(f"Forensic Notice: Detected {len(manip.regions)} anomalous compression patch(es).")
        else:
            st.success("No abnormal compression or edge-splicing artifacts detected.")

    with tab_preproc:
        st.markdown("#### Computer Vision Preprocessing Comparison")
        st.markdown("Observe how noise removal, CLAHE contrast enhancement, deskewing, and Otsu binarization prepare the scan for OCR.")

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.image(res["primary_image"], caption="1. Original Raw Document", use_container_width=True)
        with pcol2:
            st.image(res["preprocessed_image"], caption="2. Enhanced & Thresholded Output", use_container_width=True)

        with st.expander("View All Intermediate OpenCV Stages"):
            stages = res.get("stages", {})
            st_cols = st.columns(len(stages))
            for idx, (st_name, st_img) in enumerate(stages.items()):
                with st_cols[idx]:
                    st.image(st_img, caption=st_name.title(), use_container_width=True)

    with tab_ocr:
        st.markdown("#### Extracted Optical Character Recognition Text")
        st.caption(f"Engine: `{ocr.engine_name}` • Confidence: `{ocr.confidence:.1f}%` • Pages: `{ocr.page_count}`")
        if ocr.warning:
            st.info(ocr.warning)
        st.text_area("OCR Raw Stream", ocr.text, height=300)

    # Mandatory Academic Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <b>Academic System Disclaimer:</b> This system provides an automated risk assessment and does not constitute 
        definitive legal proof of document fraud or authenticity. Results are intended for triage and verification decision support.
    </div>
    """, unsafe_allow_html=True)


def main():
    render_sidebar()

    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ AI-Powered Document Verification & Fraud Detection</h1>
        <p>Optical Character Recognition • Computer Vision Image Forensics • Machine Learning Classification • Explainable Risk Scoring</p>
    </div>
    """, unsafe_allow_html=True)

    tab_verify, tab_gallery, tab_history, tab_evaluation, tab_viva = st.tabs([
        "📄 Upload & Verify",
        "🧪 Interactive Sample Gallery",
        "📊 Verification History",
        "📈 Model Evaluation",
        "🎓 Architecture & College Viva Guide"
    ])

    # -------------------------------------------------------------
    # TAB 1: UPLOAD & VERIFY
    # -------------------------------------------------------------
    with tab_verify:
        st.subheader("Upload Document for Automated Verification")
        st.markdown("Select an image (**JPG, PNG**) or multi-page **PDF** to execute automated analysis.")

        uploaded_file = st.file_uploader(
            "Choose a document",
            type=["jpg", "jpeg", "png", "pdf"],
            help="Supports JPG, JPEG, PNG, and PDF up to 10MB."
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name

            col_btn, col_info = st.columns([1, 3])
            with col_btn:
                run_btn = st.button("🚀 Analyze & Verify Document", type="primary", use_container_width=True)
            with col_info:
                st.write(f"Selected: **{filename}** ({len(file_bytes) / 1024:.1f} KB)")

            if run_btn:
                with st.spinner("Processing document through multi-stage AI pipeline..."):
                    res = process_document_pipeline(file_bytes, filename)
                    if res:
                        st.session_state["latest_result"] = res
                        st.success("Verification Pipeline Execution Completed!")

        # Render latest result if available in session state
        if "latest_result" in st.session_state:
            render_analysis_results(st.session_state["latest_result"], key_prefix="upload_tab")

    # -------------------------------------------------------------
    # TAB 2: INTERACTIVE SAMPLE GALLERY
    # -------------------------------------------------------------
    with tab_gallery:
        st.subheader("🧪 Instant Verification Demo: Preloaded Samples")
        st.markdown("Test genuine and deliberately modified/tampered document samples without uploading files.")

        sample_options = {
            "Genuine Invoice (Valid math, clean ELA)": GENUINE_SAMPLES_DIR / "sample_genuine_invoice.png",
            "Suspicious Tampered Invoice (Math mismatch + ELA artifact)": SUSPICIOUS_SAMPLES_DIR / "sample_suspicious_invoice.png",
            "Genuine Bank Statement (Valid balances)": GENUINE_SAMPLES_DIR / "sample_genuine_bank_statement.png",
            "Genuine Educational Certificate": GENUINE_SAMPLES_DIR / "sample_genuine_certificate.png",
            "Genuine Government ID Card": GENUINE_SAMPLES_DIR / "sample_genuine_id_card.png",
            "Genuine University Application Form": GENUINE_SAMPLES_DIR / "sample_genuine_application_form.png",
        }

        selected_sample_label = st.selectbox("Select a demo document to analyze:", list(sample_options.keys()))
        selected_sample_path = sample_options[selected_sample_label]

        col_samp_prev, col_samp_act = st.columns([1.5, 1])

        with col_samp_prev:
            if selected_sample_path.exists():
                st.image(str(selected_sample_path), caption=selected_sample_label, width=420)
            else:
                st.warning("Sample document image not found. Run prepare_dataset.py to generate samples.")

        with col_samp_act:
            st.markdown("#### Sample Verification Details")
            if "Suspicious" in selected_sample_label:
                st.error("⚠️ **Expected Result**: HIGH RISK\n• Mathematical mismatch (Calculated 5900 != Stated 12500)\n• Forensic ELA compression anomaly in total amount region")
            else:
                st.success("✓ **Expected Result**: LOW RISK\n• All required fields present\n• Arithmetic consistency verified\n• Zero compression anomalies")

            if st.button("▶️ Run Analysis on Selected Sample", type="primary", use_container_width=True):
                if selected_sample_path.exists():
                    with open(selected_sample_path, "rb") as f:
                        s_bytes = f.read()
                    with st.spinner("Analyzing sample document..."):
                        s_res = process_document_pipeline(s_bytes, selected_sample_path.name)
                        if s_res:
                            st.session_state["gallery_result"] = s_res
                            st.success("Sample Analysis Completed!")

        if "gallery_result" in st.session_state:
            render_analysis_results(st.session_state["gallery_result"], key_prefix="gallery_tab")

    # -------------------------------------------------------------
    # TAB 3: VERIFICATION HISTORY
    # -------------------------------------------------------------
    with tab_history:
        st.subheader("📊 Verification Audit History & Analytics")
        st.markdown("Records persisted in local SQLite database for compliance and audit trailing.")

        records = get_all_verifications(limit=100)

        if records:
            df_hist = pd.DataFrame(records)
            hcol1, hcol2, hcol3, hcol4 = st.columns(4)

            with hcol1:
                st.metric("Total Verifications", len(df_hist))
            with hcol2:
                low_cnt = len(df_hist[df_hist["risk_level"] == "LOW RISK"])
                st.metric("Low Risk Documents", low_cnt)
            with hcol3:
                high_cnt = len(df_hist[df_hist["risk_level"] == "HIGH RISK"])
                st.metric("High Risk / Flagged", high_cnt)
            with hcol4:
                avg_score = df_hist["risk_score"].mean()
                st.metric("Average Risk Score", f"{avg_score:.1f}")

            # Visual distribution charts
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                fig_risk = px.pie(
                    df_hist, names="risk_level",
                    title="Risk Level Distribution",
                    color="risk_level",
                    color_discrete_map={
                        "LOW RISK": "#2E7D32",
                        "MEDIUM RISK": "#F57F17",
                        "HIGH RISK": "#C62828"
                    }
                )
                fig_risk.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_risk, use_container_width=True, key="history_risk_pie")

            with chart_col2:
                fig_types = px.bar(
                    df_hist, x="document_type",
                    title="Verifications by Document Category",
                    color="document_type"
                )
                fig_types.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10), showlegend=False)
                st.plotly_chart(fig_types, use_container_width=True, key="history_doc_types_bar")

            st.markdown("#### Audit Trail Log")
            display_table = df_hist[[
                "id", "filename", "timestamp", "document_type",
                "classification_confidence", "ocr_confidence", "risk_score", "risk_level"
            ]].copy()
            st.dataframe(display_table, use_container_width=True, hide_index=True)

            # Record Details & Deletion controls
            st.markdown("---")
            c_del1, c_del2 = st.columns(2)
            with c_del1:
                rec_to_delete = st.number_input("Delete Record by ID:", min_value=1, step=1)
                if st.button("🗑️ Delete Selected Record"):
                    if delete_verification(rec_to_delete):
                        st.success(f"Record #{rec_to_delete} deleted successfully.")
                        st.rerun()
                    else:
                        st.warning(f"Record #{rec_to_delete} not found.")

            with c_del2:
                st.write("Clear Audit History:")
                if st.button("⚠️ Clear Entire History Database", type="secondary"):
                    cleared = clear_all_history()
                    st.success(f"Audit history cleared ({cleared} records removed).")
                    st.rerun()
        else:
            st.info("No documents have been verified yet. Run an analysis in Tab 1 or Tab 2 to populate the audit log.")

    # -------------------------------------------------------------
    # TAB 4: MODEL EVALUATION
    # -------------------------------------------------------------
    with tab_evaluation:
        st.subheader("📈 Machine Learning Model Evaluation & Metrics")
        st.markdown("Transparent performance metrics for the TF-IDF + Logistic Regression Document Classifier.")

        eval_file = MODELS_DIR / "evaluation_results.json"
        meta_file = MODELS_DIR / "metadata.json"

        if eval_file.exists():
            with open(eval_file, "r") as f:
                eval_data = json.load(f)

            ecol1, ecol2, ecol3 = st.columns(3)
            with ecol1:
                st.metric("Overall Test Accuracy", f"{eval_data.get('overall_accuracy', 0) * 100:.1f}%")
            with ecol2:
                st.metric("Number of Classes", len(eval_data.get("classes", [])))
            with ecol3:
                st.metric("Algorithm", "TF-IDF + Calibrated Logistic Regression")

            # Confusion Matrix Plotly Heatmap
            classes = eval_data.get("classes", [])
            cm = np.array(eval_data.get("confusion_matrix", []))

            if len(cm) > 0:
                fig_cm = px.imshow(
                    cm,
                    x=classes,
                    y=classes,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted Class", y="True Ground Truth Class", color="Count"),
                    title="Confusion Matrix (Held-out Test Split)"
                )
                fig_cm.update_layout(height=420)
                st.plotly_chart(fig_cm, use_container_width=True, key="eval_confusion_matrix")

            # Per class metrics table
            st.markdown("#### Per-Category Performance Breakdown")
            metrics_df = pd.DataFrame(eval_data.get("per_class_metrics", []))
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        else:
            st.warning("Evaluation results file not found. Run `python training/evaluate_model.py` to generate metrics.")

    # -------------------------------------------------------------
    # TAB 5: ARCHITECTURE & VIVA GUIDE
    # -------------------------------------------------------------
    with tab_viva:
        st.subheader("🎓 System Architecture & College Viva Voce Preparation Guide")
        st.markdown("Comprehensive academic review materials, technical justifications, and presentation outline.")

        with st.expander("📌 1. Complete System Architecture & Pipeline Workflow", expanded=True):
            st.markdown("""
            ```
            User Upload (JPG / PNG / PDF)
                ↓
            File Integrity & Security Validation
                ↓
            PyMuPDF Image Rendering (200 DPI normalization)
                ↓
            OpenCV Computer Vision Preprocessing (Grayscale, CLAHE, Deskew, Otsu)
                ↓
            Optical Character Recognition (Tesseract / PyMuPDF fallback with confidence)
                ↓
            Machine Learning Document Classification (TF-IDF + Calibrated Softmax)
                ↓
            Structured Key Information Extraction (Regex, Masking)
                ↓
            Authenticity & Mathematical Consistency Checks (Subtotal + Tax == Total)
                ↓
            Forensic Image Forensics (Error Level Analysis - ELA & Noise Variance)
                ↓
            Explainable Composite Risk Scoring Engine (0-100 Score with Reasons)
                ↓
            SQLite Audit History & ReportLab Downloadable PDF Verification Audit Report
            ```
            """)

        with st.expander("🎯 2. Technical Justifications (Why did we choose these technologies?)"):
            st.markdown("""
            - **Why Tesseract OCR?** Open-source, highly explainable, lightweight, runs locally without cloud API keys or external server dependencies.
            - **Why OpenCV?** Standard library for image enhancement. Contrast Limited Adaptive Histogram Equalization (CLAHE) and Otsu thresholding dramatically improve OCR word recognition accuracy on degraded scans.
            - **Why TF-IDF + Logistic Regression over Heavy Deep Learning?** For document text classification, TF-IDF n-grams combined with linear models yield calibrated probabilities, instant inference on standard laptops (<20ms), and zero GPU requirements.
            - **Why Error Level Analysis (ELA)?** Re-saving lossy JPEG images at controlled compression factors highlights splicing and copy-paste edits where altered pixels possess different compression error rates.
            - **Why Streamlit?** Allows full Python interoperability, live model inference, reactive data visualization with Plotly, and clean responsive UI design suitable for academic viva demonstrations.
            """)

        with st.expander("🗣️ 3. Top 10 College Viva Voce Questions & Answers"):
            st.markdown("""
            **Q1: Can this system legally prove that a document is fraudulent?**
            *Answer*: No. The system produces an automated probabilistic risk assessment. Legally, fraud requires forensic expert testimony and intent. Our tool serves as an intelligent triage decision-support system.

            **Q2: How does Error Level Analysis (ELA) detect tampering?**
            *Answer*: Lossy JPEG compression saves images in 8x8 pixel blocks. When a document is altered by pasting text or numbers from another source, the spliced region has a different error degradation rate than the rest of the image. ELA visualizes this variance.

            **Q3: How does your risk score calculate?**
            *Answer*: It is an explainable additive penalty engine. It assesses missing fields (+15), mathematical mismatches (+20), ELA artifacts (+25), duplicate files (+20), date conflicts (+15), and low OCR confidence (+10). Total score is capped at 100.

            **Q4: How do you handle sensitive user privacy?**
            *Answer*: Sensitive identifiers like Bank Account numbers and Government IDs are masked before presentation in the UI and reports (e.g. `XXXXXXXX1012`), and temporary upload files are sanitized.

            **Q5: What is CLAHE and why is it used?**
            *Answer*: Contrast Limited Adaptive Histogram Equalization. Unlike standard global histogram equalization which can over-amplify noise, CLAHE operates on localized tiles and clips extreme contrast, normalizing uneven document lighting.
            """)


if __name__ == "__main__":
    main()
