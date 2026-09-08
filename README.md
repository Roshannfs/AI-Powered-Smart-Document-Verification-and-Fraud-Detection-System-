# AI-Powered Smart Document Verification and Fraud Detection System
> **Using Optical Character Recognition (OCR), Computer Vision (OpenCV), Machine Learning, and Forensic Error Level Analysis**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Computer Vision](https://img.shields.io/badge/Vision-OpenCV-5C3EE8.svg)](https://opencv.org/)
[![OCR](https://img.shields.io/badge/OCR-Tesseract-green.svg)](https://github.com/tesseract-ocr/tesseract)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 1. Project Overview
The **AI-Powered Smart Document Verification and Fraud Detection System** is an end-to-end, modular, and explainable AI web application designed to authenticate digital documents in **JPG, PNG, and multi-page PDF** formats. 

The system preprocesses documents using computer vision, extracts textual content via OCR, classifies document types using machine learning, extracts key fields with sensitive data masking, checks mathematical and chronological authenticity, inspects image compression forensics using **Error Level Analysis (ELA)**, calculates a transparent 0–100 **Risk Score**, displays findings on an interactive **Streamlit Dashboard**, and compiles downloadable **ReportLab PDF Verification Audit Certificates**.

---

## 🎯 2. Key Objectives
- **Automated Ingestion**: Validate and render high-resolution scans and multi-page PDFs.
- **Image Enhancement**: Denoise, deskew, and equalize lighting with CLAHE and Otsu binarization.
- **Explainable Fraud Triage**: Detect mathematical discrepancies (e.g. Subtotal + Tax $\ne$ Total) and spliced image patches.
- **Privacy Preservation**: Automatically mask sensitive personal identifiers (Bank Account numbers, ID numbers).
- **Audit-Ready Reporting**: Generate tamper-evident PDF audit summaries and log records in SQLite.
- **College Presentation Support**: 100% runnable on a normal consumer laptop with full viva voce preparation resources.

---

## 🏗️ 3. System Architecture

```
                    ┌───────────────────────────────┐
                    │       User Web Interface      │
                    │      (Streamlit Dashboard)    │
                    └───────────────┬───────────────┘
                                    │ Upload (JPG, PNG, PDF)
                                    ▼
                    ┌───────────────────────────────┐
                    │   File Validation & PyMuPDF   │
                    │   (MIME, Size, 200 DPI Render)│
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │    OpenCV Vision Preprocess   │
                    │    (Grayscale, CLAHE, Deskew, │
                    │     Noise Filter, Otsu Binar) │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       Unified OCR Engine      │
                    │  (Tesseract + Word Confidence)│
                    └───────────────┬───────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
┌───────────────────────┐ ┌───────────────────┐ ┌───────────────────────┐
│ ML Classification     │ │ Field Extraction  │ │ Forensic ELA Analysis │
│ (TF-IDF + Softmax)    │ │ (Regex, Masking)  │ │ (Compression & Edges) │
└───────────┬───────────┘ └─────────┬─────────┘ └───────────┬───────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Authenticity Check Engine   │
                    │   • Mathematical ($Sub+Tax=Tot$)│
                    │   • Date Chronology Logic     │
                    │   • SHA-256 & pHash Duplicates│
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Explainable Risk Scoring    │
                    │   (0–100 Score, Low/Med/High, │
                    │    Bullet-point Reasons)      │
                    └───────────────┬───────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌───────────────────────┐                       ┌───────────────────────┐
│  SQLite Audit History │                       │ ReportLab PDF Audit   │
│  (Database Log & Purge)                       │ Verification Report   │
└───────────────────────┘                       └───────────────────────┘
```

---

## 🗂️ 4. Supported Document Categories
1. **Invoice / Bill**: Subtotal, taxes, vendor, customer, grand total arithmetic verification.
2. **Bank Statement**: Account holder, masked account number, opening/closing balance verification.
3. **Educational Certificate**: Candidate name, degree, institution, roll number, issue date.
4. **Identity Card**: Full name, masked ID number, date of birth, expiry date.
5. **Application Form**: Reference number, applicant details, email, submission date.

---

## 💻 5. Technology Stack
- **Frontend / Web UI**: [Streamlit](https://streamlit.io/) with custom dark-mode CSS and [Plotly](https://plotly.com/) charts.
- **Image Processing**: [OpenCV](https://opencv.org/) (CLAHE, deskewing, Otsu) & [Pillow](https://python-pillow.org/).
- **Optical Character Recognition (OCR)**: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) via `pytesseract` with PyMuPDF fallback stream.
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (TF-IDF Vectorizer + Calibrated Logistic Regression).
- **PDF Processing**: [PyMuPDF](https://pymupdf.readthedocs.io/) (`pymupdf`) for rendering; [ReportLab](https://www.reportlab.com/) for audit reporting.
- **Hashing & Duplicate Detection**: `hashlib` (SHA-256) & [ImageHash](https://github.com/JohannesBuchner/imagehash) (Perceptual pHash).
- **Database**: SQLite3 for audit trailing and duplicate lookup.
- **Testing**: Pytest (37 automated unit and integration tests).

---

## ⚙️ 6. Installation & Setup

### Prerequisites
- Python 3.10 to 3.14 installed.
- (Optional but Recommended) Tesseract OCR.

### Step 1: Clone or Navigate to Project Directory
```powershell
cd "c:\Projects\AI-Powered Smart Document Verification and Fraud Detection System Using OCR, Computer Vision and Machine Learning"
```

### Step 2: Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: (Optional) Install Tesseract OCR
- **Windows**:
  Run via Windows Package Manager:
  ```powershell
  winget install UB-Mannheim.TesseractOCR
  ```
  Or download the installer from [UB-Mannheim Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).
  Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`.
- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt update && sudo apt install -y tesseract-ocr
  ```
- **macOS**:
  ```bash
  brew install tesseract
  ```

> *Note: If Tesseract is not installed, the application automatically uses its built-in PyMuPDF stream fallback so it runs without crashing.*

---

## 🚀 7. Running the Application

### 1. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Prepare Synthetic Dataset & Test Samples (Optional - Already Built)
```powershell
python training/prepare_dataset.py
python training/train_classifier.py
python training/evaluate_model.py
```

### 3. Run Automated Tests
```powershell
python -m pytest tests/ -v
```
All **37 unit and integration tests** will execute and pass.

---

## 🧪 8. Interactive Features in Web Dashboard

### Tab 1: Upload & Verify
- Upload any JPG, PNG, or multi-page PDF document.
- Click **"Analyze & Verify Document"**.
- View:
  - Classification badge & confidence.
  - Plotly interactive Risk Score Gauge (0–100).
  - Pass/Warn/Fail authenticity checklist.
  - Forensic Error Level Analysis (ELA) map highlighting altered image patches.
  - Downloadable official **PDF Audit Report**.

### Tab 2: Interactive Demo Gallery
- Instant one-click testing using built-in samples:
  - **Genuine Invoice**: Arithmetic consistent, clean ELA $\to$ **LOW RISK**.
  - **Tampered Invoice**: Total fraudulently altered from $5,900 to $12,500 with ELA compression patch $\to$ **HIGH RISK**.
  - **Genuine Bank Statement, Certificate, ID Card, Application Form**.

### Tab 3: Verification History
- View SQLite audit log.
- Risk distribution pie chart and document category bar chart.
- Delete individual records or clear history.

### Tab 4: Machine Learning Model Evaluation
- Interactive Plotly Confusion Matrix heatmap.
- Accuracy, Precision, Recall, and F1-Score breakdown.

### Tab 5: College Presentation & Viva Guide
- Interactive slides outline and top viva voce Q&A directly inside the application!

---

## ⚖️ 9. Risk Scoring Engine Logic
The risk engine uses an explainable additive penalty formulation capped at 100:

| Violation / Indicator | Penalty Points | Severity |
| :--- | :---: | :---: |
| **Possible Image Manipulation (ELA Artifacts)** | `+25` | HIGH |
| **Mathematical Mismatch ($Subtotal + Tax \ne Total$)** | `+20` | HIGH |
| **Duplicate Document (SHA256 / pHash match)** | `+20` | HIGH |
| **Missing Mandatory Fields** | `+15` | MEDIUM |
| **Chronological Date Inconsistency** | `+15` | MEDIUM |
| **Low OCR Quality ($< 60\%$)** | `+10` | LOW |
| **Low Classification Confidence ($< 50\%$)** | `+10` | LOW |

### Risk Categories:
- `0 to 30`: **LOW RISK** $\to$ Direct approval.
- `31 to 60`: **MEDIUM RISK** $\to$ Secondary review advised.
- `61 to 100`: **HIGH RISK** $\to$ Escalation and manual verification required.

---

## 🔒 10. Privacy & Security
- **Sensitive Data Masking**: Bank Account numbers (`XXXXXXXX1012`) and Government IDs (`XXXXX1100`) are masked in both UI tables and generated PDF reports.
- **Upload Sanitization**: Uploaded files are processed in memory or ephemeral paths; no raw documents are permanently retained unless requested.
- **Injection Safety**: SQL parameters are prepared and parameterized to prevent SQL injection.

---

## ⚠️ 11. Disclaimer
This system provides an **automated probabilistic risk assessment** and does not constitute definitive proof of document fraud or legal authenticity. It is developed as an academic college project prototype to assist human verifiers in document triage.

---

## 📄 12. College Project Documentation Links
- [Detailed College Project Report](docs/COLLEGE_PROJECT_REPORT.md)
- [15-Slide Presentation Outline & 25 Viva Voce Q&A](docs/PRESENTATION_AND_VIVA_GUIDE.md)
