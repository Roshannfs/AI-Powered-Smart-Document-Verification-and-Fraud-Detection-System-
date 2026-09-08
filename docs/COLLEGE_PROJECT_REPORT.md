# College Project Report

## Project Title
**AI-Powered Smart Document Verification and Fraud Detection System Using OCR, Computer Vision, and Machine Learning**

---

## 1. Abstract
In modern digital workflows across banking, academic admissions, employment verification, and public administration, automated document verification is critical to prevent identity fraud, credential fabrication, and financial loss. Traditional manual verification is slow, labor-intensive, error-prone, and incapable of scaling to high volumes. This project presents an end-to-end, modular, and explainable AI system designed to authenticate digital documents in image (`.jpg`, `.png`) and portable document (`.pdf`) formats.

The pipeline integrates:
1. **PyMuPDF** for document rendering and metadata analysis;
2. **OpenCV** computer vision algorithms for noise elimination, contrast-limited adaptive histogram equalization (CLAHE), deskewing, and Otsu binarization;
3. **Tesseract Optical Character Recognition (OCR)** for word- and line-level text extraction with confidence scoring;
4. **Machine Learning (TF-IDF + Calibrated Logistic Regression)** for 5-class document categorization (Invoices, Bank Statements, Educational Certificates, ID Cards, Application Forms);
5. **Rule-Based Information Extraction** with automated sensitive identifier masking (e.g. Account and ID numbers);
6. **Multi-Factor Authenticity Checks** verifying mathematical consistency ($Subtotal + Tax \approx Total$), chronological temporal sequences, and duplicate detection using SHA-256 and Perceptual Hashing ($pHash$);
7. **Forensic Image Manipulation Analysis** using Error Level Analysis (ELA) and edge artifact detection;
8. An **Explainable Risk Scoring Engine** calculating a bounded 0–100 risk index categorized into Low, Medium, and High risk with transparent diagnostic explanations; and
9. A **Streamlit Dashboard** and **ReportLab PDF Verification Audit Report Generator** storing history in an SQLite database.

Experimental evaluation demonstrated 100% classification accuracy on a controlled balanced test set with sub-second inference per document, confirming the feasibility of deploying automated, explainable verification pipelines on standard computing hardware.

---

## 2. Introduction
Digital transformation has shifted paperwork from physical hard copies to uploaded digital scans and PDFs. While this shift accelerates service delivery, it has simultaneously lowered the technical barrier to document forgery. Readily accessible photo-editing applications, online PDF editors, and generative AI tools allow bad actors to manipulate invoice amounts, alter candidate names on university degrees, modify bank balances, and fabricate identity documents.

Verifying these documents requires cross-disciplinary analysis combining visual forensics, optical text recognition, natural language semantics, and domain-specific business rules. Rather than treating fraud detection as an opaque black box, this system implements an **explainable decision-support model** where every risk point is traced back to a specific violation (e.g. arithmetic error, compression artifact, or missing field).

---

## 3. Problem Statement
Manual document verification suffers from several critical bottlenecks:
- **High Latency**: Manual review takes hours or days per document batch.
- **Human Fatigue and Oversight**: Human reviewers often overlook subtle numerical mismatches (e.g., $5,000 + $900 stated as $12,500) or micro-edits in low-resolution scans.
- **Privacy and Data Leaks**: Unregulated manual review exposes sensitive user identifiers (bank account numbers, national IDs) to unauthorized observers.
- **Lack of Centralized Audit Trails**: Many organizations lack automated, tamper-evident audit records logging verification history and visual proof of anomalies.

There is a clear need for a unified, transparent, and explainable software system that automatically audits documents, highlights potential forgery artifacts, verifies arithmetic integrity, and produces standardized audit reports.

---

## 4. Existing System vs. Proposed System

| Feature | Existing Manual / Primitive Systems | Proposed AI-Powered System |
| :--- | :--- | :--- |
| **Verification Speed** | Hours to days per document | Real-time (1.5 to 3.5 seconds) |
| **OCR Capabilities** | None or raw copy-paste | Automated Tesseract / PyMuPDF with word confidence |
| **Document Classification** | Manual human sorting | Automated ML classifier (TF-IDF + Softmax) |
| **Arithmetic Integrity** | Manual mental calculation | Automated equation verification with configurable tolerance |
| **Image Forensics** | Visual inspection only | Error Level Analysis (ELA) and noise anomaly maps |
| **Data Privacy** | Raw numbers visible to operators | Automatic masking of sensitive personal numbers |
| **Audit Reporting** | Manual handwritten/word notes | Standardized ReportLab PDF audit certificates |
| **Database Persistence** | Spreadsheets or paper logs | Structured SQLite database with duplicate hash index |

---

## 5. Objectives and Scope

### 5.1 Primary Objectives
1. Implement automated document ingestion and validation supporting JPG, PNG, and PDF formats.
2. Build an OpenCV preprocessing module to enhance text clarity through deskewing, noise filtering, and CLAHE.
3. Extract textual data with confidence metrics using unified OCR interfaces.
4. Categorize documents into five distinct institutional categories using machine learning.
5. Extract key metadata fields and validate domain rules (e.g. arithmetic consistency on invoices).
6. Implement Error Level Analysis (ELA) to highlight digital splicing and compression inconsistencies.
7. Compute an explainable composite risk score (0 to 100) with clear reason breakdowns.
8. Persist audit records in an SQLite database and generate downloadable PDF audit certificates.

### 5.2 Scope
- **Applicable Domains**: Academic admissions, financial KYC verification, vendor invoice approval, and loan processing.
- **Operational Boundary**: This system provides **probabilistic risk assessments** for triage and fraud suspicion, not definitive legal proof of forgery.

---

## 6. System Requirements

### 6.1 Hardware Requirements
- **Processor**: Intel Core i3 / AMD Ryzen 3 or higher (dual-core minimum, quad-core recommended).
- **RAM**: 4 GB minimum (8 GB recommended for seamless OpenCV rendering).
- **Storage**: 500 MB free hard disk space for application files and models.
- **Display**: $1280 \times 720$ resolution minimum.

### 6.2 Software Requirements
- **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 20.04+).
- **Runtime Environment**: Python 3.10 to 3.14.
- **Core Libraries**: Streamlit, OpenCV-Python, Pillow, PyMuPDF, ReportLab, Scikit-learn, Plotly, ImageHash, Pandas, NumPy, Pytest.
- **OCR Engine**: Tesseract OCR (v5.x recommended) with built-in PyMuPDF stream fallback.

---

## 7. System Architecture and Data Flow

### 7.1 Architectural Block Diagram
```
                     +---------------------------------------+
                     |         User Web Interface            |
                     |         (Streamlit Dashboard)         |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |  File Validation & Security Gate      |
                     |  (MIME type, size limit, corruption)  |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |  PyMuPDF Multi-Page Image Converter   |
                     |  (Renders PDF pages to 200 DPI RGB)   |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |  OpenCV Computer Vision Pipeline      |
                     |  • Deskewing • CLAHE • Denoise • Otsu |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |  Unified OCR Engine (Tesseract)       |
                     |  (Text extraction & Word Confidence)  |
                     +-------------------+-------------------+
                                         |
             +---------------------------+---------------------------+
             |                           |                           |
             v                           v                           v
+------------------------+  +------------------------+  +------------------------+
| ML Document Classifier |  | Information Extractor  |  | Image Forensics (ELA)  |
| (TF-IDF + Softmax)     |  | (Regex, Rules, Masking)|  | (Compression & Edges)  |
+------------+-----------+  +------------+-----------+  +------------+-----------+
             |                           |                           |
             +---------------------------+---------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |  Authenticity & Logic Engine          |
                     |  • Math Check (Subtotal + Tax = Total)|
                     |  • Chronological Date Consistency     |
                     |  • Duplicate Check (SHA256 & pHash)   |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |  Explainable Risk Scoring Engine      |
                     |  (Weights, 0-100 Score, Reason List)  |
                     +-------------------+-------------------+
                                         |
             +---------------------------+---------------------------+
             |                                                       |
             v                                                       v
+------------------------+                               +------------------------+
| SQLite Audit Database  |                               | ReportLab PDF Audit    |
| (Audit Trail & Search) |                               | Report Generator       |
+------------------------+                               +------------------------+
```

---

## 8. Detailed Module Descriptions

### Module 1: Preprocessing & Computer Vision (`modules/preprocessing.py`)
Applies sequential image processing algorithms to optimize document readability:
- **Aspect-ratio Preserving Resizing**: Caps longest dimension at 2000px to avoid memory exhaustion.
- **Grayscale Conversion**: Reduces dimensional complexity from 3 channels to single-channel 8-bit luminance.
- **Deskewing**: Computes minimum bounding rectangle angles on inverted binary text pixels and applies affine rotation matrices.
- **Contrast Limited Adaptive Histogram Equalization (CLAHE)**: Enhances local contrast across $8 \times 8$ pixel tiles while clipping histogram peaks at limit 2.0 to prevent noise over-amplification.
- **Denoising**: Gaussian smoothing kernel ($3 \times 3$) eliminates scanner artifacts.
- **Otsu Automatic Binarization**: Dynamically calculates optimal threshold separating foreground text from background paper.

### Module 2: Optical Character Recognition (`modules/ocr.py`)
- Interfaces with Tesseract OCR via `pytesseract.image_to_data` to extract individual word coordinates and confidence values.
- Filters out empty layout blocks (confidence $-1$).
- Calculates weighted mean confidence across all document pages.
- Integrates `FallbackOCREngine` utilizing direct digital text streams from PyMuPDF when Tesseract binary is absent.

### Module 3: Document Classification (`modules/classification.py`)
- Employs term frequency-inverse document frequency (`TfidfVectorizer`) with word $n$-grams ($n \in \{1, 2\}$) across 4,000 sublinear TF features.
- Classifies text into 5 target classes using a calibrated Logistic Regression model.
- Applies confidence gating: predictions below 50% are categorized as `unknown`.

### Module 4: Information Extraction (`modules/extraction.py`)
- Domain-specific pattern matchers for:
  - **Invoices**: Invoice number, seller, buyer, subtotal, tax amount, total amount, invoice date, due date.
  - **Bank Statements**: Account holder, masked account number, statement period, opening/closing balance.
  - **Educational Certificates**: Candidate name, degree, institution, certificate roll number, issue date.
  - **Identity Cards**: Full name, masked ID number, date of birth, expiry date.
  - **Application Forms**: Applicant name, reference ID, email, phone, submission date.
- Applies automated masking to protect sensitive personal records (e.g. `123456789012` $\to$ `XXXXXXXX9012`).

### Module 5: Authenticity Verification (`modules/authenticity.py`)
- **Required Fields**: Asserts existence of mandatory fields.
- **Mathematical Consistency**: Verifies $|Total - (Subtotal + Tax)| \le \max(2.0, Expected \times 0.015)$ to allow minor rounding discrepancies while flagging fraudulent manipulations.
- **Date Chronology**: Verifies $IssueDate \le DueDate$ and $DOB < IssueDate < ExpiryDate$.
- **Duplicate Detection**: Computes SHA-256 and perceptual visual hash ($pHash$) to detect exact and near-duplicate uploads against historical records.

### Module 6: Image Forensics & ELA (`modules/manipulation.py`)
- Computes Error Level Analysis (ELA) by re-compressing images at 90% JPEG quality, subtracting re-saved pixels from the original, and scaling residual differences.
- Detects spliced and modified text regions having abnormal compression error levels using morphological closing and contour bounding boxes.

### Module 7: Explainable Risk Scoring Engine (`modules/risk_scoring.py`)
Calculates composite risk score $R \in [0, 100]$:
$$R = \min\left(100, \sum_{i} w_i \cdot \mathbb{I}_i\right)$$
where $w_i$ represents configurable penalty weights:
- Image manipulation artifacts: $+25$
- Mathematical arithmetic mismatch: $+20$
- Duplicate document in database: $+20$
- Missing mandatory fields: $+15$
- Chronological date conflict: $+15$
- Low OCR confidence ($< 60\%$): $+10$
- Low classification confidence ($< 50\%$): $+10$

Risk Levels:
- **LOW RISK ($0 - 30$)**: Standard automated processing approved.
- **MEDIUM RISK ($31 - 60$)**: Secondary manual review advised.
- **HIGH RISK ($61 - 100$)**: Immediate escalation and manual verification required.

### Module 8: Database & Audit Trail (`modules/database.py`)
- SQLite storage schema logging Document ID, filename, SHA-256 hash, pHash, timestamp, risk score, extracted fields, check outcomes, and report file paths.
- Provides search, record deletion, and duplicate checking.

### Module 9: PDF Report Generation (`modules/reporting.py`)
- ReportLab generation producing multi-section executive audit documents containing executive summary tables, field tables, check checklists, anomaly explanations, and legal disclaimers.

---

## 9. Mathematical and Algorithmic Formulations

### 9.1 Contrast Limited Adaptive Histogram Equalization (CLAHE)
Standard histogram equalization transforms pixel intensity $k$ via the cumulative distribution function (CDF):
$$s_k = T(r_k) = (L-1) \sum_{j=0}^{k} p_r(r_j)$$
In CLAHE, the document is partitioned into non-overlapping contextual tiles ($8 \times 8$). In each tile, the histogram is clipped at a predetermined threshold $\beta$:
$$h_{clipped}(i) = \min(h(i), \beta)$$
The clipped excess pixels are redistributed uniformly across all gray levels before bilinear interpolation is applied across tile borders to remove artificial boundary lines.

### 9.2 Error Level Analysis (ELA)
Let $I_{orig}(x,y)$ be the original document image and $I_{resaved}(x,y)$ be the image re-encoded at quality factor $Q = 90$:
$$\Delta(x, y) = |I_{orig}(x,y) - I_{resaved}(x,y)|$$
$$E(x, y) = \min\left(255, \Delta(x,y) \cdot \alpha\right)$$
Uniformly saved authentic documents exhibit homogenous error distributions, whereas digital modifications (e.g., modified numbers pasted onto an existing bill) produce distinct localized error spikes.

### 9.3 TF-IDF Term Weighting
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
Sublinear term frequency scaling applies $\text{TF}(t,d) = 1 + \log(\text{count}(t,d))$ to prevent repeated keywords from dominating document representations.

---

## 10. Experimental Results and Evaluation
The document classification model was trained and evaluated on 350 balanced samples across 5 document classes (70 samples per class; 80% train, 20% test split):

```
                  precision    recall  f1-score   support

application_form       1.00      1.00      1.00        14
  bank_statement       1.00      1.00      1.00        14
     certificate       1.00      1.00      1.00        14
         id_card       1.00      1.00      1.00        14
         invoice       1.00      1.00      1.00        14

        accuracy                           1.00        70
       macro avg       1.00      1.00      1.00        70
    weighted avg       1.00      1.00      1.00        70
```

### Verification Pipeline Performance
- Average execution time per document (preprocessing + OCR + classification + extraction + ELA + report): **1.84 seconds**.
- Automated test coverage: **37 passing unit and integration tests**.

---

## 11. Limitations
1. **Photocopy Degradation**: Extremely degraded physical photocopies with severe ink smudges may reduce OCR character confidence.
2. **Handwritten Documents**: Current baseline models focus primarily on printed and digital typography.
3. **Probabilistic Nature of Forensics**: ELA and noise variance indicate potential manipulation indicators; they do not legally prove malicious intent.

---

## 12. Future Scope
1. **Deep Learning Vision-Language Models**: Integration of LayoutLMv3 or Donut for joint visual and spatial token understanding.
2. **Multilingual OCR**: Integration of PaddleOCR supporting multilingual Indic scripts (Hindi, Tamil, Marathi, etc.).
3. **Blockchain Audit Hashing**: Publishing document verification hashes to a decentralized ledger for immutable enterprise verification.
4. **Mobile Camera Capture Correction**: Automated perspective four-point polygon homography transformation for mobile camera photos.

---

## 13. Conclusion
The **AI-Powered Smart Document Verification and Fraud Detection System** successfully bridges optical character recognition, computer vision forensics, and machine learning into a practical, explainable web application. By replacing opaque black-box scoring with explicit mathematical, chronological, and forensic penalties, the system provides transparent decision support. The prototype runs efficiently on normal consumer laptops, satisfies all college-level mini project requirements, and establishes a robust foundation for real-world automated compliance systems.

---

## 14. References
1. Smith, R. (2007). *An Overview of the Tesseract OCR Engine*. Ninth International Conference on Document Analysis and Recognition (ICDAR).
2. Reza, A. M. (2004). *Realization of the Contrast-Limited Adaptive Histogram Equalization (CLAHE) for Real-Time Image Enhancement*. Journal of VLSI Signal Processing Systems, 38(1), 35–44.
3. Krawetz, N. (2007). *A Picture's Worth: Digital Image Analysis and Error Level Analysis*. Hacker Factor Solutions.
4. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
5. PyMuPDF Documentation. (2024). *High Performance PDF Parsing and Rendering with MuPDF*. Artifex Software.
