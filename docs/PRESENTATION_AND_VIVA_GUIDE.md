# College Presentation & Viva Voce Preparation Guide

## Project Title
**AI-Powered Smart Document Verification and Fraud Detection System Using OCR, Computer Vision and Machine Learning**

---

# Part 1: 15-Slide Presentation (PPT) Outline

### Slide 1: Title Slide
- **Title**: AI-Powered Smart Document Verification and Fraud Detection System
- **Subtitle**: Optical Character Recognition • Computer Vision • Machine Learning
- **Presented By**: [Student Name / Roll Number]
- **Department**: Department of Computer Science & Engineering / AI & Data Science
- **College/Institution**: [University Name]
- **Academic Year**: 2025 – 2026
- *Speaker Note*: Introduce the project topic and emphasize that it is an automated, explainable system designed for real-world document verification and fraud risk assessment.

### Slide 2: Problem Statement & Motivation
- **Exponential Growth of Digital Documents**: Banking (KYC), college admissions, job recruitment, insurance.
- **Vulnerability to Tampering**: Readily available digital photo editors and online tools make forging invoices, certificates, and IDs effortless.
- **Limitations of Manual Verification**:
  - Time-consuming (hours to days)
  - Prone to human fatigue and oversights
  - Privacy risks when human reviewers view sensitive personal details
- *Speaker Note*: Highlight that humans cannot easily spot localized compression artifacts or micro-edits in digital documents.

### Slide 3: Project Objectives
- Build an automated end-to-end verification web application runnable on a standard laptop.
- Support **JPG, PNG, and multi-page PDF** document uploads.
- Enhance scans using **OpenCV** (CLAHE, deskewing, noise reduction).
- Extract textual data and per-word confidence using **Tesseract OCR**.
- Categorize documents using **Machine Learning** (Invoices, Bank Statements, Certificates, IDs, Forms).
- Detect tampering via **Error Level Analysis (ELA)** and cross-field arithmetic checks.
- Generate an explainable **Risk Score (0–100)** and a **Downloadable PDF Audit Report**.

### Slide 4: System Architecture & Workflow
- *Visual*: Architecture flowchart diagram (User Upload $\to$ Validation $\to$ Preprocessing $\to$ OCR $\to$ ML Classification $\to$ Extraction $\to$ Forensics $\to$ Risk Engine $\to$ Dashboard & Report).
- **Core Pipeline Stages**:
  1. Input Ingestion & PDF-to-Image Rendering
  2. Computer Vision Enhancement
  3. OCR & Text Analytics
  4. Fraud Forensics & Consistency Verification
  5. Explainable Risk Scoring & Audit Trail

### Slide 5: Supported Document Categories
- **Invoice / Bill**: Stated amounts, taxes, vendor, buyer, arithmetic consistency ($Subtotal + Tax = Total$).
- **Bank Statement**: Account holder, masked account number, balance integrity.
- **Educational Certificate**: Candidate name, degree, institution, issue date, roll number.
- **Identity Card**: Cardholder name, masked ID number, date of birth, expiry date.
- **Application Form**: Reference number, applicant details, contact validation.

### Slide 6: Computer Vision & Image Preprocessing
- **Resizing**: Capped at 2000px to maintain aspect ratio and prevent memory overload.
- **Grayscale Conversion**: Reduces dimensional complexity for faster filtering.
- **Deskewing**: Min-area bounding rectangle angle detection and affine rotation.
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization normalizes uneven scan lighting without noise over-amplification.
- **Otsu Binarization**: Computes optimal global threshold separating foreground text from paper.

### Slide 7: Optical Character Recognition (OCR)
- Unified OCR engine wrapping **Tesseract OCR** with fallback capability.
- **Word-Level Confidence Scoring**: Filters layout noise (confidence $-1$) and computes true document-level recognition confidence.
- **Multi-Page Handling**: Automatically extracts and concatenates multi-page PDF documents.
- Pluggable design ready for PaddleOCR or EasyOCR addition.

### Slide 8: Machine Learning Document Classification
- **Feature Extraction**: TF-IDF Vectorizer with unigrams and bigrams ($n \in \{1, 2\}$) and sublinear TF scaling across 4,000 features.
- **Classifier Algorithm**: Calibrated Logistic Regression yielding reliable posterior probability distributions ($p(y|x)$).
- **Confidence Gating**: Any prediction with confidence $< 50\%$ or short text is gated as `unknown`.
- **Performance**: 100% test accuracy on balanced 5-class evaluation dataset.

### Slide 9: Structured Information Extraction & Privacy
- **Pattern Matching**: Document-specific regular expressions and keyword boundaries.
- **Data Privacy by Design**: Automated masking of sensitive fields:
  - Bank Account Number: `987654321012` $\to$ `XXXXXXXX1012`
  - National ID: `DL-9922-1100` $\to$ `XXXXX1100`
- Extracted data returned in clean structured dictionaries.

### Slide 10: Document Authenticity Verification
- **Required Fields**: Asserts whether mandatory keys exist for the document type.
- **Mathematical Consistency**: Flags invoices where $Subtotal + Tax \ne Total$.
- **Chronological Logic**: Flags chronological anomalies (e.g. Invoice Date after Due Date, or Date of Birth after Card Issue Date).
- **Duplicate Detection**: SHA-256 cryptographic digest + perceptual hash ($pHash$) to prevent replay fraud.

### Slide 11: Image Manipulation Forensics (ELA)
- **Error Level Analysis (ELA)**:
  - Re-compresses the image at 90% JPEG quality.
  - Computes pixel-level absolute difference against original scan.
  - Digitally modified or spliced regions show higher error variance due to mismatched compression levels.
- Annotated bounding boxes generated over anomalous regions.
- Transparent disclaimer: *Manipulation indicator, not legal proof.*

### Slide 12: Explainable Risk Scoring Engine
- Composite score from 0 to 100 based on additive penalties:
  - Possible Image Manipulation: $+25$
  - Mathematical Inconsistency: $+20$
  - Duplicate Detected: $+20$
  - Missing Required Fields: $+15$
  - Date Logic Conflict: $+15$
  - Low OCR Confidence: $+10$
- **Risk Categorization**:
  - `0 – 30`: **LOW RISK** (Verified)
  - `31 – 60`: **MEDIUM RISK** (Review Advised)
  - `61 – 100`: **HIGH RISK** (Manual Verification Required)

### Slide 13: Web Dashboard & PDF Audit Reports
- **Streamlit Frontend**:
  - Modern dark-mode interface with responsive cards and Plotly risk gauge.
  - Interactive Gallery with preloaded genuine and tampered samples for instant viva demos.
  - Side-by-side comparison of original vs preprocessed images and ELA maps.
- **ReportLab PDF Generator**: Downloads official verification audit certificates.
- **SQLite Database**: Persistent audit trail with search and record deletion.

### Slide 14: Testing & Experimental Evaluation
- **Automated Test Suite**: 37 unit and integration tests passing in under 3 seconds.
- Validated modules: preprocessing, OCR, classification, extraction, arithmetic checks, ELA, risk engine, database, and report generation.
- Full end-to-end integration test confirming pipeline stability.

### Slide 15: Conclusion & Future Scope
- **Key Achievements**:
  - Built a complete, runnable, modular AI verification system.
  - Explainable, transparent scoring replacing black-box models.
  - Privacy-preserving architecture with sensitive data masking.
- **Future Work**:
  - Integration of Vision-Language Models (LayoutLMv3).
  - Multilingual Indic script OCR (PaddleOCR).
  - Mobile camera auto-perspective polygon homography correction.

---

# Part 2: Top 25 College Viva Voce Questions & Answers

### General & Project Concept
**Q1: What is the main objective of this project?**
*Answer*: The objective is to build an automated, explainable web application that accepts documents in JPG, PNG, or PDF format, extracts text via OCR, classifies the document type with machine learning, validates arithmetic and chronological consistency, detects image manipulation using Error Level Analysis, computes an explainable 0–100 risk score, and generates a downloadable PDF audit report.

**Q2: Can this system legally prove that a document is fraudulent?**
*Answer*: No. The system provides an automated *risk and suspicion assessment*. In legal terminology, fraud requires forensic intent and expert testimony. Our system serves as an intelligent triage tool to highlight suspicious characteristics (like arithmetic mismatches or compression anomalies) for human review.

**Q3: Why did you choose Streamlit for the user interface?**
*Answer*: Streamlit allows building reactive, data-centric web applications natively in Python without requiring a separate Node.js/React frontend. It provides seamless integration with OpenCV, PIL, Plotly interactive charts, and Python ML pipelines, making it ideal for prototyping and college project demonstrations.

---

### Optical Character Recognition (OCR) & Preprocessing
**Q4: What is OCR and which engine does this project use?**
*Answer*: Optical Character Recognition converts visual pixels containing text into machine-readable character strings. This project uses Tesseract OCR (via `pytesseract`) which extracts text, line hierarchies, and per-word confidence scores, with an intelligent PyMuPDF fallback for direct digital text extraction.

**Q5: Why is image preprocessing necessary before OCR?**
*Answer*: Real-world document scans often suffer from noise, uneven lighting, skew (rotation), and poor contrast. Preprocessing with OpenCV cleans the image, levels contrast via CLAHE, removes tilt through deskewing, and binarizes pixels with Otsu's thresholding, dramatically improving OCR character recognition rates.

**Q6: What is CLAHE and how does it differ from standard Histogram Equalization?**
*Answer*: Contrast Limited Adaptive Histogram Equalization (CLAHE) divides the image into small tiles (e.g. $8 \times 8$). Unlike global histogram equalization which stretches contrast across the entire image and often amplifies background noise, CLAHE limits contrast enhancement by clipping the histogram at a predefined limit, resulting in uniform, artifact-free text contrast.

**Q7: How does your deskewing algorithm work?**
*Answer*: We invert the document so text characters become white pixels against a black background. We extract all text coordinates and compute a minimum-area bounding rectangle using OpenCV's `cv2.minAreaRect()`. The detected orientation angle is then used to construct an affine rotation matrix that rotates the document back to horizontal alignment.

**Q8: What does OCR confidence score represent?**
*Answer*: Tesseract computes a confidence value (0 to 100) for every recognized word based on how closely the character shapes match its internal trained language models. Our system aggregates these confidences into a document-level average. Low confidence signals blurry, low-resolution, or corrupted text.

---

### Machine Learning & Document Classification
**Q9: What machine learning algorithm is used for document classification?**
*Answer*: We use a pipeline consisting of a TF-IDF Vectorizer with unigrams and bigrams ($n \in \{1, 2\}$) and a Logistic Regression classifier trained with softmax cross-entropy loss.

**Q10: Why use TF-IDF and Logistic Regression instead of a Deep Learning model like ResNet or BERT?**
*Answer*: For a college mini-project running on a standard laptop:
1. TF-IDF + Logistic Regression requires minimal computational resources, training in seconds and inferencing in under 10 milliseconds without a GPU.
2. It produces well-calibrated class probability scores directly via the softmax function.
3. It is explainable: we can inspect the exact n-gram token weights that led to a classification decision.
4. Deep learning models like BERT require gigabytes of GPU memory and heavy downloads that are impractical for lightweight local execution.

**Q11: How do you evaluate your classification model?**
*Answer*: We evaluate on a stratified held-out test split (20%) using standard statistical metrics: Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix. We serialize these metrics into `metadata.json` and display them via an interactive Plotly heatmap in the dashboard.

**Q12: What happens if an uploaded document does not match any known category?**
*Answer*: The classifier outputs probability distributions. If the top class confidence is below our configured reliability threshold (50%) or if the text is insufficient, the document is classified as `unknown / low confidence`, which adds a penalty to the risk score.

---

### Information Extraction & Authenticity Checks
**Q13: How does the system extract structured information like invoice numbers and dates?**
*Answer*: We use document-specific regular expression pattern matchers and keyword boundary anchors. For example, for invoices, the system searches for patterns following `"Invoice Number:"`, `"Subtotal:"`, and date formats (`DD/MM/YYYY`).

**Q14: How does the system verify mathematical consistency?**
*Answer*: For invoices, it parses the subtotal, tax amount, and grand total. It computes $Expected = Subtotal + Tax$ and checks whether $|StatedTotal - Expected| \le \text{tolerance}$. If an invoice has $Subtotal = 5000$, $Tax = 900$, but states $Total = 12500$, the system flags a mathematical mismatch and adds a major risk penalty (+20).

**Q15: How does the system check for chronological date consistency?**
*Answer*: Dates are parsed into Python `datetime` objects. The system asserts logical temporal rules: an invoice date cannot be after its due date; an ID card date of birth must precede its issue date; and an issue date must precede its expiration date.

**Q16: How does duplicate detection work?**
*Answer*: We implement two complementary hashing techniques:
1. **SHA-256**: A cryptographic hash function that produces a 64-character hex digest. Any byte-level identical upload produces an exact match.
2. **Perceptual Hash ($pHash$)**: Computes visual frequency fingerprints using discrete cosine transform (DCT). Images with a Hamming distance $\le 4$ are identified as near-duplicate visual scans even if re-saved.

---

### Image Forensics & Manipulation Detection
**Q17: What is Error Level Analysis (ELA)?**
*Answer*: ELA is a computer vision forensic technique used to identify image regions with inconsistent compression levels. Lossy JPEG compression saves images at a known quality factor. When a portion of an image is altered or copy-pasted from another image, that region compresses at a different rate than the surrounding authentic background. ELA reveals these variations by subtracting a re-compressed version from the original.

**Q18: How does your code detect suspicious regions in the ELA map?**
*Answer*: After computing the ELA absolute difference and scaling brightness, we apply Otsu thresholding, morphological closing (`cv2.morphologyEx`), and contour detection (`cv2.findContours`). Clusters of high-error pixels exceeding a minimum area threshold are boxed and highlighted on the annotated output image.

**Q19: What are the limitations of ELA?**
*Answer*: ELA is sensitive to overall image compression and works best on lossy formats (JPEG). High-resolution uncompressed scans (TIFF/PNG) or images that have been repeatedly re-saved multiple times may display uniform degradation, which can mask subtle splices. Therefore, ELA is treated as an indicator, not definitive proof.

---

### Risk Scoring & Database
**Q20: How is the risk score calculated?**
*Answer*: The risk score is calculated by an additive penalty engine capped strictly between 0 and 100:
$$Score = \min(100, \sum \text{penalties})$$
Penalties are awarded for:
- Possible image manipulation: $+25$
- Mathematical mismatch: $+20$
- Duplicate document: $+20$
- Missing required fields: $+15$
- Date logic conflict: $+15$
- Low OCR confidence: $+10$
- Low classification confidence: $+10$

**Q21: What are the risk categories and recommended actions?**
*Answer*:
- **0 to 30 (LOW RISK)**: Document verified. Standard automated processing approved.
- **31 to 60 (MEDIUM RISK)**: Secondary review advised. Moderate flags require human attention.
- **61 to 100 (HIGH RISK)**: Manual verification required. High fraud suspicion detected.

**Q22: Why is the risk scoring engine explainable?**
*Answer*: In many AI systems, a single number is produced with no explanation. In our system, every point added to the risk score is accompanied by an explicit, human-readable reason (e.g. *"Mathematical error: Total amount does not match sum of subtotal and tax"*), which is displayed in the dashboard and printed on the PDF report.

**Q23: How does the system protect user privacy?**
*Answer*: Sensitive identifiers such as Bank Account numbers, Aadhaar, PAN, and Driver License numbers are masked in the presentation layer (e.g. `XXXXXXXX1012`), ensuring sensitive customer data is never exposed in plain view.

**Q24: What database is used and what does it store?**
*Answer*: SQLite is used via Python's built-in `sqlite3` module. It stores document verification metadata, SHA-256 hashes, perceptual hashes, timestamps, classification labels, risk scores, extracted JSON fields, and generated PDF report paths for audit compliance.

**Q25: What happens if Tesseract is not installed on a client's computer?**
*Answer*: The system is designed with a resilient fallback architecture. If the Tesseract binary is not detected on PATH or standard directories, the system uses a `FallbackOCREngine` that extracts direct text streams from PDFs via PyMuPDF and informs the user via the UI, preventing application crashes.
