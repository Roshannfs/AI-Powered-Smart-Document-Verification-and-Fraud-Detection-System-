"""
Configuration module for AI-Powered Smart Document Verification and Fraud Detection System.
Centralizes all directories, risk weights, thresholds, supported document types, and OCR settings.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
from typing import Set, Optional
from dotenv import load_dotenv

# Load environment variables if .env exists
load_dotenv()

# Base paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
SAMPLE_DOCS_DIR: Path = DATA_DIR / "sample_documents"
GENUINE_SAMPLES_DIR: Path = SAMPLE_DOCS_DIR / "genuine"
SUSPICIOUS_SAMPLES_DIR: Path = SAMPLE_DOCS_DIR / "suspicious"
DATASET_DIR: Path = DATA_DIR / "dataset"
MODELS_DIR: Path = BASE_DIR / "models"
MODEL_PATH: Path = MODELS_DIR / "document_classifier.pkl"
METADATA_PATH: Path = MODELS_DIR / "metadata.json"
REPORTS_DIR: Path = BASE_DIR / "reports"
DB_PATH: Path = DATA_DIR / "verification_history.db"

# Ensure all essential directories exist
for folder in [RAW_DATA_DIR, GENUINE_SAMPLES_DIR, SUSPICIOUS_SAMPLES_DIR, DATASET_DIR, MODELS_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


@dataclass
class RiskWeights:
    """Configurable weights for the risk scoring engine. Total contributes to risk score (capped at 100)."""
    missing_required_fields: int = 15
    mathematical_mismatch: int = 20
    possible_image_manipulation: int = 25
    duplicate_document: int = 20
    date_inconsistency: int = 15
    low_ocr_confidence: int = 10
    low_classification_confidence: int = 10


@dataclass
class RiskThresholds:
    """Risk classification boundaries."""
    LOW_MAX: int = 30       # 0 to 30 => LOW RISK
    MEDIUM_MAX: int = 60    # 31 to 60 => MEDIUM RISK
    HIGH_MAX: int = 100     # 61 to 100 => HIGH RISK

    LOW_LABEL: str = "LOW RISK"
    MEDIUM_LABEL: str = "MEDIUM RISK"
    HIGH_LABEL: str = "HIGH RISK"


def detect_tesseract_binary() -> Optional[str]:
    """
    Intelligently discover Tesseract OCR binary across standard Windows and Unix paths,
    falling back to system PATH or custom environment variable.
    """
    # 1. Environment variable override
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).is_file():
        return env_cmd

    # 2. Check system PATH
    which_cmd = shutil.which("tesseract")
    if which_cmd:
        return which_cmd

    # 3. Common Windows install locations
    windows_candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%USERPROFILE%\AppData\Local\Tesseract-OCR\tesseract.exe"),
    ]
    for path in windows_candidates:
        if Path(path).is_file():
            return path

    return None


@dataclass
class Settings:
    """Application global settings."""
    app_title: str = "AI Document Verification & Fraud Detection System"
    app_subtitle: str = "OCR • Computer Vision • Machine Learning • Explainable Risk Scoring"
    version: str = "1.0.0"

    # Supported Document Categories
    DOC_TYPES: list = field(default_factory=lambda: [
        "invoice",
        "bank_statement",
        "certificate",
        "id_card",
        "application_form",
        "unknown"
    ])

    # Display friendly names
    DOC_TYPE_LABELS: dict = field(default_factory=lambda: {
        "invoice": "Invoice / Bill",
        "bank_statement": "Bank Statement",
        "certificate": "Educational Certificate",
        "id_card": "Identity Card",
        "application_form": "Application Form",
        "unknown": "Unknown / Unclassified"
    })

    # Allowed uploads
    allowed_extensions: Set[str] = field(default_factory=lambda: {".jpg", ".jpeg", ".png", ".pdf"})
    max_file_size_mb: int = 10
    max_pdf_pages: int = 10

    # OCR Settings
    tesseract_cmd: Optional[str] = field(default_factory=detect_tesseract_binary)
    ocr_min_confidence_warning: float = 60.0
    classification_min_confidence_warning: float = 0.50

    # Risk Scoring Config
    risk_weights: RiskWeights = field(default_factory=RiskWeights)
    risk_thresholds: RiskThresholds = field(default_factory=RiskThresholds)

    # Database
    db_path: Path = DB_PATH


# Global settings singleton
settings = Settings()
