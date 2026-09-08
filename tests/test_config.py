"""Unit tests for configuration and environment setup."""

from pathlib import Path
from config.config import settings, RiskWeights, RiskThresholds


def test_settings_initialization():
    assert settings.app_title is not None
    assert "invoice" in settings.DOC_TYPES
    assert "bank_statement" in settings.DOC_TYPES
    assert "certificate" in settings.DOC_TYPES
    assert "id_card" in settings.DOC_TYPES
    assert "application_form" in settings.DOC_TYPES
    assert settings.max_file_size_mb == 10
    assert ".pdf" in settings.allowed_extensions
    assert ".png" in settings.allowed_extensions
    assert ".jpg" in settings.allowed_extensions


def test_risk_weights():
    weights = RiskWeights()
    assert weights.missing_required_fields == 15
    assert weights.mathematical_mismatch == 20
    assert weights.possible_image_manipulation == 25
    assert weights.duplicate_document == 20


def test_risk_thresholds():
    thresholds = RiskThresholds()
    assert thresholds.LOW_MAX == 30
    assert thresholds.MEDIUM_MAX == 60
    assert thresholds.HIGH_MAX == 100


def test_directory_creation():
    from config.config import RAW_DATA_DIR, SAMPLE_DOCS_DIR, MODELS_DIR, REPORTS_DIR
    assert RAW_DATA_DIR.exists()
    assert SAMPLE_DOCS_DIR.exists()
    assert MODELS_DIR.exists()
    assert REPORTS_DIR.exists()
