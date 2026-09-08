"""
Document Classification Inference Module.
Uses the trained TF-IDF + Logistic Regression model to classify OCR text
into document categories with confidence estimation and threshold gating.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
import logging
from pathlib import Path
import joblib

from config.config import MODEL_PATH, settings

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Document classification prediction details."""
    document_type: str
    display_name: str
    confidence: float  # Scale 0.0 to 1.0 (e.g. 0.94 = 94%)
    confidence_percentage: float  # Scale 0.0 to 100.0 (e.g. 94.0)
    probabilities: Dict[str, float]
    is_confident: bool
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "display_name": self.display_name,
            "confidence": round(self.confidence_percentage, 1),
            "is_confident": self.is_confident,
            "warning": self.warning
        }


# Global model cache to avoid reloading from disk on every inference
_CACHED_MODEL = None


def load_classifier():
    """Loads and caches the serialized document classifier pipeline."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    if not MODEL_PATH.exists():
        logger.warning(f"Document classifier model not found at {MODEL_PATH}")
        return None

    try:
        _CACHED_MODEL = joblib.load(MODEL_PATH)
        return _CACHED_MODEL
    except Exception as e:
        logger.error(f"Failed to load classifier model: {e}")
        return None


def classify_document(ocr_text: str) -> ClassificationResult:
    """
    Classifies extracted document text into a supported category.
    Applies confidence thresholding: if confidence is below 50% or text is negligible,
    marks document as 'unknown' or low-confidence.
    """
    cleaned_text = (ocr_text or "").strip()

    # If text is too short to be meaningful
    if len(cleaned_text) < 20:
        return ClassificationResult(
            document_type="unknown",
            display_name=settings.DOC_TYPE_LABELS.get("unknown", "Unknown / Low Confidence"),
            confidence=0.0,
            confidence_percentage=0.0,
            probabilities={},
            is_confident=False,
            warning="Extracted text is insufficient for reliable classification."
        )

    model = load_classifier()
    if model is None:
        return ClassificationResult(
            document_type="unknown",
            display_name="Unknown (Model Not Loaded)",
            confidence=0.0,
            confidence_percentage=0.0,
            probabilities={},
            is_confident=False,
            warning="Classification model is missing or could not be loaded."
        )

    try:
        probs = model.predict_proba([cleaned_text])[0]
        classes = list(model.classes_)

        prob_dict = {cls_name: round(float(p), 4) for cls_name, p in zip(classes, probs)}
        top_idx = int(probs.argmax())
        top_class = classes[top_idx]
        top_conf = float(probs[top_idx])

        # Check against minimum confidence threshold (50%)
        is_confident = top_conf >= settings.classification_min_confidence_warning
        doc_type = top_class if is_confident else "unknown"
        display_name = settings.DOC_TYPE_LABELS.get(doc_type, doc_type.replace("_", " ").title())

        warning = None
        if not is_confident:
            warning = f"Classification confidence ({top_conf * 100:.1f}%) is below the reliability threshold (50%)."

        return ClassificationResult(
            document_type=doc_type,
            display_name=display_name,
            confidence=top_conf,
            confidence_percentage=top_conf * 100.0,
            probabilities=prob_dict,
            is_confident=is_confident,
            warning=warning
        )

    except Exception as e:
        logger.error(f"Classification failed during prediction: {e}")
        return ClassificationResult(
            document_type="unknown",
            display_name=settings.DOC_TYPE_LABELS.get("unknown", "Unknown"),
            confidence=0.0,
            confidence_percentage=0.0,
            probabilities={},
            is_confident=False,
            warning=f"Inference error: {str(e)}"
        )
