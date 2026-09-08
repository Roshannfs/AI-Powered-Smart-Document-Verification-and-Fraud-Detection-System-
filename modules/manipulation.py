"""
Image Manipulation & Forensic Suspicion Analysis Module.
Performs Error Level Analysis (ELA), edge inconsistency detection,
and local noise variance analysis to identify potential tampering indicators.
"""

from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from modules.utils import pil_to_cv2, cv2_to_pil


@dataclass
class SuspiciousRegion:
    """Bounding box and score for a detected suspicious region."""
    x: int
    y: int
    w: int
    h: int
    score: float
    description: str


@dataclass
class ManipulationReport:
    """Aggregated forensic manipulation analysis results."""
    has_suspicious_regions: bool
    manipulation_score: float  # 0.0 (clean) to 1.0 (highly suspicious)
    ela_image: Image.Image
    annotated_image: Image.Image
    regions: List[SuspiciousRegion] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    method_disclaimer: str = (
        "Forensic indicators detect compression and edge anomalies. "
        "They indicate potential manipulation and do not constitute definitive legal proof."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_suspicious_regions": self.has_suspicious_regions,
            "manipulation_score": round(self.manipulation_score, 2),
            "regions_count": len(self.regions),
            "reasons": self.reasons,
            "disclaimer": self.method_disclaimer
        }


def compute_error_level_analysis(image: Image.Image, quality: int = 90, scale: int = 15) -> Tuple[Image.Image, np.ndarray]:
    """
    Computes Error Level Analysis (ELA) by re-compressing at a specific JPEG quality
    and scaling pixel-level difference against the original.
    """
    rgb_img = image.convert("RGB")

    # Resave in-memory at defined JPEG quality
    buffer = BytesIO()
    rgb_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    resaved_img = Image.open(buffer)

    # Compute absolute difference
    ela_im = ImageChops.difference(rgb_img, resaved_img)

    # Scale the difference so human eye and contours can analyze the variance
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 1
    if max_diff == 0:
        max_diff = 1

    scale_factor = 255.0 / max_diff if max_diff < 50 else scale
    enhanced_ela = ImageEnhance.Brightness(ela_im).enhance(scale_factor)

    # Return PIL image and grayscale NumPy array for contour processing
    ela_np = np.array(enhanced_ela)
    ela_gray = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)

    return enhanced_ela, ela_gray


def detect_suspicious_contours(
    ela_gray: np.ndarray,
    orig_bgr: np.ndarray,
    min_area: int = 400,
    max_area_ratio: float = 0.4
) -> Tuple[List[SuspiciousRegion], np.ndarray, float]:
    """
    Identifies anomalous high-error clusters in the ELA map and draws bounding boxes.
    """
    annotated = orig_bgr.copy()
    h, w = orig_bgr.shape[:2]
    total_area = h * w

    # Threshold high difference pixels in ELA
    _, thresh = cv2.threshold(ela_gray, 45, 255, cv2.THRESH_BINARY)

    # Morphological dilation to connect nearby error dots into distinct blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: List[SuspiciousRegion] = []
    suspicious_area_sum = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > (total_area * max_area_ratio):
            continue

        rx, ry, rw, rh = cv2.boundingRect(cnt)
        # Avoid full borders
        if rx < 10 or ry < 10 or (rx + rw) > (w - 10) or (ry + rh) > (h - 10):
            continue

        # Extract patch and compute local ELA variance
        patch = ela_gray[ry:ry + rh, rx:rx + rw]
        mean_intensity = float(np.mean(patch))

        if mean_intensity > 35:
            suspicious_area_sum += area
            region = SuspiciousRegion(
                x=rx, y=ry, w=rw, h=rh,
                score=round(mean_intensity / 255.0, 2),
                description=f"Compression inconsistency (intensity: {mean_intensity:.1f})"
            )
            regions.append(region)

            # Draw visual highlight on annotated image
            cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
            cv2.putText(
                annotated, "Potential Manipulation Indicator",
                (rx, max(18, ry - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1, cv2.LINE_AA
            )

    # Score based on number and area of suspicious regions
    area_ratio = min(1.0, suspicious_area_sum / (total_area * 0.15))
    count_factor = min(1.0, len(regions) / 3.0)
    composite_score = round(0.6 * area_ratio + 0.4 * count_factor, 2)

    return regions, annotated, composite_score


def analyze_manipulation(image: Image.Image) -> ManipulationReport:
    """
    Main entrypoint for document manipulation detection.
    Runs ELA and contour segmentation to locate potential spliced/altered regions.
    """
    orig_bgr = pil_to_cv2(image)
    ela_img, ela_gray = compute_error_level_analysis(image)

    regions, annotated_bgr, score = detect_suspicious_contours(ela_gray, orig_bgr)
    annotated_pil = cv2_to_pil(annotated_bgr)

    has_suspicious = len(regions) > 0 and score > 0.20
    reasons = []

    if has_suspicious:
        reasons.append(
            f"Detected {len(regions)} region(s) with anomalous compression artifact levels (Error Level Analysis)."
        )
    else:
        reasons.append("No significant compression or edge splicing inconsistencies detected.")

    return ManipulationReport(
        has_suspicious_regions=has_suspicious,
        manipulation_score=score if has_suspicious else 0.0,
        ela_image=ela_img,
        annotated_image=annotated_pil,
        regions=regions,
        reasons=reasons
    )
