"""
OpenCV Image Preprocessing and Enhancement Module.
Prepares documents for OCR, removes noise, corrects skew, and enhances contrast.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

from modules.utils import pil_to_cv2, cv2_to_pil


@dataclass
class PreprocessingOptions:
    """Configurable options for image preprocessing."""
    apply_deskew: bool = True
    apply_denoise: bool = True
    apply_clahe: bool = True
    apply_threshold: bool = True
    threshold_method: str = "otsu"  # "otsu", "adaptive", or "none"
    target_max_dim: int = 2000


def resize_image(image: np.ndarray, max_dimension: int = 2000) -> np.ndarray:
    """
    Resizes image maintaining aspect ratio so the longest side does not exceed max_dimension.
    Prevents OCR slowdown on oversized scans.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dimension:
        return image.copy()

    scale = max_dimension / float(max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts BGR image to single-channel 8-bit grayscale."""
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(gray_image: np.ndarray, method: str = "gaussian") -> np.ndarray:
    """
    Denoises grayscale document image using Gaussian or Median filter.
    Preserves text edges while eliminating scanner speckles.
    """
    if method == "median":
        return cv2.medianBlur(gray_image, 3)
    return cv2.GaussianBlur(gray_image, (3, 3), 0)


def enhance_contrast(gray_image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to normalize lighting variations across document regions.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_image)


def threshold_image(gray_image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Binarizes grayscale image to high-contrast black-and-white for OCR optimization.
    Supports Otsu automatic global thresholding and Adaptive Gaussian local thresholding.
    """
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    # Default Otsu
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def deskew_image(gray_image: np.ndarray, max_angle: float = 45.0) -> Tuple[np.ndarray, float]:
    """
    Detects text orientation angle and deskews the document.
    Returns (deskewed_image, detected_angle_in_degrees).
    """
    # Invert image: text becomes white, background black
    inv = cv2.bitwise_not(gray_image)
    coords = np.column_stack(np.where(inv > 0))

    if len(coords) < 100:
        return gray_image.copy(), 0.0

    # Determine bounding box angle
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    # Ignore micro-tilts or extreme erroneous detections
    if abs(angle) < 0.3 or abs(angle) > max_angle:
        return gray_image.copy(), 0.0

    # Rotate around center
    (h, w) = gray_image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray_image, m, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


def sharpen_image(gray_image: np.ndarray) -> np.ndarray:
    """Applies unsharp mask kernel to sharpen blurred characters."""
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    return cv2.filter2D(gray_image, -1, kernel)


def preprocess_image(
    image: Image.Image,
    options: Optional[PreprocessingOptions] = None
) -> Tuple[Image.Image, Dict[str, Image.Image]]:
    """
    Comprehensive document preprocessing pipeline.
    
    Returns:
        (final_preprocessed_image, dictionary_of_intermediate_stages)
    """
    if options is None:
        options = PreprocessingOptions()

    cv_bgr = pil_to_cv2(image)
    stages: Dict[str, Image.Image] = {"original": image}

    # 1. Resize
    resized = resize_image(cv_bgr, max_dimension=options.target_max_dim)
    stages["resized"] = cv2_to_pil(resized)

    # 2. Grayscale
    gray = convert_to_grayscale(resized)
    stages["grayscale"] = cv2_to_pil(gray)

    # 3. Deskew
    deskewed = gray
    angle = 0.0
    if options.apply_deskew:
        deskewed, angle = deskew_image(gray)
        stages["deskewed"] = cv2_to_pil(deskewed)

    # 4. Contrast Enhancement (CLAHE)
    enhanced = deskewed
    if options.apply_clahe:
        enhanced = enhance_contrast(deskewed)
        stages["enhanced"] = cv2_to_pil(enhanced)

    # 5. Denoise
    denoised = enhanced
    if options.apply_denoise:
        denoised = remove_noise(enhanced)
        stages["denoised"] = cv2_to_pil(denoised)

    # 6. Sharpening
    sharpened = sharpen_image(denoised)
    stages["sharpened"] = cv2_to_pil(sharpened)

    # 7. Thresholding / Binarization
    final_cv = sharpened
    if options.apply_threshold and options.threshold_method != "none":
        final_cv = threshold_image(sharpened, method=options.threshold_method)
        stages["thresholded"] = cv2_to_pil(final_cv)

    final_pil = cv2_to_pil(final_cv)
    return final_pil, stages
