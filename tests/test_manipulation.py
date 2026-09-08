"""Unit tests for image manipulation and ELA analysis."""

import numpy as np
from PIL import Image
import pytest
from modules.manipulation import (
    compute_error_level_analysis,
    analyze_manipulation,
    ManipulationReport
)


def test_compute_error_level_analysis():
    img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    ela_im, ela_gray = compute_error_level_analysis(img)
    assert isinstance(ela_im, Image.Image)
    assert isinstance(ela_gray, np.ndarray)
    assert ela_gray.shape == (300, 300)


def test_analyze_manipulation_clean_image():
    # Plain solid image has uniform compression
    img = Image.new("RGB", (400, 400), color=(240, 240, 240))
    report = analyze_manipulation(img)
    assert isinstance(report, ManipulationReport)
    assert isinstance(report.annotated_image, Image.Image)
    assert isinstance(report.manipulation_score, float)
    assert report.method_disclaimer is not None
