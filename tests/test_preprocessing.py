"""Unit tests for OpenCV preprocessing module."""

import numpy as np
from PIL import Image
import pytest

from modules.preprocessing import (
    resize_image,
    convert_to_grayscale,
    remove_noise,
    enhance_contrast,
    threshold_image,
    deskew_image,
    sharpen_image,
    preprocess_image,
    PreprocessingOptions
)


@pytest.fixture
def sample_test_image():
    # Create a synthetic 800x600 RGB image with some lines and text-like blocks
    arr = np.full((600, 800, 3), 255, dtype=np.uint8)
    # Add dark shapes
    arr[100:200, 100:500] = 50
    arr[300:350, 100:700] = 30
    return Image.fromarray(arr)


def test_resize_image():
    # Create large image 3000x2000
    large_arr = np.zeros((2000, 3000, 3), dtype=np.uint8)
    resized = resize_image(large_arr, max_dimension=1500)
    assert max(resized.shape[:2]) <= 1500


def test_convert_to_grayscale(sample_test_image):
    rgb_arr = np.array(sample_test_image)
    gray = convert_to_grayscale(rgb_arr)
    assert len(gray.shape) == 2
    assert gray.shape == (600, 800)


def test_remove_noise(sample_test_image):
    gray = convert_to_grayscale(np.array(sample_test_image))
    denoised = remove_noise(gray)
    assert denoised.shape == gray.shape


def test_enhance_contrast(sample_test_image):
    gray = convert_to_grayscale(np.array(sample_test_image))
    enhanced = enhance_contrast(gray)
    assert enhanced.shape == gray.shape


def test_threshold_image(sample_test_image):
    gray = convert_to_grayscale(np.array(sample_test_image))
    thresh_otsu = threshold_image(gray, method="otsu")
    thresh_adapt = threshold_image(gray, method="adaptive")
    assert thresh_otsu.shape == gray.shape
    assert thresh_adapt.shape == gray.shape
    # Check binary values
    unique_vals = set(np.unique(thresh_otsu))
    assert unique_vals.issubset({0, 255})


def test_deskew_image(sample_test_image):
    gray = convert_to_grayscale(np.array(sample_test_image))
    deskewed, angle = deskew_image(gray)
    assert deskewed.shape == gray.shape
    assert isinstance(angle, float)


def test_preprocess_pipeline(sample_test_image):
    final_img, stages = preprocess_image(sample_test_image)
    assert isinstance(final_img, Image.Image)
    assert "original" in stages
    assert "grayscale" in stages
    assert "thresholded" in stages
