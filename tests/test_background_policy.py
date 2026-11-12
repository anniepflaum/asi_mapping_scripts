"""Calibration must select background from native elevation, not the display mask."""
import numpy as np
import pytest
from core.calibration import calibrate_image, calibrate_image_cached, calibration_metadata


def test_background_is_independent_of_display_mask_and_color():
    elevation = np.zeros((200, 200))
    elevation[:, 100:] = 10.0
    image = np.full(elevation.shape, 20.0)
    image[:, 50:] = 10000.0
    for color in ('red', 'green', 'blue'):
        outputs = []
        for display_mask in (elevation < 22, elevation < 0):
            calibrated, bg = calibrate_image('VEE', image, display_mask, color, elevation=elevation)
            assert bg['center'] == 20.0
            assert bg['n_pixels'] == 10000
            outputs.append(calibrated)
        np.testing.assert_array_equal(*outputs)
        metadata = calibration_metadata(['VEE'], color)
        assert metadata['background_elevation_cutoff_deg'] == 0
        assert metadata['background_edge_buffer_px'] == 50


def test_missing_elevation_cannot_fall_back_to_display_mask():
    with pytest.raises(ValueError, match='Native elevation is required'):
        calibrate_image('VEE', np.ones((200, 200)), np.ones((200, 200), bool), 'green')


def test_cached_background_changes_when_buffer_changes():
    elevation = np.zeros((200, 200))
    elevation[:, 100:] = 10
    image = np.ones(elevation.shape)
    cache = {}
    _, a = calibrate_image_cached('VEE', image, elevation < 22, 'green', 1, cache, elevation=elevation)
    _, b = calibrate_image_cached('VEE', image, elevation < 22, 'green', 1, cache, edge_buffer_px=60, elevation=elevation)
    assert a['n_pixels'] == 10000
    assert b['n_pixels'] == 8000
