"""Detector offsets are applied once, in DN, before downstream processing."""
import datetime as dt
import numpy as np
import tifffile
from asi_shared.detector import correct_detector_frame, ARV_BLUE_CORRECTION
from core.tiff_utils import load_best_frame_from_tiffs, load_best_frame_from_cached_tiffs, build_tiff_metadata
from core.series_utils import load_tiff_frame_with_metadata
from core.select_tiffs import read_selected_frame


def test_exact_offsets_and_scope():
    raw = np.zeros((512, 512), dtype=np.uint16)
    fixed = correct_detector_frame(raw, site='ARV', color='blue')
    expected = np.zeros_like(fixed)
    expected[153:, 281] = -77
    expected[153:, 282] = -47
    expected[146:, 342] = -47
    expected[146:, 343] = -429
    expected[146:, 344] = -10
    np.testing.assert_array_equal(fixed, expected)
    assert not raw.any()
    for site, color in [('VEE','blue'), ('BVR','blue'), ('ARV','green'), ('ARV','red')]:
        assert correct_detector_frame(raw, site=site, color=color) is raw


def test_all_mapping_readers_and_corrected_export(tmp_path):
    directory = tmp_path / 'blue' / 'ARV'
    directory.mkdir(parents=True)
    source = directory / 'ARV_4278_20260210_102400.tiff'
    raw = np.full((512,512), 500, dtype=np.uint16)
    tifffile.imwrite(source, raw)
    target = dt.datetime(2026,2,10,10,24)
    expected = correct_detector_frame(raw, source)
    metadata = build_tiff_metadata([str(source)], .9)
    loaded, _ = load_best_frame_from_tiffs('ARV',[str(source)],target,.9,color='blue')
    np.testing.assert_array_equal(loaded, expected)
    np.testing.assert_array_equal(load_best_frame_from_cached_tiffs('ARV',metadata,target,.9), expected)
    loaded, _ = load_tiff_frame_with_metadata('ARV',metadata,target,.9)
    np.testing.assert_array_equal(loaded, expected)
    np.testing.assert_array_equal(read_selected_frame({'path':source,'idx':0}), expected)
    untouched, _ = load_best_frame_from_tiffs('ARV',[str(source)],target,.9,color='blue',correct_detector=False)
    np.testing.assert_array_equal(untouched, raw)
    exported = directory / 'ARV_GNEISS_blue_102400_102400.tiff'
    tifffile.imwrite(exported, expected, metadata={'detector_correction':ARV_BLUE_CORRECTION})
    np.testing.assert_array_equal(read_selected_frame({'path':exported,'idx':0}), expected)
