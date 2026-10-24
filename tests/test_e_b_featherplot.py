"""Keep ExB vectors and geographic rocket positions on the same UTC clock."""

import datetime as dt

import numpy as np
import pytest

import e_b_featherplot as feather


def write_exb(path, rocket="397", t0="10:19:00.3167", rows=None):
    if rows is None:
        rows = "1,100,-50,4\n16,200,-100,8\n31,300,-150,12\n"
    path.write_text(
        f"# Time since t0 {rocket} t0 (s): {t0}, "
        "ExB_east (m/s), ExB_north (m/s), ExB_up (m/s)\n" + rows,
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "rocket,t0,offset",
    [("397", "10:19:00.3167", 0.3167), ("398", "10:19:30.4242", 30.4242)],
)
def test_exb_time_uses_each_source_header_and_preserves_components(tmp_path, rocket, t0, offset):
    path = write_exb(tmp_path / "exb.csv", rocket=rocket, t0=t0)
    time_s, east, north = feather.load_exb(path, rocket)
    np.testing.assert_allclose(time_s, np.array([1, 16, 31]) + offset, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(east, [100, 200, 300])
    np.testing.assert_array_equal(north, [-50, -100, -150])


def test_exb_reference_is_read_from_header_not_hardcoded_for_rocket(tmp_path):
    path = write_exb(tmp_path / "exb.csv", t0="10:19:00.8167")
    time_s, _east, _north = feather.load_exb(path, "397")
    np.testing.assert_allclose(time_s, [1.8167, 16.8167, 31.8167], rtol=0, atol=1e-8)


def test_exb_rejects_different_rocket_header(tmp_path):
    path = write_exb(tmp_path / "exb.csv", rocket="397")
    with pytest.raises(ValueError):
        feather.load_exb(path, "398")


def test_exb_rejects_missing_north_component(tmp_path):
    path = write_exb(tmp_path / "exb.csv", rows="1,100\n16,200\n")
    with pytest.raises(ValueError):
        feather.load_exb(path, "397")


@pytest.mark.parametrize(
    "rocket,start,end,official_t0_offset",
    [("397", 0.0, 558.0, 0.6), ("398", 30.0, 588.0, 30.9)],
)
def test_trajectory_uses_gps_utc_and_direct_geographic_columns(
    tmp_path, rocket, start, end, official_t0_offset
):
    tg_times = [start - 1, start, start + 15, end, end + 1]
    reference = dt.datetime(2026, 2, 10, 10, 19)
    rows = []
    for index, tg in enumerate(tg_times):
        utc = reference + dt.timedelta(seconds=tg)
        # Official T0 differs from the ExB source's local reference time.
        rows.append(
            f"041 {utc:%H:%M:%S.%f},{tg - official_t0_offset},"
            f"{65 + index / 10},{-147 + index / 10},{100 + index * 20}\n"
        )
    path = tmp_path / "trajectory.csv"
    path.write_text(
        "Time (UTC / From GPS Receiver),Flight Time (Official T0),"
        "Latitude,Longitude,Altitude (km)\n" + "".join(rows),
        encoding="utf-8",
    )

    trajectory = feather.load_trajectory(path, rocket)
    np.testing.assert_allclose(trajectory["time_s"], [start, start + 15, end], atol=1e-8)
    np.testing.assert_allclose(trajectory["latitude_deg"], [65.1, 65.2, 65.3])
    np.testing.assert_allclose(trajectory["longitude_deg"], [-146.9, -146.8, -146.7])
    np.testing.assert_array_equal(trajectory["altitude_km"], [120, 140, 160])


def test_interpolation_preserves_missing_data_and_does_not_cross_gaps_or_extrapolate():
    time_s = np.array([0, 1, 2, 10, 11, 12], dtype=float)
    values = np.array([0, 10, 20, 100, np.nan, 120], dtype=float)
    targets = np.array([-1, 0, 0.5, 1.5, 2, 5, 10, 10.5, 11, 11.5, 12, 13])
    expected = [np.nan, 0, 5, 15, 20, np.nan, 100, np.nan, np.nan, np.nan, 120, np.nan]

    result = feather.interpolate_samples(time_s, values, targets, max_gap_s=2.0)

    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_velocity_offsets_preserve_directions_zero_and_scale_in_geographic_degrees():
    # WGS84 metres per degree at the equator, independent reference values.
    metres_per_degree_lon = 111319.49079327358
    metres_per_degree_lat = 110574.27582159436
    east = np.array([1000, 0, 0, -1000, 0], dtype=float)
    north = np.array([0, 1000, 0, 0, -1000], dtype=float)

    dlon, dlat = feather.velocity_to_degree_offsets(np.zeros(5), east, north, scale_seconds=10)

    np.testing.assert_allclose(dlon, east * 10 / metres_per_degree_lon, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(dlat, north * 10 / metres_per_degree_lat, rtol=1e-8, atol=1e-12)


def test_same_eastward_velocity_spans_more_longitude_at_high_latitude():
    dlon, dlat = feather.velocity_to_degree_offsets(
        np.array([0.0, 60.0]), np.array([1000.0, 1000.0]), np.zeros(2), scale_seconds=1
    )
    assert 1.99 < dlon[1] / dlon[0] < 2.01
    np.testing.assert_array_equal(dlat, [0.0, 0.0])
