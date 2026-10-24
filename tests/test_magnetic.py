"""Mission epochs must survive cache reuse and other Apex model instances."""

import datetime as dt

from apexpy import Apex
import numpy as np
import pytest

from core import magnetic, traj_utils
from core.ezie import load_ezie_mem_tracks
from core.missions import mission_trajectory_paths, resolve_mission_and_date


@pytest.fixture(autouse=True)
def clear_magnetic_caches():
    magnetic._apex_for_date.cache_clear()
    traj_utils._load_mapped_traj.cache_clear()
    yield
    magnetic._apex_for_date.cache_clear()
    traj_utils._load_mapped_traj.cache_clear()


@pytest.mark.parametrize(
    "mission,rocket,date_key,year",
    [
        ("GNEISS", None, "20260210", 2026 + 40 / 365),
        ("GIRAFF", None, "20250202", 2025 + 32 / 365),
        ("GIRAFF", "380", "20250209", 2025 + 39 / 365),
    ],
)
def test_selected_mission_epoch(mission, rocket, date_key, year):
    selected_mission, selected_date = resolve_mission_and_date(mission, rocket)
    assert magnetic.mission_date_key(selected_date, selected_mission) == date_key
    assert magnetic.get_mission_apex(selected_date, selected_mission).year == pytest.approx(year)
    if rocket is None:
        assert magnetic.mission_date_key(mission=mission) == date_key
        assert magnetic.get_mission_apex(mission=mission).year == pytest.approx(year)


@pytest.mark.parametrize(
    "date",
    ["20260210", "2026-02-10", dt.date(2026, 2, 10), dt.datetime(2026, 2, 10, 12, 30)],
)
def test_date_forms_share_one_model(date):
    assert magnetic.mission_date_key(date) == "20260210"
    assert magnetic.get_mission_apex(date) is magnetic.get_mission_apex("20260210")


@pytest.mark.parametrize("date", ["20260230", "2026", "not-a-date"])
def test_invalid_model_date_is_rejected(date):
    with pytest.raises(ValueError):
        magnetic.get_mission_apex(date)


def test_cached_model_restores_epoch_after_external_apex_instance():
    model = magnetic.get_mission_apex("20260210")
    expected = model.geo2apex(65.0, -147.0, 110.0)

    other_model = Apex(date=2020)
    other_coordinates = other_model.geo2apex(65.0, -147.0, 110.0)
    assert not np.allclose(other_coordinates, expected)

    restored = magnetic.get_mission_apex("2026-02-10")
    assert restored is model
    np.testing.assert_allclose(restored.geo2apex(65.0, -147.0, 110.0), expected, rtol=0, atol=1e-6)


def test_cached_model_restores_requested_date_after_epoch_mutation():
    model = magnetic.get_mission_apex("20260210")
    expected_year = model.year
    expected_coordinates = model.geo2apex(65.0, -147.0, 110.0)

    model.set_epoch(2020)
    assert not np.allclose(model.geo2apex(65.0, -147.0, 110.0), expected_coordinates)

    restored = magnetic.get_mission_apex("20260210")
    assert restored is model
    assert restored.year == expected_year
    np.testing.assert_allclose(
        restored.geo2apex(65.0, -147.0, 110.0), expected_coordinates, rtol=0, atol=1e-6
    )


@pytest.fixture
def tiny_trajectory(tmp_path):
    path = tmp_path / "36397_GPS.csv"
    path.write_text(
        "Time (UTC / From GPS Receiver),Flight Time (Official T0),Latitude,Longitude,Altitude (km)\n"
        "041 10:19:00.0,0,65,-147,300\n"
        "041 10:19:01.0,1,65.1,-147.1,310\n",
        encoding="utf-8",
    )
    return path


def test_trajectory_cache_separates_dates_and_colors(monkeypatch, tiny_trajectory):
    calls = []
    original = Apex.map_to_height

    def count_mapping(self, lat, lon, height, newheight, *args, **kwargs):
        calls.append((self.year, newheight))
        return original(self, lat, lon, height, newheight, *args, **kwargs)

    monkeypatch.setattr(Apex, "map_to_height", count_mapping)
    first = traj_utils.load_traj(tiny_trajectory, date="20260210")
    repeat = traj_utils.load_traj(tiny_trajectory, date="2026-02-10", map_time="101901")
    assert len(calls) == 1
    np.testing.assert_array_equal(repeat[0], first[0])
    assert repeat[6] == first[0][1]

    other_date = traj_utils.load_traj(tiny_trajectory, date="20250209")
    red = traj_utils.load_traj(tiny_trajectory, date="20260210", color="red")
    assert len(calls) == 3
    assert [height for _, height in calls] == [110.0, 110.0, 180.0]
    assert np.max(np.abs(other_date[0] - first[0])) > 1e-4
    assert np.max(np.abs(red[0] - first[0])) > 1e-4

    # Returning to an earlier date still uses that date's cached trajectory.
    again = traj_utils.load_traj(tiny_trajectory, date="20260210")
    assert len(calls) == 3
    np.testing.assert_array_equal(again[0], first[0])


@pytest.mark.parametrize("date", ["20250202", "20250209"])
def test_giraff_trajectory_infers_rocket_date(monkeypatch, tiny_trajectory, date):
    raw_records = traj_utils.load_traj_records(tiny_trajectory)
    monkeypatch.setattr(traj_utils, "load_traj_records", lambda _path: raw_records)
    configured_path = mission_trajectory_paths("GIRAFF", date)["main"]

    inferred = traj_utils.load_traj(configured_path)
    expected = magnetic.get_mission_apex(date).map_to_height(*raw_records[2:], 110.0)
    np.testing.assert_allclose(inferred[0], expected[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(inferred[1], expected[1], rtol=0, atol=1e-6)


def test_build_lookup_uses_requested_date(tiny_trajectory):
    lookup = traj_utils.build_traj_lookup(tiny_trajectory, date="20250209", color="red")
    expected = magnetic.get_mission_apex("20250209").map_to_height(
        lookup["raw_lats"], lookup["raw_lons"], lookup["raw_alts_km"], 180.0
    )
    np.testing.assert_allclose(lookup["lats"], expected[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(lookup["lons"], expected[1], rtol=0, atol=1e-6)
    assert lookup["launch_sec"] == 10 * 3600 + 19 * 60


def test_ezie_tracks_use_requested_date(tmp_path):
    path = tmp_path / "ezie.csv"
    path.write_text(
        "spacecraft,mem_index,time_utc,latitude_deg,longitude_deg,altitude_km\n"
        "EZIE-B,1,2026-02-10T10:19:01,65.1,-147.1,310\n"
        "EZIE-B,1,2026-02-10T10:19:00,65,-147,300\n",
        encoding="utf-8",
    )
    tracks = load_ezie_mem_tracks(path, date="20250209")
    expected = magnetic.get_mission_apex("20250209").map_to_height(
        [65.0, 65.1], [-147.0, -147.1], [300.0, 310.0], 110.0
    )
    assert len(tracks) == 1
    np.testing.assert_allclose(tracks[0]["lat"], expected[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(tracks[0]["lon"], expected[1], rtol=0, atol=1e-6)


def test_custom_red_height_changes_mapping_and_separates_cache(tiny_trajectory):
    default = traj_utils.load_traj(tiny_trajectory, color="red", date="20260210")
    custom = traj_utils.load_traj(tiny_trajectory, color="red", date="20260210", map_alt_km=250.0)
    assert np.max(np.abs(custom[0] - default[0])) > 1e-4
    repeat = traj_utils.load_traj(tiny_trajectory, color="red", date="20260210")
    np.testing.assert_array_equal(repeat[0], default[0])
