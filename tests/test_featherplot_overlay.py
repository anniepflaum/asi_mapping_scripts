from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

from core.plotting import draw_featherplot, nearest_feather_vector_index, nearest_feather_vector_indices
import map_asi_archive_series
from e_b_featherplot import velocity_to_degree_offsets
from map_asi_archive import map_featherplot_geometry


def tiny_context():
    trajectory = {
        "longitude_deg": np.array([-147.0, -146.9]),
        "latitude_deg": np.array([66.0, 66.1]),
    }
    vectors = {
        "time_s": np.array([100.0, 110.0, 120.0]),
        "utc_seconds": np.array([37240.0, 37250.0, 37260.0]),
        "longitude_deg": np.array([-147.0, -146.9, -146.8]),
        "latitude_deg": np.array([66.0, 66.1, 66.2]),
        "dlon": np.array([0.1, 0.2, 0.3]),
        "dlat": np.array([0.0, 0.1, 0.2]),
    }
    return {"397": {"trajectory": trajectory, "vectors": vectors}}


def test_nearest_vector_uses_map_utc_and_resolves_tie_to_earlier_sample():
    vectors = tiny_context()["397"]["vectors"]
    assert nearest_feather_vector_index(vectors, "102050") == 1
    assert nearest_feather_vector_index(vectors, "102045") == 0


def test_draw_uses_pale_noncurrent_vectors_and_saturated_nearest_vector():
    fig, ax = plt.subplots()
    draw_featherplot(ax, tiny_context(), "102050", ax.transData)

    assert len(ax.collections) == 2
    faint, nearest = ax.collections
    np.testing.assert_allclose(faint.get_facecolors()[0], to_rgba("#a6cee3", alpha=0.75))
    np.testing.assert_allclose(nearest.get_facecolors()[0], to_rgba("tab:blue"))
    assert len(faint.U) == 2
    assert len(nearest.U) == 1
    plt.close(fig)


def test_series_passes_featherplot_flag_to_each_map_call():
    args = SimpleNamespace(
        mission="GNEISS", color="green", red_wavelength="6300",
        colorbar_scale="log", vmax=99.0, render_mode="pcolormesh",
        rocket=None, sites=None, no_shared_norm=False, bounds=None,
        pretty=False, plot_receivers=False, plot_ipps=False,
        plot_geodetic_traj=False, featherplot=True, plot_ezie=False, step=5.0,
    )
    result = map_asi_archive_series.build_map_args(args, "102400")
    assert "--featherplot" in result
    assert result[result.index("--featherplot-highlight-count") + 1] == "5"


def test_arrow_is_geodetic_at_target_height_on_apex_mapped_tail():
    class FakeApex:
        @staticmethod
        def map_to_height(lat, lon, altitude, target_height):
            lat = np.asarray(lat, dtype=float)
            lon = np.asarray(lon, dtype=float)
            altitude = np.asarray(altitude, dtype=float)
            return lat + 0.1 * lon, lon + 0.2 * lat, altitude * 0 + target_height

    trajectory = {
        "latitude_deg": np.array([66.0, 66.1]),
        "longitude_deg": np.array([-147.0, -146.9]),
        "altitude_km": np.array([200.0, 210.0]),
    }
    vectors = {
        "time_s": np.array([100.0]),
        "utc_seconds": np.array([37240.0]),
        "latitude_deg": np.array([66.0]),
        "longitude_deg": np.array([-147.0]),
        "altitude_km": np.array([200.0]),
        "east_m_s": np.array([1000.0]),
        "north_m_s": np.array([500.0]),
    }
    mapped = map_featherplot_geometry(FakeApex(), trajectory, vectors, 110.0)
    tail_lat = 66.0 + 0.1 * -147.0
    tail_lon = -147.0 + 0.2 * 66.0
    expected_dlon, expected_dlat = velocity_to_degree_offsets(
        np.array([tail_lat]), vectors["east_m_s"], vectors["north_m_s"],
        scale_seconds=160.0, altitude_km=110.0,
    )

    np.testing.assert_allclose(mapped["vectors"]["latitude_deg"], [tail_lat])
    np.testing.assert_allclose(mapped["vectors"]["longitude_deg"], [tail_lon])
    np.testing.assert_allclose(mapped["vectors"]["dlat"], expected_dlat)
    np.testing.assert_allclose(mapped["vectors"]["dlon"], expected_dlon)


def test_multiple_nearest_feathers_and_available_vector_limit():
    context = tiny_context()
    vectors = context["397"]["vectors"]
    np.testing.assert_array_equal(nearest_feather_vector_indices(vectors, "102050", 2), [1, 0])
    for count, expected in [(2, 2), (5, 3)]:
        fig, ax = plt.subplots()
        draw_featherplot(ax, context, "102050", ax.transData, highlight_count=count)
        assert len(ax.collections[-1].U) == expected
        np.testing.assert_allclose(ax.collections[-1].get_facecolors()[0], to_rgba("tab:blue"))
        plt.close(fig)


def test_five_nearest_vectors_follow_rocket_time():
    vectors = {"utc_seconds": np.arange(37240.0, 37254.0, 2.0)}
    selected = nearest_feather_vector_indices(vectors, "102046", 5)
    np.testing.assert_array_equal(selected, [3, 2, 4, 1, 5])
