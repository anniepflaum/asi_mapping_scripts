"""Check mission selection for maps using local images."""

import datetime as dt

from apexpy.helpers import toYearFraction
import numpy as np
import requests

import map_asi_archive as mapping


def test_local_map_uses_selected_rocket_date_without_network(monkeypatch):
    plots = []
    network_calls = []

    def unexpected_request(*args, **kwargs):
        network_calls.append((args, kwargs))
        raise AssertionError("Mapping local images must not make network requests")

    monkeypatch.setattr(requests.sessions.Session, "request", unexpected_request)
    monkeypatch.setattr(mapping, "cached_skymaps", lambda *args: {})
    monkeypatch.setattr(mapping, "get_site_tiff_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        mapping, "load_best_frame_from_tiffs",
        lambda *args, **kwargs: (np.ones((2, 2)), None),
    )
    monkeypatch.setattr(
        mapping, "calibrate_image_for_map",
        lambda site, image, skymaps, color: (image, None),
    )

    def plot(skymaps, images, **kwargs):
        date = kwargs["map_date"]
        assert kwargs["apex"].year == toYearFraction(dt.datetime.strptime(date, "%Y%m%d"))
        plots.append((date, kwargs["color"]))

    monkeypatch.setattr(mapping, "plot_map", plot)

    for rocket, color in [("397", "green"), ("381", "green"), ("380", "green"),
                          ("397", "green"), ("397", "red")]:
        mapping.main([
            "--rocket", rocket, "--time", "102400", "--sites", "VEE",
            "--color", color, "--no-shared-norm",
        ])

    assert plots == [
        ("20260210", "green"),
        ("20250202", "green"),
        ("20250209", "green"),
        ("20260210", "green"),
        ("20260210", "red"),
    ]
    assert not network_calls


def test_red_alt_reaches_skymaps_plot_and_feathers(monkeypatch):
    calls = {}
    def skymaps(sites, color, mission, altitude):
        calls["skymap_alt"] = altitude
        return {}
    def feathers(color, date, **kwargs):
        calls["feather_alt"] = kwargs["map_alt_km"]
        return {}
    monkeypatch.setattr(mapping, "cached_skymaps", skymaps)
    monkeypatch.setattr(mapping, "cached_featherplot_context", feathers)
    monkeypatch.setattr(mapping, "plot_map", lambda *args, **kwargs: calls.update(plot_alt=kwargs["map_alt_km"]))
    mapping.main(["--mission", "GNEISS", "--time", "102400", "--sites", "--color", "red",
                  "--red-alt", "250", "--no-shared-norm", "--featherplot"])
    assert calls == {"skymap_alt": 250.0, "feather_alt": 250.0, "plot_alt": 250.0}


def test_invalid_red_alt_rejected():
    import pytest
    for options in [["--color", "red", "--red-alt", "0"],
                    ["--color", "red", "--red-alt", "nan"],
                    ["--color", "green", "--red-alt", "250"]]:
        with pytest.raises(SystemExit) as exc:
            mapping.main(["--time", "102400", *options])
        assert exc.value.code == 2
