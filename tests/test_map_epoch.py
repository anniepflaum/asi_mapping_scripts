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
