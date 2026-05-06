#!/usr/bin/env python3
"""Geometric coordinate-conversion helpers."""

import numpy as np
import pymap3d as pm

from core.constants import DEFAULT_GREEN_ALT_KM


def calc_ipp(site, rocket, rockcoords="geo", height=DEFAULT_GREEN_ALT_KM):
    """
    Calculate the ionospheric pierce point for a receiver-rocket line of sight.

    Parameters
    ----------
    site : sequence of float
        Ground receiver geodetic coordinates [latitude, longitude, altitude_m].
    rocket : sequence of float
        rocket coordinates in the representation specified by ``rockcoords``.
        Supported forms are [azimuth_deg, elevation_deg], [x, y, z] ECEF meters,
        or [latitude, longitude, altitude_m] geodetic.
    rockcoords : {"azel", "ecef", "geo"}, optional
        Coordinate system used by ``rocket``.
    height : float, optional
        Ionospheric shell height in kilometers.

    Returns
    -------
    tuple[float, float]
        Geodetic latitude and longitude of the ionospheric pierce point.
    """
    lat0, lon0, alt0 = site
    x, y, z = pm.geodetic2ecef(lat0, lon0, alt0)

    if rockcoords == "azel":
        e, n, u = pm.aer2enu(*rocket, 1.0)
        vx, vy, vz = pm.enu2uvw(e, n, u, lat0, lon0)
    elif rockcoords == "ecef":
        vx = rocket[0] - x
        vy = rocket[1] - y
        vz = rocket[2] - z
    elif rockcoords == "geo":
        sx, sy, sz = pm.geodetic2ecef(*rocket)
        vx = sx - x
        vy = sy - y
        vz = sz - z
    else:
        raise ValueError(f"rockcoords={rockcoords} is not a valid coordinate option")

    earth = pm.Ellipsoid.from_name("wgs84")
    a2 = (earth.semimajor_axis + height * 1000.0) ** 2
    b2 = (earth.semimajor_axis + height * 1000.0) ** 2
    c2 = (earth.semiminor_axis + height * 1000.0) ** 2

    A = vx**2 / a2 + vy**2 / b2 + vz**2 / c2
    B = x * vx / a2 + y * vy / b2 + z * vz / c2
    C = x**2 / a2 + y**2 / b2 + z**2 / c2 - 1

    alpha = (np.sqrt(B**2 - A * C) - B) / A
    lat, lon, _alt = pm.ecef2geodetic(x + alpha * vx, y + alpha * vy, z + alpha * vz)
    return lat, lon
