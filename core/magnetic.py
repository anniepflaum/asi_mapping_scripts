"""Mission-dated Apex models shared by the mapping and sampling scripts."""

import datetime as dt
from functools import lru_cache

from apexpy import Apex
from apexpy.helpers import toYearFraction

from core.missions import default_date, normalize_date_key


def mission_date_key(date=None, mission="GNEISS"):
    """Return a validated YYYYMMDD model date; never default to today's date."""
    if date is None:
        date = default_date(mission)
    if isinstance(date, dt.date):
        date = date.strftime("%Y%m%d")
    return dt.datetime.strptime(normalize_date_key(date), "%Y%m%d").strftime("%Y%m%d")


@lru_cache(maxsize=16)
def _apex_for_date(date_key):
    return Apex(date=dt.datetime.strptime(date_key, "%Y%m%d"))


def get_mission_apex(date=None, mission="GNEISS"):
    """Reuse the date's model and activate it before a batch of conversions.

    ApexPy's Fortran coefficients are shared across instances. Reactivation is
    necessary even for a cache hit after another model (including one created
    by a dependency) has selected a different epoch. Call this again before
    reusing a retained model if another date has been used in the meantime.
    """
    date_key = mission_date_key(date, mission)
    apex = _apex_for_date(date_key)
    # Derive the epoch from the key, since callers can change an Apex instance
    # with set_epoch as well as changing the process-wide Fortran state.
    apex.set_epoch(toYearFraction(dt.datetime.strptime(date_key, "%Y%m%d")))
    return apex
