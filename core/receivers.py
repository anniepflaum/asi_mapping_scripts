"""Receiver CSV loading and mission filtering helpers."""

import csv

from core.paths import RECEIVERS_PATH
from core.missions import mission_receiver_acronyms


def load_receivers(path=RECEIVERS_PATH, warn=None):
    try:
        with open(path, "r", encoding="utf-8", newline="") as fd:
            reader = csv.DictReader(fd)
            return [
                {
                    "name": row["Name"].strip(),
                    "acronym": row["acronym"].strip(),
                    "lon": float(row["Lon"]),
                    "lat": float(row["Lat"]),
                    "alt_m": float(row.get("Alt_m", 0.0) or 0.0),
                }
                for row in reader
                if row.get("Lon") and row.get("Lat")
            ]
    except FileNotFoundError:
        if warn is not None:
            warn(f"Receiver file not found: {path}. Receiver and IPP overlays will be skipped.")
        return []


def filter_receivers_for_mission(receivers, mission):
    acronyms = mission_receiver_acronyms(mission)
    if acronyms is None:
        return receivers
    receiver_map = {receiver.get("acronym", "").upper(): receiver for receiver in receivers}
    return [receiver_map[acronym] for acronym in acronyms if acronym in receiver_map]
