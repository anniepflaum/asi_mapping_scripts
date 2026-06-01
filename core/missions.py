"""Mission-specific configuration and small policy helpers."""

import datetime as dt
from dataclasses import dataclass

from core.paths import (
    GIRAFF_380_TRAJECTORY_PATH,
    GIRAFF_381_TRAJECTORY_PATH,
    GNEISS_LEFT_TRAJECTORY_PATH,
    GNEISS_RIGHT_TRAJECTORY_PATH,
    MAPPED_DIR,
)


GNEISS_DEFAULT_DATE = "20260210"
GIRAFF_DEFAULT_DATE = "20250202"
GIRAFF_RECEIVER_ACRONYMS = ("VEE", "TOO", "PKR")
GIRAFF_DEFAULT_SITE_ACRONYMS = ("VEE",)
GIRAFF_SITE_ACRONYMS = ("VEE", "PKR")
GNEISS_TIFF_SITE_ACRONYMS = ("ARV", "VEE", "BVR")
GNEISS_ALL_SITE_ACRONYMS = ("ARV", "BVR", "VEE", "PKR")
ROCKET_TIME_WINDOWS = {
    "397": ("101900", "102818"),
    "398": ("101930", "102848"),
    "381": ("070715", "071636"),
    "380": ("083501", "084410"),
}
ROCKET_DEFAULT_DATES = {
    "397": GNEISS_DEFAULT_DATE,
    "398": GNEISS_DEFAULT_DATE,
    "381": "20250202",
    "380": "20250209",
}
MISSION_ROCKETS = {
    "GNEISS": ("397", "398"),
    "GIRAFF": ("380", "381"),
}


@dataclass(frozen=True)
class TrajectoryConfig:
    key: str
    tag: str
    label: str
    path: object


def mission_key(mission):
    return str(mission).upper()


def normalize_date_key(date=None):
    if date is None:
        return None
    return str(date).replace("-", "")


def default_date(mission):
    return GIRAFF_DEFAULT_DATE if mission_key(mission) == "GIRAFF" else GNEISS_DEFAULT_DATE


def mission_for_rocket(rocket_tag):
    rocket_tag = str(rocket_tag)
    for mission, rockets in MISSION_ROCKETS.items():
        if rocket_tag in rockets:
            return mission
    raise ValueError(f"Unsupported rocket: {rocket_tag}")


def resolve_mission_and_date(mission=None, rocket_tag=None, default_mission="GNEISS"):
    """Return a validated mission key and default date for a mission/rocket selection."""
    if mission is None:
        mission = mission_for_rocket(rocket_tag) if rocket_tag is not None else default_mission
    key = mission_key(mission)
    if rocket_tag is not None and str(rocket_tag) not in MISSION_ROCKETS.get(key, ()):
        allowed = " or ".join(MISSION_ROCKETS.get(key, ()))
        raise ValueError(f"--mission {key} only supports --rocket {allowed}")
    date_key = rocket_default_date(rocket_tag) if rocket_tag is not None else default_date(key)
    return key, date_key


def default_sites(mission, include_pkr=False):
    if mission_key(mission) == "GIRAFF":
        return list(GIRAFF_SITE_ACRONYMS if include_pkr else GIRAFF_DEFAULT_SITE_ACRONYMS)
    return list(GNEISS_ALL_SITE_ACRONYMS if include_pkr else GNEISS_TIFF_SITE_ACRONYMS)


def default_time_range(mission, date=None, rocket_tags=None):
    """Return the default HHMMSS start/end window for a mission product."""
    key = mission_key(mission)
    if rocket_tags is None:
        rocket_tags = [giraff_rocket_id_for_date(date)] if key == "GIRAFF" else ["397", "398"]
    windows = [ROCKET_TIME_WINDOWS[str(tag)] for tag in rocket_tags]
    return min(start for start, _end in windows), max(end for _start, end in windows)


def rocket_launch_start(rocket_tag):
    """Return mission-configured T0 as HHMMSS for a rocket."""
    return ROCKET_TIME_WINDOWS[str(rocket_tag)][0]


def rocket_default_date(rocket_tag):
    """Return the default YYYYMMDD date for a rocket."""
    return ROCKET_DEFAULT_DATES[str(rocket_tag)]


def rocket_launch_datetime(rocket_tag):
    """Return mission-configured T0 as a datetime for a rocket."""
    date_key = rocket_default_date(rocket_tag)
    launch_start = rocket_launch_start(rocket_tag)
    return dt.datetime(
        int(date_key[:4]),
        int(date_key[4:6]),
        int(date_key[6:8]),
        int(launch_start[:2]),
        int(launch_start[2:4]),
        int(launch_start[4:6]),
    )


def rocket_time_window_datetimes(rocket_tag):
    """Return mission-configured start/end datetimes for a rocket."""
    date_key = rocket_default_date(rocket_tag)
    start, end = ROCKET_TIME_WINDOWS[str(rocket_tag)]
    return (
        dt.datetime(
            int(date_key[:4]),
            int(date_key[4:6]),
            int(date_key[6:8]),
            int(start[:2]),
            int(start[2:4]),
            int(start[4:6]),
        ),
        dt.datetime(
            int(date_key[:4]),
            int(date_key[4:6]),
            int(date_key[6:8]),
            int(end[:2]),
            int(end[2:4]),
            int(end[4:6]),
        ),
    )


def giraff_rocket_id_for_trajectory_path(path):
    """Return the GIRAFF rocket ID implied by a trajectory path, if known."""
    path_name = getattr(path, "name", None) or str(path)
    if path_name == GIRAFF_380_TRAJECTORY_PATH.name or "36380" in path_name:
        return "380"
    if path_name == GIRAFF_381_TRAJECTORY_PATH.name or "36381" in path_name:
        return "381"
    return None


def validate_color_and_sites(parser, mission, color, sites, giraff_message_site="VEE"):
    """Apply mission-specific constraints to color/site arguments."""
    if mission_key(mission) != "GIRAFF":
        return
    selected_sites = {site.upper() for site in sites}
    invalid_sites = sorted(selected_sites - set(GIRAFF_SITE_ACRONYMS))
    if invalid_sites:
        parser.error(f"--mission GIRAFF only supports {giraff_message_site}; remove site(s): {', '.join(invalid_sites)}")


def giraff_rocket_id_for_date(date=None):
    return "380" if normalize_date_key(date) == "20250209" else "381"


def mission_output_dir(mission, color="green", date=None):
    base = MAPPED_DIR / str(color).lower()
    key = mission_key(mission)
    if key == "GIRAFF":
        return base / "GIRAFF" / giraff_rocket_id_for_date(date)
    return base / key


def mission_receiver_acronyms(mission):
    if mission_key(mission) == "GIRAFF":
        return GIRAFF_RECEIVER_ACRONYMS
    return None


def mission_trajectory_paths(mission, date=None):
    key = mission_key(mission)
    date_key = normalize_date_key(date)
    if key == "GIRAFF":
        if date_key == "20250209":
            return {"main": GIRAFF_380_TRAJECTORY_PATH}
        return {"main": GIRAFF_381_TRAJECTORY_PATH}
    if key == "GNEISS":
        return {
            "left": GNEISS_LEFT_TRAJECTORY_PATH,
            "right": GNEISS_RIGHT_TRAJECTORY_PATH,
        }
    raise ValueError(f"Unsupported mission: {mission}")


def trajectory_display_labels(mission, date=None):
    if mission_key(mission) == "GIRAFF":
        rocket = f"36{giraff_rocket_id_for_date(date)}"
        return {"main": f"{rocket} Main", "main_tag": rocket}
    return {"left": "36.397", "right": "36.398", "left_tag": "397", "right_tag": "398"}


def trajectory_configs(mission, date=None):
    paths = mission_trajectory_paths(mission, date=date)
    labels = trajectory_display_labels(mission, date=date)
    if mission_key(mission) == "GIRAFF":
        return [TrajectoryConfig("main", labels["main_tag"], labels["main"], paths["main"])]
    return [
        TrajectoryConfig("left", labels["left_tag"], labels["left"], paths["left"]),
        TrajectoryConfig("right", labels["right_tag"], labels["right"], paths["right"]),
    ]


def trajectory_config_tuples(mission, date=None):
    return [(cfg.key, cfg.tag, cfg.label, cfg.path) for cfg in trajectory_configs(mission, date=date)]
