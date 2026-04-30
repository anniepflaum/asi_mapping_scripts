"""Mission-specific configuration and small policy helpers."""

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
GIRAFF_SITE_ACRONYMS = ("VEE",)
GNEISS_TIFF_SITE_ACRONYMS = ("ARV", "VEE", "BVR")
GNEISS_ALL_SITE_ACRONYMS = ("ARV", "BVR", "VEE", "PKR")


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


def default_sites(mission, include_pkr=False):
    if mission_key(mission) == "GIRAFF":
        return list(GIRAFF_SITE_ACRONYMS)
    return list(GNEISS_ALL_SITE_ACRONYMS if include_pkr else GNEISS_TIFF_SITE_ACRONYMS)


def validate_color_and_sites(parser, mission, color, sites, giraff_message_site="VEE"):
    """Apply mission-specific constraints to color/site arguments."""
    if mission_key(mission) != "GIRAFF":
        return
    if str(color).lower() != "green":
        parser.error("--mission GIRAFF only supports --color green")
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
