from pathlib import Path


CORE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CORE_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent

GNEISS_TRAJECTORIES_DIR = WORKSPACE_DIR / "trajectories" / "GNEISS"
GIRAFF_TRAJECTORIES_DIR = WORKSPACE_DIR / "trajectories" / "GIRAFF"
MAPPED_DIR = WORKSPACE_DIR / "mapped"
COAST_DIR = PROJECT_DIR / "coast"
RECEIVERS_PATH = WORKSPACE_DIR / "receivers.csv"
GIRAFF_RECEIVER_ACRONYMS = ("VEE", "TOO", "PKR")

GNEISS_LEFT_TRAJECTORY_PATH = GNEISS_TRAJECTORIES_DIR / "36397_GPS_Time_Export_01.csv"
GNEISS_RIGHT_TRAJECTORY_PATH = GNEISS_TRAJECTORIES_DIR / "36398_GPS_Time_Export_00.csv"
GIRAFF_381_TRAJECTORY_PATH = GIRAFF_TRAJECTORIES_DIR / "36381_MAIN_PAYLOAD_GPS.xlsx"
GIRAFF_380_TRAJECTORY_PATH = GIRAFF_TRAJECTORIES_DIR / "36380_MAIN_PAYLOAD_GPS.xlsx"
COAST_LON_PATH = COAST_DIR / "coastlon.txt"
COAST_LAT_PATH = COAST_DIR / "coastlat.txt"


MISSION_TRAJECTORY_PATHS = {
    "GNEISS": {
        "left": GNEISS_LEFT_TRAJECTORY_PATH,
        "right": GNEISS_RIGHT_TRAJECTORY_PATH,
    },
    "GIRAFF": {
        "main": GIRAFF_381_TRAJECTORY_PATH,
    },
}

GIRAFF_TRAJECTORY_PATHS_BY_DATE = {
    "20250202": {
        "main": GIRAFF_381_TRAJECTORY_PATH,
    },
    "20250209": {
        "main": GIRAFF_380_TRAJECTORY_PATH,
    },
}


def normalize_date_key(date=None):
    if date is None:
        return None
    return str(date).replace("-", "")


def giraff_rocket_id_for_date(date=None):
    date_key = normalize_date_key(date)
    if date_key == "20250209":
        return "380"
    return "381"


def mission_output_dir(mission, color="green", date=None):
    mission_key = str(mission).upper()
    base = MAPPED_DIR / str(color).lower()
    if mission_key == "GIRAFF":
        return base / "GIRAFF" / giraff_rocket_id_for_date(date)
    return base / mission_key


def mission_receiver_acronyms(mission):
    if str(mission).upper() == "GIRAFF":
        return GIRAFF_RECEIVER_ACRONYMS
    return None


def filter_receivers_for_mission(receivers, mission):
    acronyms = mission_receiver_acronyms(mission)
    if acronyms is None:
        return receivers
    receiver_map = {receiver.get("acronym", "").upper(): receiver for receiver in receivers}
    return [receiver_map[acronym] for acronym in acronyms if acronym in receiver_map]
