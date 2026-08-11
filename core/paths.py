import os
from pathlib import Path


CORE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CORE_DIR.parent
WORKSPACE_DIR = Path(
    os.environ.get("ASI_WORKSPACE_ROOT", PROJECT_DIR.parent)
).expanduser()
LAB317_ROOT = Path(
    os.environ.get("LAB317_ROOT", PROJECT_DIR.parents[1])
).expanduser()
DATA_ROOT = Path(
    os.environ.get("LAB317_DATA_ROOT", LAB317_ROOT / "data")
).expanduser()
OUTPUT_ROOT = Path(
    os.environ.get("LAB317_OUTPUT_ROOT", LAB317_ROOT / "outputs")
).expanduser()
IMAGE_DIR = Path(
    os.environ.get("ASI_IMAGE_ROOT", DATA_ROOT / "raw" / "asi" / "images")
).expanduser()
if output_root := os.environ.get("ASI_OUTPUT_ROOT"):
    MAPPED_DIR = Path(output_root).expanduser()
else:
    MAPPED_DIR = OUTPUT_ROOT / "asi-mapping" / "mapped"

TRAJECTORY_DIR = Path(
    os.environ.get("ASI_TRAJECTORY_ROOT", DATA_ROOT / "raw" / "rocket" / "trajectories")
).expanduser()
GNEISS_TRAJECTORIES_DIR = TRAJECTORY_DIR / "GNEISS"
GIRAFF_TRAJECTORIES_DIR = TRAJECTORY_DIR / "GIRAFF"
COAST_DIR = PROJECT_DIR / "coast"
RECEIVERS_PATH = Path(
    os.environ.get("ASI_RECEIVERS_PATH", DATA_ROOT / "reference" / "asi-mapping" / "receivers.csv")
).expanduser()
STARMAP_DIR = Path(
    os.environ.get("ASI_STARMAP_ROOT", DATA_ROOT / "reference" / "starmaps")
).expanduser()
LEGACY_STARMAPS_DIR = STARMAP_DIR

GNEISS_LEFT_TRAJECTORY_PATH = GNEISS_TRAJECTORIES_DIR / "36397_GPS_Time_Export_01.csv"
GNEISS_RIGHT_TRAJECTORY_PATH = GNEISS_TRAJECTORIES_DIR / "36398_GPS_Time_Export_00.csv"
EZIE_MEM_EPHEMERIS_PATH = GNEISS_TRAJECTORIES_DIR / "all_spacecraft_20260210_mem_ephemeris.csv"
GIRAFF_381_TRAJECTORY_PATH = GIRAFF_TRAJECTORIES_DIR / "36381_MAIN_PAYLOAD_GPS.xlsx"
GIRAFF_380_TRAJECTORY_PATH = GIRAFF_TRAJECTORIES_DIR / "36380_MAIN_PAYLOAD_GPS.xlsx"
COAST_LON_PATH = COAST_DIR / "coastlon.txt"
COAST_LAT_PATH = COAST_DIR / "coastlat.txt"


def starmap_path(color, *parts):
    color_key = str(color).lower()
    relative_path = Path(*parts)
    path = STARMAP_DIR / color_key / relative_path
    if path.exists():
        return path
    legacy_path = LEGACY_STARMAPS_DIR / color_key / relative_path
    if legacy_path.exists():
        return legacy_path
    if color_key != "green":
        green_path = STARMAP_DIR / "green" / relative_path
        if green_path.exists():
            return green_path
        legacy_green_path = LEGACY_STARMAPS_DIR / "green" / relative_path
        if legacy_green_path.exists():
            return legacy_green_path
    return path
