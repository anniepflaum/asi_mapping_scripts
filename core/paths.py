from pathlib import Path


CORE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CORE_DIR.parent

TRAJECTORIES_DIR = PROJECT_DIR / "trajectories"
COAST_DIR = PROJECT_DIR / "coast"

LEFT_TRAJECTORY_PATH = TRAJECTORIES_DIR / "36397_GPS_Time_Export_01.csv"
RIGHT_TRAJECTORY_PATH = TRAJECTORIES_DIR / "36398_GPS_Time_Export_00.csv"
COAST_LON_PATH = COAST_DIR / "coastlon.txt"
COAST_LAT_PATH = COAST_DIR / "coastlat.txt"
