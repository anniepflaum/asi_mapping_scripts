import re
from datetime import datetime, timedelta
from urllib.parse import urljoin, unquote
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import requests
import concurrent.futures


def closest_amisr_png_url(site: str, date: str, time: str) -> str:
    """
    Return the URL of the PNG image whose timestamp is closest to the requested
    (site, date, time), searching ONLY images within ±60 seconds of the requested HHMMSS.

    Args:
        site: "PKR", "VEE", "BVR", or "ARV"
        date: "YYYYMMDD"
        time: "HHMMSS"
    """
    site = site.upper().strip()
    if site not in {"PKR", "VEE", "BVR", "ARV"}:
        raise ValueError("site must be one of: PKR, VEE, BVR, ARV")

    date = re.sub(r"\D", "", date)
    time = re.sub(r"\D", "", time)
    if len(date) != 8 or len(time) != 6:
        raise ValueError("date must be YYYYMMDD and time must be HHMMSS (digits only)")

    target = datetime.strptime(date + time, "%Y%m%d%H%M%S")
    tol_sec = 15  # Only consider PNGs within ±15 seconds of the target time

    def _dt_from_filename(fname: str) -> datetime | None:
        # Extract first YYYYMMDD_HHMMSS occurrence anywhere in the filename
        m = re.search(r"(\d{8})_(\d{6})", fname)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            return None

    def _list_png_files(dir_url: str, target_dt: datetime, tol_seconds: int, site: str) -> list[str]:
        """
        List PNG filenames in a directory listing, but ONLY those whose embedded
        timestamp is within ±tol_seconds of target_dt.
        """
        r = requests.get(dir_url, timeout=20, verify=False)
        r.raise_for_status()
        html = r.text

        if site == "PKR":
            pat = r'href=["\'](PFRR_\d{8}_\d{6}_0558\.png)["\']'
        else:
            pat = rf'href=["\']({site}_558_\d{{8}}_\d{{6}}\.png)["\']'

        hrefs = re.findall(pat, html, flags=re.IGNORECASE)

        out: list[str] = []
        seen = set()
        for h in hrefs:
            h = unquote(h)
            fname = h.split("/")[-1]
            if not fname.lower().endswith(".png"):
                continue
            if fname in seen:
                continue

            fdt = _dt_from_filename(fname)
            if fdt is None:
                continue
            if abs((fdt - target_dt).total_seconds()) <= tol_seconds:
                seen.add(fname)
                out.append(fname)

        return out

    def _pick_best_from_dir_specs(dir_specs: list[str]) -> str | None:
        best = None  # (abs_seconds, url)
        found_exact = None
        def fetch_and_check(durl):
            try:
                files = _list_png_files(durl, target, tol_sec, site)
            except Exception:
                return []
            return [(f, durl) for f in files]

        # Fetch all directories in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(fetch_and_check, dir_specs))
        # Flatten results
        all_files = [(f, durl) for sub in results for (f, durl) in sub]
        for f, durl in all_files:
            fdt = _dt_from_filename(f)
            if fdt is None:
                continue
            diff = abs((fdt - target).total_seconds())
            if diff == 0:
                return urljoin(durl, f)  # Return immediately on exact match
            if best is None or diff < best[0]:
                best = (diff, urljoin(durl, f))
        return None if best is None else best[1]

    if site == "PKR":
        # Only check adjacent hourly directories if the second is within 8 seconds of the start/end of the hour
        sec = target.second
        minute = target.minute
        hour = target.hour
        dir_urls: list[str] = []
        check_prev = (minute == 0 and sec < 8)
        check_next = (minute == 59 and sec > 52)
        y = target.strftime("%Y")
        ymd = target.strftime("%Y%m%d")
        hh = target.strftime("%H")
        # Always check the requested hour
        dir_urls.append(f"https://optics.gi.alaska.edu/amisr_archive/PKR/DASC/PNG/{y}/{ymd}/{hh}/")
        if check_prev:
            dt_prev = target - timedelta(hours=1)
            y_prev = dt_prev.strftime("%Y")
            ymd_prev = dt_prev.strftime("%Y%m%d")
            hh_prev = dt_prev.strftime("%H")
            dir_urls.append(f"https://optics.gi.alaska.edu/amisr_archive/PKR/DASC/PNG/{y_prev}/{ymd_prev}/{hh_prev}/")
        if check_next:
            dt_next = target + timedelta(hours=1)
            y_next = dt_next.strftime("%Y")
            ymd_next = dt_next.strftime("%Y%m%d")
            hh_next = dt_next.strftime("%H")
            dir_urls.append(f"https://optics.gi.alaska.edu/amisr_archive/PKR/DASC/PNG/{y_next}/{ymd_next}/{hh_next}/")

        url = _pick_best_from_dir_specs(dir_urls)
        if url:
            return url
        raise FileNotFoundError("No PKR PNGs found within ±15 seconds in the checked hour directories.")

    else:
        # VEE/BVR/ARV: ONLY check the requested date (no adjacent days).
        ymd = target.strftime("%Y%m%d")
        durl = f"https://optics.gi.alaska.edu/amisr_archive/{site}/GASI_5577/png/{ymd}/"

        url = _pick_best_from_dir_specs([durl])
        if url:
            return url
        raise FileNotFoundError(f"No {site} PNGs found within ±15 seconds in the {ymd} daily directory.")

    
def main():
    import sys
    if len(sys.argv) != 4:
        print("Usage: python fetch_url.py <site> <date: YYYYMMDD> <time: HHMMSS>")
        sys.exit(1)
    site = sys.argv[1]
    date = sys.argv[2]
    time = sys.argv[3]
    try:
        url = closest_amisr_png_url(site, date, time)
        print(f"Closest PNG URL: {url}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    end = time.time()
    print(f"Runtime: {end - start:.2f} seconds")