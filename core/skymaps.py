from core import generate_skymap as skymap


def load_skymaps(selected_sites=None, color="green", mission="GNEISS"):
    """
    Load latitude/longitude mapping arrays for each site using the skymap module.
    """
    skymaps = {}
    if selected_sites is None:
        selected_sites = {"ARV", "PKR", "VEE", "BVR"}
    if "ARV" in selected_sites:
        lat, lon, az, el, mask = skymap.load_ARV(color=color)
        skymaps["ARV"] = {"site_lat": lat, "site_lon": lon, "azmt": az, "elev": el, "mask": mask}
    if "VEE" in selected_sites:
        vee_payload = skymap.load_VEE(mission=mission)
        if len(vee_payload) == 7:
            lat, lon, az, el, mask, mapped_lat, mapped_lon = vee_payload
            skymaps["VEE"] = {
                "site_lat": lat,
                "site_lon": lon,
                "azmt": az,
                "elev": el,
                "mask": mask,
                "lat": mapped_lat,
                "lon": mapped_lon,
            }
        else:
            lat, lon, az, el, mask = vee_payload
            skymaps["VEE"] = {"site_lat": lat, "site_lon": lon, "azmt": az, "elev": el, "mask": mask}
    if "BVR" in selected_sites:
        lat, lon, az, el, mask = skymap.load_BVR()
        skymaps["BVR"] = {"site_lat": lat, "site_lon": lon, "azmt": az, "elev": el, "mask": mask}
    if "PKR" in selected_sites:
        lat, lon, az, el, mask = skymap.load_PKR()
        skymaps["PKR"] = {"site_lat": lat, "site_lon": lon, "azmt": az, "elev": el, "mask": mask}
    map_alt_km = 200.0 if str(color).lower() == "red" else 110.0
    for sm in skymaps.values():
        if "lat" in sm and "lon" in sm:
            continue
        lat, lon = skymap.azel2geo(sm["site_lat"], sm["site_lon"], sm["azmt"], sm["elev"], alt=map_alt_km)
        sm["lat"] = lat
        sm["lon"] = lon
    return skymaps
