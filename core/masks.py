from core import check_intersect as ci


def build_overlap_masks(skymaps):
    """
    For each site, compute masks that remove the pixels overlapping every other site.
    The masks are stored in-place under ``skymaps[site]["extra_masks"]``.
    """
    sites = list(skymaps.keys())
    for site in sites:
        other_sites = [other_site for other_site in sites if other_site != site]
        skymaps[site]["extra_masks"] = {}
        for other_site in other_sites:
            site_mask, _other_mask = ci.calculate_masks(
                skymaps[site]["site_lat"],
                skymaps[site]["site_lon"],
                skymaps[site]["azmt"],
                skymaps[site]["elev"],
                skymaps[other_site]["site_lat"],
                skymaps[other_site]["site_lon"],
                skymaps[other_site]["azmt"],
                skymaps[other_site]["elev"],
            )
            skymaps[site]["extra_masks"][other_site] = site_mask
