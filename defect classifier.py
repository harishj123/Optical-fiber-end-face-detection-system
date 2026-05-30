# ============================================================
# defect_classifier.py
# Classifies each defect blob as Scratch, Dust, or Chip
# Same as MATLAB's regionprops() + blob analysis
# ============================================================

from skimage import measure


def analyze_defects(defect_mask, zones, image):
    """
    Labels each connected defect blob and classifies it.

    Classification Rules:
    ┌──────────┬────────────────────────────────────────┐
    │ Type     │ Rule                                   │
    ├──────────┼────────────────────────────────────────┤
    │ Scratch  │ Eccentricity > 0.92 (long and thin)   │
    │ Dust     │ Area < 50px (small and round)          │
    │ Chip     │ Large area, irregular shape            │
    └──────────┴────────────────────────────────────────┘

    Eccentricity:
        0.0 = perfect circle
        1.0 = straight line
    So high eccentricity = elongated = scratch.

    Parameters:
        defect_mask : binary mask of all defects
        zones       : dict of zone masks
        image       : enhanced grayscale image

    Returns:
        list of defect dicts with type, zone, area, eccentricity
    """
    labeled = measure.label(defect_mask)
    props = measure.regionprops(labeled, intensity_image=image)

    defects = []

    for prop in props:
        area  = prop.area
        ecc   = prop.eccentricity
        bbox  = prop.bbox
        cy_d, cx_d = prop.centroid

        # Skip tiny noise
        if area < 4:
            continue

        # Classify defect type
        if ecc > 0.92:
            defect_type = "Scratch"
        elif area < 50:
            defect_type = "Dust"
        else:
            defect_type = "Chip"

        # Find which zone this defect belongs to
        zone_name = "Outside"
        for name, mask in zones.items():
            if mask[int(cy_d), int(cx_d)]:
                zone_name = name
                break

        defects.append({
            'type':  defect_type,
            'zone':  zone_name,
            'area':  area,
            'ecc':   round(ecc, 3),
            'cx':    int(cx_d),
            'cy':    int(cy_d),
            'bbox':  bbox
        })

    print(f"[INFO] Total defects classified: {len(defects)}")
    return defects
