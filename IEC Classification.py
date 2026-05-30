# ============================================================
# iec_classifier.py
# IEC 61300-3-35 Pass/Fail classifier
# Replicates Nokia-grade optical transceiver QC logic
# ============================================================

from collections import defaultdict


# ============================================================
# IEC 61300-3-35 Standard Limits
# ============================================================
#
# These are the maximum allowed defects per zone:
#
# Zone      | Max Defects | Max Scratch Area
# ----------|-------------|------------------
# Core      |      0      |        0
# Cladding  |      3      |       80 px
# Adhesive  |     10      |      200 px
# Contact   |     20      |      500 px
#
# Any violation = FAIL
# ============================================================

IEC_LIMITS = {
    'Core':     {'max_defects': 0,  'max_scratch_area': 0},
    'Cladding': {'max_defects': 3,  'max_scratch_area': 80},
    'Adhesive': {'max_defects': 10, 'max_scratch_area': 200},
    'Contact':  {'max_defects': 20, 'max_scratch_area': 500},
}


def iec_pass_fail(defects):
    """
    Checks all defects against IEC 61300-3-35 limits.

    Parameters:
        defects : list of defect dicts from defect_classifier.py

    Returns:
        overall       : "PASS ✅" or "FAIL ❌"
        failures      : list of failure reason strings
        zone_counts   : dict of defect counts per zone
    """
    zone_counts = defaultdict(int)
    zone_scratch_area = defaultdict(int)
    failures = []

    # Count defects per zone
    for d in defects:
        zone = d['zone']
        zone_counts[zone] += 1
        if d['type'] == 'Scratch':
            zone_scratch_area[zone] += d['area']

    overall = "PASS ✅"

    # Check against IEC limits
    for zone, limits in IEC_LIMITS.items():
        count        = zone_counts[zone]
        scratch_area = zone_scratch_area[zone]

        if count > limits['max_defects']:
            failures.append(
                f"FAIL: {count} defects in {zone} zone (limit: {limits['max_defects']})"
            )
            overall = "FAIL ❌"

        if scratch_area > limits['max_scratch_area']:
            failures.append(
                f"FAIL: Scratch area {scratch_area}px in {zone} (limit: {limits['max_scratch_area']}px)"
            )
            overall = "FAIL ❌"

    print(f"[INFO] IEC 61300-3-35 Result: {overall}")
    return overall, failures, dict(zone_counts)
