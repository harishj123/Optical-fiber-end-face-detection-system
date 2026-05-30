# ============================================================
# main.py
# Optical Fiber End-Face Defect Detection System
# IEC 61300-3-35 Compliance Checker
#
# How to run:
#   python main.py
#
# Requirements:
#   pip install opencv-python numpy matplotlib scikit-image
#
# Place your fiber image as fiber.png in the same folder.
# ============================================================

from image_loader      import load_image
from preprocessing     import apply_clahe
from zone_segmentation import find_fiber_center, create_zone_masks
from defect_detection  import detect_defects
from defect_classifier import analyze_defects
from iec_classifier    import iec_pass_fail
from visualizer        import visualize_results


def main():
    print("=" * 50)
    print("  Fiber Optic End-Face Inspection System")
    print("  IEC 61300-3-35 Compliance Check")
    print("=" * 50)

    # Step 1: Load image
    image = load_image("fiber.png")
    print("[1/6] Image loaded ✓")

    # Step 2: Enhance contrast using CLAHE
    enhanced = apply_clahe(image)
    print("[2/6] CLAHE enhancement done ✓")

    # Step 3: Find fiber center using Hough Circle Transform
    cx, cy, outer_r = find_fiber_center(enhanced)
    print(f"[3/6] Fiber center found at ({cx}, {cy}), radius={outer_r} ✓")

    # Step 4: Segment into IEC zones
    zones = create_zone_masks(image.shape, cx, cy, outer_r)
    print("[4/6] IEC zones segmented ✓")

    # Step 5: Detect and classify defects
    defect_mask, edges, dark = detect_defects(enhanced, zones)
    defects = analyze_defects(defect_mask, zones, enhanced)
    print(f"[5/6] {len(defects)} defect(s) detected ✓")

    # Step 6: IEC Pass/Fail classification
    result, failures, zone_counts = iec_pass_fail(defects)
    print(f"[6/6] IEC Classification: {result}")

    # Print defect details
    print("\n--- DEFECT DETAILS ---")
    for i, d in enumerate(defects, 1):
        print(f"  {i}. {d['type']:7s} | Zone: {d['zone']:9s} | "
              f"Area: {d['area']:4d}px | Ecc: {d['ecc']}")

    if failures:
        print("\n--- FAILURE REASONS ---")
        for f in failures:
            print(f"  ⚠ {f}")

    # Show and save visual results
    visualize_results(
        image, enhanced, zones, defects,
        result, failures, zone_counts,
        cx, cy, outer_r
    )


if __name__ == "__main__":
    main()
