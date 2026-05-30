# ============================================================
# defect_detection.py
# Detects defects using Canny edge detection + thresholding
# Same as MATLAB's edge() + thresholding
# ============================================================

import cv2
import numpy as np
from skimage import morphology


def detect_defects(enhanced_image, zones):
    """
    Finds all defective pixels using two methods:

    Method 1 - Dark Pixel Thresholding:
        Pixels much darker than median = dust or chips

    Method 2 - Canny Edge Detection:
        Sharp edges inside fiber = scratches

    Both results are combined into one defect mask.

    Parameters:
        enhanced_image : CLAHE-processed grayscale image
        zones          : dict of zone masks from zone_segmentation.py

    Returns:
        all_defects  : combined binary defect mask
        edges        : edge-only defect mask
        dark_defects : dark pixel defect mask
    """

    # Build full fiber mask (ignore outside fiber area)
    full_mask = (
        zones['Core']     |
        zones['Cladding'] |
        zones['Adhesive'] |
        zones['Contact']
    )

    # --- Method 1: Dark defects (dust, chips) ---
    fiber_pixels = enhanced_image[full_mask]
    median_val = np.median(fiber_pixels)
    dark_defects = (enhanced_image < (median_val * 0.65)) & full_mask

    # --- Method 2: Canny edge detection (scratches) ---
    blurred = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 90)
    edges = edges & full_mask.astype(np.uint8) * 255
    edges = edges > 0

    # --- Combine both methods ---
    all_defects = dark_defects | edges

    # Remove tiny noise pixels (less than 3px area)
    all_defects = morphology.remove_small_objects(all_defects, min_size=3)

    total = int(all_defects.sum())
    print(f"[INFO] Defect pixels found: {total}")

    return all_defects, edges, dark_defects
