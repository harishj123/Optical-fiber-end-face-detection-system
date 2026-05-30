# ============================================================
# zone_segmentation.py
# Detects fiber center and segments into IEC 61300-3-35 zones
# Same as MATLAB's imfindcircles() + manual masking
# ============================================================

import cv2
import numpy as np


def find_fiber_center(image):
    """
    Finds the circular fiber end-face using Hough Circle Transform.

    How it works:
    - Blurs the image first to reduce noise
    - Searches for circles using gradient-based Hough method
    - Returns the largest circle found (= full fiber boundary)

    Returns:
        cx, cy  : center coordinates
        r       : radius of fiber
    """
    blurred = cv2.GaussianBlur(image, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=80,
        maxRadius=250
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        largest = max(circles, key=lambda c: c[2])
        cx, cy, r = largest
        print(f"[INFO] Fiber center: ({cx}, {cy}), radius: {r}px")
        return cx, cy, r
    else:
        print("[WARNING] No circle found. Using image center as default.")
        h, w = image.shape
        return w // 2, h // 2, min(h, w) // 2 - 20


def create_zone_masks(image_shape, cx, cy, outer_radius):
    """
    Creates 4 binary masks for IEC 61300-3-35 zones.

    IEC Zone Definitions:
    ┌─────────────┬──────────────────────────────┐
    │ Zone        │ Radius Range                 │
    ├─────────────┼──────────────────────────────┤
    │ Core        │ 0%   to 12%  of outer radius │
    │ Cladding    │ 12%  to 55%  of outer radius │
    │ Adhesive    │ 55%  to 80%  of outer radius │
    │ Contact     │ 80%  to 100% of outer radius │
    └─────────────┴──────────────────────────────┘

    Returns:
        dict of boolean masks for each zone
    """
    h, w = image_shape
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

    zones = {
        'Core':     dist <= outer_radius * 0.12,
        'Cladding': (dist > outer_radius * 0.12) & (dist <= outer_radius * 0.55),
        'Adhesive': (dist > outer_radius * 0.55) & (dist <= outer_radius * 0.80),
        'Contact':  (dist > outer_radius * 0.80) & (dist <= outer_radius * 1.00),
    }

    print("[INFO] IEC zones segmented: Core, Cladding, Adhesive, Contact")
    return zones
