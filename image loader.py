# ============================================================
# image_loader.py
# Handles loading real or synthetic fiber optic images
# ============================================================

import cv2
import numpy as np


def create_synthetic_fiber_image():
    """
    Creates a fake fiber optic end-face image for testing.
    Use this if you don't have a real microscope image.
    """
    img = np.zeros((512, 512), dtype=np.uint8)
    img[:] = 30

    # Draw zones
    cv2.circle(img, (256, 256), 230, 80, -1)   # Contact zone
    cv2.circle(img, (256, 256), 180, 120, -1)  # Adhesive zone
    cv2.circle(img, (256, 256), 120, 170, -1)  # Cladding zone
    cv2.circle(img, (256, 256), 30, 220, -1)   # Core zone

    # Add fake scratches
    cv2.line(img, (150, 200), (280, 230), 40, 2)
    cv2.line(img, (200, 300), (350, 280), 35, 1)

    # Add fake dust particles
    cv2.circle(img, (180, 180), 6, 20, -1)
    cv2.circle(img, (310, 150), 4, 25, -1)
    cv2.circle(img, (260, 100), 5, 15, -1)

    # Add fake chip
    cv2.ellipse(img, (320, 310), (12, 7), 45, 0, 360, 10, -1)

    # Add noise
    noise = np.random.randint(0, 15, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    return img


def load_image(path):
    """
    Load a real grayscale image from disk.
    Falls back to synthetic image if file not found.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARNING] Could not load '{path}'. Using synthetic image instead.")
        return create_synthetic_fiber_image()
    img = cv2.resize(img, (512, 512))
    print(f"[INFO] Image loaded from: {path}")
    return img
