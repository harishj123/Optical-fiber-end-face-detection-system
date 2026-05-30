# ============================================================
# preprocessing.py
# Image enhancement using CLAHE
# Same as MATLAB's adapthisteq()
# ============================================================

import cv2


def apply_clahe(image, clip_limit=2.0, tile_size=(8, 8)):
    """
    CLAHE = Contrast Limited Adaptive Histogram Equalization.

    Why we use it:
    - Fiber images are often low contrast
    - CLAHE makes defects (scratches, dust) much more visible
    - Works tile-by-tile so local contrast is improved uniformly

    Parameters:
        image      : grayscale input image
        clip_limit : higher = more contrast (default 2.0)
        tile_size  : grid size for local histogram (default 8x8)

    Returns:
        enhanced grayscale image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(image)
    return enhanced
