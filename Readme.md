# Optical Fiber End-Face Defect Detection System

## Overview
Automated inspection system for fiber optic connectors built with Python and OpenCV.
Replicates IEC 61300-3-35 Nokia-grade quality control workflow.

## Output
![Result](fiber_inspection_result.png)

## Technologies Used
- Python
- OpenCV (CLAHE, Canny Edge Detection, Hough Transform)
- scikit-image (Blob Analysis, Region Properties)
- Matplotlib

## Features
- Detects Scratches, Dust, and Chips
- Segments fiber into 4 IEC-defined zones (Core, Cladding, Adhesive, Contact)
- Automated Pass/Fail classifier based on IEC 61300-3-35 standard

## How to Run
```bash
pip install opencv-python numpy matplotlib scikit-image
python fiber_inspect.py
```

## Result
- 315 defects detected on real fiber microscope image
- Zones: Core, Cladding, Adhesive, Contact
- IEC 61300-3-35 Pass/Fail classification
