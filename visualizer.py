# ============================================================
# visualizer.py
# Creates annotated defect map and 6-panel result display
# Replicates Nokia-grade QC output visualization
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_results(original, enhanced, zones, defects, result,
                      failures, zone_counts, cx, cy, outer_r):
    """
    Creates a professional 6-panel visualization:

    Panel 1: Original Image
    Panel 2: CLAHE Enhanced Image
    Panel 3: IEC Zone Segmentation (color coded)
    Panel 4: Defect Detection Map (hot map)
    Panel 5: Annotated Defect Map (with circles + labels)
    Panel 6: IEC 61300-3-35 Report

    Defect color coding:
        Red    = Scratch
        Yellow = Dust
        Magenta = Chip

    Zone color coding:
        Red    = Core
        Cyan   = Cladding
        Green  = Adhesive
        Yellow = Contact

    Parameters:
        original    : original grayscale image
        enhanced    : CLAHE processed image
        zones       : dict of zone masks
        defects     : list of defect dicts
        result      : "PASS ✅" or "FAIL ❌"
        failures    : list of failure reasons
        zone_counts : dict of defect counts per zone
        cx, cy      : fiber center coordinates
        outer_r     : fiber outer radius
    """

    # --- Build color zone map ---
    zone_img = np.zeros((*original.shape, 3), dtype=np.uint8)
    zone_colors = {
        'Core':     (255, 50,  50),
        'Cladding': (50,  200, 255),
        'Adhesive': (50,  255, 50),
        'Contact':  (200, 200, 50),
    }
    for name, mask in zones.items():
        zone_img[mask] = zone_colors[name]

    # --- Build annotated defect image ---
    annotated = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    defect_colors = {
        'Scratch': (0,   0,   255),
        'Dust':    (0,   255, 255),
        'Chip':    (255, 0,   255),
    }

    for d in defects:
        color = defect_colors.get(d['type'], (255, 255, 255))
        cv2.circle(annotated, (d['cy'], d['cx']), 8, color, 2)
        cv2.putText(annotated, d['type'][0],
                    (d['cy'] + 5, d['cx'] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw IEC zone boundary circles
    cv2.circle(annotated, (cx, cy), int(outer_r * 0.12), (255, 50,  50),  1)
    cv2.circle(annotated, (cx, cy), int(outer_r * 0.55), (50,  200, 255), 1)
    cv2.circle(annotated, (cx, cy), int(outer_r * 0.80), (50,  255, 50),  1)
    cv2.circle(annotated, (cx, cy), int(outer_r * 1.00), (200, 200, 50),  1)

    # --- Create 6-panel figure ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#1a1a2e')

    titles = [
        'Original Image', 'CLAHE Enhanced', 'Zone Segmentation',
        'Defect Detection', 'Annotated Map',  'IEC Report'
    ]

    for ax, title in zip(axes.flat, titles):
        ax.set_facecolor('#16213e')
        ax.set_title(title, color='white', fontsize=11, fontweight='bold')
        ax.axis('off')

    axes[0][0].imshow(original, cmap='gray')
    axes[0][1].imshow(enhanced, cmap='gray')
    axes[0][2].imshow(zone_img)

    # Defect heat map
    defect_binary = np.zeros_like(original)
    for d in defects:
        cv2.circle(defect_binary, (d['cy'], d['cx']), 5, 255, -1)
    axes[1][0].imshow(defect_binary, cmap='hot')

    axes[1][1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    # --- IEC Report Text Panel ---
    report_lines = [
        "  IEC 61300-3-35 REPORT",
        f"  {'=' * 28}",
        f"  Result: {result}",
        f"  Total Defects: {len(defects)}",
        "",
        "  Zone Breakdown:",
    ]
    for zone, count in zone_counts.items():
        report_lines.append(f"    {zone}: {count} defect(s)")

    report_lines.append("")
    if failures:
        report_lines.append("  Failure Reasons:")
        for f in failures:
            report_lines.append(f"    ⚠ {f}")
    else:
        report_lines.append("  No violations found.")

    report_lines += ["", "  Legend:", "    S=Scratch  D=Dust  C=Chip"]

    axes[1][2].text(
        0.05, 0.95, "\n".join(report_lines),
        transform=axes[1][2].transAxes,
        fontsize=9, verticalalignment='top',
        fontfamily='monospace',
        color='lime' if "PASS" in result else 'red',
        bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.8)
    )

    plt.suptitle(
        'Optical Fiber End-Face Defect Detection System',
        color='white', fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig('fiber_inspection_result.png', dpi=150,
                bbox_inches='tight', facecolor='#1a1a2e')
    plt.show()
    print("[INFO] Result saved as: fiber_inspection_result.png")
