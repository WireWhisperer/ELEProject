# Rotated Square Measurement — 发挥部分(4)

**Date:** 2026-07-14
**Project:** 2025 电赛 C 题 — 基于单目视觉的目标物测量装置

## Problem

取出最后一个发挥目标物，摆在轴线上某一指定位置。水平转动目标物，使物面与轴线间成30°~60°的夹角θ。一键启动装置，测量并显示物面上正方形的x。

## Constraints (confirmed)

- One target (the rotated square), no other squares in scene
- Target is approximately in the same plane as the A4 reference paper, only rotated horizontally
- Reuse existing functions from `detect_squares_pi_single_frame_v7.py` as much as possible
- Serial output format: `D={distance:.1f},x={side:.2f},type=square\r\n` (consistent with existing)

## Geometry Model

When a square of true side length x is rotated horizontally by angle θ:
- In the warped (A4 plane corrected) image, the square appears as a rectangle
- Height (vertical) = x (preserved)
- Width (horizontal) = x · cos(θ) (foreshortened)
- Recovery: x = height_px / SCALE, θ = arccos(width_px / height_px)

## Pipeline

```
Capture (1280×720 RGB)
  → Grayscale → Gaussian blur → Adaptive threshold
  → find_a4_outer_quad()         ← imported from detect_squares
  → order_points() → perspective warp (SCALE=20 px/cm)
  → Crop 2cm border → Otsu threshold → inner_binary
  → cv2.findContours(RETR_EXTERNAL) → filter by area
  → Take largest contour → cv2.minAreaRect()
  → w, h = rect[1]; x_true = max(w,h)/SCALE
  → θ = arccos(min(w,h)/max(w,h))
  → correct_square_size() → get_final_square_size()
  → Draw overlays + serial output
```

## Validation Gate

θ must be in [25°, 65°] (wider than 30°~60° to allow tolerance). Outside range: still output result but show warning.

## New File

`2025C_basic/rotated_square_measure_pi.py`

## Functions

| Function | Source |
|----------|--------|
| `find_a4_outer_quad()` | imported from detect_squares |
| `order_points()` | imported from detect_squares |
| `estimate_distance()` | imported from detect_squares |
| `get_outer_frame_params()` | imported from detect_squares |
| `correct_square_size()` | imported from detect_squares |
| `get_final_square_size()` | imported from detect_squares |
| `detect_rotated_square(inner_binary)` | **new** — minAreaRect + validation |
| `process_frame(frame)` | **new** — orchestration |
| `main()` | **new** — camera, serial, entry point |

## Serial Output

```
D={distance:.1f},x={side:.2f},type=square\r\n
```

Theta displayed on-screen only, not in serial output.

## On-Screen Display

- Green: A4 outer frame quadrilateral
- Red: minAreaRect bounding box of rotated square
- Text: `Dist: XXcm`, `x=XXcm`, `θ=XX°`

## Integration

- Standalone run: `python3 rotated_square_measure_pi.py`
- Optional: add `'D'` command to `central_control.py`

## Dependencies

Only existing: `cv2`, `numpy`, `picamera2`, `serial`, `time`, `os`
