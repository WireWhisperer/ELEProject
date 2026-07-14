# Rotated Square Measurement (发挥部分4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `rotated_square_measure_pi.py` — one-button measurement of a horizontally-rotated square (30°~60°) placed on the A4 plane, reusing existing A4 detection and calibration functions.

**Architecture:** Single-file script. Imports A4 pipeline functions from `detect_squares_pi_single_frame_v7.py`. Uses `cv2.minAreaRect()` on the largest dark contour in the warped inner region to measure the foreshortened square. True side x = max(w,h)/SCALE; θ = arccos(min(w,h)/max(w,h)).

**Tech Stack:** Python 3, OpenCV, NumPy, picamera2, pyserial

## Global Constraints

- Scale: 20 px/cm for perspective warp (same as detect_squares)
- Serial: `/dev/ttyAMA0`, 115200 baud
- Camera: 1280×720 RGB888 via picamera2
- Output format: `D={distance:.1f},x={side:.2f},type=square\r\n`
- θ displayed on-screen only, not in serial output
- θ validation range: [25°, 65°]
- Debug images saved to `images/` directory

---

### Task 1: Create rotated_square_measure_pi.py — complete implementation

**Files:**
- Create: `2025C_basic/rotated_square_measure_pi.py`

**Interfaces:**
- Consumes: `find_a4_outer_quad`, `order_points`, `estimate_distance`, `get_outer_frame_params`, `correct_square_size`, `get_final_square_size` from `detect_squares_pi_single_frame_v7`
- Produces: standalone script, run via `python3 rotated_square_measure_pi.py`

- [ ] **Step 1: Write the complete file**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025 电赛 C 题 发挥部分(4) — 旋转正方形测量。

目标物摆在轴线上，水平旋转使物面与轴线成30°~60°夹角。
一键启动，测量并显示正方形边长 x。

核心思路：在 A4 透视校正图中，水平旋转的正方形呈现为矩形。
  - 高度（垂直）= 真实边长 x（未被压缩）
  - 宽度（水平）= x · cos(θ)（被压缩）
  - 恢复：x = max(w,h)/SCALE，θ = arccos(min(w,h)/max(w,h))

复用 detect_squares_pi_single_frame_v7.py 的 A4 检测与标定函数。
"""

import os
import time
import serial
import cv2
import numpy as np
from picamera2 import Picamera2

from detect_squares_pi_single_frame_v7 import (
    find_a4_outer_quad,
    order_points,
    estimate_distance,
    get_outer_frame_params,
    correct_square_size,
    get_final_square_size,
)

IMAGES_DIR = 'images'
SCALE = 20.0  # px/cm for perspective warp
A4_W_CM, A4_H_CM = 21.0, 29.7
BORDER_CM = 2.0

# 旋转正方形边长应在 6~12 cm 范围内（与题目一致）
MIN_SIDE_CM, MAX_SIDE_CM = 5.5, 12.5
MIN_SIDE_PX = MIN_SIDE_CM * SCALE
MAX_SIDE_PX = MAX_SIDE_CM * SCALE

# θ 验证范围（略宽于30°~60°，允许测量容差）
THETA_MIN_DEG, THETA_MAX_DEG = 25.0, 65.0


def detect_rotated_square(inner_binary):
    """
    在透视校正后的 inner 二值图中检测唯一旋转正方形。

    参数:
        inner_binary: 二值图 (黑色=255, 白色=0)，已裁剪 2cm 边框

    返回:
        dict: {
            'width': float,       # minAreaRect 短边 (px)
            'height': float,      # minAreaRect 长边 (px) = 真实正方形边长
            'box': np.ndarray,    # 旋转矩形的4个角点 (4,2)
            'cos_theta': float,   # min(w,h)/max(w,h) = cos(θ)
            'theta_deg': float,   # 估算旋转角（度）
            'theta_valid': bool,  # θ 是否在 25°~65° 范围内
            'center': tuple,      # 矩形中心 (cx, cy)
        }
        若未找到合格正方形，返回 None
    """
    # 形态学清理小断裂
    clean = cv2.morphologyEx(inner_binary, cv2.MORPH_CLOSE,
                             np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 过滤面积过小的轮廓，取面积最大的作为目标
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.4 * MIN_SIDE_PX * MIN_SIDE_PX:
            continue
        candidates.append((area, cnt))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    largest_cnt = candidates[0][1]

    # minAreaRect 获取旋转矩形
    rect = cv2.minAreaRect(largest_cnt)
    box = cv2.boxPoints(rect)  # (4,2) float32
    rw, rh = rect[1]  # width, height (OpenCV 约定 w< h? 不一定，直接取)

    if min(rw, rh) < MIN_SIDE_PX or max(rw, rh) > MAX_SIDE_PX:
        return None

    # 长边 = 真实正方形边长（垂直方向未被压缩）
    true_side_px = max(rw, rh)
    compressed_px = min(rw, rh)

    cos_theta = compressed_px / max(true_side_px, 1.0)
    cos_theta = float(np.clip(cos_theta, 0.30, 0.98))
    theta_deg = float(np.degrees(np.arccos(cos_theta)))

    theta_valid = THETA_MIN_DEG <= theta_deg <= THETA_MAX_DEG
    center = tuple(rect[0])

    return {
        'width': float(compressed_px),
        'height': float(true_side_px),
        'box': box,
        'cos_theta': cos_theta,
        'theta_deg': theta_deg,
        'theta_valid': theta_valid,
        'center': center,
    }


def draw_rotated_box_on_original(img, box_points, border_pix, M_inv,
                                 color, thickness=3):
    """将 inner 坐标系的旋转矩形框绘制回原始图像。"""
    pts = box_points.copy().astype(np.float32)
    pts[:, 0] += border_pix
    pts[:, 1] += border_pix
    orig = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), M_inv)
    orig = np.round(orig.reshape(-1, 2)).astype(np.int32)
    cv2.polylines(img, [orig], True, color, thickness)
    return orig


def process_frame(frame):
    """
    处理单帧：A4检测 → 透视校正 → 旋转正方形检测 → 计算x和θ。

    返回:
        img:       带标注的结果图像
        result:    dict 或 None
    """
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    outer_binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)

    # --- A4 外框检测 ---
    _, approx = find_a4_outer_quad(outer_binary)
    if approx is None:
        cv2.putText(img, 'ERROR: A4 frame not found', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img, None

    cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

    # --- 距离估算 ---
    left_pix, right_pix = get_outer_frame_params(approx)
    outer_height = (left_pix + right_pix) / 2.0
    distance_cm = float(estimate_distance(outer_height)) if outer_height > 0 else 0.0
    cv2.putText(img, f'Dist: {distance_cm:.1f}cm', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # --- 透视变换 ---
    src_pts = order_points(approx.reshape(4, 2).astype(np.float32))
    dw = int(A4_W_CM * SCALE)
    dh = int(A4_H_CM * SCALE)
    dst_pts = np.float32([[0, 0], [dw, 0], [dw, dh], [0, dh]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = np.linalg.inv(M)

    warped = cv2.warpPerspective(gray, M, (dw, dh))
    warped_blur = cv2.GaussianBlur(warped, (3, 3), 0)

    # --- 裁剪内部区域 + Otsu ---
    border_px = int(BORDER_CM * SCALE)
    inner_gray = warped_blur[border_px:dh - border_px,
                             border_px:dw - border_px]
    otsu_value, inner_binary = cv2.threshold(
        inner_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    inner_binary = cv2.morphologyEx(inner_binary, cv2.MORPH_CLOSE,
                                    np.ones((3, 3), np.uint8))

    # --- 旋转正方形检测 ---
    square = detect_rotated_square(inner_binary)
    if square is None:
        cv2.putText(img, 'ERROR: rotated square not found', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img, None

    # --- 计算尺寸 ---
    raw_cm = square['height'] / SCALE
    side_cm = get_final_square_size(raw_cm)
    theta_deg = square['theta_deg']
    theta_valid = square['theta_valid']

    # --- 绘制旋转矩形框 ---
    color = (0, 0, 255) if theta_valid else (0, 165, 255)  # 橙色警告
    draw_rotated_box_on_original(img, square['box'], border_px,
                                 M_inv, color, 3)

    # --- 文字叠加 ---
    cv2.putText(img, f'x={side_cm:.2f}cm', (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

    theta_text = f'theta={theta_deg:.1f}deg'
    if not theta_valid:
        theta_text += ' (WARN: out of 30-60deg)'
    cv2.putText(img, theta_text, (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255) if theta_valid else (0, 165, 255), 2)

    print(f'D={distance_cm:.1f} cm, x_raw={raw_cm:.3f}cm, x={side_cm:.3f}cm, '
          f'theta={theta_deg:.1f}deg, valid={theta_valid}, '
          f'side_px=({square["width"]:.1f}, {square["height"]:.1f})')

    result = {
        'distance_cm': distance_cm,
        'side_cm': side_cm,
        'raw_cm': raw_cm,
        'theta_deg': theta_deg,
        'theta_valid': theta_valid,
        'otsu': otsu_value,
    }
    return img, result


def main():
    """一键测量：初始化相机，拍摄并检测旋转正方形，串口输出结果。"""
    # 确保 images 目录存在
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # 初始化相机
    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={'size': (1280, 720), 'format': 'RGB888'})
    picam2.configure(config)
    picam2.start()
    print('Camera started; waiting for exposure/AWB...')
    time.sleep(2.0)

    # 打开串口
    ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.1)

    try:
        # 拍摄一帧
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(IMAGES_DIR, 'rotated_00_original.jpg'),
                    frame_bgr)
        print(f'Captured: {frame_bgr.shape[1]}x{frame_bgr.shape[0]}')

        # 处理
        t0 = time.perf_counter()
        result_img, result = process_frame(frame_bgr)
        elapsed = (time.perf_counter() - t0) * 1000.0

        cv2.imwrite(os.path.join(IMAGES_DIR, 'rotated_result.jpg'),
                    result_img)
        print(f'Processing time: {elapsed:.1f} ms')

        # 串口输出
        if result is not None:
            msg = (f"D={result['distance_cm']:.1f},"
                   f"x={result['side_cm']:.2f},type=square\r\n")
            ser.write(msg.encode('utf-8'))
            print(f'Sent via serial: {msg.strip()}')
        else:
            print('RESULT: measurement failed')
            msg = "D=0.0,x=0.00,type=square\r\n"
            ser.write(msg.encode('utf-8'))

        print('Waiting 1s before exit...')
        time.sleep(1.0)

    finally:
        ser.close()
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify the file exists and has correct imports**

Run: `python3 -c "import ast; ast.parse(open('2025C_basic/rotated_square_measure_pi.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 3: Verify imports resolve (check detect_squares module is importable)**

Run: `cd 2025C_basic && python3 -c "from detect_squares_pi_single_frame_v7 import find_a4_outer_quad, order_points, estimate_distance, get_outer_frame_params, correct_square_size, get_final_square_size; print('Imports OK')"`
Expected: `Imports OK`

- [ ] **Step 4: Commit**

```bash
git add 2025C_basic/rotated_square_measure_pi.py
git commit -m "feat: add rotated square measurement for 发挥部分(4)

- Uses minAreaRect to detect horizontally-rotated square in warped A4 view
- Recovers true square size x from uncompressed vertical dimension
- Estimates rotation angle theta from width/height ratio
- Reuses A4 detection and calibration from detect_squares_pi_single_frame_v7.py
- Serial output: D={distance},x={side},type=square

Co-Authored-By: Claude <noreply@anthropic.com>"
```
