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
        print('  detect_rotated_square: no contours found in inner binary')
        return None

    # 过滤面积过小的轮廓，取面积最大的作为目标
    candidates = []
    min_area_threshold = 0.4 * MIN_SIDE_PX * MIN_SIDE_PX
    print(f'  detect_rotated_square: found {len(contours)} contours, '
          f'area threshold={min_area_threshold:.0f}px^2')
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 始终打印前5大轮廓的面积，方便诊断
        candidates.append((area, cnt))

    # 按面积排序，打印前5个
    candidates.sort(key=lambda x: x[0], reverse=True)
    for i, (area, _) in enumerate(candidates[:5]):
        side_if_square = np.sqrt(area)
        print(f'    contour[{i}]: area={area:.0f}px^2 '
              f'(~sqrt={side_if_square:.0f}px = {side_if_square/SCALE:.1f}cm)')

    # 过滤面积
    all_by_area = [(a, c) for a, c in candidates]
    candidates = [(a, c) for a, c in all_by_area if a >= min_area_threshold]

    if not candidates:
        largest_info = f'largest={all_by_area[0][0]:.0f}px^2' if all_by_area else 'none'
        print(f'  detect_rotated_square: no contour with area >= '
              f'{min_area_threshold:.0f}px^2 ({largest_info})')
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    largest_cnt = candidates[0][1]
    largest_area = candidates[0][0]

    # minAreaRect 获取旋转矩形
    rect = cv2.minAreaRect(largest_cnt)
    box = cv2.boxPoints(rect)  # (4,2) float32
    rw, rh = rect[1]  # width, height

    print(f'  detect_rotated_square: largest contour area={largest_area:.0f}px^2, '
          f'minAreaRect=({rw:.1f}, {rh:.1f})px = '
          f'({rw/SCALE:.2f}, {rh/SCALE:.2f})cm')

    # 长边 = 真实正方形边长（垂直方向未被压缩）
    true_side_px = max(rw, rh)
    compressed_px = min(rw, rh)

    # 验证：长边须在 6~12cm 范围；短边因旋转被压缩，
    # 最极端情况 (6cm 正方形 @ 65°) 仅约 46px，需放宽下限。
    if true_side_px < MIN_SIDE_PX or true_side_px > MAX_SIDE_PX:
        print(f'  detect_rotated_square: true_side {true_side_px:.1f}px '
              f'out of range [{MIN_SIDE_PX:.0f}, {MAX_SIDE_PX:.0f}]')
        return None
    min_compressed = MIN_SIDE_PX * np.cos(np.radians(THETA_MAX_DEG))
    if compressed_px < min_compressed:
        print(f'  detect_rotated_square: compressed {compressed_px:.1f}px '
              f'< min {min_compressed:.1f}px')
        return None

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
        img:        带标注的结果图像
        result:     dict 或 None
        error_msg:  失败原因字符串（成功时为 None）
    """
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 使用与 calibrate_stable_pi.py 一致的参数：更大窗口 + MORPH_CLOSE 连接断边
    outer_binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    outer_binary = cv2.morphologyEx(outer_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # --- 始终保存灰度图和外框二值图，方便诊断 ---
    cv2.imwrite(os.path.join(IMAGES_DIR, 'rotated_01_gray.jpg'), gray)
    cv2.imwrite(os.path.join(IMAGES_DIR, 'rotated_02_outer_binary.jpg'), outer_binary)
    print(f'Gray shape: {gray.shape}, outer_binary foreground: '
          f'{cv2.countNonZero(outer_binary)}px')

    # --- A4 外框检测 ---
    # 先做诊断：统计 outer_binary 中可能的四边形轮廓
    diag_contours, _ = cv2.findContours(outer_binary, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = outer_binary.shape[:2]
    img_area = float(h_img * w_img)
    print(f'  A4 diag: {len(diag_contours)} contours in outer_binary, '
          f'image={w_img}x{h_img}, area_threshold=[{0.025*img_area:.0f}, {0.9*img_area:.0f}]')

    quad_candidates = []
    for cnt in diag_contours:
        area = cv2.contourArea(cnt)
        if area < 0.025 * img_area or area > 0.90 * img_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        q = order_points(approx.reshape(4, 2).astype(np.float32))
        w = (np.linalg.norm(q[0]-q[1]) + np.linalg.norm(q[3]-q[2])) / 2.0
        h = (np.linalg.norm(q[0]-q[3]) + np.linalg.norm(q[1]-q[2])) / 2.0
        ratio = max(w, h) / max(min(w, h), 1)
        recty = area / max(w * h, 1)
        quad_candidates.append((area, ratio, recty))
    if quad_candidates:
        print(f'  A4 diag: {len(quad_candidates)} quads passed area+convex check: '
              f'{[f"a={a:.0f} r={r:.3f} ry={y:.3f}" for a, r, y in quad_candidates[:5]]}')
    else:
        print(f'  A4 diag: NO quads passed area+convex check')

    _, approx = find_a4_outer_quad(outer_binary)

    # 回退方案1：放宽长宽比限制，直接用最大合格四边形
    if approx is None:
        print('  A4 fallback1: trying relaxed aspect ratio...')
        # 重复 find_a4_outer_quad 逻辑但放宽 ratio 到 [1.10, 2.20]
        contours_fb, _ = cv2.findContours(outer_binary, cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_SIMPLE)
        fb_candidates = []
        for cnt in contours_fb:
            area = cv2.contourArea(cnt)
            if area < 0.025 * img_area or area > 0.90 * img_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx_fb = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx_fb) != 4 or not cv2.isContourConvex(approx_fb):
                continue
            q = order_points(approx_fb.reshape(4, 2).astype(np.float32))
            w = (np.linalg.norm(q[0]-q[1]) + np.linalg.norm(q[3]-q[2])) / 2.0
            h = (np.linalg.norm(q[0]-q[3]) + np.linalg.norm(q[1]-q[2])) / 2.0
            ratio = max(w, h) / max(min(w, h), 1)
            recty = area / max(w * h, 1)
            if 1.10 <= ratio <= 2.20 and recty >= 0.60:
                fb_candidates.append((area, approx_fb, ratio, recty))
        if fb_candidates:
            fb_candidates.sort(key=lambda x: x[0], reverse=True)
            best = fb_candidates[0]
            approx = best[1]
            print(f'  A4 fallback1 OK: area={best[0]:.0f} ratio={best[2]:.3f} recty={best[3]:.3f}')

    # 回退方案2：连四边形都找不到，直接取最大外部轮廓
    if approx is None:
        print('  A4 fallback2: trying largest external contour...')
        contours_ext, _ = cv2.findContours(outer_binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
        if contours_ext:
            largest = max(contours_ext, key=cv2.contourArea)
            peri = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
            if len(approx) == 4:
                print(f'  A4 fallback2 OK: area={cv2.contourArea(largest):.0f}')
            else:
                # 即使是>4边形也强制取凸包四边形
                hull = cv2.convexHull(largest)
                peri_h = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.02 * peri_h, True)
                if len(approx) != 4:
                    approx = None
                    print(f'  A4 fallback2: convex hull has {len(approx) if approx is not None else 0} pts')

    if approx is None:
        cv2.putText(img, 'ERROR: A4 frame not found', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img, None, 'A4外框未检测到，请检查A4纸是否在画面中、光照是否充足'

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

    # 保存中间调试图片
    cv2.imwrite(os.path.join(IMAGES_DIR, 'rotated_01_warped.jpg'), warped)
    cv2.imwrite(os.path.join(IMAGES_DIR, 'rotated_02_inner_gray.jpg'), inner_gray)
    cv2.imwrite(os.path.join(IMAGES_DIR, 'rotated_03_inner_binary.jpg'), inner_binary)
    print(f'Otsu threshold: {otsu_value:.1f}, inner_binary shape: {inner_binary.shape}')
    print(f'Binary stats: foreground={cv2.countNonZero(inner_binary)}px, '
          f'ratio={cv2.countNonZero(inner_binary)/inner_binary.size:.4f}')

    # --- 旋转正方形检测 ---
    square = detect_rotated_square(inner_binary)
    if square is None:
        cv2.putText(img, 'ERROR: rotated square not found', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img, None, ('旋转正方形未检测到，可能原因：\n'
                           '  1) 目标物未摆入A4纸面内\n'
                           '  2) 旋转角度超出范围(需30~60度)\n'
                           '  3) 正方形边长不在6~12cm范围\n'
                           '  4) 二值化阈值不理想(查看images/debug图片)')

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
    return img, result, None


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
        result_img, result, error_msg = process_frame(frame_bgr)
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
            print(f'\n===== 测量失败 =====')
            print(f'原因: {error_msg}')
            print(f'====================\n')
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
