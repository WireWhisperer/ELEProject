#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025 电赛 C 题发挥部分（3）：按输入编号测量对应正方形。
依赖：detect_squares_pi_single_frame_v6.py 与本文件放在同一目录。
运行：python3 develop_3_numbered_square.py
可选：python3 develop_3_numbered_square.py --digit 7 --no-gui
"""
import argparse
import time
import cv2
import numpy as np
from picamera2 import Picamera2

# 直接复用发挥 1~2 问已经封装、标定好的函数。
from detect_squares_pi_single_frame_v6 import (
    SCALE, estimate_distance, get_outer_frame_params, order_points,
    find_a4_outer_quad, detect_squares_robust, correct_square_size,
    draw_square_on_original
)

A4_W_CM = 21.0
A4_H_CM = 29.7
BORDER_CM = 2.0
OCR_SIZE = 96


def make_hog():
    return cv2.HOGDescriptor(
        (48, 64), (16, 16), (8, 8), (8, 8), 9,
        1, -1, 0, 0.2, False, 64, True)


def normalize_digit(mask):
    """把白字二值图归一化到 48x64，保持字形比例。"""
    mask = (mask > 0).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    # 保留主要连通域；兼容阈值化造成的笔画小断裂。
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = float(areas.max())
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= max(5.0, 0.035 * largest):
            keep[labels == i] = 255
    pts = cv2.findNonZero(keep)
    if pts is None:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    roi = keep[y:y+h, x:x+w]
    canvas = np.zeros((64, 48), np.uint8)
    scale = min(38.0 / max(w, 1), 54.0 / max(h, 1))
    nw, nh = max(1, int(round(w*scale))), max(1, int(round(h*scale)))
    roi = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
    ox, oy = (48-nw)//2, (64-nh)//2
    canvas[oy:oy+nh, ox:ox+nw] = roi
    return canvas


def build_digit_classifier():
    """用 OpenCV 自带字体合成训练集，无需联网、Tesseract 或模型文件。"""
    hog = make_hog()
    samples, labels = [], []
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
             cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX,
             cv2.FONT_HERSHEY_PLAIN]
    rng = np.random.default_rng(2025)
    for digit in range(10):
        text = str(digit)
        for font in fonts:
            for thickness in (2, 3, 4, 5):
                for _ in range(10):
                    image = np.zeros((110, 90), np.uint8)
                    fs = float(rng.uniform(1.7, 2.5))
                    (tw, th), base = cv2.getTextSize(text, font, fs, thickness)
                    x = int((90-tw)/2 + rng.integers(-4, 5))
                    y = int((110+th)/2 + rng.integers(-4, 5))
                    cv2.putText(image, text, (x, y), font, fs, 255,
                                thickness, cv2.LINE_AA)
                    angle = float(rng.uniform(-4.0, 4.0))
                    M = cv2.getRotationMatrix2D((45, 55), angle, 1.0)
                    image = cv2.warpAffine(image, M, (90, 110))
                    _, image = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
                    norm = normalize_digit(image)
                    samples.append(hog.compute(norm).reshape(-1))
                    labels.append(digit)
    knn = cv2.ml.KNearest_create()
    knn.train(np.asarray(samples, np.float32), cv2.ml.ROW_SAMPLE,
              np.asarray(labels, np.float32))
    return hog, knn


def rectify_square_gray(inner_gray, square, size=OCR_SIZE):
    """将某个检测到的正方形单独拉正，供白色编号识别。"""
    src = order_points(np.asarray(square.vertices, np.float32))
    dst = np.float32([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]])
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(inner_gray, H, (size, size))


def extract_white_digit(square_gray):
    """黑方块内提取白字；只取中央区域，避免正方形外部白纸进入。"""
    g = cv2.GaussianBlur(square_gray, (3, 3), 0)
    # 方块内部通常接近黑色，白字明显更亮；Otsu 后再限制到中央 76%。
    _, white = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    margin = int(0.12 * white.shape[0])
    central = np.zeros_like(white)
    central[margin:-margin, margin:-margin] = 255
    white = cv2.bitwise_and(white, central)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return normalize_digit(white)


def recognize_digit(square_gray, hog, knn):
    norm = extract_white_digit(square_gray)
    if norm is None or cv2.countNonZero(norm) < 25:
        return -1, 0.0, None
    feature = hog.compute(norm).reshape(1, -1).astype(np.float32)
    _, result, neighbours, distances = knn.findNearest(feature, k=7)
    votes = neighbours.astype(np.int32).ravel()
    counts = np.bincount(votes, minlength=10)
    digit = int(np.argmax(counts))
    vote_conf = float(counts[digit] / len(votes))
    # 距离只作相对质量项，最终置信度主要由近邻投票决定。
    d = distances.ravel()[votes == digit]
    quality = 1.0 / (1.0 + (float(np.mean(d)) if len(d) else 1e6) / 250.0)
    confidence = 0.80 * vote_conf + 0.20 * quality
    return digit, confidence, norm


def process_frame(frame, requested_digit, hog, knn, save_debug=False):
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    outer_binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)

    _, approx = find_a4_outer_quad(outer_binary)
    if approx is None:
        return img, None, '未找到 A4 黑色外框'
    cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

    left_px, right_px = get_outer_frame_params(approx)
    outer_height = (left_px + right_px) / 2.0
    distance_cm = estimate_distance(outer_height)

    src = order_points(approx.reshape(4, 2).astype(np.float32))
    dw, dh = int(A4_W_CM*SCALE), int(A4_H_CM*SCALE)
    dst = np.float32([[0, 0], [dw, 0], [dw, dh], [0, dh]])
    H = cv2.getPerspectiveTransform(src, dst)
    H_inv = np.linalg.inv(H)
    warped = cv2.warpPerspective(gray, H, (dw, dh))
    warped = cv2.GaussianBlur(warped, (3, 3), 0)

    border_px = int(BORDER_CM*SCALE)
    inner_gray = warped[border_px:dh-border_px, border_px:dw-border_px]
    otsu, inner_binary = cv2.threshold(
        inner_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    inner_binary = cv2.morphologyEx(
        inner_binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # 白色数字会在黑方块内形成小孔，但 detect_squares_robust 使用外轮廓，
    # 且已有填充率容差，因此可以直接复用发挥 1~2 问检测器。
    squares, clean_binary = detect_squares_robust(inner_binary)
    if not squares:
        return img, None, '未找到有效正方形'

    records = []
    for i, sq in enumerate(squares):
        patch = rectify_square_gray(inner_gray, sq)
        digit, conf, norm = recognize_digit(patch, hog, knn)
        raw_cm = sq.avg_side_length / SCALE
        corrected_cm = correct_square_size(raw_cm)
        records.append(dict(index=i, square=sq, digit=digit,
                            confidence=conf, side_cm=corrected_cm,
                            patch=patch, norm=norm))

    matches = [r for r in records if r['digit'] == requested_digit]
    if not matches:
        # 没匹配时不胡乱输出尺寸，屏幕显示所有识别结果便于排错。
        summary = ', '.join(f"{r['digit']}({r['confidence']:.2f})" for r in records)
        return img, None, f'未识别到编号 {requested_digit}; 当前: {summary}'
    selected = max(matches, key=lambda r: r['confidence'])

    for r in records:
        chosen = (r is selected)
        color = (0, 0, 255) if chosen else (255, 0, 0)
        label = f"ID={r['digit']}" + (f" x={r['side_cm']:.2f}" if chosen else '')
        draw_square_on_original(img, r['square'], border_px, H_inv, color, label)

    cv2.rectangle(img, (8, 8), (570, 112), (0, 0, 0), -1)
    cv2.putText(img, f'D={distance_cm:.1f} cm', (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, .85, (0, 255, 255), 2)
    cv2.putText(img, f'ID={requested_digit}  x={selected["side_cm"]:.2f} cm',
                (20, 82), cv2.FONT_HERSHEY_SIMPLEX, .85, (0, 255, 255), 2)
    cv2.putText(img, f'OCR confidence={selected["confidence"]:.2f}', (20, 106),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1)

    if save_debug:
        cv2.imwrite('dev3_01_outer_binary.jpg', outer_binary)
        cv2.imwrite('dev3_02_warped.jpg', warped)
        cv2.imwrite('dev3_03_inner_gray.jpg', inner_gray)
        cv2.imwrite('dev3_04_inner_binary.jpg', inner_binary)
        cv2.imwrite('dev3_05_clean_binary.jpg', clean_binary)
        for r in records:
            cv2.imwrite(f"dev3_square_{r['index']}_id_{r['digit']}.jpg", r['patch'])
            if r['norm'] is not None:
                cv2.imwrite(f"dev3_digit_{r['index']}_id_{r['digit']}.jpg", r['norm'])

    result = dict(distance_cm=distance_cm, digit=requested_digit,
                  side_cm=selected['side_cm'], confidence=selected['confidence'],
                  all_records=records, otsu=otsu)
    return img, result, 'OK'


def read_digit(value):
    if value is not None:
        return value
    while True:
        s = input('请输入指定正方形编号 0~9：').strip()
        if len(s) == 1 and s.isdigit():
            return int(s)
        print('输入无效。')


def main():
    parser = argparse.ArgumentParser(description='2025 C题发挥(3)：指定编号正方形测量')
    parser.add_argument('--digit', type=int, choices=range(10), help='指定编号 0~9')
    parser.add_argument('--no-gui', action='store_true', help='不打开 OpenCV 窗口')
    parser.add_argument('--debug', action='store_true', help='保存中间调试图')
    parser.add_argument('--warmup', type=float, default=2.0, help='相机预热秒数')
    args = parser.parse_args()

    requested = read_digit(args.digit)
    print('正在建立离线数字分类器...')
    hog, knn = build_digit_classifier()

    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={'size': (1280, 720), 'format': 'RGB888'})
    picam2.configure(config)
    picam2.start()
    time.sleep(max(0.0, args.warmup))
    input(f'编号已设为 {requested}。摆好目标物后按 Enter 一键测量...')

    try:
        # 连拍 3 帧，丢弃前两帧，减小按键后曝光波动；仍属于一次启动测量。
        frame_rgb = None
        for _ in range(3):
            frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        output, result, message = process_frame(
            frame_bgr, requested, hog, knn, save_debug=args.debug)
        cv2.imwrite('develop_3_result.jpg', output)
        if result is None:
            print('测量失败：' + message)
            cv2.putText(output, message[:35], (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 255), 2)
        else:
            print(f"测量完成: D={result['distance_cm']:.1f} cm, "
                  f"ID={result['digit']}, x={result['side_cm']:.2f} cm, "
                  f"OCR置信度={result['confidence']:.2f}")
        if not args.no_gui:
            cv2.imshow('Develop Part 3', output)
            cv2.waitKey(0)
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
