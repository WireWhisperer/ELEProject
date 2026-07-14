#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025 电赛 C 题发挥(3) V3——数字优先、支持局部重叠正方形。

核心：先在透视校正图中检测白色编号，以编号中心约束正方形中心，
再在 6~12 cm 范围搜索边长；不再要求先恢复所有实心正方形。

依赖：本文件与 detect_squares_pi_single_frame_v7.py 放在同一目录。
运行：python3 numbered_square_pi_v3.py
      python3 numbered_square_pi_v3.py --digit 7 --debug
"""
import argparse
import glob
import os
import time
import serial
from dataclasses import dataclass

import cv2
import numpy as np
from picamera2 import Picamera2

from detect_squares_pi_single_frame_v7 import (
    find_a4_outer_quad, order_points, correct_square_size,
    estimate_distance, get_outer_frame_params
)

A4_W_CM, A4_H_CM = 21.0, 29.7
BORDER_CM = 2.0
# 第三问单独使用更高透视分辨率，保留更多数字笔画。
PROCESS_SCALE = 40.0
MIN_SIDE_CM, MAX_SIDE_CM = 5.5, 12.5
NORM_W, NORM_H = 48, 64

# 当前暂不使用距离标定，避免显示无意义的 D。
DISPLAY_DISTANCE = False


@dataclass
class DigitCandidate:
    center: tuple
    bbox: tuple
    mask: np.ndarray
    predicted: int = -1
    confidence: float = 0.0
    requested_score: float = 0.0
    class_scores: np.ndarray = None


@dataclass
class SquareResult:
    center: tuple
    side_px: float
    raw_cm: float
    corrected_cm: float
    score: float
    side_supports: tuple
    corners: np.ndarray


def normalize_digit(mask):
    """将单个白字归一化为 48x64 白字黑底图。"""
    b = (mask > 0).astype(np.uint8) * 255
    pts = cv2.findNonZero(b)
    if pts is None:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    if w < 2 or h < 5:
        return None
    roi = b[y:y+h, x:x+w]
    canvas = np.zeros((NORM_H, NORM_W), np.uint8)
    scale = min(38.0 / w, 54.0 / h)
    nw, nh = max(1, int(round(w*scale))), max(1, int(round(h*scale)))
    roi = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
    ox, oy = (NORM_W-nw)//2, (NORM_H-nh)//2
    canvas[oy:oy+nh, ox:ox+nw] = roi
    return canvas


def count_holes(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return sum(1 for h in hierarchy[0] if h[3] >= 0)


def make_hog():
    return cv2.HOGDescriptor((48, 64), (16, 16), (8, 8), (8, 8), 9)


def feature_of(mask, hog):
    f = hog.compute(mask).reshape(-1).astype(np.float32)
    n = float(np.linalg.norm(f))
    return f / max(n, 1e-8)


def available_fonts():
    """优先使用接近目标物的无衬线字体；没有 Pillow 时自动回退。"""
    patterns = [
        '/usr/share/fonts/truetype/dejavu/*.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans*.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans*.ttf',
        '/usr/share/fonts/opentype/noto/NotoSans*.ttf',
    ]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    # 粗体与普通字体均加入，最多取16个，避免启动过慢。
    return sorted(set(paths))[:16]


def build_template_bank():
    """生成多字体、多粗细模板库；返回 HOG 和按数字分类的特征。"""
    hog = make_hog()
    bank = {d: [] for d in range(10)}
    try:
        from PIL import Image, ImageDraw, ImageFont
        fonts = available_fonts()
        for path in fonts:
            for size in (52, 60, 68, 76):
                try:
                    font = ImageFont.truetype(path, size)
                except Exception:
                    continue
                for d in range(10):
                    for dx in (-2, 0, 2):
                        for dy in (-2, 0, 2):
                            im = Image.new('L', (100, 120), 0)
                            draw = ImageDraw.Draw(im)
                            box = draw.textbbox((0, 0), str(d), font=font)
                            tw, th = box[2]-box[0], box[3]-box[1]
                            x = (100-tw)//2-box[0]+dx
                            y = (120-th)//2-box[1]+dy
                            draw.text((x, y), str(d), fill=255, font=font)
                            a = np.asarray(im)
                            _, a = cv2.threshold(a, 40, 255, cv2.THRESH_BINARY)
                            n = normalize_digit(a)
                            if n is not None:
                                bank[d].append(feature_of(n, hog))
    except ImportError:
        pass

    # OpenCV 字体作为兜底及补充。
    faces = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
             cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX]
    for face in faces:
        for thick in (1, 2, 3, 4):
            for fs in (1.6, 1.9, 2.2):
                for d in range(10):
                    a = np.zeros((100, 90), np.uint8)
                    (tw, th), _ = cv2.getTextSize(str(d), face, fs, thick)
                    cv2.putText(a, str(d), ((90-tw)//2, (100+th)//2),
                                face, fs, 255, thick, cv2.LINE_AA)
                    _, a = cv2.threshold(a, 40, 255, cv2.THRESH_BINARY)
                    n = normalize_digit(a)
                    bank[d].append(feature_of(n, hog))
    # 优先加入本机采集的真实模板：digit_templates/0/*.png ... /9/*.png
    # 模板应为白字黑底；彩色/灰度图片也会自动二值化和归一化。
    real_counts = [0]*10
    for d in range(10):
        patterns = [f'digit_templates/{d}/*.png', f'digit_templates/{d}/*.jpg',
                    f'digit_templates/{d}/*.jpeg']
        for pattern in patterns:
            for filename in glob.glob(pattern):
                a = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                if a is None:
                    continue
                # 自动判断极性：模板默认要求白字黑底。
                if float(np.mean(a)) > 127:
                    a = cv2.bitwise_not(a)
                _, a = cv2.threshold(a, 0, 255,
                                     cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                n = normalize_digit(a)
                if n is not None:
                    bank[d].append(feature_of(n, hog))
                    real_counts[d] += 1
    print('Real OCR template counts:', real_counts)
    for d in bank:
        bank[d] = np.asarray(bank[d], np.float32)
    print('OCR template counts:', [len(bank[d]) for d in range(10)])
    return hog, bank


def classify_digit(mask, hog, bank):
    """模板余弦相似度分类，并使用孔洞数修正 0/6/8/9。"""
    f = feature_of(mask, hog)
    scores = np.zeros(10, np.float32)
    holes = count_holes(mask)
    expected_holes = {0: 1, 1: 0, 2: 0, 3: 0, 4: 0,
                      5: 0, 6: 1, 7: 0, 8: 2, 9: 1}
    for d in range(10):
        sims = bank[d] @ f
        k = min(5, len(sims))
        base = float(np.mean(np.partition(sims, -k)[-k:]))
        # 拍摄模糊会封闭/打开小孔，拓扑只作弱修正。
        diff = abs(holes - expected_holes[d])
        scores[d] = base + (0.035 if diff == 0 else -0.025*diff)
    order = np.argsort(scores)[::-1]
    pred = int(order[0])
    margin = float(scores[order[0]] - scores[order[1]])
    confidence = float(np.clip(0.55 + 3.0*margin, 0.0, 0.99))
    return pred, confidence, scores


def refine_digit_from_gray(inner_gray, bbox):
    """在候选位置回到灰度图局部阈值，恢复Otsu丢掉的细笔画和孔洞。"""
    x, y, w, h = bbox
    pad_x = max(4, int(0.55*h))
    pad_y = max(4, int(0.35*h))
    x1, y1 = max(0, x-pad_x), max(0, y-pad_y)
    x2 = min(inner_gray.shape[1], x+w+pad_x)
    y2 = min(inner_gray.shape[0], y+h+pad_y)
    roi = inner_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    # 黑方块背景取低分位，白字取高分位；局部阈值比整图Otsu稳定。
    bg = float(np.percentile(roi, 25))
    core = inner_gray[y:y+h, x:x+w]
    fg = float(np.percentile(core, 88)) if core.size else float(np.percentile(roi, 99))
    if fg-bg < 15:
        return None
    threshold = bg + 0.48*(fg-bg)
    bright = np.uint8(roi > threshold)*255
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                              np.ones((2, 2), np.uint8))

    n, labels, stats, centers = cv2.connectedComponentsWithStats(bright, 8)
    wanted = np.array([x+w/2-x1, y+h/2-y1], np.float32)
    choices = []
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if area < 8 or bh < 5:
            continue
        # 排除进入ROI的大片白纸。
        if area > 0.42*roi.size or bw > 0.80*roi.shape[1]:
            continue
        dist = float(np.linalg.norm(centers[i]-wanted))
        overlap_x = max(0, min(bx+bw, x+w-x1)-max(bx, x-x1))
        overlap_y = max(0, min(by+bh, y+h-y1)-max(by, y-y1))
        overlap = overlap_x*overlap_y
        choices.append((overlap > 0, -dist, area, i))
    if not choices:
        return None
    choices.sort(reverse=True)
    chosen = choices[0][3]
    component = np.uint8(labels == chosen)*255
    return normalize_digit(component)


def detect_white_digits(inner_gray, black_mask):
    """
    从整幅物面中检测被黑色包围的白色字符，不依赖正方形候选。
    black_mask: 黑色区域为255。
    """
    # 白色候选由“不是黑色”得到；纸面和字符都为白色，随后用黑色环绕率区分。
    white = cv2.bitwise_not(black_mask)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(white, 8)
    candidates = []
    h, w = white.shape
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        # 20 px/cm下，实际白字通常在8~65px高；保留少量余量。
        if not (5 <= cw <= 55 and 9 <= ch <= 80 and 15 <= area <= 2200):
            continue
        if x < 2 or y < 2 or x+cw >= w-2 or y+ch >= h-2:
            continue
        component = (labels == i).astype(np.uint8)*255
        # 字符周围应该主要是黑色。使用膨胀环，不要求四周全部黑，兼容细笔画。
        radius = max(3, int(round(0.22*ch)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (2*radius+1, 2*radius+1))
        ring = cv2.subtract(cv2.dilate(component, kernel), component)
        ring_n = cv2.countNonZero(ring)
        surround = (cv2.countNonZero(cv2.bitwise_and(ring, black_mask)) /
                    max(ring_n, 1))
        if surround < 0.58:
            continue
        # 去除很扁、很细的反光或边缘噪声。
        aspect = cw / max(ch, 1)
        if not 0.10 <= aspect <= 1.15:
            continue
        crop = component[y:y+ch, x:x+cw]
        norm = refine_digit_from_gray(inner_gray, (x, y, cw, ch))
        if norm is None:
            norm = normalize_digit(crop)
        if norm is None:
            continue
        candidates.append(DigitCandidate(
            center=(float(centers[i][0]), float(centers[i][1])),
            bbox=(int(x), int(y), int(cw), int(ch)), mask=norm))
    return candidates


def fill_digit_holes(black_mask, digits):
    """测边前把所有白色编号填回黑色，避免编号产生假边缘。"""
    filled = black_mask.copy()
    for d in digits:
        x, y, w, h = d.bbox
        pad = max(2, int(0.10*h))
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(filled.shape[1], x+w+pad), min(filled.shape[0], y+h+pad)
        filled[y1:y2, x1:x2] = 255
    return filled


def strip_transition_score(black, orientation, coord, lo, hi, band=3):
    """计算理论边界处“内黑外白”的比例，重叠处自然不给分。"""
    h, w = black.shape
    lo, hi = int(max(0, lo)), int(hi)
    if orientation == 'v':
        x = int(round(coord))
        if x-band-2 < 0 or x+band+2 >= w or hi <= lo:
            return 0.0
        # 两种方向均计算，调用者无需指定该边是左边还是右边。
        left = black[lo:hi, x-band:x]
        right = black[lo:hi, x+1:x+band+1]
    else:
        y = int(round(coord))
        if y-band-2 < 0 or y+band+2 >= h or hi <= lo:
            return 0.0
        left = black[y-band:y, lo:hi]
        right = black[y+1:y+band+1, lo:hi]
    if left.size == 0 or right.size == 0:
        return 0.0
    a = np.mean(left > 0, axis=0 if orientation == 'h' else 1)
    b = np.mean(right > 0, axis=0 if orientation == 'h' else 1)
    # 边界两侧一黑一白，不限定方向。
    return float(np.mean(np.abs(a-b)))


def square_candidate_score(black, cx, cy, side):
    half = side/2.0
    x1, x2, y1, y2 = cx-half, cx+half, cy-half, cy+half
    h, w = black.shape
    if x1 < 2 or y1 < 2 or x2 >= w-2 or y2 >= h-2:
        return -1.0, (0, 0, 0, 0)
    # 去掉两端约7%，避免相邻方块在角部造成过大干扰。
    cut = 0.07*side
    top = strip_transition_score(black, 'h', y1, x1+cut, x2-cut)
    bottom = strip_transition_score(black, 'h', y2, x1+cut, x2-cut)
    left = strip_transition_score(black, 'v', x1, y1+cut, y2-cut)
    right = strip_transition_score(black, 'v', x2, y1+cut, y2-cut)
    supports = (top, right, bottom, left)
    ss = sorted(supports, reverse=True)
    # 局部重叠时允许1~2条边消失，主要相信最清晰的两条边。
    edge_score = 0.48*ss[0] + 0.32*ss[1] + 0.15*ss[2] + 0.05*ss[3]

    ix1, ix2 = int(x1+0.10*side), int(x2-0.10*side)
    iy1, iy2 = int(y1+0.10*side), int(y2-0.10*side)
    inside = float(np.mean(black[iy1:iy2, ix1:ix2] > 0))

    # 四角应属于黑色方块；取角点内侧小块黑色率。
    r = max(3, int(0.05*side))
    corner_values = []
    for x, y in ((x1+r, y1+r), (x2-r, y1+r),
                 (x2-r, y2-r), (x1+r, y2-r)):
        xa, xb = int(x-r), int(x+r+1)
        ya, yb = int(y-r), int(y+r+1)
        corner_values.append(float(np.mean(black[ya:yb, xa:xb] > 0)))
    corner_score = float(np.mean(corner_values))
    score = 0.72*edge_score + 0.18*inside + 0.10*corner_score
    return score, supports


def estimate_square_from_digit(black, digit_center):
    """在数字中心附近小范围平移，并遍历6~12cm边长寻找最佳正方形。"""
    base_x, base_y = digit_center
    min_side = int(round(MIN_SIDE_CM*PROCESS_SCALE))
    max_side = int(round(MAX_SIDE_CM*PROCESS_SCALE))
    candidates = []
    # 数字印刷中心和二值质心可能相差数像素。
    for dy in range(-16, 17, 4):
        for dx in range(-16, 17, 4):
            cx, cy = base_x+dx, base_y+dy
            for side in range(min_side, max_side+1, 4):
                score, supports = square_candidate_score(black, cx, cy, side)
                score -= 0.0004*np.hypot(dx, dy)
                candidates.append((score, side, cx, cy, supports))
    candidates.sort(reverse=True, key=lambda z: z[0])
    if not candidates or candidates[0][0] < 0.18:
        return None

    # 在最佳值附近加权平均，减小2px步进和阈值噪声。
    best = candidates[0]
    nearby = [c for c in candidates[:80]
              if abs(c[1]-best[1]) <= 10 and
              np.hypot(c[2]-best[2], c[3]-best[3]) <= 8 and
              c[0] >= best[0]-0.025]
    weights = np.array([max(c[0]-best[0]+0.03, 0.002) for c in nearby])
    side = float(np.average([c[1] for c in nearby], weights=weights))
    cx = float(np.average([c[2] for c in nearby], weights=weights))
    cy = float(np.average([c[3] for c in nearby], weights=weights))
    half = side/2
    corners = np.float32([[cx-half, cy-half], [cx+half, cy-half],
                          [cx+half, cy+half], [cx-half, cy+half]])
    raw_cm = side/PROCESS_SCALE
    return SquareResult((cx, cy), side, raw_cm,
                        correct_square_size(raw_cm), best[0], best[4], corners)


def draw_inner_polygon_on_original(img, corners, border_px, H_inv,
                                   color, thickness=3):
    pts = corners.copy().astype(np.float32)
    pts[:, 0] += border_px
    pts[:, 1] += border_px
    orig = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H_inv)
    orig = np.round(orig.reshape(-1, 2)).astype(np.int32)
    cv2.polylines(img, [orig], True, color, thickness)
    return orig


def put_ascii_status(img, lines, color=(0, 255, 255)):
    width = min(img.shape[1]-10, 620)
    height = 18 + 34*len(lines)
    cv2.rectangle(img, (8, 8), (width, height), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (20, 38+32*i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.78, color, 2, cv2.LINE_AA)


def process_frame(frame, requested_digit, hog, bank, debug=False):
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    outer_binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    _, approx = find_a4_outer_quad(outer_binary)
    if approx is None:
        put_ascii_status(img, ['ERROR: A4 frame not found'], (0, 0, 255))
        return img, None
    cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

    left_pix, right_pix = get_outer_frame_params(approx)
    outer_height = (left_pix + right_pix) / 2.0
    distance_cm = float(estimate_distance(outer_height)) if outer_height > 0 else 0.0

    src = order_points(approx.reshape(4, 2).astype(np.float32))
    dw, dh = int(A4_W_CM*PROCESS_SCALE), int(A4_H_CM*PROCESS_SCALE)
    dst = np.float32([[0, 0], [dw-1, 0], [dw-1, dh-1], [0, dh-1]])
    H = cv2.getPerspectiveTransform(src, dst)
    H_inv = np.linalg.inv(H)
    warped = cv2.warpPerspective(gray, H, (dw, dh))
    warped_blur = cv2.GaussianBlur(warped, (3, 3), 0)
    border_px = int(BORDER_CM*PROCESS_SCALE)
    inner_gray = warped_blur[border_px:dh-border_px,
                             border_px:dw-border_px]

    otsu, black = cv2.threshold(inner_gray, 0, 255,
                                cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE,
                             np.ones((3, 3), np.uint8))

    digits = detect_white_digits(inner_gray, black)
    if not digits:
        put_ascii_status(img, ['ERROR: no white digits found'], (0, 0, 255))
        return img, None

    for d in digits:
        pred, conf, scores = classify_digit(d.mask, hog, bank)
        d.predicted, d.confidence, d.class_scores = pred, conf, scores
        d.requested_score = float(scores[requested_digit])

    # 目标选择V3：优先硬分类匹配；禁止直接跨字符比较未经校准的原始类别分数。
    def relative_score(d):
        target_score = float(d.class_scores[requested_digit])
        other_best = max(float(d.class_scores[i]) for i in range(10)
                         if i != requested_digit)
        return target_score-other_best

    hard_matches = [d for d in digits if d.predicted == requested_digit]
    if hard_matches:
        target = max(hard_matches, key=relative_score)
        selection_mode = 'hard'
    else:
        # 无硬匹配才回退到相对优势；例如真实5稍被分成3时，
        # 其5/3分差通常仍比数字8的5/8分差更接近0。
        target = max(digits, key=relative_score)
        selection_mode = 'fallback'
    target_margin = relative_score(target)

    black_filled = fill_digit_holes(black, digits)
    square = estimate_square_from_digit(black_filled, target.center)
    if square is None:
        put_ascii_status(img, [f'ERROR: square for ID={requested_digit} not found'],
                         (0, 0, 255))
        return img, None

    # 绘制全部数字中心与硬分类结果。
    for d in digits:
        x, y, w, h = d.bbox
        box = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
        color = (0, 0, 255) if d is target else (255, 0, 0)
        orig = draw_inner_polygon_on_original(img, box, border_px, H_inv,
                                              color, 1)
        tx, ty = tuple(orig[0])
        cv2.putText(img, str(d.predicted), (tx, max(15, ty-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)

    orig_square = draw_inner_polygon_on_original(
        img, square.corners, border_px, H_inv, (0, 0, 255), 3)
    label_pos = tuple(np.mean(orig_square, axis=0).astype(int))
    cv2.putText(img, f'ID={requested_digit} x={square.corrected_cm:.2f}cm',
                label_pos, cv2.FONT_HERSHEY_SIMPLEX, .72, (0, 0, 255), 2)

    lines = [f'ID={requested_digit}  x={square.corrected_cm:.2f} cm',
             f'OCR={target.predicted} mode={selection_mode} margin={target_margin:.3f}',
             f'geometry score={square.score:.3f}']
    put_ascii_status(img, lines)

    print('All digits:', [(d.predicted, round(d.confidence, 2),
                           round(d.requested_score, 3),
                           tuple(round(v, 1) for v in d.center)) for d in digits])
    print(f'Selected ID={requested_digit}: OCR hard={target.predicted}, mode={selection_mode}, '
          f'x_raw={square.raw_cm:.3f}cm, x={square.corrected_cm:.3f}cm, '
          f'geometry={square.score:.3f}, supports='
          f'{[round(v, 3) for v in square.side_supports]}')

    if debug:
        cv2.imwrite('d3_01_gray.jpg', gray)
        cv2.imwrite('d3_02_outer_binary.jpg', outer_binary)
        cv2.imwrite('d3_03_warped.jpg', warped)
        cv2.imwrite('d3_04_inner_gray.jpg', inner_gray)
        cv2.imwrite('d3_05_black.jpg', black)
        cv2.imwrite('d3_06_black_digits_filled.jpg', black_filled)
        debug_warp = cv2.cvtColor(inner_gray, cv2.COLOR_GRAY2BGR)
        for d in digits:
            x, y, w, h = d.bbox
            color = (0, 0, 255) if d is target else (255, 0, 0)
            cv2.rectangle(debug_warp, (x, y), (x+w, y+h), color, 1)
            cv2.putText(debug_warp, str(d.predicted), (x, max(10, y-2)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        cv2.polylines(debug_warp, [np.round(square.corners).astype(np.int32)],
                      True, (0, 0, 255), 2)
        cv2.imwrite('d3_07_digit_and_square.jpg', debug_warp)
        for i, d in enumerate(digits):
            cv2.imwrite(f'd3_digit_{i}_pred_{d.predicted}.png', d.mask)

    result = {
        'requested_digit': requested_digit,
        'ocr_hard_prediction': target.predicted,
        'selection_mode': selection_mode,
        'ocr_margin': target_margin,
        'distance_cm': distance_cm,
        'raw_cm': square.raw_cm,
        'side_cm': square.corrected_cm,
        'geometry_score': square.score,
        'supports': square.side_supports,
        'digit_count': len(digits),
        'otsu': otsu,
    }
    return img, result


def read_digit(value):
    if value is not None:
        return value
    while True:
        text = input('Input requested square ID (0-9): ').strip()
        if len(text) == 1 and text.isdigit():
            return int(text)
        print('Invalid input.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--digit', type=int, choices=range(10))
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no-gui', action='store_true')
    parser.add_argument('--warmup', type=float, default=2.0)
    args = parser.parse_args()

    requested = read_digit(args.digit)
    print('Building offline OCR templates...')
    hog, bank = build_template_bank()

    camera = Picamera2()
    config = camera.create_still_configuration(
        main={'size': (1280, 720), 'format': 'RGB888'})
    camera.configure(config)
    camera.start()
    time.sleep(max(args.warmup, 0.0))

    ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.1)

    input(f'ID={requested}. Place target, then press Enter to measure...')
    try:
        frame = None
        for _ in range(3):
            frame = camera.capture_array()
        # RGB888在部分Picamera2版本返回RGB；若你们实测颜色正常，保留此转换。
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output, result = process_frame(bgr, requested, hog, bank, args.debug)
        cv2.imwrite('numbered_square_pi_v3_result.jpg', output)
        if result:
            print(f"RESULT: ID={requested}, x={result['side_cm']:.2f} cm")
            msg = f"D={result['distance_cm']:.1f},x={result['side_cm']:.2f},type=square\r\n"
            ser.write(msg.encode('utf-8'))
            print(f'Sent via serial: {msg.strip()}')
        else:
            print('RESULT: measurement failed; use --debug to inspect images.')
        if not args.no_gui:
            cv2.imshow('Numbered Square Pi V3', output)
            cv2.waitKey(0)
    finally:
        ser.close()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
