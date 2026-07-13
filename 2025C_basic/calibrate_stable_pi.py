import cv2
import numpy as np
import json
import time
from collections import deque
from picamera2 import Picamera2

# ==================== 可调参数 ====================
CAMERA_SIZE = (1280, 720)       # 必须与 basic_pi.py 保持一致
START_DISTANCE_CM = 100.0       # 启动时的默认标定距离
COLLECTION_FRAMES = 40          # 每个标定点采集的有效帧数
MIN_AREA_RATIO = 0.05           # A4轮廓至少占画面的比例
MAX_AREA_RATIO = 0.92           # 防止把整个画面边界识别为A4
MAX_SAMPLE_STD = 2.0            # 采集标准差超过此值则拒绝保存
LOCK_CAMERA_CONTROLS = True     # 预热后锁定曝光和白平衡
OUTPUT_PY = "new_distance_calibration.py"
OUTPUT_JSON = "calibration_data.json"


# ==================== 几何辅助函数 ====================
def point_distance(p1, p2):
    return float(np.linalg.norm(np.asarray(p1, dtype=np.float32) -
                                np.asarray(p2, dtype=np.float32)))


def order_points(pts):
    """将四点排序为：左上、右上、右下、左下。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def get_outer_frame_params(approx_outer):
    """返回A4左边、右边和中轴的像素长度。"""
    tl, tr, br, bl = order_points(approx_outer.reshape(4, 2))
    left_pixels = point_distance(tl, bl)
    right_pixels = point_distance(tr, br)
    top_mid = (tl + tr) / 2.0
    bottom_mid = (bl + br) / 2.0
    axis_pixels = point_distance(top_mid, bottom_mid)
    return left_pixels, right_pixels, axis_pixels


def refine_corners(gray, approx):
    """对四个整数角点进行亚像素优化。"""
    if approx is None or len(approx) != 4:
        return approx
    corners = approx.reshape(4, 2).astype(np.float32)
    h, w = gray.shape[:2]
    # cornerSubPix要求初始点位于图像内部。
    corners[:, 0] = np.clip(corners[:, 0], 8, w - 9)
    corners[:, 1] = np.clip(corners[:, 1], 8, h - 9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30, 0.01)
    try:
        refined = cv2.cornerSubPix(gray, corners.reshape(-1, 1, 2),
                                   (7, 7), (-1, -1), criteria)
        return refined.reshape(4, 1, 2)
    except cv2.error:
        return corners.reshape(4, 1, 2)


def is_valid_a4(approx, frame_shape):
    """过滤噪声轮廓、画面边界和明显不是A4的四边形。"""
    if approx is None or len(approx) != 4:
        return False
    draw_pts = np.round(approx).astype(np.int32)
    if not cv2.isContourConvex(draw_pts):
        return False

    h, w = frame_shape[:2]
    pts = approx.reshape(4, 2).astype(np.float32)
    tl, tr, br, bl = order_points(pts)
    wt = point_distance(tl, tr)
    wb = point_distance(bl, br)
    hl = point_distance(tl, bl)
    hr = point_distance(tr, br)
    avg_w = (wt + wb) / 2.0
    avg_h = (hl + hr) / 2.0
    if avg_w < 20 or avg_h < 20:
        return False

    ratio = max(avg_w, avg_h) / min(avg_w, avg_h)
    if not 1.12 <= ratio <= 1.75:
        return False

    # 防止四角贴边：贴边时通常代表A4没有完整进入画面。
    margin = 6
    xs, ys = pts[:, 0], pts[:, 1]
    if (xs.min() <= margin or xs.max() >= w - margin or
            ys.min() <= margin or ys.max() >= h - margin):
        return False

    # 两条对应边差异过大通常是错误轮廓或透视过强。
    h_diff = abs(hl - hr) / max(hl, hr)
    w_diff = abs(wt - wb) / max(wt, wb)
    return h_diff <= 0.40 and w_diff <= 0.40


def find_a4(gray, frame_shape):
    """检测A4四边形，返回亚像素角点、二值图。"""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame_shape[0] * frame_shape[1]
    min_area = frame_area * MIN_AREA_RATIO
    max_area = frame_area * MAX_AREA_RATIO

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        area = cv2.contourArea(contour)
        if not min_area <= area <= max_area:
            continue
        peri = cv2.arcLength(contour, True)
        candidate = cv2.approxPolyDP(contour, 0.015 * peri, True)
        if len(candidate) != 4:
            continue
        candidate = refine_corners(gray, candidate)
        if is_valid_a4(candidate, frame_shape):
            return candidate, thresh
    return None, thresh


# ==================== 实时稳定器 ====================
class HeightStabilizer:
    def __init__(self, window_size=15, max_jump=15.0,
                 alpha=0.25, deadband=0.4):
        self.history = deque(maxlen=window_size)
        self.max_jump = float(max_jump)
        self.alpha = float(alpha)
        self.deadband = float(deadband)
        self.filtered = None

    def reset(self):
        self.history.clear()
        self.filtered = None

    def update(self, value):
        if value is None or not np.isfinite(value):
            return self.filtered, False
        value = float(value)
        if len(self.history) >= 5:
            historical_median = float(np.median(self.history))
            if abs(value - historical_median) > self.max_jump:
                return self.filtered, False
        self.history.append(value)
        med = float(np.median(self.history))
        if self.filtered is None:
            self.filtered = med
        else:
            candidate = self.alpha * med + (1.0 - self.alpha) * self.filtered
            if abs(candidate - self.filtered) >= self.deadband:
                self.filtered = candidate
        return self.filtered, True


# ==================== 数据保存 ====================
def save_calibration(data):
    if not data:
        print("未采集任何数据。")
        return

    # 同一距离若重复采集，保留每个样本；最终按像素高度排序。
    data_sorted = sorted(data, key=lambda x: x[0])
    with open(OUTPUT_PY, "w", encoding="utf-8") as f:
        f.write("# 自动生成：[(像素高度, 距离cm), ...]\\n")
        f.write("DISTANCE_CALIBRATION = [\\n")
        for pixels, dist in data_sorted:
            f.write(f"    ({pixels:.2f}, {dist:.2f}),\\n")
        f.write("]\\n")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump([[round(p, 2), round(d, 2)] for p, d in data_sorted],
                  f, ensure_ascii=False, indent=2)

    print(f"已保存 {len(data_sorted)} 个标定点：{OUTPUT_PY}")
    print(f"JSON备份：{OUTPUT_JSON}")


def lock_camera_after_warmup(picam2):
    """锁定自动曝光/白平衡；不支持时自动跳过。"""
    if not LOCK_CAMERA_CONTROLS:
        return
    print("摄像头预热中……")
    time.sleep(2.0)
    try:
        metadata = picam2.capture_metadata()
        controls = {"AeEnable": False, "AwbEnable": False}
        if metadata.get("ExposureTime") is not None:
            controls["ExposureTime"] = int(metadata["ExposureTime"])
        if metadata.get("AnalogueGain") is not None:
            controls["AnalogueGain"] = float(metadata["AnalogueGain"])
        if metadata.get("ColourGains") is not None:
            controls["ColourGains"] = tuple(metadata["ColourGains"])
        picam2.set_controls(controls)
        print("已锁定曝光和白平衡。")
    except Exception as exc:
        print(f"无法锁定摄像头参数，继续使用自动模式：{exc}")


# ==================== 主标定程序 ====================
def collect_distance_pixels(picam2):
    data = []
    stabilizer = HeightStabilizer(window_size=15, max_jump=15.0,
                                  alpha=0.25, deadband=0.4)
    current_distance = START_DISTANCE_CM
    collecting = False
    samples = []

    print("\n=== 稳定版距离-像素标定 ===")
    print("确保 basic_pi.py 使用相同分辨率和 axis_pixels 高度定义。")
    print("按键：")
    print("  c       采集当前距离（40个有效帧）")
    print("  + / -   距离增加/减少 5 cm")
    print("  ] / [   距离增加/减少 1 cm")
    print("  r       重置滤波器（移动A4后建议按一次）")
    print("  d       删除最后一个标定点")
    print("  q       保存并退出\n")

    while True:
        frame_rgb = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        approx, _ = find_a4(gray, frame.shape)

        raw_height = None
        stable_height = None
        accepted = False

        if approx is not None:
            _, _, raw_height = get_outer_frame_params(approx)
            stable_height, accepted = stabilizer.update(raw_height)
            draw_pts = np.round(approx).astype(np.int32)
            cv2.drawContours(display, [draw_pts], -1, (0, 255, 0), 3)
        else:
            cv2.putText(display, "A4 not detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if raw_height is not None:
            cv2.putText(display, f"Raw: {raw_height:.2f}px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        if stable_height is not None:
            cv2.putText(display, f"Stable: {stable_height:.2f}px", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(display, f"Distance: {current_distance:.1f}cm", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display, f"Saved points: {len(data)}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 只采集通过突变检查的原始高度，最终再做MAD过滤。
        if collecting:
            if raw_height is not None and accepted:
                samples.append(float(raw_height))
            cv2.putText(display,
                        f"Collecting: {len(samples)}/{COLLECTION_FRAMES}",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 255), 2)

            if len(samples) >= COLLECTION_FRAMES:
                values = np.asarray(samples, dtype=np.float32)
                median = float(np.median(values))
                deviation = np.abs(values - median)
                mad = float(np.median(deviation))
                threshold = max(3.0 * mad, 1.0)
                valid = values[deviation <= threshold]
                final_height = float(np.median(valid))
                std = float(np.std(valid))

                print(f"采集结果：{final_height:.2f}px，标准差={std:.2f}px，"
                      f"有效帧={len(valid)}/{len(values)}")
                if std <= MAX_SAMPLE_STD:
                    data.append((final_height, current_distance))
                    print(f"已记录：{final_height:.2f}px -> "
                          f"{current_distance:.1f}cm")
                else:
                    print("波动超过阈值，本次结果已丢弃，请保持A4静止后重试。")
                collecting = False
                samples.clear()

        cv2.imshow("Stable Distance Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            if collecting:
                print("正在采集中，请等待。")
            elif approx is None or stable_height is None:
                print("尚未稳定检测到完整A4。")
            elif len(stabilizer.history) < 10:
                print("滤波器尚未稳定，请稍候再按 c。")
            else:
                samples.clear()
                collecting = True
                print(f"开始采集 {current_distance:.1f}cm，请保持静止。")
        elif key in (ord('+'), ord('=')):
            current_distance += 5.0
            stabilizer.reset()
        elif key in (ord('-'), ord('_')):
            current_distance = max(1.0, current_distance - 5.0)
            stabilizer.reset()
        elif key == ord(']'):
            current_distance += 1.0
            stabilizer.reset()
        elif key == ord('['):
            current_distance = max(1.0, current_distance - 1.0)
            stabilizer.reset()
        elif key == ord('r'):
            stabilizer.reset()
            collecting = False
            samples.clear()
            print("滤波器已重置。")
        elif key == ord('d'):
            if data:
                removed = data.pop()
                print(f"已删除最后一点：{removed[0]:.2f}px -> {removed[1]:.1f}cm")
            else:
                print("没有可删除的数据。")

    save_calibration(data)
    return data


if __name__ == "__main__":
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": CAMERA_SIZE, "format": "RGB888"},
        buffer_count=4)
    picam2.configure(config)
    picam2.start()

    try:
        lock_camera_after_warmup(picam2)
        collect_distance_pixels(picam2)
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
