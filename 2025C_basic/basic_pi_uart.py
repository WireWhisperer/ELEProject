import numpy as np
import argparse
import time
import cv2
from picamera2 import Picamera2


# ==================== UART measurement output ====================
try:
    import serial
except ImportError:
    serial = None

class MeasurementUART:
    """115200 8N1 UART; one ASCII record per measurement."""
    def __init__(self, port='/dev/serial0', baudrate=115200):
        self.ser = None
        if serial is None:
            print('WARNING: pyserial not installed; UART disabled.')
            return
        try:
            self.ser = serial.Serial(port=port, baudrate=baudrate,
                                     bytesize=serial.EIGHTBITS,
                                     parity=serial.PARITY_NONE,
                                     stopbits=serial.STOPBITS_ONE,
                                     timeout=0, write_timeout=0.3)
            print(f'UART ready: {port}, {baudrate} baud, 8N1')
        except (serial.SerialException, OSError) as exc:
            print(f'WARNING: UART disabled: {exc}')

    def send(self, distance_cm, size_cm, object_type):
        if self.ser is None or distance_cm is None or size_cm is None:
            return False
        line = f'D={float(distance_cm):.1f},x={float(size_cm):.2f},type={object_type}\r\n'
        try:
            self.ser.write(line.encode('ascii'))
            self.ser.flush()
            print('UART TX:', line.strip())
            return True
        except (serial.SerialException, OSError) as exc:
            print(f'UART send failed: {exc}')
            return False

    def close(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()


DISTANCE_CALIBRATION = [
    (232.21,200),(237.94,195),(243.98,190),(250.35,185),(257.06,180),
    (264.21,175),(271.52,170),(279.79,165),(288.04,160),(297.37,155),
    (306.64,150),(318.05,145),(328.08,140),(341.49,135),(353.16,130),
    (367.41,125),(383.57,120),(404.31,115),(421.94,110),(441.94,105),(464.26,100)
]
DISTANCE_CALIBRATION.sort(key=lambda x: x[0])
CALIB_PIXELS    = np.array([p for p, _ in DISTANCE_CALIBRATION], dtype=np.float32)
CALIB_DISTANCES = np.array([d for _, d in DISTANCE_CALIBRATION], dtype=np.float32)

COMPENSATION_TABLE = {
    (100, 120): { 'normal':1.006, 'normal2':1.0035, 'slight':1.005, 'moderate':1.000, 'large':1.000, 'circle':0.999 },
    (120, 140): { 'normal':1.005, 'normal2':1.0035, 'slight':1.004, 'moderate':1.009, 'large':1.009, 'circle':0.999 },
    (140, 160): { 'normal':1.005, 'normal2':1.0035, 'slight':1.003, 'moderate':1.008, 'large':1.008, 'circle':0.999 },
    (160, 180): { 'normal':1.005, 'normal2':1.0034, 'slight':1.002, 'moderate':1.007, 'large':1.007, 'circle':0.999 },
    (180, 200): { 'normal':1.0053, 'normal2':1.004, 'slight':1.001, 'moderate':1.006, 'large':1.006, 'circle':0.999 },
    (200, 300): { 'normal':1.006, 'normal2':1.0045, 'slight':1.000, 'moderate':1.005, 'large':1.005, 'circle':0.999 },
}

def estimate_distance(pixel_height):
    """根据外框高度插值估算距离（cm）"""
    if pixel_height <= CALIB_PIXELS[0]:
        k = (CALIB_DISTANCES[1] - CALIB_DISTANCES[0]) / (CALIB_PIXELS[1] - CALIB_PIXELS[0])
        return CALIB_DISTANCES[0] + k * (pixel_height - CALIB_PIXELS[0])
    elif pixel_height >= CALIB_PIXELS[-1]:
        k = (CALIB_DISTANCES[-1] - CALIB_DISTANCES[-2]) / (CALIB_PIXELS[-1] - CALIB_PIXELS[-2])
        return CALIB_DISTANCES[-1] + k * (pixel_height - CALIB_PIXELS[-1])
    else:
        return float(np.interp(pixel_height, CALIB_PIXELS, CALIB_DISTANCES))

def get_compensation(distance_cm):
    """根据测得距离选择对应段的补偿系数"""
    for (d_min, d_max), tbl in COMPENSATION_TABLE.items():
        if d_min <= distance_cm < d_max:
            return tbl
    # 超出范围时用最近段
    return COMPENSATION_TABLE[max(COMPENSATION_TABLE.keys(), key=lambda r: r[0])]

    
def get_contour_center(contour):
    """计算轮廓中心点"""
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return (0, 0)
    return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
	
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_outer_frame_params(approx_outer):
    """获取外框关键参数：左右边像素数、中轴像素数"""
    pts = approx_outer.reshape(4, 2).astype(np.float32)
    #数组4行2列，每行一个点的坐标

    # 按x坐标排序区分左右边
    x_sorted = pts[np.argsort(pts[:, 0])]
    left_pts = x_sorted[:2]  # 左边两点
    right_pts = x_sorted[2:]  # 右边两点

    # 计算左右边像素数（竖直方向长度）
    left_pixels = distance(left_pts[0], left_pts[1])
    right_pixels = distance(right_pts[0], right_pts[1])

    # 按y坐标排序区分上下边
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_pts = y_sorted[:2]  # 上边两点
    bottom_pts = y_sorted[2:]  # 下边两点

    # 计算上下边中点（中轴端点）
    top_mid = ((top_pts[0][0] + top_pts[1][0]) / 2, (top_pts[0][1] + top_pts[1][1]) / 2)
    bottom_mid = ((bottom_pts[0][0] + bottom_pts[1][0]) / 2, (bottom_pts[0][1] + bottom_pts[1][1]) / 2)

    # 计算中轴像素数（上下中点连线）
    axis_pixels = distance(top_mid, bottom_mid)

    return left_pixels, right_pixels, axis_pixels

def get_inner_square_axis(approx_inner):
    """获取内部正方形中轴像素数（上下边中点连线）"""
    pts = approx_inner.reshape(4, 2).astype(np.float32)

    # 按y坐标排序区分上下边
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_pts = y_sorted[:2]  # 上边两点
    bottom_pts = y_sorted[2:]  # 下边两点

    # 计算上下边中点
    top_mid = ((top_pts[0][0] + top_pts[1][0]) / 2, (top_pts[0][1] + top_pts[1][1]) / 2)
    bottom_mid = ((bottom_pts[0][0] + bottom_pts[1][0]) / 2, (bottom_pts[0][1] + bottom_pts[1][1]) / 2)

    # 计算中轴像素数
    return distance(top_mid, bottom_mid)

def order_points(pts):
    """将4个点排序为[左上, 右上, 右下, 左下]（顺时针）"""
    rect = np.zeros((4, 2), dtype=np.float32)
    # 按x+y排序，最小为左上，最大为右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 按x-y排序，最小为右上，最大为左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def process_frame(frame):
    img = frame.copy()
    measurement = {
        'valid': False, 'distance_cm': None, 'size_cm': None,
        'object_type': 'unknown'
    }
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # ---------- 1. 外框检测 ----------
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, measurement
    a4_outer = max(contours, key=cv2.contourArea)
    cv2.drawContours(img, [a4_outer], -1, (0, 255, 0), 2)

    peri = cv2.arcLength(a4_outer, True)
    approx = cv2.approxPolyDP(a4_outer, 0.02 * peri, True)
    if len(approx) != 4:
        cv2.putText(img, "A4 not rectangle", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img, measurement

    # ---------- 2. 测距（原功能） ----------
    # 计算外框高度，用于距离估算
    left_pix, right_pix, axis_pix = get_outer_frame_params(approx)
    if current_mode == "normal":
        outer_height = (left_pix + right_pix) / 2
    else:
        outer_height = axis_pix

    distance_cm = 0.0
    if outer_height > 0:
        distance_cm = estimate_distance(outer_height)
        comp = get_compensation(distance_cm)  # 补偿系数（可后续使用）
        # 显示距离在图像上
        cv2.putText(img, f"Dist: {distance_cm:.1f}cm", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        print(f"外框高度: {outer_height:.1f}像素 -> 距离估算: {distance_cm:.1f}cm")

    # ---------- 3. 透视变换到标准平面 ----------
    src_pts = order_points(approx.reshape(4, 2).astype(np.float32))
    scale = 20  # 每厘米像素数
    dst_width = int(21 * scale)
    dst_height = int(29.7 * scale)
    dst_pts = np.float32([[0, 0], [dst_width, 0],
                          [dst_width, dst_height], [0, dst_height]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = np.linalg.inv(M)  # 用于逆变换

    warped = cv2.warpPerspective(gray, M, (dst_width, dst_height))
    warped_blur = cv2.GaussianBlur(warped, (5, 5), 0)
    warped_thresh = cv2.adaptiveThreshold(warped_blur, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)

    # ---------- 4. 裁剪内部区域（去除2cm边框） ----------
    border_pix = int(2 * scale)   # 2cm 对应像素
    inner = warped_thresh[border_pix : dst_height - border_pix,
                          border_pix : dst_width - border_pix]

    # ---------- 5. 寻找内部目标 ----------
    cnts_inner, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_inner:
        cv2.putText(img, "No object", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img, measurement

    target = max(cnts_inner, key=cv2.contourArea)
    area = cv2.contourArea(target)
    if area < 50:   # 过滤噪声
        cv2.putText(img, "Too small", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img, measurement

    # ---------- 6. 形状判定与尺寸测量 ----------
    peri_obj = cv2.arcLength(target, True)
    approx_obj = cv2.approxPolyDP(target, 0.02 * peri_obj, True)
    n = len(approx_obj)

    size_cm = None
    object_type = 'unknown'
    if n == 3:   # 三角形
        sides = [np.linalg.norm(approx_obj[i][0] - approx_obj[(i+1)%3][0]) for i in range(3)]
        avg = sum(sides) / 3 / scale
        size_cm = float(avg)
        object_type = 'triangle'
        label = f"Tri={avg:.2f}cm"
        print(f"三角形边长: {avg:.2f}cm")
    elif n == 4: # 四边形（视为正方形）
        sides = [np.linalg.norm(approx_obj[i][0] - approx_obj[(i+1)%4][0]) for i in range(4)]
        avg = sum(sides) / 4 / scale
        size_cm = float(avg)
        object_type = 'square'
        label = f"S={avg:.2f}cm"
        print(f"正方形边长: {avg:.2f}cm")
    else:        # 圆形
        (cx, cy), radius = cv2.minEnclosingCircle(target)
        diameter = 2 * radius / scale
        size_cm = float(diameter)
        object_type = 'circle'
        label = f"DIA={diameter:.2f}cm"
        print(f"圆形直径: {diameter:.2f}cm")

    # ---------- 7. 在原始图像上绘制蓝色目标轮廓 ----------
    # 将目标轮廓点从裁剪区域坐标转换回原始图像坐标
    # 步骤：从裁剪区域坐标 → warped图像坐标 → 原始图像坐标
    if n > 0:
        # 获取轮廓点（相对于裁剪区域）
        pts = target.reshape(-1, 2).astype(np.float32)
        # 加上边框偏移
        pts[:, 0] += border_pix
        pts[:, 1] += border_pix
        # 变换到原始图像
        pts_orig = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), M_inv)
        pts_orig = pts_orig.reshape(-1, 2).astype(np.int32)
        # 绘制蓝色轮廓
        cv2.polylines(img, [pts_orig], isClosed=True, color=(255, 0, 0), thickness=2)

    # 在图像上显示测量结果（位置固定）
    cv2.putText(img, label, (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    measurement = {
        'valid': distance_cm > 0 and size_cm is not None,
        'distance_cm': float(distance_cm),
        'size_cm': float(size_cm) if size_cm is not None else None,
        'object_type': object_type
    }
    return img, measurement

if __name__ == "__main__":
    # 正式模式：实时预览；按空格一键触发7帧融合；串口只发送最终结果一次。
    current_mode = "normal"
    SAMPLE_COUNT = 7
    MIN_VALID_COUNT = 4

    uart = MeasurementUART('/dev/serial0', 115200)
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(2.0)
    print("实时预览：空格键测量，s保存，q退出")

    last_display = None
    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            preview = frame_bgr.copy()
            cv2.putText(preview, "SPACE: measure   Q: quit", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Basic Measurement", preview if last_display is None else last_display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                cv2.imwrite('saved_frame.jpg', preview if last_display is None else last_display)
                continue
            if key != ord(' '):
                # 测量结果锁屏约1秒后恢复实时预览。
                if last_display is not None and time.monotonic() - display_since > 1.0:
                    last_display = None
                continue

            valid = []
            output = preview
            t0 = time.perf_counter()
            for _ in range(SAMPLE_COUNT):
                rgb = picam2.capture_array()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                output, m = process_frame(bgr)
                if (m['valid'] and 90.0 <= m['distance_cm'] <= 210.0
                        and 9.0 <= m['size_cm'] <= 17.0):
                    valid.append(m)

            if len(valid) < MIN_VALID_COUNT:
                cv2.putText(output, f"FAILED valid={len(valid)}/{SAMPLE_COUNT}",
                            (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)
                print(f'Measurement failed: valid={len(valid)}/{SAMPLE_COUNT}')
            else:
                # 形状多数表决；只融合多数形状的帧。
                kinds = [m['object_type'] for m in valid]
                final_type = max(set(kinds), key=kinds.count)
                same = [m for m in valid if m['object_type'] == final_type]
                if len(same) < 3:
                    same = valid
                final_d = float(np.median([m['distance_cm'] for m in same]))
                final_x = float(np.median([m['size_cm'] for m in same]))
                cv2.rectangle(output, (10, 130), (560, 230), (0, 0, 0), -1)
                cv2.putText(output, f"FINAL D={final_d:.1f}cm", (20, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(output, f"FINAL x={final_x:.2f}cm {final_type}", (20, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                uart.send(final_d, final_x, final_type)
                print(f'FINAL: D={final_d:.1f}cm, x={final_x:.2f}cm, type={final_type}, '
                      f'valid={len(valid)}/{SAMPLE_COUNT}, time={time.perf_counter()-t0:.2f}s')
            last_display = output
            display_since = time.monotonic()
    finally:
        picam2.stop()
        uart.close()
        cv2.destroyAllWindows()
