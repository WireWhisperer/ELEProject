import numpy as np
import argparse
import cv2
from picamera2 import Picamera2

DISTANCE_CALIBRATION = [
    (761.0, 200),(781.5, 195), (801.0, 190), (822.0, 185), (844.0, 180),
    (868.0, 175), (893.0, 170), (919.0, 165), (946.5, 160),
    (975.0, 155), (1006.5, 150), (1042.0, 145), (1075.5, 140),
    (1114.5, 135), (1156.5, 130), (1200.5, 125), (1249.0, 120),
    (1301.0, 115), (1357.0, 110), (1419.5, 105), (1488.5, 100)
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # ---------- 1. 外框检测 ----------
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    a4_outer = max(contours, key=cv2.contourArea)
    cv2.drawContours(img, [a4_outer], -1, (0, 255, 0), 2)

    peri = cv2.arcLength(a4_outer, True)
    approx = cv2.approxPolyDP(a4_outer, 0.02 * peri, True)
    if len(approx) != 4:
        cv2.putText(img, "A4 not rectangle", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img

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
        return img

    target = max(cnts_inner, key=cv2.contourArea)
    area = cv2.contourArea(target)
    if area < 50:   # 过滤噪声
        cv2.putText(img, "Too small", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img

    # ---------- 6. 形状判定与尺寸测量 ----------
    peri_obj = cv2.arcLength(target, True)
    approx_obj = cv2.approxPolyDP(target, 0.02 * peri_obj, True)
    n = len(approx_obj)

    if n == 3:   # 三角形
        sides = [np.linalg.norm(approx_obj[i][0] - approx_obj[(i+1)%3][0]) for i in range(3)]
        avg = sum(sides) / 3 / scale
        label = f"Tri={avg:.2f}cm"
        print(f"三角形边长: {avg:.2f}cm")
    elif n == 4: # 四边形（视为正方形）
        sides = [np.linalg.norm(approx_obj[i][0] - approx_obj[(i+1)%4][0]) for i in range(4)]
        avg = sum(sides) / 4 / scale
        label = f"S={avg:.2f}cm"
        print(f"正方形边长: {avg:.2f}cm")
    else:        # 圆形
        (cx, cy), radius = cv2.minEnclosingCircle(target)
        diameter = 2 * radius / scale
        label = f"D={diameter:.2f}cm"
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

    return img

if __name__ == "__main__":
    # ---------- 使用 picamera2 初始化 ----------
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1280, 720)})
    picam2.configure(config)
    picam2.start()
    print("摄像头已启动，按 'q' 键退出，按 's' 键保存当前帧")

    # 确保 current_mode 在全局作用域定义（如果尚未定义，在此处设置默认值）
    current_mode = "normal"    # 或者您需要的模式

    while True:
        # 捕获 RGB 图像（numpy 数组）
        frame_rgb = picam2.capture_array()
        # 转为 OpenCV 常用的 BGR 格式
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 调用您的处理函数
        result_img = process_frame(frame_bgr)

        # 显示
        cv2.imshow("Real-time Detection", result_img)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("saved_frame.jpg", result_img)
            print("已保存当前帧")

    # 释放资源
    picam2.stop()
    cv2.destroyAllWindows()