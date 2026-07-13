import cv2
import numpy as np
import json
import os
from collections import defaultdict
from picamera2 import Picamera2   # 新增导入

# ========== 辅助函数（保持不变） ==========
def get_contour_center(contour):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return (0, 0)
    return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_outer_frame_params(approx_outer):
    pts = approx_outer.reshape(4, 2).astype(np.float32)
    x_sorted = pts[np.argsort(pts[:, 0])]
    left_pts = x_sorted[:2]
    right_pts = x_sorted[2:]
    left_pixels = distance(left_pts[0], left_pts[1])
    right_pixels = distance(right_pts[0], right_pts[1])
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_pts = y_sorted[:2]
    bottom_pts = y_sorted[2:]
    top_mid = ((top_pts[0][0] + top_pts[1][0]) / 2, (top_pts[0][1] + top_pts[1][1]) / 2)
    bottom_mid = ((bottom_pts[0][0] + bottom_pts[1][0]) / 2, (bottom_pts[0][1] + bottom_pts[1][1]) / 2)
    axis_pixels = distance(top_mid, bottom_mid)
    return left_pixels, right_pixels, axis_pixels

def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# ========== 1. 距离-像素标定 ==========
def collect_distance_pixels(picam2):
    """采集不同距离下的外框像素高度（使用 picamera2）"""
    data = []  # 存储 (像素高度, 距离cm)
    print("\n=== 距离-像素标定模式 ===")
    print("将A4纸放在摄像头前方，确保完整可见。")
    print("按键: [c] 采集当前帧  [q] 退出标定\n")

    while True:
        # 捕获图像（RGB格式）
        frame_rgb = picam2.capture_array()
        # 转为 BGR 供 OpenCV 使用
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            a4 = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(a4, True)
            approx = cv2.approxPolyDP(a4, 0.02*peri, True)
            if len(approx) == 4:
                left_pix, right_pix, _ = get_outer_frame_params(approx)
                height_pix = (left_pix + right_pix) / 2
                cv2.drawContours(display, [approx], -1, (0,255,0), 3)
                cv2.putText(display, f"Height: {height_pix:.1f}px", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            else:
                height_pix = None
                cv2.putText(display, "Not a rectangle", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            height_pix = None
        cv2.imshow("Calibration - Distance", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if height_pix is None:
                print("未检测到有效A4外框，请调整位置或光照。")
                continue
            print(f"当前外框高度: {height_pix:.1f} 像素")
            try:
                dist = float(input("请输入当前A4纸到摄像头的实际距离 (cm): "))
            except ValueError:
                print("输入无效，跳过此帧。")
                continue
            data.append((height_pix, dist))
            print(f"已记录: 像素 {height_pix:.1f} -> 距离 {dist:.1f} cm")
            cv2.putText(display, f"Points: {len(data)}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    if data:
        data_sorted = sorted(data, key=lambda x: x[0])
        print("\n采集完成，共 {} 个点".format(len(data_sorted)))
        print("排序后数据 (像素, 距离):")
        for p, d in data_sorted:
            print(f"  ({p:.1f}, {d:.0f})")
        calib_list = [(p, int(d)) for p, d in data_sorted]
        with open("new_distance_calibration.py", "w") as f:
            f.write("# 新标定数据，格式：[(像素高度, 距离cm), ...]\n")
            f.write("DISTANCE_CALIBRATION = [\n")
            for p, d in calib_list:
                f.write(f"    ({p:.1f}, {d}),\n")
            f.write("]\n")
        print("已保存到 new_distance_calibration.py")
        with open("calibration_data.json", "w") as f:
            json.dump(calib_list, f, indent=2)
        print("备份JSON已保存到 calibration_data.json")
    else:
        print("未采集任何数据。")
    return data

# ========== 2. 补偿系数测量 ==========
def measure_compensation(picam2):
    """
    在固定距离下，测量不同旋转/倾斜状态下的尺寸误差，生成补偿系数。
    使用 picamera2 捕获图像。
    """
    print("\n=== 补偿系数测量模式 ===")
    print("请确保A4纸中心有已知尺寸的图形（正方形/圆形），且真实尺寸已知。")
    print("步骤：")
    print("  1) 将A4纸正对摄像头（正视），按 [c] 记录基准尺寸")
    print("  2) 将A4纸倾斜一定角度（如轻微倾斜），按 [c] 记录")
    print("  3) 继续尝试不同倾斜程度（中度、大角度）")
    print("  4) 按 [q] 结束并计算补偿系数")
    print("提示：记录时需输入该次测量的真实尺寸值（cm）和当前姿态名称。\n")

    samples = defaultdict(list)

    while True:
        frame_rgb = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            cv2.putText(display, "No A4 detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow("Calibration - Compensation", display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        a4 = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(a4, True)
        approx = cv2.approxPolyDP(a4, 0.02*peri, True)
        if len(approx) != 4:
            cv2.putText(display, "Not rectangle", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow("Calibration - Compensation", display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        src_pts = order_points(approx.reshape(4,2).astype(np.float32))
        scale = 20
        dst_w = int(21*scale)
        dst_h = int(29.7*scale)
        dst_pts = np.float32([[0,0],[dst_w,0],[dst_w,dst_h],[0,dst_h]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(gray, M, (dst_w, dst_h))
        warped_blur = cv2.GaussianBlur(warped, (5,5), 0)
        warped_thresh = cv2.adaptiveThreshold(warped_blur, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2)
        border_pix = int(2*scale)
        inner = warped_thresh[border_pix:dst_h-border_pix, border_pix:dst_w-border_pix]

        cnts_inner, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts_inner:
            cv2.putText(display, "No object inside", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow("Calibration - Compensation", display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        target = max(cnts_inner, key=cv2.contourArea)
        area = cv2.contourArea(target)
        if area < 50:
            cv2.putText(display, "Object too small", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow("Calibration - Compensation", display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        peri_obj = cv2.arcLength(target, True)
        approx_obj = cv2.approxPolyDP(target, 0.02*peri_obj, True)
        n = len(approx_obj)
        if n == 3:
            shape = "triangle"
            sides = [np.linalg.norm(approx_obj[i][0] - approx_obj[(i+1)%3][0]) for i in range(3)]
            measured = sum(sides)/3/scale
        elif n == 4:
            shape = "square"
            sides = [np.linalg.norm(approx_obj[i][0] - approx_obj[(i+1)%4][0]) for i in range(4)]
            measured = sum(sides)/4/scale
        else:
            shape = "circle"
            (cx, cy), radius = cv2.minEnclosingCircle(target)
            measured = 2*radius/scale

        cv2.putText(display, f"{shape}: {measured:.2f}cm", (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.imshow("Calibration - Compensation", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            try:
                real = float(input("请输入该图形的真实尺寸 (cm): "))
                pose = input("请输入姿态名称 (normal, slight, moderate, large): ").strip()
                if pose not in ["normal", "slight", "moderate", "large"]:
                    print("姿态名称无效，使用默认 'normal'")
                    pose = "normal"
                samples[pose].append((real, measured, shape))
                print(f"已记录: {pose} 姿态, 真实 {real:.2f}cm, 测量 {measured:.2f}cm")
            except ValueError:
                print("输入无效，跳过。")

    if samples:
        print("\n=== 补偿系数计算结果 ===")
        compensation = {}
        for pose, data_list in samples.items():
            ratios = [real/meas for real, meas, _ in data_list]
            avg_ratio = np.mean(ratios)
            compensation[pose] = avg_ratio
            print(f"{pose}: 平均补偿系数 = {avg_ratio:.4f} (基于 {len(data_list)} 个样本)")
        print("\n生成的补偿系数建议（单距离区间）:")
        print(compensation)
        with open("new_compensation_table.py", "w") as f:
            f.write("# 新补偿系数 (仅示例，需结合距离区间)\n")
            f.write("COMPENSATION_TABLE = {\n")
            f.write("    (100, 300): {\n")
            for pose, coeff in compensation.items():
                f.write(f"        '{pose}': {coeff:.4f},\n")
            f.write("    },\n")
            f.write("}\n")
        print("已保存到 new_compensation_table.py")
    else:
        print("未采集任何补偿数据。")

    return samples

# ========== 主程序 ==========
if __name__ == "__main__":
    # ---------- 初始化 picamera2 ----------
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1280, 720)})
    picam2.configure(config)
    picam2.start()
    print("摄像头已启动。")

    print("请选择标定模式:")
    print("  1 - 距离-像素标定 (生成 DISTANCE_CALIBRATION)")
    print("  2 - 补偿系数测量 (生成 COMPENSATION_TABLE)")
    choice = input("请输入数字 (1 或 2): ").strip()

    if choice == '1':
        collect_distance_pixels(picam2)
    elif choice == '2':
        measure_compensation(picam2)
    else:
        print("无效选择")

    picam2.stop()
    cv2.destroyAllWindows()