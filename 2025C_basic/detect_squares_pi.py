import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple
from picamera2 import Picamera2   # 新增导入

# ==================== 距离标定与补偿（保留原功能） ====================
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
    """根据测得距离选择补偿系数（本版本未实际使用）"""
    for (d_min, d_max), tbl in COMPENSATION_TABLE.items():
        if d_min <= distance_cm < d_max:
            return tbl
    return COMPENSATION_TABLE[max(COMPENSATION_TABLE.keys(), key=lambda r: r[0])]

# ==================== 辅助几何函数 ====================
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_outer_frame_params(approx_outer):
    """获取外框左右边像素数（用于测距）"""
    pts = approx_outer.reshape(4, 2).astype(np.float32)
    x_sorted = pts[np.argsort(pts[:, 0])]
    left_pts = x_sorted[:2]
    right_pts = x_sorted[2:]
    left_pixels = distance(left_pts[0], left_pts[1])
    right_pixels = distance(right_pts[0], right_pts[1])
    return left_pixels, right_pixels

def order_points(pts):
    """将4个点排序为[左上, 右上, 右下, 左下]（顺时针）"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_contour_center(contour):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return (0, 0)
    return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

# ==================== 正方形数据结构与全局管理 ====================
@dataclass
class SquareInfo:
    vertices: List[Tuple[int, int]]
    center: Tuple[int, int] = None
    avg_side_length: float = 0.0
    detection_method: str = ""
    is_valid: bool = True

    def __post_init__(self):
        if not self.center and self.vertices:
            x_coords = [v[0] for v in self.vertices]
            y_coords = [v[1] for v in self.vertices]
            self.center = (int(sum(x_coords) / len(x_coords)),
                           int(sum(y_coords) / len(y_coords)))

all_squares: List[SquareInfo] = []

def reset_square_collection():
    global all_squares
    all_squares = []

def add_square_to_collection(vertices, avg_side, method, center=None):
    square = SquareInfo(vertices=vertices, avg_side_length=avg_side,
                        detection_method=method, center=center)
    all_squares.append(square)
    return square

def get_all_squares():
    return all_squares

# ==================== 角度与斜率计算 ====================
def angle_between_vectors(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    n1 = (v1[0]**2 + v1[1]**2)**0.5
    n2 = (v2[0]**2 + v2[1]**2)**0.5
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = dot / (n1*n2)
    cos = max(min(cos, 1.0), -1.0)
    return abs(np.degrees(np.arccos(cos)))

def calculate_slope(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if abs(dx) < 1e-6:
        return float('inf')
    return dy / dx

def angle_between_slopes(s1, s2):
    if s1 == float('inf'): s1 = 1e10
    if s2 == float('inf'): s2 = 1e10
    tan = abs((s2 - s1) / (1 + s1*s2))
    return np.degrees(np.arctan(tan))

def sort_square_vertices(p1, p2, p3, p4):
    center = np.mean([p1, p2, p3, p4], axis=0)
    def angle(p):
        return np.arctan2(p[1]-center[1], p[0]-center[0])
    pts = [p1, p2, p3, p4]
    pts.sort(key=angle, reverse=True)
    return pts

# ==================== 正方形检测核心算法（完美边 + 落单角） ====================
# 以下函数均保持原 detect_squares_V1 的实现不变
def calculate_vectors(p_prev, p_curr, p_next):
    v_prev = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
    v_next = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
    return v_prev, v_next

def detect_right_angles(contour, img, return_details=False):
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.003 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = approx.reshape(-1, 2)
    num = len(vertices)
    if num < 3:
        return img, [] 
    right_angles = []
    for i in range(num):
        p_curr = vertices[i]
        p_prev = vertices[(i-1)%num]
        p_next = vertices[(i+1)%num]
        v_prev, v_next = calculate_vectors(p_prev, p_curr, p_next)
        angle = angle_between_vectors(v_prev, v_next)
        if 82 <= angle <= 97:
            slope_prev = calculate_slope(p_prev, p_curr)
            slope_next = calculate_slope(p_curr, p_next)
            right_angles.append({
                'point': tuple(p_curr),
                'index': i,
                'angle': angle,
                'slopes': [slope_prev, slope_next],
                'neighbors': [tuple(p_prev), tuple(p_next)]
            })
            if not return_details:
                cv2.circle(img, tuple(p_curr), 8, (0,0,255), -1)
                cv2.putText(img, f"{angle:.1f}°", (p_curr[0]+10, p_curr[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    if not return_details:
        cv2.putText(img, f"直角点: {len(right_angles)}", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    if return_details:
        return img, right_angles
    return img, right_angles

def check_direction_consistency(p0, p1, p2, p3):
    s_prev = calculate_slope(p0, p1)
    s_curr = calculate_slope(p1, p2)
    s_next = calculate_slope(p2, p3)
    def is_on_side(point, line_p1, line_p2):
        return (line_p2[0]-line_p1[0])*(point[1]-line_p1[1]) - \
               (line_p2[1]-line_p1[1])*(point[0]-line_p1[0])
    prev_ref = ((p0[0]+p1[0])//2, (p0[1]+p1[1])//2)
    next_ref = ((p2[0]+p3[0])//2, (p2[1]+p3[1])//2)
    prev_side = np.sign(is_on_side(prev_ref, p1, p2))
    next_side = np.sign(is_on_side(next_ref, p1, p2))
    same_side = (prev_side == next_side) and (prev_side != 0)
    return same_side, prev_side

def check_black_region(img, p1, p2, prev_side, step_pixels=30, threshold=0.70):
    curr_dx = p2[0] - p1[0]
    curr_dy = p2[1] - p1[1]
    edge_length = np.hypot(curr_dx, curr_dy)
    if edge_length < 1e-6:
        return False, 0.0
    # 采样点
    if edge_length < 50:
        points = [((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)]
        required = 1
    elif edge_length < 100:
        points = [(p1[0]+int(curr_dx*i/3), p1[1]+int(curr_dy*i/3)) for i in [1,2]]
        required = 1
    else:
        points = [(p1[0]+int(curr_dx*i/6), p1[1]+int(curr_dy*i/6)) for i in [1,2,3,4,5]]
        required = 3
    raw_normal = (-curr_dy, curr_dx)
    norm = np.hypot(raw_normal[0], raw_normal[1])
    unit = (raw_normal[0]/norm, raw_normal[1]/norm)
    final = (unit[0]*prev_side, unit[1]*prev_side)
    passed = 0
    ratios = []
    h, w = img.shape[:2]
    for (x,y) in points:
        tx = int(x + final[0]*step_pixels)
        ty = int(y + final[1]*step_pixels)
        tx = max(0, min(w-1, tx))
        ty = max(0, min(h-1, ty))
        # 取区域
        size = 20
        x1 = max(0, tx-size//2)
        x2 = min(w, tx+size//2+1)
        y1 = max(0, ty-size//2)
        y2 = min(h, ty+size//2+1)
        region = img[y1:y2, x1:x2]
        if len(region.shape)==3:
            region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(region, 100, 255, cv2.THRESH_BINARY)
        total = binary.size
        if total==0:
            ratio=0.0
        else:
            dark = total - cv2.countNonZero(binary)
            ratio = dark / total
        ratios.append(ratio)
        if ratio >= threshold:
            passed += 1
    return passed >= required, np.mean(ratios) if ratios else 0.0

def infer_square_points(p1, p2, prev_side):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    norm = np.hypot(dx, dy)
    if norm == 0:
        return None, None
    perp_dx = -dy / norm * prev_side
    perp_dy = dx / norm * prev_side
    length = norm
    p3 = (int(p1[0] + perp_dx*length), int(p1[1] + perp_dy*length))
    p4 = (int(p2[0] + perp_dx*length), int(p2[1] + perp_dy*length))
    return p3, p4

def find_matching_node(point, all_nodes, threshold=20):
    best_dist = float('inf')
    best_idx = None
    best_node = None
    for idx, node in enumerate(all_nodes):
        d = np.hypot(node[0]-point[0], node[1]-point[1])
        if d <= threshold and d < best_dist:
            best_dist = d
            best_idx = idx
            best_node = node
    return best_idx, best_node

def get_square_side_lengths(p1, p2, p3, p4):
    s1 = distance(p1,p2); s2 = distance(p2,p3); s3 = distance(p3,p4); s4 = distance(p4,p1)
    return (s1+s2+s3+s4)/4

def is_edge_part_of_existing_square(edge, square_groups, threshold=30):
    p1,p2 = edge
    for sq in square_groups:
        p1_in = any(np.hypot(p1[0]-sp[0], p1[1]-sp[1]) <= threshold for sp in sq)
        p2_in = any(np.hypot(p2[0]-sp[0], p2[1]-sp[1]) <= threshold for sp in sq)
        if p1_in and p2_in:
            return True
    return False

def detect_perfect_edges_and_squares(contour, img, step_pixels=30):
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.003 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    original_vertices = approx.reshape(-1, 2)
    num_original = len(original_vertices)
    all_vertices = original_vertices.copy().tolist()
    processed_edges = set()
    square_groups = []
    vertex_status = {i: "未核验" for i in range(num_original)}
    edge_status = {}
    cv2.drawContours(img, [approx], -1, (147,112,219), 2)
    for i in range(num_original):
        cv2.putText(img, f"{i+1}", (original_vertices[i][0]+5, original_vertices[i][1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.circle(img, tuple(original_vertices[i]), 5, (0,255,0), -1)
    perfect_edges = []
    for i in range(num_original):
        p1_idx = i
        p2_idx = (i+1)%num_original
        p1 = original_vertices[p1_idx]
        p2 = original_vertices[p2_idx]
        edge_name = f"l{p1_idx+1}l{p2_idx+1}"
        p0 = original_vertices[(i-1)%num_original]
        p3 = original_vertices[(i+2)%num_original]
        prev_len = np.hypot(p1[0]-p0[0], p1[1]-p0[1])
        next_len = np.hypot(p3[0]-p2[0], p3[1]-p2[1])
        s_prev = calculate_slope(p0, p1)
        s_curr = calculate_slope(p1, p2)
        s_next = calculate_slope(p2, p3)
        angle_p1 = angle_between_slopes(s_prev, s_curr)
        angle_p2 = angle_between_slopes(s_curr, s_next)
        is_right_p1 = 80 <= angle_p1 <= 100 if prev_len < 100 else 85 <= angle_p1 <= 95
        is_right_p2 = 80 <= angle_p2 <= 100 if next_len < 100 else 85 <= angle_p2 <= 95
        dir_consistent, prev_side = check_direction_consistency(p0, p1, p2, p3)
        black_valid = False
        if is_right_p1 and is_right_p2 and dir_consistent:
            black_valid, _ = check_black_region(img, p1, p2, prev_side, step_pixels=step_pixels)
        if is_right_p1 and is_right_p2 and dir_consistent and black_valid:
            perfect_edges.append((p1_idx, p2_idx, p1, p2, prev_side))
            edge_status[edge_name] = "已核验"
            vertex_status[p1_idx] = "已核验"
            vertex_status[p2_idx] = "已核验"
            cv2.line(img, tuple(p1), tuple(p2), (255,0,0), 3)
        else:
            edge_status[edge_name] = "未核验"
            cv2.line(img, tuple(p1), tuple(p2), (0,0,255), 3)
        edge_mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2 - 10)
        cv2.putText(img, edge_name, edge_mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    for edge in perfect_edges:
        p1_idx, p2_idx, p1, p2, prev_side = edge
        edge_key = frozenset([p1_idx, p2_idx])
        if edge_key in processed_edges or is_edge_part_of_existing_square((p1,p2), square_groups):
            processed_edges.add(edge_key)
            continue
        p3, p4 = infer_square_points(p1, p2, prev_side)
        if p3 is None or p4 is None:
            continue
        match3_idx, match3 = find_matching_node(p3, all_vertices)
        match4_idx, match4 = find_matching_node(p4, all_vertices)
        original_matched_indices = set([p1_idx, p2_idx])
        if match3_idx is not None and match3_idx < num_original:
            p3_final = match3
            original_matched_indices.add(match3_idx)
            vertex_status[match3_idx] = "已核验"
        else:
            all_vertices.append(p3)
            p3_final = p3
            cv2.circle(img, p3_final, 5, (0,255,255), -1)
        if match4_idx is not None and match4_idx < num_original:
            p4_final = match4
            original_matched_indices.add(match4_idx)
            vertex_status[match4_idx] = "已核验"
        else:
            all_vertices.append(p4)
            p4_final = p4
            cv2.circle(img, p4_final, 5, (0,255,255), -1)
        sorted_points = sort_square_vertices(p1, p2, p3_final, p4_final)
        s1,s2,s3,s4 = sorted_points
        square = (s1,s2,s3,s4)
        square_groups.append(square)
        cv2.line(img, tuple(s1), tuple(s2), (255,255,0), 2)
        cv2.line(img, tuple(s2), tuple(s3), (255,255,0), 2)
        cv2.line(img, tuple(s3), tuple(s4), (255,255,0), 2)
        cv2.line(img, tuple(s4), tuple(s1), (255,255,0), 2)
        avg_len = get_square_side_lengths(*square)
        add_square_to_collection(vertices=[(s1[0],s1[1]),(s2[0],s2[1]),(s3[0],s3[1]),(s4[0],s4[1])],
                                 avg_side=avg_len, method="perfect_edge")
        mid_square = ((s1[0]+s3[0])//2, (s1[1]+s3[1])//2)
        cv2.putText(img, f"= {avg_len:.1f}", mid_square,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        for idx in original_matched_indices:
            prev_idx = (idx-1)%num_original
            next_idx = (idx+1)%num_original
            vertex_status[prev_idx] = "已核验"
            vertex_status[next_idx] = "已核验"
        for i in range(num_original):
            pi = i
            pj = (i+1)%num_original
            en = f"l{pi+1}l{pj+1}"
            if vertex_status[pi] == "已核验" and vertex_status[pj] == "已核验":
                edge_status[en] = "已核验"
                cv2.line(img, tuple(original_vertices[pi]), tuple(original_vertices[pj]), (255,0,0), 3)
        square_edges = [edge_key,
                        frozenset([p2_idx, match3_idx]) if match3_idx is not None and match3_idx < num_original else None,
                        frozenset([match3_idx, match4_idx]) if match3_idx is not None and match4_idx is not None and match3_idx < num_original and match4_idx < num_original else None,
                        frozenset([match4_idx, p1_idx]) if match4_idx is not None and match4_idx < num_original else None]
        for se in square_edges:
            if se is not None:
                processed_edges.add(se)
    unverified_vertices = [i for i, status in vertex_status.items() if status == "未核验" and i < num_original]
    if unverified_vertices:
        cv2.putText(img, "未核验: "+",".join(map(str,[i+1 for i in unverified_vertices])),
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    return img, vertex_status, edge_status, square_groups, original_vertices

def extend_line(p1, p2, length_ratio=2.0):
    dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
    return (int(p2[0]+dx*length_ratio), int(p2[1]+dy*length_ratio))

def line_intersection(p1,p2,p3,p4):
    x1,y1 = p1; x2,y2 = p2; x3,y3 = p3; x4,y4 = p4
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0: return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (int(x1+t*(x2-x1)), int(y1+t*(y2-y1)))
    return None

def check_black_pixel_percentage(img, point, radius=25):
    h,w = img.shape[:2]
    x,y = point
    if x<0 or x>=w or y<0 or y>=h:
        return 0.0
    x1 = max(0, x-radius); x2 = min(w, x+radius+1)
    y1 = max(0, y-radius); y2 = min(h, y+radius+1)
    if x1>=x2 or y1>=y2:
        return 0.0
    region = img[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    if len(region.shape)==3:
        region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(region, 100, 255, cv2.THRESH_BINARY)
    total = binary.size
    if total == 0: return 0.0
    dark = total - cv2.countNonZero(binary)
    return dark / total

def check_slope_match(slopes1, slopes2, threshold=15):
    if len(slopes1)!=2 or len(slopes2)!=2: return False
    match1 = (angle_between_slopes(slopes1[0], slopes2[1]) < threshold and
              angle_between_slopes(slopes1[1], slopes2[0]) < threshold)
    match2 = (angle_between_slopes(slopes1[0], slopes2[0]) < threshold and
              angle_between_slopes(slopes1[1], slopes2[1]) < threshold)
    return match1 or match2

def orphan_corner_method(img, contour, unverified_indices, original_vertices, square_groups):
    # 将未核验索引转为 Python 整数集合，避免 NumPy 类型问题
    unverified_set = {int(i) for i in unverified_indices}
    # 获取所有直角点
    _, all_right_angles = detect_right_angles(contour, img, return_details=True)
    # 筛选出未核验的直角点，索引也转为 int
    orphan_corners = [
        corner for corner in all_right_angles 
        if int(corner['index']) in unverified_set
    ]
    if len(orphan_corners) < 2:
        return img, square_groups
    cv2.putText(img, f"落单角数量: {len(orphan_corners)}", (20,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    ORPHAN_ANGLE_THRESHOLD = 15
    SIDE_LENGTH_TOLERANCE = 0.3
    for i in range(len(orphan_corners)):
        for j in range(i+1, len(orphan_corners)):
            corner1 = orphan_corners[i]
            corner2 = orphan_corners[j]
            if not check_slope_match(corner1['slopes'], corner2['slopes'], ORPHAN_ANGLE_THRESHOLD):
                continue
            cv2.circle(img, corner1['point'], 10, (255,0,255), 2)
            cv2.circle(img, corner2['point'], 10, (255,0,255), 2)
            cv2.line(img, corner1['point'], corner2['point'], (255,0,255), 1)
            p1 = corner1['point']; p1a, p1b = corner1['neighbors']
            p2 = corner2['point']; p2a, p2b = corner2['neighbors']
            slope1a = calculate_slope(p1, p1a); slope1b = calculate_slope(p1, p1b)
            slope2a = calculate_slope(p2, p2a); slope2b = calculate_slope(p2, p2b)
            ext_p1a = extend_line(p1, p1a, 3.0); ext_p1b = extend_line(p1, p1b, 3.0)
            ext_p2a = extend_line(p2, p2a, 3.0); ext_p2b = extend_line(p2, p2b, 3.0)
            # 配对正负斜率射线
            def get_pos_neg(ray1, ray2):
                # 简化处理：斜率正负配对
                slopes = [slope1a, slope1b]
                if (slope1a<0 and slope1b>0) or (slope1a>0 and slope1b<0):
                    neg = (p1, ext_p1a) if slope1a<0 else (p1, ext_p1b)
                    pos = (p1, ext_p1b) if slope1a<0 else (p1, ext_p1a)
                else:
                    neg = (p1, ext_p1a) if abs(slope1a)>abs(slope1b) else (p1, ext_p1b)
                    pos = (p1, ext_p1b) if abs(slope1a)>abs(slope1b) else (p1, ext_p1a)
                return neg, pos
            neg1, pos1 = get_pos_neg(slope1a, slope1b)
            neg2, pos2 = get_pos_neg(slope2a, slope2b)
            intersect1 = line_intersection(neg1[0], neg1[1], pos2[0], pos2[1])
            intersect2 = line_intersection(pos1[0], pos1[1], neg2[0], neg2[1])
            valid = []
            if intersect1: valid.append(intersect1)
            if intersect2: valid.append(intersect2)
            if len(valid) >= 2:
                # 对角点场景
                i1, i2 = valid[:2]
                sides = [distance(p1,i1), distance(i1,p2), distance(p2,i2), distance(i2,p1)]
                max_len = max(sides); min_len = min(sides)
                if (max_len - min_len)/max_len <= SIDE_LENGTH_TOLERANCE:
                    square_pts = [p1, i1, p2, i2]
                    sorted_pts = sort_square_vertices(*square_pts)
                    avg_len = get_square_side_lengths(*sorted_pts)
                    cv2.line(img, sorted_pts[0], sorted_pts[1], (128,0,128), 2)
                    cv2.line(img, sorted_pts[1], sorted_pts[2], (128,0,128), 2)
                    cv2.line(img, sorted_pts[2], sorted_pts[3], (128,0,128), 2)
                    cv2.line(img, sorted_pts[3], sorted_pts[0], (128,0,128), 2)
                    mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
                    cv2.putText(img, f"= {avg_len:.1f}", mid,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
                    add_square_to_collection(vertices=sorted_pts, avg_side=avg_len, method="orphan_corner")
                    square_groups.append(sorted_pts)
                    continue
            # 同边点场景
            side_length = distance(p1, p2)
            dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
            norm = np.hypot(dx, dy)
            if norm == 0: continue
            perp_dx = -dy/norm; perp_dy = dx/norm
            p3_1 = (int(p1[0]+perp_dx*side_length), int(p1[1]+perp_dy*side_length))
            p4_1 = (int(p2[0]+perp_dx*side_length), int(p2[1]+perp_dy*side_length))
            p3_2 = (int(p1[0]-perp_dx*side_length), int(p1[1]-perp_dy*side_length))
            p4_2 = (int(p2[0]-perp_dx*side_length), int(p2[1]-perp_dy*side_length))
            black_thresh = 0.3
            g1_valid = (check_black_pixel_percentage(img, p3_1) >= black_thresh and
                        check_black_pixel_percentage(img, p4_1) >= black_thresh)
            g2_valid = (check_black_pixel_percentage(img, p3_2) >= black_thresh and
                        check_black_pixel_percentage(img, p4_2) >= black_thresh)
            if g1_valid and g2_valid:
                avg1 = (check_black_pixel_percentage(img,p3_1)+check_black_pixel_percentage(img,p4_1))/2
                avg2 = (check_black_pixel_percentage(img,p3_2)+check_black_pixel_percentage(img,p4_2))/2
                selected = (p3_1, p4_1) if avg1 > avg2 else (p3_2, p4_2)
            elif g1_valid:
                selected = (p3_1, p4_1)
            elif g2_valid:
                selected = (p3_2, p4_2)
            else:
                avg1 = (check_black_pixel_percentage(img,p3_1)+check_black_pixel_percentage(img,p4_1))/2
                avg2 = (check_black_pixel_percentage(img,p3_2)+check_black_pixel_percentage(img,p4_2))/2
                selected = (p3_1, p4_1) if avg1 > avg2 else (p3_2, p4_2)
            square_pts = [p1, p2, selected[1], selected[0]]
            sorted_pts = sort_square_vertices(*square_pts)
            avg_len = get_square_side_lengths(*sorted_pts)
            cv2.line(img, sorted_pts[0], sorted_pts[1], (128,0,128), 2)
            cv2.line(img, sorted_pts[1], sorted_pts[2], (128,0,128), 2)
            cv2.line(img, sorted_pts[2], sorted_pts[3], (128,0,128), 2)
            cv2.line(img, sorted_pts[3], sorted_pts[0], (128,0,128), 2)
            mid = ((p1[0]+selected[0][0])//2, (p1[1]+selected[0][1])//2)
            cv2.putText(img, f"= {avg_len:.1f}", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
            add_square_to_collection(vertices=sorted_pts, avg_side=avg_len, method="orphan_corner")
            square_groups.append(sorted_pts)
    return img, square_groups

# ==================== 主处理函数（整合透视变换） ====================
def process_frame(frame):
    """ 
    处理单帧：
      1. 检测 A4 外框，透视变换校正图像
      2. 裁剪内部区域（去除2cm边框）
      3. 在内部区域检测正方形（使用原来的完美边+落单角法）
      4. 将检测到的正方形顶点逆变换回原图并绘制
    """
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # ---------- 外框检测 ----------
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cv2.putText(img, "No A4 paper", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return img
    a4_outer = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(a4_outer, True)
    approx = cv2.approxPolyDP(a4_outer, 0.02*peri, True)
    if len(approx) != 4:
        cv2.putText(img, "Not rectangle", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return img
    cv2.drawContours(img, [a4_outer], -1, (0,255,0), 2)

    # ---------- 测距 ----------
    left_pix, right_pix = get_outer_frame_params(approx)
    outer_height = (left_pix + right_pix) / 2.0
    distance_cm = estimate_distance(outer_height) if outer_height > 0 else 0.0
    cv2.putText(img, f"Dist: {distance_cm:.1f}cm", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

    # ---------- 透视变换 ----------
    src_pts = order_points(approx.reshape(4,2).astype(np.float32))
    scale = 20   # 像素/厘米，可根据需要调整
    dst_w = int(21 * scale)
    dst_h = int(29.7 * scale)
    dst_pts = np.float32([[0,0], [dst_w,0], [dst_w,dst_h], [0,dst_h]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = np.linalg.inv(M)

    warped = cv2.warpPerspective(gray, M, (dst_w, dst_h))
    warped_blur = cv2.GaussianBlur(warped, (5,5), 0)
    warped_thresh = cv2.adaptiveThreshold(warped_blur, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)

    # ---------- 裁剪内部区域（去除2cm边框） ----------
    border_pix = int(2 * scale)
    inner = warped_thresh[border_pix : dst_h - border_pix,
                          border_pix : dst_w - border_pix]
    # 将inner转为彩色（供绘制函数使用）
    inner_color = cv2.cvtColor(inner, cv2.COLOR_GRAY2BGR)

    # ---------- 在内部区域检测正方形 ----------
    # 查找内部轮廓
    inner_contours, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reset_square_collection()   # 清空上一帧的正方形列表

    for cnt in inner_contours:
        area = cv2.contourArea(cnt)
        if area < 50:   # 忽略过小噪声
            continue
        # 使用与原代码相同的逼近参数
        peri_cnt = cv2.arcLength(cnt, True)
        epsilon = 0.003 * peri_cnt
        approx_obj = cv2.approxPolyDP(cnt, epsilon, True)
        # 调用完美边检测
        img_temp, vertex_status, edge_status, square_groups, orig_verts = detect_perfect_edges_and_squares(
            approx_obj, inner_color, step_pixels=9
        )
        # 检查是否有未核验顶点
        unverified = [i for i, status in vertex_status.items() if status == "未核验"]
        if unverified:
            inner_color, square_groups = orphan_corner_method(
                inner_color, approx_obj, unverified, orig_verts, square_groups
            )
        # 注意：检测到的正方形已经通过 add_square_to_collection 加入全局 all_squares

    # ---------- 将检测结果映射回原图 ----------
    squares = get_all_squares()
    for sq in squares:
        # sq.vertices 是 inner 坐标系中的点
        # 转换到 warped 坐标系：加上边框偏移
        warped_pts = np.array(sq.vertices, dtype=np.float32) + [border_pix, border_pix]
        # 逆透视变换到原图
        orig_pts = cv2.perspectiveTransform(warped_pts.reshape(-1,1,2), M_inv).reshape(-1,2).astype(np.int32)
        # 绘制在原图上（蓝色）
        cv2.polylines(img, [orig_pts], isClosed=True, color=(255,0,0), thickness=2)
        # 显示边长（像素值，可转换为cm）
        center = np.mean(orig_pts, axis=0).astype(int)
        length_cm = sq.avg_side_length / scale
        cv2.putText(img, f"{length_cm:.1f}cm", tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    return img

# ==================== 主函数：摄像头实时流（使用 picamera2） ====================
if __name__ == "__main__":
    # ---------- 初始化 picamera2 ----------
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})   # 可调整分辨率
    picam2.configure(config)
    picam2.start()
    print("摄像头已启动，按 'q' 键退出，按 's' 键保存当前帧")

    while True:
        # 捕获 RGB 图像（picamera2 返回 RGB）
        frame_rgb = picam2.capture_array()
        # 转为 BGR 供 OpenCV 使用
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 处理当前帧
        processed = process_frame(frame_bgr)

        # 显示
        cv2.imshow("Square Detection", processed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("saved_detection.jpg", processed)
            print("已保存当前帧")

    # 释放资源
    picam2.stop()
    cv2.destroyAllWindows()